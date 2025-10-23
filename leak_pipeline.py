#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leak pipeline: dataset -> supervised learning (RF) with multi-leak (0..2), 6/12h,
hydraulic timestep 300s, report 600s, sensor subset, parallel generation.

NEW:
- Leak area "scaled_by_D2": A = k_scale * (π/4) * D^2 * jitter
- Controlled distribution of 0/1/2 leaks with an upper cap for 0-leak
- Baseline perturbation to diversify 0-leak episodes
- Sensor subset by fraction or absolute number
- Parallelization with SubprocVecEnv (default 6 workers)
- RF: separate estimators (classifier/regressor)
- Robust inference + top-k with labels
- Utilities to test single episodes
- Flag --infer-only

Dependencies:
  pip install wntr numpy pandas scikit-learn joblib gymnasium stable-baselines3 tqdm
"""

import os
import sys
import copy
import math
import argparse
import warnings
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import wntr
from gymnasium import Env, spaces

# Parallel env for dataset generation
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# ML
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, top_k_accuracy_score
import joblib

warnings.filterwarnings("ignore")

# -----------------------------
# Config & utilities
# -----------------------------

@dataclass
class SimConfig:
    # network / timing
    inp_path: str = "Net3.inp"
    hours: float = 6.0
    hyd_step_s: int = 300
    report_step_s: int = 600
    min_pressure_m: float = 10.0
    req_pressure_m: float = 20.0

    # multi-leak (allowed range)
    min_leaks: int = 0
    max_leaks: int = 2
    allow_same_pipe_multiple_events: bool = False
    min_leak_start_h: float = 0.0
    min_gap_h: float = 0.0

    # class distribution for n_leaks + cap on 0-leak
    allow_zero_leak: bool = True
    p0: float = 0.10
    p1: float = 0.50
    p2: float = 0.40
    max_zero_frac: Optional[float] = 0.10  # None = no cap

    # diversify 0-leak episodes
    baseline_perturb: float = 0.0  # e.g., 0.02 => ±2%

    # sensors
    sensor_fraction: Optional[float] = 1.0  # 0<frac<=1 or integer (exact # of sensors)
    seed: int = 0

    # leak sizing
    leak_area_mode: str = "scaled_by_D2"  # "scaled_by_D2" | "range"
    leak_k_scale: float = 0.05            # fraction of SECTION (π/4 D^2)
    leak_area_jitter: float = 0.2         # ±20% on k_scale
    area_min: float = 1e-4                 # used only if mode="range"
    area_max: float = 8e-4                 # used only if mode="range"
    diameter_units: str = "m"             # "m" | "mm" | "in"

    # dataset generation
    n_samples: int = 2000
    n_workers: int = 6                     # <-- local default: 6 env
    out_csv: str = "leak_supervised_dataset.csv"

    # training
    test_size: float = 0.2
    rf_clf_estimators: int = 300           # classifier
    rf_reg_estimators: int = 300           # regressor
    rf_random_state: int = 0

    # output directory
    out_dir: str = "artifacts"


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_base_network(inp_path: str,
                      hours: float,
                      hyd_step_s: int,
                      report_step_s: int,
                      min_pressure_m: float,
                      req_pressure_m: float) -> wntr.network.model.WaterNetworkModel:
    wn = wntr.network.WaterNetworkModel(inp_path)
    wn.options.hydraulic.demand_model = 'PDD'
    wn.options.time.duration = int(hours * 3600)
    wn.options.time.hydraulic_timestep = int(hyd_step_s)
    wn.options.time.report_timestep = int(report_step_s)
    wn.options.hydraulic.minimum_pressure = float(min_pressure_m)
    wn.options.hydraulic.required_pressure = float(req_pressure_m)
    return wn


def _diam_to_meters(d_val: float, units: str) -> float:
    u = (units or "m").lower()
    if u == "m":
        return float(d_val)
    if u == "mm":
        return float(d_val) / 1000.0
    if u in ("in", "inch", "inches"):
        return float(d_val) * 0.0254
    return float(d_val)


def choose_sensors(wn: wntr.network.model.WaterNetworkModel,
                   sensor_fraction: Optional[float],
                   rnd: np.random.Generator) -> List[str]:
    # sensors = junctions with demand > 0
    demand_nodes = wntr.metrics.population(wn)
    nodes = demand_nodes[demand_nodes > 0].index.tolist()

    if sensor_fraction is None:
        return [str(x) for x in nodes]

    if isinstance(sensor_fraction, float) and 0.0 < sensor_fraction <= 1.0:
        k = max(1, int(round(sensor_fraction * len(nodes))))
    else:
        try:
            k = int(sensor_fraction)
            if k <= 0:
                k = len(nodes)
            k = min(k, len(nodes))
        except Exception:
            k = len(nodes)
    return [str(x) for x in rnd.choice(nodes, size=k, replace=False)]


def _perturb_base_demands(wn: wntr.network.model.WaterNetworkModel,
                          rng_: np.random.Generator,
                          amplitude: float):
    """
    Multiplies junction base demands by a uniform factor in [1-amplitude, 1+amplitude].
    Use only if amplitude>0. Helps diversify 0-leak episodes.
    """
    if amplitude <= 0:
        return
    for jn in wn.junction_name_list:
        j = wn.get_node(jn)
        if j.demand_timeseries_list:
            ts = j.demand_timeseries_list[0]
            if hasattr(ts, "base_value"):
                u = float(rng_.uniform(1.0 - amplitude, 1.0 + amplitude))
                ts.base_value = float(ts.base_value) * u


def sample_n_leaks(cfg: SimConfig, rnd: np.random.Generator) -> int:
    """
    Samples n_leaks in {min_leaks..max_leaks} respecting p0,p1,p2 when possible.
    If 0 is not allowed/desired, renormalizes over the remaining classes.
    """
    allowed = list(range(cfg.min_leaks, cfg.max_leaks + 1))
    cand = []
    if 0 in allowed and cfg.allow_zero_leak:
        cand.append((0, cfg.p0))
    if 1 in allowed:
        cand.append((1, cfg.p1))
    if 2 in allowed:
        cand.append((2, cfg.p2))
    if not cand:
        return int(rnd.integers(cfg.min_leaks, cfg.max_leaks + 1))
    vals, probs = zip(*cand)
    probs = np.array(probs, dtype=float)
    if probs.sum() <= 0:
        return int(rnd.integers(cfg.min_leaks, cfg.max_leaks + 1))
    probs = probs / probs.sum()
    return int(rnd.choice(vals, p=probs))


def sample_leak_schedule(wn: wntr.network.model.WaterNetworkModel,
                         cfg: SimConfig,
                         num_leaks: int,
                         steps: int,
                         step_hours: float,
                         rnd: np.random.Generator) -> List[Tuple[str, float, int]]:
    """
    Returns a list [(pipe_name, area_m2, start_step), ...] sorted by start_step.
    Area computed via:
      - "scaled_by_D2": A = k_scale * (π/4)*D^2 * jitter
      - "range": A ~ U[area_min, area_max]
    """
    pipes = wn.pipe_name_list
    chosen: List[Tuple[str, float, int]] = []
    used_pipes: set = set()
    used_steps: List[int] = []

    valid_steps = list(range(0, steps))
    min_gap_steps = int(round(cfg.min_gap_h / step_hours))

    for _ in range(num_leaks):
        # pipe
        if cfg.allow_same_pipe_multiple_events:
            pipe = str(rnd.choice(pipes))
        else:
            avail = [p for p in pipes if p not in used_pipes]
            pipe = str(rnd.choice(avail if avail else pipes))
            used_pipes.add(pipe)

        # area
        if cfg.leak_area_mode.lower() == "scaled_by_d2":
            pipe_obj = wn.get_link(pipe)
            D_raw = float(pipe_obj.diameter)
            D_m = _diam_to_meters(D_raw, cfg.diameter_units)
            jitter_u = float(rnd.uniform(1.0 - cfg.leak_area_jitter, 1.0 + cfg.leak_area_jitter))
            area = cfg.leak_k_scale * (math.pi * 0.25 * D_m * D_m) * jitter_u
        else:
            area = float(rnd.uniform(cfg.area_min, cfg.area_max))

        # start step with minimal gap enforcement
        candidates = valid_steps
        if min_gap_steps > 0 and used_steps:
            good = [s for s in candidates if all(abs(s - u) >= min_gap_steps for u in used_steps)]
            candidates = good or valid_steps
        start_step = int(rnd.choice(candidates))
        used_steps.append(start_step)

        chosen.append((pipe, area, start_step))

    chosen.sort(key=lambda t: t[2])
    return chosen


def apply_leaks_to_network(wn: wntr.network.model.WaterNetworkModel,
                           schedule: List[Tuple[str, float, int]],
                           report_step_s: int):
    """
    Applies leaks: for each pipe, splits the pipe to insert a node and calls .add_leak
    with start_time aligned to the report grid.
    """
    for (pipe, area, start_step) in schedule:
        split_node = f"{pipe}_leak_node_{start_step}"
        new_pipe = f"{pipe}_B_{start_step}"
        wn = wntr.morph.split_pipe(wn,
                                   pipe_name_to_split=pipe,
                                   new_pipe_name=new_pipe,
                                   new_junction_name=split_node)
        wn.get_node(split_node).add_leak(
            wn,
            area=area,
            start_time=int(start_step * report_step_s),
            end_time=wn.options.time.duration
        )
    return wn

# -----------------------------
# Gym Env for parallel generation
# -----------------------------

class LeakSimEnv(Env):
    """
    Minimal Env to call simulate_once() via VecEnv and produce dataset rows.
    This is not RL: VecEnv is used only to parallelize simulation.
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg: SimConfig, seed_offset: int = 0):
        super().__init__()
        self.cfg = cfg
        self.rnd = rng(cfg.seed + seed_offset)

        # base network & sensors
        self.wn_template = make_base_network(cfg.inp_path, cfg.hours,
                                             cfg.hyd_step_s, cfg.report_step_s,
                                             cfg.min_pressure_m, cfg.req_pressure_m)
        self.sensor_list = choose_sensors(self.wn_template, cfg.sensor_fraction, self.rnd)

        # temporal dimensions
        self.step_hours = cfg.report_step_s / 3600.0
        self.steps = int(round(cfg.hours / self.step_hours))
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(1)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, True, False, {}

    # ------ custom method invoked via env_method ------
    def simulate_once(self) -> Optional[Dict[str, Any]]:
        cfg = self.cfg
        wn = copy.deepcopy(self.wn_template)

        # number of leaks with controlled distribution
        num_leaks = sample_n_leaks(cfg, self.rnd)

        # if 0 leaks and you want to avoid "identical" samples, slightly perturb demands
        if num_leaks == 0 and cfg.baseline_perturb > 0.0:
            _perturb_base_demands(wn, self.rnd, cfg.baseline_perturb)

        # schedule
        if num_leaks == 0:
            schedule = []
        else:
            schedule = sample_leak_schedule(wn, cfg, num_leaks, self.steps, self.step_hours, self.rnd)

        # apply leaks and simulate
        wn_eff = apply_leaks_to_network(wn, schedule, cfg.report_step_s) if schedule else wn
        sim = wntr.sim.WNTRSimulator(wn_eff)
        try:
            results = sim.run_sim()
        except Exception:
            return None  # discard this sample

        # pressures for selected sensors
        press_df = results.node['pressure'][self.sensor_list]  # (time_index, sensors)
        press_df.columns = press_df.columns.map(lambda x: str(x).strip())
        sens_sorted = sorted([str(s).strip() for s in self.sensor_list])
        press_df = press_df[sens_sorted]

        feat = press_df.to_numpy(dtype=float)                   # shape: [steps, n_sens]
        if feat.shape[0] != self.steps:
            feat = feat[:self.steps, :]
        feature_flat = feat.reshape(-1)                         # [steps*n_sens]

        # features
        row: Dict[str, Any] = {}
        for t in range(self.steps):
            for j, n in enumerate(sens_sorted):
                row[f"P_{n}_t{t}"] = feature_flat[t*len(sens_sorted) + j]

        # global targets
        row["n_leaks"] = num_leaks

        # leak slots (max 2)
        for k in range(2):
            if k < num_leaks:
                p, a, s = schedule[k]
                row[f"leak{k+1}_pipe"] = p
                row[f"leak{k+1}_area"] = float(a)
                row[f"leak{k+1}_start_step"] = int(s)
            else:
                row[f"leak{k+1}_pipe"] = "None"
                row[f"leak{k+1}_area"] = 0.0
                row[f"leak{k+1}_start_step"] = -1

        # metadata
        row["_steps"] = self.steps
        row["_report_step_s"] = self.cfg.report_step_s
        row["_hours"] = self.cfg.hours
        row["_sensors_used"] = "|".join(sens_sorted)

        return row

# -----------------------------
# Dataset generation (with 0-leak cap)
# -----------------------------

def _make_env(i: int, cfg: SimConfig):
    """Picklable factory for SubprocVecEnv (Windows/spawn compatible)."""
    def _thunk():
        return LeakSimEnv(cfg, seed_offset=10_000 + i)
    return _thunk


def generate_dataset(cfg: SimConfig) -> pd.DataFrame:
    os.makedirs(os.path.dirname(cfg.out_csv) or ".", exist_ok=True)

    if cfg.n_workers > 1:
        vec = SubprocVecEnv([_make_env(i, cfg) for i in range(cfg.n_workers)])
    else:
        vec = DummyVecEnv([_make_env(0, cfg)])

    rows: List[Dict[str, Any]] = []
    zero_count = 0
    max_zero = None if cfg.max_zero_frac is None else int(round(cfg.max_zero_frac * cfg.n_samples))
    pbar = tqdm(total=cfg.n_samples, desc="Simulating")

    try:
        while len(rows) < cfg.n_samples:
            batch = vec.env_method("simulate_once")
            for b in batch:
                if b is None:
                    continue
                n_leaks_here = int(b.get("n_leaks", 0))
                if n_leaks_here == 0 and max_zero is not None and zero_count >= max_zero:
                    continue  # discard beyond the cap
                rows.append(b)
                if n_leaks_here == 0:
                    zero_count += 1
                pbar.update(1)
                if len(rows) >= cfg.n_samples:
                    break
    finally:
        # SAFE WORKER SHUTDOWN
        if hasattr(vec, "close"):
            vec.close()

    pbar.close()
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_csv, index=False)
    return df

# -----------------------------
# Model training
# -----------------------------

@dataclass
class ModelArtifacts:
    le_pipe1_path: str
    le_pipe2_path: str
    feature_cols_path: str
    meta_path: str
    clf_nleaks_path: str
    clf_pipe1_path: str
    clf_pipe2_path: str
    clf_t1_path: str
    clf_t2_path: str
    reg_a1_path: str
    reg_a2_path: str


def train_models(cfg: SimConfig, df: pd.DataFrame) -> ModelArtifacts:
    os.makedirs(cfg.out_dir, exist_ok=True)

    # feature columns = all P_* fields (ordered by t then by sensor name)
    feature_cols = sorted([c for c in df.columns if c.startswith("P_")],
                          key=lambda x: (int(x.split("_t")[-1]), x.split("_t")[0]))
    X = df[feature_cols].values
    y_n = df["n_leaks"].values  # 0/1/2

    X_train, X_test, y_n_train, y_n_test, df_train, df_test = train_test_split(
        X, y_n, df, test_size=cfg.test_size, random_state=cfg.rf_random_state, stratify=y_n
    )

    # ----- 1) #leaks -----
    clf_n = RandomForestClassifier(
        n_estimators=cfg.rf_clf_estimators, class_weight="balanced",
        random_state=cfg.rf_random_state, n_jobs=-1
    )
    clf_n.fit(X_train, y_n_train)
    y_n_pred = clf_n.predict(X_test)
    print("\n=== Classificazione numero di leak (0/1/2) ===")
    print(classification_report(y_n_test, y_n_pred, digits=4))

    # Helper for slot k (1..2)
    def subset_k(df_split: pd.DataFrame, k: int):
        mask = df_split["n_leaks"] >= k
        Xk = df_split.loc[mask, feature_cols].values
        y_pipe = df_split.loc[mask, f"leak{k}_pipe"].values
        y_area = df_split.loc[mask, f"leak{k}_area"].values
        y_t = df_split.loc[mask, f"leak{k}_start_step"].values
        return Xk, y_pipe, y_area, y_t

    steps = int(df["_steps"].iloc[0])

    # ----- 2) Pipe/time/area slot 1 -----
    X1_tr, ypipe1_tr, yarea1_tr, yt1_tr = subset_k(df_train, 1)
    X1_te, ypipe1_te, yarea1_te, yt1_te = subset_k(df_test, 1)

    le1 = LabelEncoder().fit(ypipe1_tr)
    ypipe1_tr_enc = le1.transform(ypipe1_tr)
    ypipe1_te_enc = le1.transform(ypipe1_te)

    clf_p1 = RandomForestClassifier(
        n_estimators=cfg.rf_clf_estimators, class_weight="balanced",
        random_state=cfg.rf_random_state, n_jobs=-1
    )
    clf_p1.fit(X1_tr, ypipe1_tr_enc)
    yp1_pred = clf_p1.predict(X1_te)
    print("\n=== Classificazione pipe leak #1 ===")
    print(classification_report(ypipe1_te_enc, yp1_pred, digits=4))
    if hasattr(clf_p1, "predict_proba"):
        proba1 = clf_p1.predict_proba(X1_te)
        top3 = top_k_accuracy_score(ypipe1_te_enc, proba1, k=3, labels=np.arange(proba1.shape[1]))
        print(f"Top-3 accuracy pipe#1: {top3:.4f}")

    clf_t1 = RandomForestClassifier(
        n_estimators=cfg.rf_clf_estimators, random_state=cfg.rf_random_state, n_jobs=-1
    )
    clf_t1.fit(X1_tr, yt1_tr)
    yt1_pred = clf_t1.predict(X1_te)
    print("\n=== Classificazione start step leak #1 ===")
    print(f"Accuracy: {(yt1_pred == yt1_te).mean():.4f}")

    reg_a1 = RandomForestRegressor(
        n_estimators=cfg.rf_reg_estimators, random_state=cfg.rf_random_state, n_jobs=-1
    )
    reg_a1.fit(X1_tr, yarea1_tr)
    a1_pred = reg_a1.predict(X1_te)
    print("\n=== Regressione area leak #1 ===")
    mse1 = mean_squared_error(yarea1_te, a1_pred)
    print(f"MSE: {mse1:.6f}")

    # ----- 3) Pipe/time/area slot 2 -----
    X2_tr, ypipe2_tr, yarea2_tr, yt2_tr = subset_k(df_train, 2)
    X2_te, ypipe2_te, yarea2_te, yt2_te = subset_k(df_test, 2)

    if len(X2_tr) > 0 and len(X2_te) > 0:
        le2 = LabelEncoder().fit(ypipe2_tr)
        ypipe2_tr_enc = le2.transform(ypipe2_tr)
        ypipe2_te_enc = le2.transform(ypipe2_te)

        clf_p2 = RandomForestClassifier(
            n_estimators=cfg.rf_clf_estimators, class_weight="balanced",
            random_state=cfg.rf_random_state, n_jobs=-1
        )
        clf_p2.fit(X2_tr, ypipe2_tr_enc)
        yp2_pred = clf_p2.predict(X2_te)
        print("\n=== Classificazione pipe leak #2 ===")
        print(classification_report(ypipe2_te_enc, yp2_pred, digits=4))
        if hasattr(clf_p2, "predict_proba"):
            proba2 = clf_p2.predict_proba(X2_te)
            top3_2 = top_k_accuracy_score(ypipe2_te_enc, proba2, k=3, labels=np.arange(proba2.shape[1]))
            print(f"Top-3 accuracy pipe#2: {top3_2:.4f}")

        clf_t2 = RandomForestClassifier(
            n_estimators=cfg.rf_clf_estimators, random_state=cfg.rf_random_state, n_jobs=-1
        )
        clf_t2.fit(X2_tr, yt2_tr)
        yt2_pred = clf_t2.predict(X2_te)
        print("\n=== Classificazione start step leak #2 ===")
        print(f"Accuracy: {(yt2_pred == yt2_te).mean():.4f}")

        reg_a2 = RandomForestRegressor(
            n_estimators=cfg.rf_reg_estimators, random_state=cfg.rf_random_state, n_jobs=-1
        )
        reg_a2.fit(X2_tr, yarea2_tr)
        a2_pred = reg_a2.predict(X2_te)
        print("\n=== Regressione area leak #2 ===")
        mse2 = mean_squared_error(yarea2_te, a2_pred)
        print(f"MSE: {mse2:.6f}")
    else:
        print("\n[AVVISO] Pochi esempi con 2 leak: salvo modelli 'stub' per slot #2.")
        le2 = LabelEncoder().fit(["None"])
        # minimal stub models to keep the pipeline consistent
        clf_p2 = RandomForestClassifier(n_estimators=1, random_state=cfg.rf_random_state).fit(
            X1_tr[:1], np.zeros(1, dtype=int)
        )
        clf_t2 = RandomForestClassifier(n_estimators=1, random_state=cfg.rf_random_state).fit(
            X1_tr[:1], np.zeros(1, dtype=int)
        )
        reg_a2 = RandomForestRegressor(n_estimators=1, random_state=cfg.rf_random_state).fit(
            X1_tr[:1], np.zeros(1, dtype=float)
        )

    # ------- save artifacts -------
    paths = ModelArtifacts(
        le_pipe1_path=os.path.join(cfg.out_dir, "le_pipe1.joblib"),
        le_pipe2_path=os.path.join(cfg.out_dir, "le_pipe2.joblib"),
        feature_cols_path=os.path.join(cfg.out_dir, "feature_cols.joblib"),
        meta_path=os.path.join(cfg.out_dir, "meta.json"),
        clf_nleaks_path=os.path.join(cfg.out_dir, "rf_clf_nleaks.joblib"),
        clf_pipe1_path=os.path.join(cfg.out_dir, "rf_clf_pipe1.joblib"),
        clf_pipe2_path=os.path.join(cfg.out_dir, "rf_clf_pipe2.joblib"),
        clf_t1_path=os.path.join(cfg.out_dir, "rf_clf_t1.joblib"),
        clf_t2_path=os.path.join(cfg.out_dir, "rf_clf_t2.joblib"),
        reg_a1_path=os.path.join(cfg.out_dir, "rf_reg_a1.joblib"),
        reg_a2_path=os.path.join(cfg.out_dir, "rf_reg_a2.joblib"),
    )

    joblib.dump(le1, paths.le_pipe1_path)
    joblib.dump(le2, paths.le_pipe2_path)
    joblib.dump(feature_cols, paths.feature_cols_path)

    meta = {
        "cfg": asdict(cfg),
        "steps": int(df["_steps"].iloc[0]),
        "report_step_s": int(df["_report_step_s"].iloc[0]),
        "hours": float(df["_hours"].iloc[0]),
        "hyd_step_s": int(cfg.hyd_step_s),  # <<< SAVED IN META
        "sensors": df["_sensors_used"].iloc[0].split("|"),
        # useful class counters
        "count_n0": int((df["n_leaks"] == 0).sum()),
        "count_n1": int((df["n_leaks"] == 1).sum()),
        "count_n2": int((df["n_leaks"] == 2).sum()),
    }
    pd.Series(meta).to_json(paths.meta_path)

    joblib.dump(clf_n, paths.clf_nleaks_path)
    joblib.dump(clf_p1, paths.clf_pipe1_path)
    joblib.dump(clf_t1, paths.clf_t1_path)
    joblib.dump(reg_a1, paths.reg_a1_path)
    joblib.dump(clf_p2, paths.clf_pipe2_path)
    joblib.dump(clf_t2, paths.clf_t2_path)
    joblib.dump(reg_a2, paths.reg_a2_path)

    print(f"\n💾 Salvati modelli e metadati in: {cfg.out_dir}")
    return paths

# -----------------------------
# Inference (demo) – robust to sensor mismatches
# -----------------------------

def predict_leaks(inp_path: str, artifacts: ModelArtifacts, top_k: int = 3) -> Dict[str, Any]:
    # load
    clf_n = joblib.load(artifacts.clf_nleaks_path)
    clf_p1 = joblib.load(artifacts.clf_pipe1_path)
    clf_t1 = joblib.load(artifacts.clf_t1_path)
    reg_a1 = joblib.load(artifacts.reg_a1_path)
    clf_p2 = joblib.load(artifacts.clf_pipe2_path)
    clf_t2 = joblib.load(artifacts.clf_t2_path)
    reg_a2 = joblib.load(artifacts.reg_a2_path)
    le1 = joblib.load(artifacts.le_pipe1_path)
    le2 = joblib.load(artifacts.le_pipe2_path)
    feature_cols: List[str] = joblib.load(artifacts.feature_cols_path)
    meta = pd.read_json(artifacts.meta_path, typ='series')
    sensors = list(meta["sensors"])
    hours = float(meta["hours"])
    report_step_s = int(meta["report_step_s"])
    hyd_step_s = int(meta["hyd_step_s"])  # <<< REUSE TRAINING TIMESTEP

    # build network
    wn = make_base_network(inp_path, hours, hyd_step_s, report_step_s, 10.0, 20.0)

    # simulate baseline (no leak)
    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()

    # 1) all pressures with clean string columns
    press_all = results.node['pressure']
    press_all.columns = press_all.columns.map(lambda x: str(x).strip())

    # 2) normalize sensor list
    sensors_str = [str(s).strip() for s in sensors]

    # 3) intersection of available sensors
    have = set(press_all.columns)
    want = list(sensors_str)
    missing = sorted(set(want) - have)
    if missing:
        print(f"[WARN] {len(missing)} sensori mancanti in inferenza (esempio: {missing[:5]}). "
              f"Uso l'intersezione, le feature mancanti verranno riempite a 0.")
    use_cols = [c for c in want if c in have]
    press_df = press_all[use_cols].copy()

    # 4) temporal trim/fill
    steps = int(round(hours * 3600 / report_step_s))
    if press_df.shape[0] < steps:
        press_df = press_df.reindex(range(steps)).ffill().bfill()
    elif press_df.shape[0] > steps:
        press_df = press_df.iloc[:steps, :]

    # 5) rebuild X in the exact feature order; if a column is missing, fill with 0
    X_vec = []
    for col in feature_cols:
        name, t_str = col.split("_t")
        node = name.split("P_")[1].strip()
        t = int(t_str)
        if node in press_df.columns:
            X_vec.append(float(press_df.iloc[t][node]))
        else:
            X_vec.append(0.0)
    X = np.array(X_vec, dtype=float).reshape(1, -1)

    # #leaks
    n_pred = int(clf_n.predict(X)[0])
    prob_n = clf_n.predict_proba(X)[0] if hasattr(clf_n, "predict_proba") else None

    out: Dict[str, Any] = {"n_leaks_pred": n_pred, "n_leaks_proba": prob_n}

    def topk_from_clf(clf, le, X, k):
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[0]
            idx = np.argsort(proba)[::-1][:k]
            labs = le.inverse_transform(idx)
            return [(labs[i], float(proba[idx[i]])) for i in range(len(idx))]
        pred = le.inverse_transform([int(clf.predict(X)[0])])[0]
        return [(pred, None)]

    if n_pred >= 1:
        top1 = topk_from_clf(clf_p1, le1, X, top_k)
        t1 = int(clf_t1.predict(X)[0])
        a1 = float(reg_a1.predict(X)[0])
        out.update({
            "leak1_pipe_topk": top1,
            "leak1_start_step": t1,
            "leak1_start_time_min": t1 * report_step_s / 60.0,
            "leak1_area_pred": a1
        })
    if n_pred >= 2:
        top2 = topk_from_clf(clf_p2, le2, X, top_k)
        t2 = int(clf_t2.predict(X)[0])
        a2 = float(reg_a2.predict(X)[0])
        out.update({
            "leak2_pipe_topk": top2,
            "leak2_start_step": t2,
            "leak2_start_time_min": t2 * report_step_s / 60.0,
            "leak2_area_pred": a2
        })
    return out

# -----------------------------
# Single-episode testing utilities
# -----------------------------

def _area_scaled_by_D2_for_pipe(wn, pipe_name: str, k_scale: float, jitter: float, diameter_units: str) -> float:
    pipe_obj = wn.get_link(pipe_name)
    D_raw = float(pipe_obj.diameter)
    D_m = _diam_to_meters(D_raw, diameter_units)
    u = np.random.default_rng().uniform(1.0 - jitter, 1.0 + jitter)
    return k_scale * (math.pi * 0.25 * D_m * D_m) * float(u)


def simulate_one_leak_and_predict(inp_path: str,
                                  artifacts: ModelArtifacts,
                                  leak_pipe: Optional[str] = None,
                                  start_step: Optional[int] = None,
                                  k_scale: float = 0.05,
                                  jitter: float = 0.2,
                                  diameter_units: str = "m",
                                  top_k: int = 3) -> Dict[str, Any]:
    """
    Creates 1 scenario with 1 leak (specified or random pipe/start), simulates and predicts.
    Returns ground truth + predictions (slot #1).
    """
    feature_cols: List[str] = joblib.load(artifacts.feature_cols_path)
    meta = pd.read_json(artifacts.meta_path, typ='series')
    sensors = list(meta["sensors"])
    hours = float(meta["hours"])
    report_step_s = int(meta["report_step_s"])
    hyd_step_s = int(meta["hyd_step_s"])  # <<< REUSE TRAINING TIMESTEP
    steps = int(round(hours*3600 / report_step_s))

    # models
    clf_n = joblib.load(artifacts.clf_nleaks_path)
    clf_p1 = joblib.load(artifacts.clf_pipe1_path)
    clf_t1 = joblib.load(artifacts.clf_t1_path)
    reg_a1 = joblib.load(artifacts.reg_a1_path)
    le1 = joblib.load(artifacts.le_pipe1_path)

    # base network
    wn = make_base_network(inp_path, hours, hyd_step_s, report_step_s, 10.0, 20.0)
    pipes = wn.pipe_name_list
    rng_local = np.random.default_rng()

    if leak_pipe is None:
        leak_pipe = str(rng_local.choice(pipes))
    if start_step is None:
        start_step = int(rng_local.integers(0, steps))

    area = _area_scaled_by_D2_for_pipe(wn, leak_pipe, k_scale=k_scale, jitter=jitter, diameter_units=diameter_units)
    schedule = [(leak_pipe, area, start_step)]
    wn_leak = apply_leaks_to_network(wn, schedule, report_step_s)

    sim = wntr.sim.WNTRSimulator(wn_leak)
    results = sim.run_sim()

    press_all = results.node['pressure']
    press_all.columns = press_all.columns.map(lambda x: str(x).strip())
    sensors_str = [str(s).strip() for s in sensors]
    have = set(press_all.columns)
    use_cols = [c for c in sensors_str if c in have]
    press_df = press_all[use_cols].copy()
    if press_df.shape[0] < steps:
        press_df = press_df.reindex(range(steps)).ffill().bfill()
    elif press_df.shape[0] > steps:
        press_df = press_df.iloc[:steps, :]

    X_vec = []
    for col in feature_cols:
        name, t_str = col.split("_t")
        node = name.split("P_")[1].strip()
        t = int(t_str)
        X_vec.append(float(press_df.iloc[t][node]) if node in press_df.columns else 0.0)
    X = np.array(X_vec, dtype=float).reshape(1, -1)

    def topk_from_clf(clf, le, X, k):
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[0]
            idx = np.argsort(proba)[::-1][:k]
            labs = le.inverse_transform(idx)
            return [(labs[i], float(proba[idx[i]])) for i in range(len(idx))]
        pred = le.inverse_transform([int(clf.predict(X)[0])])[0]
        return [(pred, None)]

    n_pred = int(clf_n.predict(X)[0])
    top1 = topk_from_clf(clf_p1, le1, X, top_k)
    t1 = int(clf_t1.predict(X)[0])
    a1 = float(reg_a1.predict(X)[0])

    return {
        "ground_truth": {
            "pipe": leak_pipe,
            "start_step": start_step,
            "start_time_min": start_step * report_step_s / 60.0,
            "area_m2": area
        },
        "prediction": {
            "n_leaks": n_pred,
            "leak1_pipe_topk": top1,
            "leak1_start_step": t1,
            "leak1_start_time_min": t1 * report_step_s / 60.0,
            "leak1_area_pred": a1
        }
    }


def run_many_simulations_and_report(inp_path: str,
                                    artifacts: ModelArtifacts,
                                    n: int = 20,
                                    top_k: int = 3,
                                    k_scale: float = 0.05,
                                    jitter: float = 0.2,
                                    diameter_units: str = "m") -> pd.DataFrame:
    """
    Runs n scenarios (each with 1 random leak), predicts and prints a table
    with ground truth vs prediction + hit@k for the pipe and start_step correctness.
    """
    rows = []
    hits_topk = 0
    hits_t = 0
    for i in range(n):
        res = simulate_one_leak_and_predict(inp_path, artifacts, top_k=top_k,
                                            k_scale=k_scale, jitter=jitter,
                                            diameter_units=diameter_units)
        gt = res["ground_truth"]
        pr = res["prediction"]

        pred_pipe = pr["leak1_pipe_topk"][0][0]
        topk_list = [p for p, _ in pr["leak1_pipe_topk"]]
        hit_k = int(gt["pipe"] in topk_list)
        hit_t = int(gt["start_step"] == pr["leak1_start_step"])
        hits_topk += hit_k
        hits_t += hit_t

        rows.append({
            "gt_pipe": gt["pipe"],
            "pred_pipe_top1": pred_pipe,
            "pred_pipe_topk": topk_list,
            "hit_pipe@k": hit_k,
            "gt_start_step": gt["start_step"],
            "pred_start_step": pr["leak1_start_step"],
            "hit_start": hit_t,
            "gt_area_m2": gt["area_m2"],
            "pred_area_m2": pr["leak1_area_pred"]
        })

    df_out = pd.DataFrame(rows)
    k_acc = hits_topk / n
    t_acc = hits_t / n
    print(f"\n== Riepilogo su {n} simulazioni ==")
    print(f"Pipe hit@{top_k}: {k_acc:.3f}")
    print(f"Start step accuracy: {t_acc:.3f}")
    return df_out

# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leak dataset & supervised learning pipeline (WNTR + RF)")

    # network/timing
    p.add_argument("--inp", type=str, default="Net3.inp", help="Percorso file EPANET .inp")
    p.add_argument("--hours", type=float, default=6.0, help="Durata simulazione (tipicamente 6 o 12)")
    p.add_argument("--hyd-step", type=int, default=300, help="Hydraulic timestep [s] (default 300)")
    p.add_argument("--report-step", type=int, default=600, help="Report timestep [s] (default 600)")

    # leak sizing
    p.add_argument("--leak-area-mode", type=str, default="scaled_by_D2",
                   choices=["scaled_by_D2", "range"], help="Modo di calcolo area leak")
    p.add_argument("--leak-k-scale", type=float, default=0.05,
                   help="Frazione della SEZIONE tubo (π/4 D^2) per 'scaled_by_D2'")
    p.add_argument("--leak-area-jitter", type=float, default=0.2,
                   help="± frazione di jitter su k_scale (es. 0.2 => ±20%)")
    p.add_argument("--diameter-units", type=str, default="m",
                   choices=["m", "mm", "in"], help="Unità dei diametri pipe nel .inp")
    p.add_argument("--area-min", type=float, default=1e-4,
                   help="Area minima [m^2] (solo se mode='range')")
    p.add_argument("--area-max", type=float, default=8e-4,
                   help="Area massima [m^2] (solo se mode='range')")

    # multi-leak (range)
    p.add_argument("--min-leaks", type=int, default=0, help="Min numero leak per episodio")
    p.add_argument("--max-leaks", type=int, default=2, help="Max numero leak per episodio")
    p.add_argument("--allow-same-pipe", action="store_true", help="Consenti eventi multipli sulla stessa pipe")
    p.add_argument("--min-gap-h", type=float, default=0.0, help="Gap minimo tra start dei leak [ore]")

    # class distribution + 0-leak cap + perturb
    p.add_argument("--allow-zero-leak", action="store_true",
                   help="Permetti anche episodi senza perdite (classe 0 leak)")
    p.add_argument("--p0", type=float, default=0.10, help="Probabilità di 0 leak (se consentito)")
    p.add_argument("--p1", type=float, default=0.50, help="Probabilità di 1 leak")
    p.add_argument("--p2", type=float, default=0.40, help="Probabilità di 2 leak")
    p.add_argument("--max-zero-frac", type=float, default=0.10,
                   help="Tetto massimo frazione di esempi 0-leak nel dataset (0–1). Usa -1 per nessun tetto.")
    p.add_argument("--baseline-perturb", type=float, default=0.0,
                   help="± perturbazione sulle domande base quando n_leaks=0 (es. 0.02 = ±2%)")

    # sensors
    p.add_argument("--sensor-frac", type=float, default=1.0,
                   help="Frazione sensori (0<frac<=1) oppure numero se usi --sensor-num")
    p.add_argument("--sensor-num", type=int, default=None, help="Numero esatto di sensori (prioritario su --sensor-frac)")

    # parallel dataset
    p.add_argument("--samples", type=int, default=2000, help="Numero di campioni da simulare")
    p.add_argument("--workers", type=int, default=6, help="Numero di processi paralleli per simulazione")  # <-- default 6
    p.add_argument("--out-csv", type=str, default="leak_supervised_dataset.csv", help="File CSV dataset")

    # training
    p.add_argument("--out-dir", type=str, default="artifacts", help="Directory salvataggio modelli")
    p.add_argument("--seed", type=int, default=0, help="Seed riproducibilità")
    p.add_argument("--rf-clf-estimators", type=int, default=300, help="# alberi RF per i classificatori")
    p.add_argument("--rf-reg-estimators", type=int, default=300, help="# alberi RF per i regressori")
    p.add_argument("--test-size", type=float, default=0.2, help="Quota di test set (default 0.2)")

    # inference only
    p.add_argument("--infer-only", action="store_true",
                   help="Esegui solo l'inferenza demo (step 3/3) caricando i modelli da --out-dir")

    return p.parse_args()


def main():
    args = parse_args()

    # sensors: --sensor-num has priority
    if args.sensor_num is not None:
        sensor_fraction = float(args.sensor_num)  # the resolver also accepts integers
    else:
        sensor_fraction = float(args.sensor_frac)

    cfg = SimConfig(
        inp_path=args.inp,
        hours=float(args.hours),
        hyd_step_s=int(args.hyd_step),
        report_step_s=int(args.report_step),
        min_pressure_m=10.0,
        req_pressure_m=20.0,
        min_leaks=int(args.min_leaks),
        max_leaks=int(args.max_leaks),
        allow_same_pipe_multiple_events=bool(args.allow_same_pipe),
        min_leak_start_h=0.0,
        min_gap_h=float(args.min_gap_h),
        sensor_fraction=sensor_fraction,
        seed=int(args.seed),
        n_samples=int(args.samples),
        n_workers=int(args.workers),
        out_csv=str(args.out_csv),
        out_dir=str(args.out_dir),
        test_size=float(args.test_size),

        # class distribution
        allow_zero_leak=bool(args.allow_zero_leak),
        p0=float(args.p0),
        p1=float(args.p1),
        p2=float(args.p2),
        max_zero_frac=(None if args.max_zero_frac is not None and args.max_zero_frac < 0 else float(args.max_zero_frac)),
        baseline_perturb=float(args.baseline_perturb),

        # sizing
        leak_area_mode=str(args.leak_area_mode),
        leak_k_scale=float(args.leak_k_scale),
        leak_area_jitter=float(args.leak_area_jitter),
        area_min=float(args.area_min),
        area_max=float(args.area_max),
        diameter_units=str(args.diameter_units),

        # RF
        rf_clf_estimators=int(args.rf_clf_estimators),
        rf_reg_estimators=int(args.rf_reg_estimators),
    )

    # If requested: inference-only demo (step 3/3)
    if args.infer_only:
        print("== Inferenza solo ==")
        print("Carico artifact da:", cfg.out_dir)
        artifacts = ModelArtifacts(
            le_pipe1_path=os.path.join(cfg.out_dir, "le_pipe1.joblib"),
            le_pipe2_path=os.path.join(cfg.out_dir, "le_pipe2.joblib"),
            feature_cols_path=os.path.join(cfg.out_dir, "feature_cols.joblib"),
            meta_path=os.path.join(cfg.out_dir, "meta.json"),
            clf_nleaks_path=os.path.join(cfg.out_dir, "rf_clf_nleaks.joblib"),
            clf_pipe1_path=os.path.join(cfg.out_dir, "rf_clf_pipe1.joblib"),
            clf_pipe2_path=os.path.join(cfg.out_dir, "rf_clf_pipe2.joblib"),
            clf_t1_path=os.path.join(cfg.out_dir, "rf_clf_t1.joblib"),
            clf_t2_path=os.path.join(cfg.out_dir, "rf_clf_t2.joblib"),
            reg_a1_path=os.path.join(cfg.out_dir, "rf_reg_a1.joblib"),
            reg_a2_path=os.path.join(cfg.out_dir, "rf_reg_a2.joblib"),
        )
        out = predict_leaks(cfg.inp_path, artifacts, top_k=3)
        print("Predizione demo:", out)
        print("Fatto.")
        return

    print("== Config ==")
    for k, v in asdict(cfg).items():
        print(f"{k}: {v}")

    # 1) Dataset
    print("\n[1/3] Generazione dataset...")
    df = generate_dataset(cfg)
    # class counts report
    n0 = int((df["n_leaks"] == 0).sum())
    n1 = int((df["n_leaks"] == 1).sum())
    n2 = int((df["n_leaks"] == 2).sum())
    print(f"Class counts -> n0: {n0}, n1: {n1}, n2: {n2}")
    print(f"✅ Dataset salvato: {cfg.out_csv}  ({len(df)} righe)")

    # 2) Training
    print("\n[2/3] Training modelli...")
    artifacts = train_models(cfg, df)

    # 3) Quick inference test
    print("\n[3/3] Inferenza demo...")
    out = predict_leaks(cfg.inp_path, artifacts, top_k=3)
    print("Predizione demo:", out)
    print("\nFatto.")


if __name__ == "__main__":
    main()
