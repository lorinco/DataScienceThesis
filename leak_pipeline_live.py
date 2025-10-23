#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
leak_pipeline_live.py
Supervised, causal, sliding-window pipeline for live single-leak detection/localization/area.

Steps:
1) Baseline simulation (no leak) -> reference pressures
2) Unit-signature precompute (per pipe, small area, start at t=0) -> signatures (sensors x W) [PARALLEL]
3) Dataset generation [PARALLEL]:
   - Simulate episodes with 0 or 1 leak
   - Build causal sliding windows of length W (stride 1)
   - Features = [cosine similarity with each pipe's signature] + global stats of residuals
   - Labels:
       detector_y ∈ {0,1} (1 if leak has started by window end)
       pipe_y ∈ {pipe_id} (only for positive windows)
       area_y ∈ ℝ⁺      (only for positive windows)
4) Train models: RF detector (binary), RF pipe (multiclass), RF area (regression)
5) Live inference loop with hysteresis and warm-up.

Dependencies:
  pip install wntr numpy pandas scikit-learn joblib tqdm
"""

import os
import math
import json
import argparse
import warnings
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import wntr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, average_precision_score,
                             roc_auc_score, mean_absolute_error)
import joblib
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")


# -----------------------------
# Config
# -----------------------------

@dataclass
class LiveConfig:
    # Network/time
    inp_path: str = "Net3.inp"
    hours: float = 6.0
    hyd_step_s: int = 300
    report_step_s: int = 600

    # Sensors
    sensor_fraction: float = 1.0  # fraction in (0,1] OR integer (exact #) via CLI override
    seed: int = 0

    # Windows
    win: int = 6                  # window length in steps (e.g., 6 -> last 60 min if report=10 min)
    warmup_steps: int = 3         # ignore alarms before this
    stride: int = 1

    # Simulation episodes (dataset)
    n_episodes: int = 1000
    p_leak: float = 0.5           # probability of 1 leak vs 0-leak per episode
    leak_k_scale: float = 0.05    # area = k*(pi/4) D^2 * jitter
    leak_area_jitter: float = 0.2
    diameter_units: str = "m"     # "m" | "mm" | "in"
    sensor_noise_std: float = 0.0 # Gaussian noise std added to measured pressures

    # Unit-signatures
    unit_area: float = 2e-4       # area for unit leak used to compute signatures

    # Models
    rf_trees: int = 300
    rf_random_state: int = 0
    test_size: float = 0.2

    # Output
    out_dir: str = "artifacts_live"
    save_csv: bool = False
    out_csv_dataset: str = "live_windows_dataset.csv"

    # Inference (live)
    alarm_threshold: float = 0.8
    hysteresis: int = 2

    # Parallel
    workers: int = 6              # number of parallel processes


# -----------------------------
# Utils
# -----------------------------

def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_base_network(inp_path: str,
                      hours: float,
                      hyd_step_s: int,
                      report_step_s: int,
                      min_pressure_m: float = 10.0,
                      req_pressure_m: float = 20.0) -> wntr.network.model.WaterNetworkModel:
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
                   sensor_fraction: float,
                   rnd: np.random.Generator) -> List[str]:
    # demand nodes > 0 as sensor candidates
    demand_nodes = wntr.metrics.population(wn)
    nodes = demand_nodes[demand_nodes > 0].index.tolist()

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


def _area_scaled_by_D2_for_pipe(wn, pipe_name: str,
                                k_scale: float, jitter: float,
                                diameter_units: str) -> float:
    pipe_obj = wn.get_link(pipe_name)
    D_raw = float(pipe_obj.diameter)
    D_m = _diam_to_meters(D_raw, diameter_units)
    u = np.random.default_rng().uniform(1.0 - jitter, 1.0 + jitter)
    return float(k_scale * (math.pi * 0.25 * D_m * D_m) * u)


def apply_single_leak(wn: wntr.network.model.WaterNetworkModel,
                      pipe_name: str,
                      area_m2: float,
                      start_step: int,
                      report_step_s: int):
    split_node = f"{pipe_name}_leak_node_{start_step}"
    new_pipe = f"{pipe_name}_B_{start_step}"
    wn = wntr.morph.split_pipe(
        wn,
        pipe_name_to_split=pipe_name,
        new_pipe_name=new_pipe,
        new_junction_name=split_node
    )
    wn.get_node(split_node).add_leak(
        wn,
        area=float(area_m2),
        start_time=int(start_step * report_step_s),
        end_time=wn.options.time.duration
    )
    return wn


def simulate_pressures(wn: wntr.network.model.WaterNetworkModel) -> pd.DataFrame:
    results = wntr.sim.WNTRSimulator(wn).run_sim()
    press = results.node['pressure'].copy()
    press.columns = press.columns.map(lambda x: str(x).strip())
    return press  # index=time, columns=nodes


def add_sensor_noise(df: pd.DataFrame, std: float, rnd: np.random.Generator) -> pd.DataFrame:
    if std <= 0:
        return df
    noise = rnd.normal(loc=0.0, scale=std, size=df.shape)
    return df + noise


# -----------------------------
# Baseline & signatures (PARALLEL)
# -----------------------------

def baseline_pressures(cfg: LiveConfig,
                       sensors: List[str]) -> Tuple[pd.DataFrame, wntr.network.model.WaterNetworkModel]:
    wn = make_base_network(cfg.inp_path, cfg.hours, cfg.hyd_step_s, cfg.report_step_s)
    press = simulate_pressures(wn)
    cols = [c for c in sensors if c in press.columns]
    press = press[cols].copy()
    return press, wn


def _signature_for_pipe(pipe_name: str, cfg: LiveConfig, sensors: List[str]) -> np.ndarray:
    """Worker: compute unit signature for one pipe."""
    # Build a worker-local baseline to avoid shared-state issues
    wn_base = make_base_network(cfg.inp_path, cfg.hours, cfg.hyd_step_s, cfg.report_step_s)
    base_press = simulate_pressures(wn_base)[sensors].copy()

    wn = make_base_network(cfg.inp_path, cfg.hours, cfg.hyd_step_s, cfg.report_step_s)
    wn = apply_single_leak(wn, pipe_name=pipe_name, area_m2=cfg.unit_area, start_step=0,
                           report_step_s=cfg.report_step_s)
    press = simulate_pressures(wn)[sensors].copy()
    resid = (press - base_press).iloc[0:cfg.win, :]
    sig = resid.to_numpy(dtype=np.float32).reshape(-1)
    sig /= (np.linalg.norm(sig) + 1e-12)
    return sig


def compute_unit_signatures(cfg: LiveConfig,
                            wn_template: wntr.network.model.WaterNetworkModel,
                            sensors: List[str]) -> Tuple[np.ndarray, List[str]]:
    pipe_names = list(map(str, wn_template.pipe_name_list))
    # Parallel map over pipes
    sig_list = Parallel(n_jobs=cfg.workers, prefer="processes", verbose=0)(
        delayed(_signature_for_pipe)(p, cfg, sensors) for p in tqdm(pipe_names, desc="Computing unit signatures")
    )
    S = np.vstack(sig_list).astype(np.float32) if sig_list else np.zeros((0, len(sensors)*cfg.win), np.float32)
    return S, pipe_names


# -----------------------------
# Feature engineering
# -----------------------------

def window_features(resid_win: np.ndarray,
                    S: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    resid_win: np array shape (W, SENSORS)
    S: signatures matrix shape (n_pipes, W*SENSORS) with unit-norm rows
    Returns:
      feat = [cosine similarities per pipe] + [global stats...]
    """
    W, NS = resid_win.shape
    x = resid_win.reshape(-1).astype(np.float32)
    # global stats on window (abs)
    absx = np.abs(x)
    stats = {
        "mean_abs": float(absx.mean()),
        "max_abs": float(absx.max() if absx.size else 0.0),
        "l2": float(np.linalg.norm(x)),
        "l1": float(np.linalg.norm(x, ord=1)),
        "linf": float(np.linalg.norm(x, ord=np.inf)) if absx.size else 0.0,
    }
    # cosine similarities with signatures
    x_norm = np.linalg.norm(x) + 1e-12
    if S.size == 0:
        cos = np.zeros((0,), dtype=np.float32)
    else:
        # rows of S are unit-norm by construction
        cos = (S @ (x / x_norm)).astype(np.float32)  # shape (n_pipes,)
    feat = np.concatenate([cos, np.array(list(stats.values()), dtype=np.float32)], axis=0)
    return feat, stats


# -----------------------------
# Dataset generation (PARALLEL)
# -----------------------------

def _simulate_episode_block(block_id: int,
                            n_block: int,
                            cfg: LiveConfig,
                            sensors: List[str],
                            base_press: pd.DataFrame,
                            S: np.ndarray) -> Tuple[List[np.ndarray], List[int], List[Tuple[int,int]],
                                                    List[np.ndarray], List[str],
                                                    List[np.ndarray], List[float]]:
    """Worker: generate n_block episodes and return feature/label lists."""
    rnd = rng(cfg.seed + 1000 + block_id)
    steps = int(round(cfg.hours * 3600 / cfg.report_step_s))
    W = cfg.win

    X_det_blk, y_det_blk, meta_blk = [], [], []
    X_pipe_blk, y_pipe_blk = [], []
    X_area_blk, y_area_blk = [], []

    for ep in range(n_block):
        wn = make_base_network(cfg.inp_path, cfg.hours, cfg.hyd_step_s, cfg.report_step_s)

        is_leak = (rnd.random() < cfg.p_leak)
        leak_pipe = None
        leak_area = 0.0
        leak_start = steps

        if is_leak:
            leak_pipe = str(rnd.choice(wn.pipe_name_list))
            leak_area = _area_scaled_by_D2_for_pipe(wn, leak_pipe,
                                                    cfg.leak_k_scale, cfg.leak_area_jitter,
                                                    cfg.diameter_units)
            leak_start = int(rnd.integers(0, steps))
            wn = apply_single_leak(wn, leak_pipe, leak_area, leak_start, cfg.report_step_s)

        press = simulate_pressures(wn)[sensors].copy()
        if cfg.sensor_noise_std > 0:
            press = add_sensor_noise(press, cfg.sensor_noise_std, rnd)
        resid = press - base_press

        # sliding windows
        for t_end in range(W-1, steps, cfg.stride):
            t_start = t_end - (W-1)
            resid_win = resid.iloc[t_start:t_end+1, :].to_numpy(dtype=np.float32)
            feat, _ = window_features(resid_win, S)

            y_d = int(is_leak and (leak_start <= t_end))
            X_det_blk.append(feat)
            y_det_blk.append(y_d)
            meta_blk.append((block_id, t_end))  # block_id acts like episode-id namespace

            if y_d == 1:
                X_pipe_blk.append(feat); y_pipe_blk.append(leak_pipe)
                X_area_blk.append(feat); y_area_blk.append(float(leak_area))

    return (X_det_blk, y_det_blk, meta_blk,
            X_pipe_blk, y_pipe_blk, X_area_blk, y_area_blk)


def generate_dataset(cfg: LiveConfig) -> Dict[str, Any]:
    rnd = rng(cfg.seed)

    # base network and sensors
    wn0 = make_base_network(cfg.inp_path, cfg.hours, cfg.hyd_step_s, cfg.report_step_s)
    sensors = choose_sensors(wn0, cfg.sensor_fraction, rnd)
    steps = int(round(cfg.hours * 3600 / cfg.report_step_s))
    W = cfg.win

    # baseline (digital twin)
    base_press, wn_template = baseline_pressures(cfg, sensors)

    # signatures per pipe (unit leak) [PARALLEL]
    S, pipe_names = compute_unit_signatures(cfg, wn_template, sensors)

    # --- parallel episode simulation ---
    workers = max(1, int(cfg.workers))
    n_blocks = workers
    base_per_block = cfg.n_episodes // n_blocks
    remainder = cfg.n_episodes % n_blocks
    block_sizes = [base_per_block + (1 if i < remainder else 0) for i in range(n_blocks)]
    block_sizes = [b for b in block_sizes if b > 0]
    n_blocks = len(block_sizes)

    # Dispatch blocks
    results = Parallel(n_jobs=workers, prefer="processes", verbose=0)(
        delayed(_simulate_episode_block)(
            block_id=i,
            n_block=block_sizes[i],
            cfg=cfg,
            sensors=sensors,
            base_press=base_press,  # shared read-only
            S=S
        )
        for i in tqdm(range(n_blocks), desc="Simulating episodes (parallel blocks)")
    )

    # Collect
    X_det, y_det, meta_det = [], [], []
    X_pipe, y_pipe = [], []
    X_area, y_area = [], []

    for (Xd, yd, md, Xp, yp, Xa, ya) in results:
        X_det.extend(Xd); y_det.extend(yd); meta_det.extend(md)
        X_pipe.extend(Xp); y_pipe.extend(yp); X_area.extend(Xa); y_area.extend(ya)

    # build arrays
    X_det = np.asarray(X_det, dtype=np.float32)
    y_det = np.asarray(y_det, dtype=np.int32)
    X_pipe = np.asarray(X_pipe, dtype=np.float32) if len(X_pipe) else np.zeros((0, X_det.shape[1]), dtype=np.float32)
    y_pipe = np.asarray(y_pipe) if len(y_pipe) else np.asarray([])
    X_area = np.asarray(X_area, dtype=np.float32) if len(X_area) else np.zeros((0, X_det.shape[1]), dtype=np.float32)
    y_area = np.asarray(y_area, dtype=np.float32) if len(y_area) else np.asarray([])

    # split by "episodes": here meta_det uses block_id as episode namespace; we approximate using block_id
    meta_det = np.asarray(meta_det)
    ep_ids = meta_det[:, 0].astype(int)  # block_id as coarse episode groups
    unique_eps = np.unique(ep_ids)
    tr_eps, te_eps = train_test_split(unique_eps, test_size=cfg.test_size,
                                      random_state=cfg.rf_random_state, shuffle=True)

    tr_mask = np.isin(ep_ids, tr_eps)
    te_mask = np.isin(ep_ids, te_eps)

    def split_by_mask(X, mask):
        return X[mask], X[~mask]

    X_det_tr, X_det_te = split_by_mask(X_det, tr_mask)
    y_det_tr, y_det_te = split_by_mask(y_det, tr_mask)

    # positives mask mapped to the same split
    pos_ep_mask = (y_det == 1)
    pos_tr_mask = tr_mask[pos_ep_mask]
    pos_te_mask = te_mask[pos_ep_mask]

    X_pipe_tr, X_pipe_te = split_by_mask(X_pipe, pos_tr_mask) if len(X_pipe) else (X_pipe, X_pipe)
    y_pipe_tr, y_pipe_te = split_by_mask(y_pipe, pos_tr_mask) if len(y_pipe) else (y_pipe, y_pipe)
    X_area_tr, X_area_te = split_by_mask(X_area, pos_tr_mask) if len(X_area) else (X_area, X_area)
    y_area_tr, y_area_te = split_by_mask(y_area, pos_tr_mask) if len(y_area) else (y_area, y_area)

    # label encoder for pipe: fit on ALL pipe_names to avoid unseen labels at test time
    if len(y_pipe_tr):
        le_pipe = LabelEncoder().fit(pipe_names)  # <--- fit on ALL pipes
        y_pipe_tr_enc = le_pipe.transform(y_pipe_tr)
        y_pipe_te_enc = le_pipe.transform(y_pipe_te)
    else:
        le_pipe = LabelEncoder().fit(pipe_names)
        y_pipe_tr_enc = np.asarray([], dtype=int)
        y_pipe_te_enc = np.asarray([], dtype=int)

    # train models
    clf_det = RandomForestClassifier(
        n_estimators=cfg.rf_trees,
        random_state=cfg.rf_random_state,
        class_weight="balanced",
        n_jobs=-1
    )
    clf_det.fit(X_det_tr, y_det_tr)

    clf_pipe = RandomForestClassifier(
        n_estimators=cfg.rf_trees,
        random_state=cfg.rf_random_state,
        class_weight="balanced",
        n_jobs=-1
    )
    if len(X_pipe_tr):
        clf_pipe.fit(X_pipe_tr, y_pipe_tr_enc)

    reg_area = RandomForestRegressor(
        n_estimators=cfg.rf_trees,
        random_state=cfg.rf_random_state,
        n_jobs=-1
    )
    if len(X_area_tr):
        reg_area.fit(X_area_tr, y_area_tr)

    # quick eval
    print("\n=== Detector (leak present) ===")
    yhat = clf_det.predict(X_det_te)
    ypro = clf_det.predict_proba(X_det_te)[:, 1] if hasattr(clf_det, "predict_proba") else None
    print(classification_report(y_det_te, yhat, digits=4))
    if ypro is not None:
        try:
            print("ROC-AUC:", roc_auc_score(y_det_te, ypro))
            print("PR-AUC :", average_precision_score(y_det_te, ypro))
        except Exception:
            pass

    if len(X_pipe_te):
        print("\n=== Pipe classifier (positive windows only) ===")
        yhat_p = clf_pipe.predict(X_pipe_te)
        # Limit the report to classes the model HAS seen during training,
        # to avoid “empty” metrics for never-seen classes.
        labels_for_report = clf_pipe.classes_
        print(classification_report(y_pipe_te_enc, yhat_p, labels=labels_for_report, digits=4))

    if len(X_area_te):
        print("\n=== Area regressor (positive windows only) ===")
        yhat_a = reg_area.predict(X_area_te)
        mae = mean_absolute_error(y_area_te, yhat_a)
        print("MAE:", mae)

    # save artifacts
    os.makedirs(cfg.out_dir, exist_ok=True)
    # signatures
    np.save(os.path.join(cfg.out_dir, "signatures.npy"), S.astype(np.float32))
    with open(os.path.join(cfg.out_dir, "pipe_names.json"), "w") as f:
        json.dump(pipe_names, f)
    # models
    joblib.dump(clf_det, os.path.join(cfg.out_dir, "rf_detector.joblib"))
    joblib.dump(clf_pipe, os.path.join(cfg.out_dir, "rf_pipe.joblib"))
    joblib.dump(reg_area, os.path.join(cfg.out_dir, "rf_area.joblib"))
    joblib.dump(le_pipe, os.path.join(cfg.out_dir, "le_pipe.joblib"))
    # meta
    meta = {
        "cfg": asdict(cfg),
        "sensors": sensors,
        "steps": steps,
        "W": W,
        "report_step_s": cfg.report_step_s,
        "hyd_step_s": cfg.hyd_step_s,
        "sign_dim": int(S.shape[1]),
        "n_pipes": int(S.shape[0])
    }
    with open(os.path.join(cfg.out_dir, "meta_live.json"), "w") as f:
        json.dump(meta, f, indent=2)
    # save dataset (optional)
    if cfg.save_csv:
        det_df = pd.DataFrame(X_det)
        det_df["y_det"] = y_det
        det_df.to_csv(os.path.join(cfg.out_dir, cfg.out_csv_dataset), index=False)

    print(f"\n💾 Saved artifacts to: {cfg.out_dir}")
    return {
        "signatures": S,
        "pipe_names": pipe_names,
        "sensors": sensors
    }


# -----------------------------
# Inference live
# -----------------------------

def load_artifacts(out_dir: str):
    S = np.load(os.path.join(out_dir, "signatures.npy"))
    with open(os.path.join(out_dir, "pipe_names.json"), "r") as f:
        pipe_names = json.load(f)
    with open(os.path.join(out_dir, "meta_live.json"), "r") as f:
        meta = json.load(f)
    clf_det = joblib.load(os.path.join(out_dir, "rf_detector.joblib"))
    clf_pipe = joblib.load(os.path.join(out_dir, "rf_pipe.joblib"))
    reg_area = joblib.load(os.path.join(out_dir, "rf_area.joblib"))
    le_pipe = joblib.load(os.path.join(out_dir, "le_pipe.joblib"))
    return S, pipe_names, meta, clf_det, clf_pipe, reg_area, le_pipe


def build_features_from_buffer(buffer_resid: np.ndarray, S: np.ndarray):
    """
    buffer_resid: np array shape (W, SENSORS)
    """
    feat, stats = window_features(buffer_resid, S)
    return feat.reshape(1, -1), stats


def live_infer_loop(cfg: LiveConfig,
                    simulate_one_leak: bool = True,
                    leak_pipe: Optional[str] = None,
                    leak_start: Optional[int] = None,
                    leak_area: Optional[float] = None):
    """
    Simulate a live episode (baseline or 1 leak) and run online inference with hysteresis.
    """
    S, pipe_names, meta, clf_det, clf_pipe, reg_area, le_pipe = load_artifacts(cfg.out_dir)
    sensors = meta["sensors"]
    W = int(meta["W"])
    steps = int(meta["steps"])
    report_step_s = int(meta["report_step_s"])

    # Build baseline and episode
    wn0 = make_base_network(cfg.inp_path, cfg.hours, cfg.hyd_step_s, cfg.report_step_s)
    base_press = simulate_pressures(wn0)[sensors].copy()

    wn = make_base_network(cfg.inp_path, cfg.hours, cfg.hyd_step_s, cfg.report_step_s)
    if simulate_one_leak:
        rnd = rng(cfg.seed + 1234)
        if leak_pipe is None:
            leak_pipe = str(rnd.choice(wn.pipe_name_list))
        if leak_start is None:
            leak_start = int(rnd.integers(0, steps))
        if leak_area is None:
            leak_area = _area_scaled_by_D2_for_pipe(wn, leak_pipe,
                                                    cfg.leak_k_scale, cfg.leak_area_jitter,
                                                    cfg.diameter_units)
        wn = apply_single_leak(wn, leak_pipe, leak_area, leak_start, cfg.report_step_s)
        print(f"[GT] pipe={leak_pipe} start={leak_start} area={leak_area:.6f} m^2 "
              f"(start_min={leak_start*report_step_s/60.0:.1f})")
    press = simulate_pressures(wn)[sensors].copy()
    if cfg.sensor_noise_std > 0:
        press = add_sensor_noise(press, cfg.sensor_noise_std, rng(cfg.seed+999))

    resid = press - base_press

    # Live buffer
    buffer = np.zeros((W, len(sensors)), dtype=np.float32)
    alarm = False
    consec = 0
    t_alarm = None

    for t in range(steps):
        # update buffer with latest residual row
        buffer[:-1] = buffer[1:]
        buffer[-1] = resid.iloc[t, :].to_numpy(dtype=np.float32)

        if t < W-1 or t < cfg.warmup_steps:
            continue  # not enough data / warm-up

        X_t, stats = build_features_from_buffer(buffer, S)
        p_leak = float(clf_det.predict_proba(X_t)[0, 1]) if hasattr(clf_det, "predict_proba") else float(clf_det.predict(X_t)[0])

        if p_leak >= cfg.alarm_threshold:
            consec += 1
        else:
            consec = 0

        if (not alarm) and (consec >= cfg.hysteresis):
            alarm = True
            t_alarm = t
            # localize + area
            if hasattr(clf_pipe, "predict_proba"):
                proba_pipe = clf_pipe.predict_proba(X_t)[0]
                idx = np.argsort(proba_pipe)[::-1]
                top_k = min(5, len(idx))
                top = [(le_pipe.inverse_transform([i])[0], float(proba_pipe[i])) for i in idx[:top_k]]
            else:
                pred_idx = int(clf_pipe.predict(X_t)[0])
                top = [(le_pipe.inverse_transform([pred_idx])[0], None)]

            area_hat = float(reg_area.predict(X_t)[0]) if reg_area else None

            print(f"[ALARM] t={t} (min={t*report_step_s/60:.1f})  p_leak={p_leak:.3f}  "
                  f"top-pipes={top}  area≈{area_hat:.6f}")
            # continue running to observe stability

    if not alarm:
        print("[INFO] No alarm raised.")
    else:
        print(f"[INFO] Alarm raised at t={t_alarm} (min={t_alarm*report_step_s/60:.1f}).")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Supervised LIVE leak pipeline (single-leak, causal sliding window)")

    # Network/time
    p.add_argument("--inp", type=str, default="Net3.inp")
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--hyd-step", type=int, default=300)
    p.add_argument("--report-step", type=int, default=600)

    # Sensors
    p.add_argument("--sensor-frac", type=float, default=1.0,
                   help="Fraction of demand nodes to use as sensors (0<frac<=1) if --sensor-num not set")
    p.add_argument("--sensor-num", type=int, default=None,
                   help="Exact number of sensors (overrides --sensor-frac)")
    p.add_argument("--seed", type=int, default=0)

    # Windows
    p.add_argument("--win", type=int, default=6)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--stride", type=int, default=1)

    # Episodes
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--p-leak", type=float, default=0.5)
    p.add_argument("--leak-k-scale", type=float, default=0.05)
    p.add_argument("--leak-area-jitter", type=float, default=0.2)
    p.add_argument("--diameter-units", type=str, choices=["m", "mm", "in"], default="m")
    p.add_argument("--sensor-noise-std", type=float, default=0.0)

    # Unit signatures
    p.add_argument("--unit-area", type=float, default=2e-4)

    # Models
    p.add_argument("--rf-trees", type=int, default=300)
    p.add_argument("--rf-random-state", type=int, default=0)
    p.add_argument("--test-size", type=float, default=0.2)

    # Output
    p.add_argument("--out-dir", type=str, default="artifacts_live")
    p.add_argument("--save-csv", action="store_true")

    # Inference live options
    p.add_argument("--infer-live", action="store_true", help="Run live inference demo after training")
    p.add_argument("--alarm-threshold", type=float, default=0.8)
    p.add_argument("--hysteresis", type=int, default=2)

    # Live episode control (demo)
    p.add_argument("--simulate-one-leak", action="store_true", help="If set, the live demo simulates one leak")
    p.add_argument("--pipe", type=str, default=None)
    p.add_argument("--start-step", type=int, default=None)
    p.add_argument("--area", type=float, default=None)

    # Parallel
    p.add_argument("--workers", type=int, default=6, help="Parallel workers for signatures/episodes")

    return p


def main():
    args = parse_args().parse_args()

    # resolve sensor_fraction (num overrides frac)
    sensor_fraction = float(args.sensor_num) if args.sensor_num is not None else float(args.sensor_frac)

    cfg = LiveConfig(
        inp_path=args.inp,
        hours=float(args.hours),
        hyd_step_s=int(args.hyd_step),
        report_step_s=int(args.report_step),
        sensor_fraction=sensor_fraction,
        seed=int(args.seed),
        win=int(args.win),
        warmup_steps=int(args.warmup_steps),
        stride=int(args.stride),
        n_episodes=int(args.episodes),
        p_leak=float(args.p_leak),
        leak_k_scale=float(args.leak_k_scale),
        leak_area_jitter=float(args.leak_area_jitter),
        diameter_units=str(args.diameter_units),
        sensor_noise_std=float(args.sensor_noise_std),
        unit_area=float(args.unit_area),
        rf_trees=int(args.rf_trees),
        rf_random_state=int(args.rf_random_state),
        test_size=float(args.test_size),
        out_dir=str(args.out_dir),
        save_csv=bool(args.save_csv),
        alarm_threshold=float(args.alarm_threshold),
        hysteresis=int(args.hysteresis),
        workers=int(args.workers),
    )

    print("== Live config ==")
    for k, v in asdict(cfg).items():
        print(f"{k}: {v}")

    # Train artifacts (baseline, signatures, dataset, models)
    _ = generate_dataset(cfg)

    # Optional live demo
    if args.infer_live:
        live_infer_loop(
            cfg,
            simulate_one_leak=bool(args.simulate_one_leak),
            leak_pipe=args.pipe,
            leak_start=args.start_step,
            leak_area=args.area
        )


if __name__ == "__main__":
    main()
