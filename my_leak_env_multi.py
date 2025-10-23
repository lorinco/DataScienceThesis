# my_leak_env_multi.py
# RL environment for water networks with 12h episodes, 10' steps, and multiple random leaks per episode.
# Compatible with MaskablePPO: exposes .current_num_actions, action_mask(), and actions [NO-OP, single toggles, FULL].
# Includes minimal helpers (run_episode, summarize_episode_plus) to stay self-contained.

import os, copy
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx

import wntr
from wntr.network.controls import Control, ControlAction, SimTimeCondition
from wntr.network.base import LinkStatus

import gymnasium as gym
from gymnasium import spaces


# ============ Basic utils ============
def build_segment_adjacency(wn, node_segments: Dict[str, int], boundary_links: List[str]) -> Dict[int, List[int]]:
    """
    Build a segment adjacency graph: two segments are adjacent if
    there is a 'boundary' pipe connecting a node from the first to a node from the second.
    Returns: dict seg_id -> (ordered) list of unique adjacent segments.
    """
    adj = {}
    for bname in boundary_links:
        if bname not in wn.link_name_list:
            continue
        try:
            lk = wn.get_link(bname)
            u, v = lk.start_node.name, lk.end_node.name
            su = node_segments.get(u, None)
            sv = node_segments.get(v, None)
            if su is None or sv is None or su == sv:
                continue
            adj.setdefault(su, set()).add(sv)
            adj.setdefault(sv, set()).add(su)
        except Exception:
            continue
    return {k: sorted(list(v)) for k, v in adj.items()}

def _neutralize_pump_speed_controls_inplace(wn, open_pumps=True):
    # speed=1.0, optionally open, drop controls on pump speed/setting
    for p in getattr(wn, 'pump_name_list', []):
        try:
            pump = wn.get_link(p)
            if hasattr(pump, 'speed'):
                pump.speed = 1.0
            if open_pumps:
                for attr in ("status", "initial_status"):
                    try:
                        setattr(pump, attr, LinkStatus.Open)
                    except Exception:
                        pass
            try:
                pump.pattern = None
            except Exception:
                pass
        except Exception:
            pass
    # remove controls that touch speed/setting or target a pump
    try:
        for cname, ctrl in list(wn.controls()):
            try:
                acts = list(getattr(ctrl, "actions", []) or [getattr(ctrl, "action", None)])
                acts = [a for a in acts if a is not None]
                touches = False
                for act in acts:
                    attr = getattr(act, 'attribute', None)
                    tgt  = getattr(act, 'link', None) or getattr(act, 'target', None) or getattr(act, 'toolkit_object', None)
                    if isinstance(attr, str) and attr.lower() in ('speed', 'setting'):
                        touches = True; break
                    tname = getattr(tgt, 'name', None)
                    if tname and tname in getattr(wn, 'pump_name_list', []):
                        touches = True; break
                if touches:
                    wn.remove_control(cname)
            except Exception:
                wn.remove_control(cname)
    except Exception:
        pass

def make_base_network(inp_path: str,
                      dur_h: float,
                      hyd_step_s: int,
                      report_step_s: int,
                      min_pressure_m: float,
                      req_pressure_m: float):
    wn = wntr.network.WaterNetworkModel(inp_path)
    wn.options.time.duration = int(dur_h * 3600)
    wn.options.time.hydraulic_timestep = int(hyd_step_s)
    wn.options.time.report_timestep = int(report_step_s)
    wn.options.time.report_start = 0
    wn.options.hydraulic.demand_model = 'PDD'
    wn.options.hydraulic.minimum_pressure = float(min_pressure_m)
    wn.options.hydraulic.required_pressure = float(req_pressure_m)
    _neutralize_pump_speed_controls_inplace(wn, open_pumps=True)
    return wn

def run_simulation_strict(wn, duration_s: int, hyd_step_s: int, report_step_s: int,
                          min_pressure_m: float, req_pressure_m: float):
    wn.options.time.duration = int(duration_s)
    wn.options.time.hydraulic_timestep = int(hyd_step_s)
    wn.options.time.report_timestep = int(report_step_s)
    wn.options.time.report_start = 0
    wn.options.hydraulic.demand_model = 'PDD'
    wn.options.hydraulic.minimum_pressure = float(min_pressure_m)
    wn.options.hydraulic.required_pressure = float(req_pressure_m)
    sim = wntr.sim.WNTRSimulator(wn)
    try:
        return sim.run_sim()
    except NotImplementedError as e:
        if 'Pump speeds other than 1.0' in str(e):
            _neutralize_pump_speed_controls_inplace(wn, open_pumps=True)
            return wntr.sim.WNTRSimulator(wn).run_sim()
        raise

def expected_actual_aligned(wn, results):
    exp = wntr.metrics.expected_demand(wn)
    act = results.node.get('demand', pd.DataFrame())
    if exp is None or act is None or exp.empty or act.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    common_t = act.index.intersection(exp.index)
    cols = [c for c in wn.junction_name_list if (c in act.columns) and (c in exp.columns)]
    if len(common_t) == 0 or len(cols) == 0:
        return pd.DataFrame(), pd.DataFrame(), []
    exp_c = exp.loc[common_t, cols]
    act_c = act.loc[common_t, cols]
    active = exp_c.columns[(exp_c > 0).any()].tolist()
    return exp_c, act_c, active

def wsa_timeseries(wn, results) -> pd.Series:
    exp_c, act_c, _ = expected_actual_aligned(wn, results)
    if act_c.empty or exp_c.empty:
        return pd.Series(dtype=float)
    ratio = act_c / exp_c.replace(0, np.nan)
    return ratio.mean(axis=1).clip(upper=1.0).fillna(0.0)

def wsa_final_safe(wn, results) -> float:
    s = wsa_timeseries(wn, results)
    return float(s.iloc[-1]) if (s is not None and not s.empty) else float('nan')

def wsa_mean_safe(wn, results) -> float:
    s = wsa_timeseries(wn, results)
    return float(s.mean()) if (s is not None and not s.empty) else float('nan')

def leak_series_multi(results, leak_nodes: List[str]) -> pd.Series:
    """Sum leak_demand (or demand) over all leak nodes."""
    nodes = list(leak_nodes or [])
    if not nodes:
        return pd.Series(dtype=float)
    df = results.node.get('leak_demand', None)
    if df is None or df.empty or not any(n in df.columns for n in nodes):
        df = results.node.get('demand', pd.DataFrame())
    if df is None or df.empty:
        return pd.Series(dtype=float)
    cols = [c for c in nodes if c in df.columns]
    if not cols:
        return pd.Series(dtype=float)
    s = df[cols].sum(axis=1).sort_index()
    s.index = s.index.to_numpy(dtype=float)
    return s.astype(float)

def build_valve_segments(wn, placement_type='strategic', n=2, seed=123):
    vl = wntr.network.generate_valve_layer(wn, placement_type=placement_type, n=n, seed=seed)
    G = wn.to_graph()
    node_segments, link_segments, seg_sizes = wntr.metrics.valve_segments(G, vl)
    return vl, node_segments, link_segments, seg_sizes

def compute_all_boundary_links(wn, node_segments, link_segments) -> List[str]:
    boundaries = []
    for link_name, link in wn.links():
        if link.link_type != 'Pipe':
            continue
        u, v = link.start_node.name, link.end_node.name
        su, sv = node_segments.get(u, None), node_segments.get(v, None)
        if (su is not None) and (sv is not None) and (su != sv):
            boundaries.append(link_name)
    return sorted(set(boundaries))


# ====================  Multi-leak environment ====================

class LeakSegEnvMulti(gym.Env):
    """Multi-leak RL environment with action masking, anti-chatter, and robust cumulative feature."""
    metadata = {"render_modes": []}

    def __init__(self,
                inp_path: str = "Net3.inp",
                episode_hours: float = 12.0,
                hyd_step_s: int = 300,
                report_step_s: int = 600,
                min_pressure_m: float = 10.0,
                req_pressure_m: float = 20.0,
                # leak: sizing
                leak_area_mode: str = "scaled_by_D2",
                leak_area_m2: float = 1e-4,
                leak_k_scale: float = 0.05,
                leak_area_jitter: float = 0.2,     # ±20% on k_scale
                # multi-leak scheduling
                max_leaks: int = 3,
                min_leaks: int = 1,
                min_leak_start_h: float = 0.25,
                min_gap_h: float = 0.5,
                allow_same_pipe_multiple_events: bool = False,
                # observations (None => use ALL by default)
                n_node_sensors: Optional[float] = None,
                n_link_sensors: Optional[float] = None,
                # action limits
                k_boundaries: Optional[int] = None,
                # anti-chatter & action control
                cooldown_steps: int = 2,
                allow_full_action: bool = False,
                action_penalty_coef: float = 0.20,
                rapid_reopen_penalty: float = 0.50,
                k_max_closed: int = 5,
                # pre-leak gate
                q_gate_m3s: float = 0,
                wsa_gate_min: float = 0.90,
                idle_gate_steps: int = 2,
                # cumulative feature
                cumvol_mode: str = "saved_frac",   # "saved_frac" | "frac_baseline" | "log_norm" | "raw" | "none"
                cumvol_cap: float = 1.5,
                seed: int = 0):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.inp_path = inp_path
        self.episode_hours = float(episode_hours)
        self.hyd_step_s = int(hyd_step_s)
        self.report_step_s = int(report_step_s)
        self.min_pressure_m = float(min_pressure_m)
        self.req_pressure_m = float(req_pressure_m)

        # cumulative feature
        self.cumvol_mode = str(cumvol_mode)
        self.cumvol_cap = float(cumvol_cap)

        # leak sizing
        self.leak_area_mode = str(leak_area_mode)
        self.leak_area_m2 = float(leak_area_m2)
        self.leak_k_scale = float(leak_k_scale)
        self.leak_area_jitter = float(leak_area_jitter)

        # multi-leak
        self.max_leaks = int(max(1, max_leaks))
        self.min_leaks = int(max(1, min_leaks))
        self.min_leak_start_h = float(min_leak_start_h)
        self.min_gap_h = float(min_gap_h)
        self.allow_same_pipe_multiple_events = bool(allow_same_pipe_multiple_events)

        # anti-chatter / gate
        self.cooldown_steps = int(cooldown_steps)
        self.allow_full_action = bool(allow_full_action)
        self.action_penalty_coef = float(action_penalty_coef)
        self.rapid_reopen_penalty = float(rapid_reopen_penalty)
        self.k_max_closed = int(k_max_closed)
        self.q_gate_m3s = float(q_gate_m3s)
        self.wsa_gate_min = float(wsa_gate_min)
        self.idle_gate_steps = int(idle_gate_steps)

        self._last_toggle_step: Dict[str, int] = {}   # link -> last toggled step

        # time/step mapping
        self.step_hours = float(self.report_step_s) / 3600.0
        self.steps_per_episode = int(round(self.episode_hours / self.step_hours))

        # base network and pre-segmentation
        self.wn_template = make_base_network(self.inp_path, self.episode_hours,
                                            self.hyd_step_s, self.report_step_s,
                                            self.min_pressure_m, self.req_pressure_m)

        # --- sensor counts resolution: by default ALL ---
        wn = self.wn_template
        total_nodes = len(getattr(wn, "junction_name_list", []))
        total_links = len(getattr(wn, "pipe_name_list", []))

        def _resolve(x, total):
            # default (None) => all
            if x is None:
                return total
            # percentage (0,1]
            if isinstance(x, float) and (0.0 < x <= 1.0):
                return max(1, min(total, int(round(x * total))))
            # integer (clipped, negative => all)
            try:
                xi = int(x)
                if xi < 0:
                    return total
                return max(0, min(total, xi))
            except Exception:
                return total  # safe fallback

        self.n_node_sensors = _resolve(n_node_sensors, total_nodes)
        self.n_link_sensors = _resolve(n_link_sensors, total_links)
        self.k_boundaries = None if k_boundaries is None else int(k_boundaries)

        # valve pre-segmentation
        self.valve_layer0, self.node_segments0, self.link_segments0, self.seg_sizes0 = build_valve_segments(
            self.wn_template, placement_type='strategic', n=2, seed=123
        )
        self.boundary_links_all0 = compute_all_boundary_links(self.wn_template, self.node_segments0, self.link_segments0)
        seg_to_b = {}
        for bname in self.boundary_links_all0:
            lk = self.wn_template.get_link(bname)
            u, v = lk.start_node.name, lk.end_node.name
            su = self.node_segments0.get(u, None)
            sv = self.node_segments0.get(v, None)
            if su is not None:
                seg_to_b.setdefault(su, set()).add(bname)
            if sv is not None:
                seg_to_b.setdefault(sv, set()).add(bname)
        self.segment_to_boundary0 = {k: sorted(list(v)) for k, v in seg_to_b.items()}

        # episode state
        self.wn0 = copy.deepcopy(self.wn_template)  # without leaks
        self.wn_leak0 = None                        # with all leaks
        self.leak_nodes: List[str] = []
        self.leak_schedule: List[Tuple[str, float, float]] = []
        self.boundary_links: List[str] = []

        # flow/volume signals
        self.history: List[Dict[str, bool]] = []
        self.current_hour = 0.0
        self.cumulative_leak_vol = 0.0
        self._last_q_norm = 0.0
        self._last_q_m3s = 0.0
        self._baseline_total_ts: Optional[pd.Series] = None
        self._last_q_base_m3s = 0.0
        self._last_saved_step_m3 = 0.0
        self.q_ref = 1e-6

        # node service metrics
        self._last_any_node_service_lt_0_7 = False
        self._last_n_nodes_service_lt_0_7 = 0
        self._last_min_node_service = 1.0

        # sensors (name lists)
        self.node_sensors: List[str] = []
        self.link_sensors: List[str] = []

        # spaces (last element = cumulative feature)
        obs_dim = 4 + self.n_node_sensors + self.n_link_sensors + 2
        low = np.zeros(obs_dim, dtype=np.float32)
        high = np.ones(obs_dim, dtype=np.float32)
        press_hi = 150.0
        flow_hi = 10.0
        low[4:4+self.n_node_sensors] = 0.0
        high[4:4+self.n_node_sensors] = press_hi
        low[4+self.n_node_sensors:4+self.n_node_sensors+self.n_link_sensors] = 0.0
        high[4+self.n_node_sensors:4+self.n_node_sensors+self.n_link_sensors] = flow_hi
        high[-2] = 10.0                         # q_norm
        high[-1] = (1e6 if self.cumvol_mode == "raw" else self.cumvol_cap)  # cum_feat
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # actions: NO-OP + single toggles (B) + (optional) FULL
        num_pipes_total = len([e for e, obj in self.wn0.links() if obj.link_type == 'Pipe'])
        self.max_actions = max(3, num_pipes_total + 2)
        self.action_space = spaces.Discrete(self.max_actions)
        self.current_num_actions = 2  # updated at reset

        self._sim_cfg = dict(hyd_step_s=self.hyd_step_s,
                            report_step_s=self.report_step_s,
                            min_pressure_m=self.min_pressure_m,
                            req_pressure_m=self.req_pressure_m)


    # --------- Mask for MaskablePPO (with cool-down, k_max and pre-leak gate) ---------
    def _step_idx_next(self) -> int:
        return len(self.history) + 1

    def action_mask(self) -> np.ndarray:
        n = int(self.action_space.n)
        B = len(self.boundary_links)
        cur = int(getattr(self, "current_num_actions", B + 2))
        cur = max(1, min(cur, n))

        mask = np.zeros(n, dtype=bool)
        mask[:cur] = True

        # NO-OP always allowed
        mask[0] = True

        # FULL disabled if not allowed
        if (B + 1) < n and not self.allow_full_action:
            mask[B + 1] = False

        # current state
        state = self.history[-1] if self.history else {name: False for name in self.boundary_links}
        now_step = self._step_idx_next()
        n_closed = sum(bool(v) for k, v in state.items() if k != "__full__")

        # cool-down and k_max for each single toggle
        for i in range(B):
            a = 1 + i
            if a >= n or not mask[a]:
                continue
            link = self.boundary_links[i]
            is_closed = bool(state.get(link, False))
            since = now_step - int(self._last_toggle_step.get(link, -10**9))

            # cool-down
            if since < self.cooldown_steps:
                mask[a] = False
                continue

            # k_max closed
            if (not is_closed) and (n_closed >= self.k_max_closed):
                mask[a] = False

        # PRE-LEAK GATE
        if (len(self.history) < self.idle_gate_steps) and (self._last_q_m3s < self.q_gate_m3s) and (self._last_min_node_service >= self.wsa_gate_min):
            for i in range(1, 1 + B):
                mask[i] = False
            if (B + 1) < n:
                mask[B + 1] = False

        # --- FINAL GUARDRAILS ---
        # 1) ensure dtype/shape
        mask = np.asarray(mask, dtype=np.bool_).reshape((n,))

        # 2) NO-OP always True (even after gate)
        mask[0] = True

        # 3) if, for any reason, everything is False => unlock NO-OP
        if not mask.any():
            mask[:] = False
            mask[0] = True

        return mask


    # ---------------- RL lifecycle ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.wn0 = copy.deepcopy(self.wn_template)

        # 1) sample leaks (how many, where, when)
        pipes = [e for e, obj in self.wn0.links() if obj.link_type == 'Pipe']
        self.rng.shuffle(pipes)

        n_leaks = int(self.rng.integers(self.min_leaks, self.max_leaks + 1))
        chosen_pipes = []
        for p in pipes:
            if not self.allow_same_pipe_multiple_events and p in chosen_pipes:
                continue
            chosen_pipes.append(p)
            if len(chosen_pipes) >= n_leaks:
                break
        if not chosen_pipes:
            chosen_pipes = [pipes[0]]

        min_start = max(0.0, self.min_leak_start_h)
        latest_start = max(min_start, self.episode_hours - self.step_hours)
        starts = sorted(self.rng.uniform(min_start, latest_start, size=len(chosen_pipes)).tolist())
        if self.min_gap_h > 0 and len(starts) > 1:
            for i in range(1, len(starts)):
                starts[i] = max(starts[i], starts[i-1] + self.min_gap_h)
            starts[-1] = min(starts[-1], latest_start)
            starts = sorted(starts)

        # 2) build network with ALL leaks
        wn2 = copy.deepcopy(self.wn0)
        leak_nodes, schedule = [], []
        for p, st_h in zip(chosen_pipes, starts):
            leak_node = f"{p}_LEAK_NODE"
            new_pipe = f"{p}_B"
            if (not self.allow_same_pipe_multiple_events) or (leak_node not in wn2.node_name_list):
                wn2 = wntr.morph.split_pipe(
                    wn2, pipe_name_to_split=p,
                    new_pipe_name=new_pipe, new_junction_name=leak_node
                )
            ref_link = wn2.get_link(p) if p in wn2.link_name_list else wn2.get_link(new_pipe)
            D = float(getattr(ref_link, "diameter", 0.1))
            jitter = 1.0 + float(self.leak_area_jitter) * float(self.rng.uniform(-1.0, 1.0))
            area_eff = compute_leak_area(self.leak_area_mode, self.leak_area_m2, D, self.leak_k_scale * jitter)
            node = wn2.get_node(leak_node)
            node.add_leak(wn2, area=float(area_eff),
                        start_time=int(st_h * 3600),
                        end_time=int(self.episode_hours * 3600 + self.report_step_s))
            leak_nodes.append(leak_node)
            schedule.append((leak_node, float(st_h), float(self.episode_hours)))

        self.wn_leak0 = wn2
        self.leak_nodes = leak_nodes
        self.leak_schedule = schedule

        # 3) full NO-OP baseline
        self._baseline_total_ts = self._compute_baseline_total_ts()
        self.q_ref = self._compute_q_ref_once()
        self._last_q_base_m3s = 0.0
        self._last_saved_step_m3 = 0.0

        # 4) controllable boundaries = union of boundaries of the segments with leaks
        seg_ids = set()
        for p in chosen_pipes:
            lk = self.wn_template.get_link(p)
            u, v = lk.start_node.name, lk.end_node.name
            seg = self.node_segments0.get(u, self.node_segments0.get(v, None))
            if seg is not None:
                seg_ids.add(int(seg))
        boundaries = []
        for s in seg_ids:
            boundaries.extend(self.segment_to_boundary0.get(s, []))
        boundaries = sorted(set(boundaries))
        if self.k_boundaries is not None and len(boundaries) > self.k_boundaries:
            self.rng.shuffle(boundaries)
            boundaries = sorted(boundaries[: self.k_boundaries])
        self.boundary_links = boundaries
        self._last_toggle_step = {e: -10**9 for e in self.boundary_links}

        # 4.b) BUILD ADJACENCY BETWEEN SEGMENTS AND SELECT "UP TO 6" NEIGHBORS
        seg_adj = build_segment_adjacency(self.wn_template, self.node_segments0, self.boundary_links_all0)
        # union of neighbors of the leak segments
        neighbors = set()
        for s in seg_ids:
            for t in seg_adj.get(int(s), []):
                if t not in seg_ids:
                    neighbors.add(int(t))
        # if more than 6, order by "strength" (how many boundaries connect them to leak segments), then truncate
        if len(neighbors) > 6:
            strength = {}
            for t in neighbors:
                cnt = 0
                # Count how many total boundaries connect t with any s in seg_ids
                for bname in self.boundary_links_all0:
                    if bname not in self.wn_template.link_name_list:
                        continue
                    try:
                        lk = self.wn_template.get_link(bname)
                        u, v = lk.start_node.name, lk.end_node.name
                        su = self.node_segments0.get(u, None)
                        sv = self.node_segments0.get(v, None)
                        if su is None or sv is None or su == sv:
                            continue
                        # pair that "touches" t and one of the leak segments
                        if ((su == t and sv in seg_ids) or (sv == t and su in seg_ids)):
                            cnt += 1
                    except Exception:
                        pass
                strength[t] = cnt
            # sort descending by connections, then deterministic by id
            ordered = sorted(list(neighbors), key=lambda x: (-strength.get(x, 0), x))
            neighbors = set(ordered[:6])

        target_segments = sorted(list(seg_ids.union(neighbors)))

        # 5) sensors (NEW: targeted to the selected segments)
        self.node_sensors  = self._select_sensor_nodes_targeted(target_segments)
        self.link_sensors  = self._select_sensor_links_targeted(target_segments)

        # 6) valid actions = B + 2
        B = len(self.boundary_links)
        self.current_num_actions = min(self.max_actions, B + 2)

        # 7) episode state
        self.history.clear()
        self.current_hour = 0.0
        self.cumulative_leak_vol = 0.0
        self._last_q_norm = 0.0
        self._last_q_m3s = 0.0
        self._last_any_node_service_lt_0_7 = False
        self._last_n_nodes_service_lt_0_7 = 0
        self._last_min_node_service = 1.0

        obs = self._simulate_and_observe(self.current_hour)
        info0 = {
            "leak_nodes": list(self.leak_nodes),
            "leak_schedule": list(self.leak_schedule),
            "boundary_links": list(self.boundary_links),
            "target_segments_for_sensors": target_segments,  # <-- useful for debug
        }
        info0["action_masks"] = self.action_mask().astype(np.bool_)
        return obs, info0


    def step(self, action: int):
        # map into valid range
        try:
            action = int(action) % max(1, self.current_num_actions)
        except Exception:
            action = int(action)

        B = len(self.boundary_links)
        state = self.history[-1].copy() if self.history else {name: False for name in self.boundary_links}
        full_flag = state.get("__full__", False)

        toggle_cost = 0.0
        extra_rapid_pen = 0.0
        now_step = self._step_idx_next()

        if action == 0:
            pass  # NO-OP

        elif action == (B + 1) and B > 0:  # FULL
            if self.allow_full_action:
                full_flag = not full_flag
                for k in list(state.keys()):
                    if k != "__full__":
                        prev = state[k]
                        state[k] = full_flag
                        if prev != state[k]:
                            toggle_cost = 1.0
                            self._last_toggle_step[k] = now_step
            # if FULL not allowed => NO-OP

        else:
            idx = action - 1
            if 0 <= idx < B:
                link_name = self.boundary_links[idx]
                is_closed = bool(state.get(link_name, False))
                since = now_step - int(self._last_toggle_step.get(link_name, -10**9))

                # guardrails: cooldown and k_max_closed
                cooldown_ok = (since >= self.cooldown_steps)
                kmax_ok = True
                if (not is_closed):
                    n_closed = sum(bool(v) for k, v in state.items() if k != "__full__")
                    if n_closed >= self.k_max_closed:
                        kmax_ok = False

                if cooldown_ok and kmax_ok:
                    # toggle
                    state[link_name] = not is_closed
                    full_flag = False
                    toggle_cost = 1.0
                    # rapid reopen penalty: reopening within cooldown
                    if is_closed and (since <= self.cooldown_steps):
                        extra_rapid_pen = self.rapid_reopen_penalty
                    self._last_toggle_step[link_name] = now_step
                # else: soft NO-OP

        state["__full__"] = full_flag
        self.history.append(state)

        # advance by 1 step = report_step_s
        self.current_hour += self.step_hours
        obs = self._simulate_and_observe(self.current_hour)

        
        # ---------- REWARD BLOCK (pro-action) ----------
        # 1) episode end flags BEFORE reward
        terminated = (self.current_hour >= self.episode_hours)
        truncated = False

        # 2) unpack from observation
        t_norm, closed_ratio, wsa_t, pct_under = obs[:4].tolist()
        base_step_m3 = max(1e-6, self.q_ref * float(self.report_step_s))
        saved_norm = float(self._last_saved_step_m3 / base_step_m3)
        cum_feat = float(obs[-1])  # last feature = cumulative
        q_norm = float(getattr(self, "_last_q_norm", 0.0))

        # 3) reward components (rebalanced weights)
        #   - strong reward to saved volume
        #   - action penalty discounted at the beginning (acting early costs less)
        #   - penalty for "doing nothing" when the leak is evident
        saved_gain   = 7.5 * saved_norm

        wsa_bonus    = 1.2 * float(wsa_t)
        pressure_pen = 4.0 * float(pct_under)

        # penalty for number of active closures (softer)
        close_pen    = 0.20 * float(closed_ratio)

        # toggle cost with time discount (early episode cheaper)
        action_pen_base = self.action_penalty_coef * float(toggle_cost)
        action_pen      = action_pen_base * (0.5 + 0.5 * t_norm)  # 0.5x at start -> 1.0x at end

        # unchanged rapid reopen penalty
        rapid_pen    = float(extra_rapid_pen)

        # hard pen if WSA drops too much: more severe under 0.95
        wsa_hard_pen = 4.0 * max(0.0, 0.95 - float(wsa_t))

        # "do-nothing" penalty: if the leak is high and almost nothing is closed, encourages action
        noop_pressure = 0.6 * q_norm * (1.0 - float(closed_ratio))

        # 4) aggregation
        reward = (
            saved_gain + wsa_bonus
            - pressure_pen - close_pen - action_pen - rapid_pen - wsa_hard_pen
            - noop_pressure
        )

        # final bonus at episode end for cumulative savings (encourages long-term plans)
        if terminated:
            reward += 4.0 * (cum_feat - 0.5)

        # final clip
        reward = float(np.clip(reward, -15.0, 15.0))
        # ---------- END REWARD BLOCK ----------

        #---------- REWARD BLOCK (drop-in) ----------
        # 1) flags BEFORE reward
        # terminated = (self.current_hour >= self.episode_hours)
        # truncated = False

        # # 2) unpack from observation
        # t_norm, closed_ratio, wsa_t, pct_under = obs[:4].tolist()
        # base_step_m3 = max(1e-6, self.q_ref * float(self.report_step_s))
        # saved_norm = float(self._last_saved_step_m3 / base_step_m3)  # >= 0
        # cum_feat   = float(obs[-1])                                   # cumulative feature

        # # leak “active” from baseline viewpoint (push to act when there is real flow)
        # leak_active = (self._last_q_base_m3s > self.q_gate_m3s)

        # # 3) new closures in the step (anti flip-flop, but softer)
        # prev_state = (self.history[-2] if len(self.history) >= 2
        #             else {name: False for name in self.boundary_links})
        # curr_state = self.history[-1] if self.history else prev_state
        # prev_closed_n = sum(bool(v) for k, v in prev_state.items() if k != "__full__")
        # curr_closed_n = sum(bool(v) for k, v in curr_state.items() if k != "__full__")
        # delta_new_closures = max(0, curr_closed_n - prev_closed_n)

        # # 4) reward components
        # #   Saving with saturation (immediately rewards real savings)
        # saved_term   = 9.0 * float(np.tanh(1.5 * saved_norm))

        # #   WSA: soft bonus if >0.95, strong penalty under 0.95/0.90
        # wsa_soft     = 0.8 * max(0.0, float(wsa_t) - 0.95)
        # wsa_hard_pen = (6.0 * max(0.0, 0.90 - float(wsa_t))
        #                 + 2.0 * max(0.0, 0.95 - float(wsa_t)))

        # #   Pressure: protect it
        # pressure_pen = 10.0 * float(pct_under)

        # #   “Too isolated” state penalized but lightly (we don't want to kill exploration)
        # close_pen    = 0.10 * float(closed_ratio)

        # #   Change penalty: light (encourages trying but not spamming)
        # change_pen   = 0.25 * float(delta_new_closures)

        # #   Penalty for single toggle and rapid reopen (very light)
        # action_pen   = 0.05 * float(toggle_cost)
        # rapid_pen    = float(extra_rapid_pen)

        # #   NEW: small “no-op” penalty when the leak is active but we save nothing
        # no_op_pen = 0.0
        # if leak_active and saved_norm < 0.02 and curr_closed_n == 0:
        #     no_op_pen = 0.4

        # # 5) aggregation
        # reward = (saved_term + wsa_soft
        #         - (close_pen + change_pen + action_pen + rapid_pen + pressure_pen + wsa_hard_pen + no_op_pen))

        # # Final bonus at episode end: rewards cumulative saving only if service stayed ≥0.95
        # if terminated and float(self._last_min_node_service) >= 0.95:
        #     reward += 5.0 * (cum_feat - 0.5)

        # reward = float(np.clip(reward, -10.0, 10.0))
        # # ---------- END REWARD BLOCK ----------



        terminated = (self.current_hour >= self.episode_hours)
        truncated = False

        st = self.history[-1]
        closed_now = [k for k, v in st.items() if k != "__full__" and v]
        info = {
            "hour": self.current_hour,
            "boundary_links": list(self.boundary_links),
            "closed_links": closed_now,
            "full_isolate": bool(st.get("__full__", False)),
            "q_norm": float(self._last_q_norm),
            "q_leak_m3s": float(self._last_q_m3s),
            "q_base_m3s": float(self._last_q_base_m3s),
            "saved_step_m3": float(self._last_saved_step_m3),
            "dt_s": float(self.report_step_s),
            "wsa": float(wsa_t),
            "pct_under_min": float(pct_under),
            "any_node_service_lt_0_7": bool(self._last_any_node_service_lt_0_7),
            "n_nodes_service_lt_0_7": int(self._last_n_nodes_service_lt_0_7),
            "min_node_service": float(self._last_min_node_service),
            "leak_nodes": list(self.leak_nodes),
            "leak_schedule": list(self.leak_schedule),
            # extra cumulative (debug)
            "cum_leak_m3": float(self.cumulative_leak_vol),
            "cum_base_m3": float(self._cum_base_up_to(int(min(self.current_hour, self.episode_hours)*3600))),
        }
        info["action_masks"] = self.action_mask().astype(np.bool_)
        return obs, reward, terminated, truncated, info

    # ---------------- Sim helpers ----------------
    def _apply_history_controls(self, wn_sim):
        """Re-apply valve states recorded in self.history as WNTR controls."""
        for hour_idx, state in enumerate(self.history, start=1):
            t_s = int(hour_idx * self.report_step_s)
            for name, is_closed in state.items():
                if name == "__full__":
                    continue
                if name not in wn_sim.link_name_list:
                    continue
                try:
                    link = wn_sim.get_link(name)
                    desired = LinkStatus.Closed if is_closed else LinkStatus.Open
                    act = ControlAction(link, "status", desired)
                    cond = SimTimeCondition(wn_sim, ">=", t_s)
                    ctrl = Control(cond, act)
                    wn_sim.add_control(f"replay_{name}_{hour_idx}_{int(desired)}", ctrl)
                except Exception:
                    continue

    def _compute_baseline_total_ts(self) -> pd.Series:
        wn_sim = copy.deepcopy(self.wn_leak0)
        res = run_simulation_strict(wn_sim, int(self.episode_hours * 3600), **self._sim_cfg)
        s = leak_series_multi(res, self.leak_nodes)
        if s is None or s.empty:
            return pd.Series(dtype=float)
        s.index = s.index.astype(float)
        return s

    def _cum_base_up_to(self, t_s: int) -> float:
        """Baseline cumulative (m3) up to t_s, integrating the instantaneous flow series."""
        s = self._baseline_total_ts
        if s is None or s.empty:
            return 0.0
        s_part = s[s.index <= t_s]
        if s_part.empty:
            return 0.0
        return float((s_part * self.report_step_s).sum())

    def _compute_q_ref_once(self) -> float:
        """Baseline mean of the first step after the FIRST activation among all (scales q_norm)."""
        if self._baseline_total_ts is None or self._baseline_total_ts.empty:
            return 1e-6
        first_start_h = min([st for _, st, _ in self.leak_schedule]) if self.leak_schedule else 0.0
        start_s = int(first_start_h * 3600)
        s = self._baseline_total_ts
        mask = (s.index > start_s) & (s.index <= start_s + self.report_step_s)
        val = float(s[mask].mean()) if mask.any() else float(s.iloc[-1])
        return max(val, 1e-6)

    def _cum_feature(self, t_s: int) -> float:
        """Compute the cumulative feature in the observation according to cumvol_mode."""
        cum_act = float(self.cumulative_leak_vol)
        cum_base = float(self._cum_base_up_to(t_s))
        eps = 1e-6

        if self.cumvol_mode == "none":
            feat = 0.0
        elif self.cumvol_mode == "raw":
            feat = cum_act
        elif self.cumvol_mode == "frac_baseline":
            feat = cum_act / max(cum_base, eps)
        elif self.cumvol_mode == "log_norm":
            feat = np.log1p(cum_act) / max(np.log1p(cum_base), eps)
        else:  # "saved_frac" (default)
            feat = (cum_base - cum_act) / max(cum_base, eps)

        # cap (and non-negativity, useful for saved fraction)
        if self.cumvol_mode != "raw":
            feat = float(np.clip(feat, 0.0, self.cumvol_cap))
        else:
            feat = float(min(feat, self.cumvol_cap))
        return feat

    def _simulate_and_observe(self, hour: float) -> np.ndarray:
        end_s = int(min(hour, self.episode_hours) * 3600)
        wn_sim = copy.deepcopy(self.wn_leak0)
        self._apply_history_controls(wn_sim)
        res = run_simulation_strict(wn_sim,
                                    end_s if end_s > 0 else self.report_step_s,
                                    **self._sim_cfg)

        # total instantaneous leak
        s_tot = leak_series_multi(res, self.leak_nodes)
        q_current = float(s_tot.iloc[-1]) if (s_tot is not None and not s_tot.empty) else 0.0
        self._last_q_m3s = q_current
        step_vol = q_current * float(self.report_step_s)
        self.cumulative_leak_vol = float(self.cumulative_leak_vol + step_vol)

        # baseline @t and step saving
        t_s = int(min(hour, self.episode_hours) * 3600)
        if self._baseline_total_ts is not None and not self._baseline_total_ts.empty:
            q_base = float(self._baseline_total_ts.loc[self._baseline_total_ts.index <= t_s].iloc[-1])
        else:
            q_base = 0.0
        self._last_q_base_m3s = q_base
        self._last_saved_step_m3 = max(0.0, (q_base - q_current) * float(self.report_step_s))
        self._last_q_norm = (q_current / self.q_ref) if self.q_ref > 0 else 0.0

        # global WSA
        wsa_t = wsa_final_safe(wn_sim, res)
        if not np.isfinite(wsa_t):
            wsa_t = 1.0

        # % nodes under min_pressure
        press_df = res.node.get('pressure', pd.DataFrame())
        exp_c, act_c, _ = expected_actual_aligned(wn_sim, res)
        exp_last = exp_c.iloc[-1] if (exp_c is not None and not exp_c.empty) else pd.Series(dtype=float)
        act_last = act_c.iloc[-1] if (act_c is not None and not act_c.empty) else pd.Series(dtype=float)
        demand_nodes = exp_last[exp_last > 0].index.tolist() if len(exp_last) else []
        pct_under = 0.0
        if press_df is not None and not press_df.empty:
            p_last = press_df.iloc[-1]
            if demand_nodes:
                p_last = p_last.reindex(demand_nodes).dropna()
            if len(p_last) > 0:
                pct_under = float((p_last < self.min_pressure_m).sum()) / float(len(p_last))

        # per-node service
        any_lt_0_7, n_lt_0_7, min_node_service = False, 0, 1.0
        if len(demand_nodes) and len(act_last):
            act_pos = act_last.reindex(demand_nodes).astype(float)
            exp_pos = exp_last.reindex(demand_nodes).astype(float)
            with np.errstate(divide='ignore', invalid='ignore'):
                node_ratio = (act_pos / exp_pos).clip(upper=1.0)
            node_ratio = node_ratio.replace([np.inf, -np.inf], np.nan).dropna()
            if len(node_ratio) > 0:
                n_lt_0_7 = int((node_ratio < 0.70).sum())
                any_lt_0_7 = bool(n_lt_0_7 > 0)
                min_node_service = float(node_ratio.min())

        self._last_any_node_service_lt_0_7 = any_lt_0_7
        self._last_n_nodes_service_lt_0_7 = n_lt_0_7
        self._last_min_node_service = min_node_service

        # sensors
        pressures = []
        if press_df is not None and not press_df.empty:
            p_last_all = press_df.iloc[-1]
            for n in self.node_sensors:
                pressures.append(float(p_last_all.get(n, 0.0)))
        else:
            pressures = [0.0] * self.n_node_sensors

        flow_df = res.link.get('flowrate', pd.DataFrame())
        flows = []
        if flow_df is not None and not flow_df.empty:
            last_flow = flow_df.iloc[-1]
            for e in self.link_sensors:
                flows.append(abs(float(last_flow.get(e, 0.0))))
        else:
            flows = [0.0] * self.n_link_sensors

        # closed_ratio
        if self.history:
            st = self.history[-1]
            closed_now = [k for k, v in st.items() if k != "__full__" and v]
            closed_ratio = len(closed_now) / max(1, len(self.boundary_links))
        else:
            closed_ratio = 0.0

        # cumulative feature
        cum_feat = self._cum_feature(t_s)

        t_norm = min(hour / float(self.episode_hours), 1.0)
        obs = np.array(
            [t_norm, closed_ratio, float(wsa_t), float(pct_under)]
            + pressures + flows
            + [float(self._last_q_norm), float(cum_feat)],
            dtype=np.float32
        )
        return obs
    def _select_sensor_nodes_targeted(self, target_segments: List[int]) -> List[str]:
        """
        Return up to self.n_node_sensors nodes in wn_leak0 that belong to the target segments.
        If not enough, fill randomly with other nodes (no duplicates).
        Uses the segment mapping computed on wn_template (node_segments0).
        """
        k = int(self.n_node_sensors)
        if k <= 0:
            return []

        target_set = set(int(s) for s in target_segments)
        # candidate nodes = those from the template that are in target segments
        cand = [n for n, seg in self.node_segments0.items() if seg in target_set and n in self.wn_leak0.node_name_list]

        self.rng.shuffle(cand)
        out = cand[:k]

        if len(out) < k:
            # fallback: fill with other random nodes, avoiding duplicates
            rest = [n for n, _ in self.wn_leak0.nodes() if getattr(self.wn_leak0.get_node(n), "node_type", "") == "Junction"]
            self.rng.shuffle(rest)
            for n in rest:
                if n not in out:
                    out.append(n)
                    if len(out) >= k:
                        break

        return out[:k]
    def _select_sensor_links_targeted(self, target_segments: List[int]) -> List[str]:
        """
        Return up to self.n_link_sensors pipes in wn_leak0 that belong to the target segments.
        Criterion: pipes whose two endpoints both belong to target segments (or, if not enough,
        at least one endpoint). If still not enough, fill with random pipes avoiding duplicates.
        """
        k = int(self.n_link_sensors)
        if k <= 0:
            return []

        target_set = set(int(s) for s in target_segments)

        def _seg_of_node(nm: str) -> Optional[int]:
            # mapping from template segments; new leak nodes may be missing => None
            return self.node_segments0.get(nm, None)

        strong, weak = [], []

        for e, obj in self.wn_leak0.links():
            if getattr(obj, "link_type", "") != "Pipe":
                continue
            u, v = obj.start_node.name, obj.end_node.name
            su, sv = _seg_of_node(u), _seg_of_node(v)
            if su is None or sv is None:
                continue
            if (su in target_set) and (sv in target_set):
                strong.append(e)  # pipe "inside" target segments
            elif (su in target_set) or (sv in target_set):
                weak.append(e)    # pipe on boundary with target

        self.rng.shuffle(strong)
        self.rng.shuffle(weak)

        out = strong[:k]
        if len(out) < k:
            need = k - len(out)
            out.extend(weak[:need])

        if len(out) < k:
            # fallback: fill with other random pipes
            others = [e for e, obj in self.wn_leak0.links() if getattr(obj, "link_type", "") == "Pipe" and e not in out]
            self.rng.shuffle(others)
            out.extend(others[:(k - len(out))])

        return out[:k]



# =========== RL Helpers ===========

def compute_leak_area(mode: str, base_area_m2: float, pipe_diam_m: float, k_scale: float) -> float:
    if mode == 'scaled_by_D2':
        D = max(float(pipe_diam_m), 1e-6)
        return float(k_scale * (D**2))
    return float(base_area_m2)

def run_episode(env: LeakSegEnvMulti, policy_fn=None, name="episode"):
    # Gymnasium-compatible reset
    res = env.reset()
    obs, info0 = res if (isinstance(res, tuple) and len(res) == 2) else (res, {})
    try:
        print(f"[run_episode:{name}] leaks={len(info0.get('leak_nodes', []))}  boundary={len(info0.get('boundary_links', []))}")
    except Exception:
        pass

    rows, done = [], False
    last_info_for_policy = {}
    while not done:
        action = env.action_space.sample() if policy_fn is None else policy_fn(obs, last_info_for_policy, env)
        step_out = env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_out
        last_info_for_policy = info

        rows.append({
            "hour": info.get("hour", np.nan),
            "action": int(action),
            "reward": float(reward),
            "q_norm": float(info.get("q_norm", np.nan)),
            "wsa": float(info.get("wsa", np.nan)),
            "n_closed": len(info.get("closed_links", [])),
            "closed_links": list(info.get("closed_links", [])),
            "boundary_links": list(info.get("boundary_links", [])),
            "q_leak_m3s": float(info.get("q_leak_m3s", np.nan)),
            "q_base_m3s": float(info.get("q_base_m3s", np.nan)),
            "saved_step_m3": float(info.get("saved_step_m3", np.nan)),
            "dt_s": float(info.get("dt_s", np.nan)),
        })
    return pd.DataFrame(rows)

def summarize_episode_plus(df: pd.DataFrame, env_eval: LeakSegEnvMulti,
                           df_noop: pd.DataFrame, env_noop: LeakSegEnvMulti,
                           name: str) -> Dict[str, Any]:
    out = dict(name=name)
    out["reward_sum"] = float(df["reward"].sum()) if not df.empty else float("nan")
    out["wsa_mean"]   = float(df["wsa"].mean())  if not df.empty else float("nan")
    out["q_norm_mean"] = float(df["q_norm"].mean()) if not df.empty else float("nan")
    out["toggles"]     = int((df["action"] != 0).sum()) if not df.empty else 0

    try:
        B = max(1, len(df.iloc[-1]["boundary_links"]))
    except Exception:
        B = 1
    closed_ratio_series = df["n_closed"].astype(float) / float(B)
    out["closed_ratio_mean"] = float(closed_ratio_series.mean()) if not df.empty else float("nan")
    out["pct_steps_wsa_ge_0.9"]  = float((df["wsa"] >= 0.9).mean())  if not df.empty else float("nan")
    out["pct_steps_wsa_ge_0.95"] = float((df["wsa"] >= 0.95).mean()) if not df.empty else float("nan")
    out["min_wsa"] = float(df["wsa"].min()) if not df.empty else float("nan")

    # volumes replay (entire episode)
    def _replay_and_total_leak(env: LeakSegEnvMulti) -> Tuple[float, pd.Series]:
        wn_sim = copy.deepcopy(env.wn_leak0)
        env._apply_history_controls(wn_sim)
        res = run_simulation_strict(wn_sim, int(env.episode_hours*3600), **env._sim_cfg)
        s = leak_series_multi(res, env.leak_nodes)
        if s is None or s.empty:
            return 0.0, pd.Series(dtype=float)
        return float((s * env.report_step_s).sum()), s

    vol_eval, s_eval = _replay_and_total_leak(env_eval)
    vol_noop, s_noop = _replay_and_total_leak(env_noop)
    out["vol_m3"] = float(vol_eval)
    out["vol_saved_vs_noop_m3"] = float(max(vol_noop - vol_eval, 0.0))

    def _partial(s: pd.Series, seconds: float, dt: float) -> float:
        if s is None or s.empty:
            return 0.0
        return float((s[s.index <= seconds] * dt).sum())
    out["vol_first_1h_m3"] = _partial(s_eval, 3600.0, env_eval.report_step_s)
    out["vol_first_2h_m3"] = _partial(s_eval, 7200.0, env_eval.report_step_s)
    out["vol_saved_first_1h_m3_vs_noop"] = max(_partial(s_noop,3600.0, env_eval.report_step_s) - out["vol_first_1h_m3"], 0.0)
    out["vol_saved_first_2h_m3_vs_noop"] = max(_partial(s_noop,7200.0, env_eval.report_step_s) - out["vol_first_2h_m3"], 0.0)
    return out
