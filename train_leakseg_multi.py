import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", r"Pump .* has exceeded its maximum flow\.", UserWarning)

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from torch import nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from my_leak_env_multi import LeakSegEnvMulti, run_episode, summarize_episode_plus


# ============ UTILS ============

def _unwrap_to_base_env(env):
    """Descend through wrappers until reaching the base LeakSegEnvMulti."""
    cur = env
    for _ in range(32):
        if hasattr(cur, "env"):
            cur = cur.env
            continue
        if hasattr(cur, "venv"):
            cur = cur.venv
            continue
        break
    return getattr(cur, "unwrapped", cur)


def mask_invalid_actions(env) -> np.ndarray:
    """
    If the env exposes .action_mask(), use it (includes cool-down etc.).
    Fallback: enable 0..current_num_actions-1.
    """
    base = getattr(env, "unwrapped", env)
    if hasattr(base, "action_mask"):
        try:
            return np.asarray(base.action_mask(), dtype=bool)
        except Exception:
            pass

    n = int(env.action_space.n)
    cur = int(getattr(base, "current_num_actions", n))
    cur = max(1, min(cur, n))
    mask = np.zeros(n, dtype=bool)
    mask[:cur] = True
    return mask


def make_env_fn(env_kw: dict, seed: int):
    """Single-env constructor (for Subproc/Dummy)."""
    def _thunk():
        env = LeakSegEnvMulti(**env_kw, seed=seed)
        env = Monitor(env)
        env = ActionMasker(env, mask_invalid_actions)
        return env
    return _thunk


# ============ CALLBACKS ============

class TrainRewardLivePNGCallback(BaseCallback):
    """
    Save (and overwrite) a PNG with episode reward progression during training,
    updating it every `plot_every_episodes` episodes.
    X axis = episode index (1..N), hence no "vertical bars".
    """
    def __init__(self, save_dir: str, plot_every_episodes: int = 50, smooth_window: int = 0):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.plot_every = max(1, int(plot_every_episodes))
        self.smooth_window = max(0, int(smooth_window))
        self.ep_x = []       # episodes 1..N
        self.ep_rewards = [] # reward per episode
        self._n_done = 0

    def _maybe_save_plot(self):
        if not self.ep_rewards:
            return
        x = np.arange(1, len(self.ep_rewards) + 1, dtype=float)
        y = np.asarray(self.ep_rewards, dtype=float)

        if self.smooth_window > 1 and len(y) >= self.smooth_window:
            k = np.ones(self.smooth_window) / self.smooth_window
            y_sm = np.convolve(y, k, mode="same")
        else:
            y_sm = y

        fig = plt.figure(figsize=(9, 5))
        plt.plot(x, y, alpha=0.35, label="Reward (episodio)")
        plt.plot(x, y_sm, label=f"Smooth (w={self.smooth_window})" if self.smooth_window > 1 else "Reward")
        plt.xlabel("Episode #")
        plt.ylabel("Episode reward")
        plt.title("Training reward (aggiornato in tempo reale)")
        plt.grid(True)
        plt.legend()
        out = os.path.join(self.save_dir, "train_reward_curve_live.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close(fig)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is None or "r" not in ep:
                continue
            self._n_done += 1
            self.ep_rewards.append(float(ep["r"]))
            # save periodically
            if (self._n_done % self.plot_every) == 0:
                self._maybe_save_plot()
        return True

    def _on_training_end(self) -> None:
        # save a cleaner final plot
        if not self.ep_rewards:
            print("[TRAIN-PLOT] Nessun episodio completo registrato (run troppo breve?).")
            return

        x = np.arange(1, len(self.ep_rewards) + 1, dtype=float)
        y = np.asarray(self.ep_rewards, dtype=float)
        w = max(1, len(y)//50)
        if w > 1:
            y_sm = np.convolve(y, np.ones(w)/w, mode="same")
        else:
            y_sm = y

        fig = plt.figure(figsize=(9, 5))
        plt.plot(x, y, alpha=0.3, label="Reward (episodio)")
        plt.plot(x, y_sm, label=f"Smooth auto (w={w})" if w > 1 else "Reward")
        plt.xlabel("Episode #")
        plt.ylabel("Episode reward")
        plt.title("Training reward (finale)")
        plt.grid(True)
        plt.legend()
        out = os.path.join(self.save_dir, "train_reward_curve.png")
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(fig)
        print(f"[PLOT] Curva training reward salvata in: {out}")

        # CSV with rewards per episode (useful for analysis)
        try:
            import csv
            csv_path = os.path.join(self.save_dir, "train_rewards.csv")
            with open(csv_path, "w", newline="") as f:
                wcsv = csv.writer(f)
                wcsv.writerow(["episode_idx", "reward"])
                for i, r in enumerate(self.ep_rewards, 1):
                    wcsv.writerow([i, r])
            print(f"[LOG] train_rewards.csv salvato in: {csv_path}")
        except Exception:
            pass
class PeriodicCheckpointCallback(BaseCallback):
    """
    Save the model (and optionally VecNormalize) every `save_freq` timesteps.
    Creates files like:
      - {save_dir}/ckpt_{t}/model.zip
      - {save_dir}/ckpt_{t}/vecnorm.pkl   (if --vecnorm)
    """
    def __init__(self, save_dir: str, save_freq: int = 100_000, save_vecnorm: bool = True):
        super().__init__()
        self.save_dir = save_dir
        self.save_freq = int(save_freq)
        self.save_vecnorm = bool(save_vecnorm)
        os.makedirs(self.save_dir, exist_ok=True)
        self._last_saved_at = 0

    def _maybe_get_vecnorm(self):
        try:
            # training_env can be: VecNormalize(...) or a wrapper holding it in .venv
            from stable_baselines3.common.vec_env import VecNormalize
            env = self.training_env
            if isinstance(env, VecNormalize):
                return env
            if hasattr(env, "venv") and isinstance(env.venv, VecNormalize):
                return env.venv
        except Exception:
            pass
        return None

    def _on_step(self) -> bool:
        # Save when at least save_freq new steps have elapsed
        if (self.num_timesteps - self._last_saved_at) >= self.save_freq:
            t = int(self.num_timesteps)
            out_dir = os.path.join(self.save_dir, f"ckpt_{t}")
            os.makedirs(out_dir, exist_ok=True)
            # model
            try:
                self.model.save(os.path.join(out_dir, "model"))
                print(f"[CKPT] Modello salvato: {out_dir}/model.zip")
            except Exception as e:
                print(f"[CKPT][WARN] Salvataggio modello fallito: {e}")

            # vecnorm
            if self.save_vecnorm:
                try:
                    vecnorm = self._maybe_get_vecnorm()
                    if vecnorm is not None:
                        vecnorm.save(os.path.join(out_dir, "vecnorm.pkl"))
                        print(f"[CKPT] VecNormalize salvato: {out_dir}/vecnorm.pkl")
                except Exception as e:
                    print(f"[CKPT][WARN] Salvataggio VecNormalize fallito: {e}")

            self._last_saved_at = t
        return True
class EpisodeRewardCSVLoggerCallback(BaseCallback):
    """
    Append to a CSV file (saved in the run folder) the episode reward
    as soon as the episode ends (so data persist even if training stops).
    """
    def __init__(self, save_dir: str, filename: str = "train_rewards_live.csv"):
        super().__init__()
        self.save_dir = save_dir
        self.csv_path = os.path.join(self.save_dir, filename)
        os.makedirs(self.save_dir, exist_ok=True)
        self._header_written = False

    def _on_training_start(self) -> None:
        # If the file doesn't exist, write the header
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                f.write("episode_idx,timesteps,episode_reward\n")
            self._header_written = True
        else:
            self._header_written = True

    def _on_step(self) -> bool:
        # Monitor injects info["episode"] when an episode ends
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is None or "r" not in ep:
                continue
            ep_rew = float(ep["r"])
            # episode index can be inferred from the number of rows already written (+1)
            try:
                if not os.path.exists(self.csv_path):
                    idx = 1
                else:
                    # count existing lines (minus header)
                    with open(self.csv_path, "r") as f:
                        n_lines = sum(1 for _ in f)
                    idx = max(1, n_lines - 1 + 1)
            except Exception:
                idx = 1

            with open(self.csv_path, "a", newline="") as f:
                f.write(f"{idx},{int(self.num_timesteps)},{ep_rew}\n")
        return True


# ============ FINAL EVAL ============

def finalize_and_eval(model, env_kw, args, save_dir):
    """
    - Save the final model.
    - Run ONLY here the evaluation (K episodes) and save metrics/plots.
    """
    final_path = os.path.join(save_dir, "final_model_maskable")
    model.save(final_path)
    print(f"[DONE] Modello salvato in: {final_path}")

    # Final evaluation
    K = max(1, int(args.eval_episodes))
    rng = np.random.default_rng(seed=12345)
    vols_saved, min_wsas, toggles, rewards = [], [], [], []

    for i in range(K):
        ep_seed = int(rng.integers(0, 1 << 31))
        env_eval = make_env_fn(env_kw, seed=ep_seed)()
        env_noop = make_env_fn(env_kw, seed=ep_seed)()

        def policy_fn(obs, last_info, _env: LeakSegEnvMulti):
            m = mask_invalid_actions(_env)
            a, _ = model.predict(obs, deterministic=True, action_masks=m)
            return int(a)

        df_eval = run_episode(env_eval, policy_fn=policy_fn, name=f"eval_ep{i}")
        df_noop = run_episode(env_noop, policy_fn=lambda *_: 0, name=f"noop_ep{i}")

        base_eval = _unwrap_to_base_env(env_eval)
        base_noop = _unwrap_to_base_env(env_noop)
        summ = summarize_episode_plus(df_eval, base_eval, df_noop, base_noop, name=f"eval_ep{i}")

        vols_saved.append(summ.get("vol_saved_vs_noop_m3", 0.0))
        min_wsas.append(summ.get("min_wsa", 1.0))
        toggles.append(summ.get("toggles", 0))
        rewards.append(summ.get("reward_sum", 0.0))

    mean_saved   = float(np.mean(vols_saved)) if vols_saved else 0.0
    mean_min_wsa = float(np.mean(min_wsas)) if min_wsas else 1.0
    mean_toggles = float(np.mean(toggles)) if toggles else 0.0
    mean_reward  = float(np.mean(rewards))  if rewards  else 0.0

    print(f"[FINAL EVAL] episodes={K}  mean_eval_reward={mean_reward:.3f}  "
          f"mean_saved_vs_noop_m3={mean_saved:.2f}  mean_min_wsa={mean_min_wsa:.3f}  "
          f"mean_toggles={mean_toggles:.2f}")

    # Bar plot of eval rewards
    try:
        fig = plt.figure(figsize=(8, 4.2))
        x = np.arange(1, K + 1)
        plt.bar(x, rewards)
        plt.xlabel("Eval episode #")
        plt.ylabel("Episode reward")
        plt.title("Final evaluation (reward per episode)")
        plt.grid(True, axis="y", alpha=0.3)
        out_png = os.path.join(save_dir, "eval_reward_curve.png")
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close(fig)
        print(f"[PLOT] Curva eval salvata in: {out_png}")
    except Exception:
        pass

    # Also save a textual summary
    try:
        summary_txt = os.path.join(save_dir, "final_eval_summary.txt")
        with open(summary_txt, "w") as f:
            f.write(f"Final evaluation over {K} episode(s)\n")
            f.write(f"mean_eval_reward      : {mean_reward:.6f}\n")
            f.write(f"mean_saved_vs_noop_m3 : {mean_saved:.6f}\n")
            f.write(f"mean_min_wsa          : {mean_min_wsa:.6f}\n")
            f.write(f"mean_toggles          : {mean_toggles:.6f}\n")
        print(f"[LOG] Riassunto eval salvato in: {summary_txt}")
    except Exception:
        pass


# ============ DEFAULT ENV CONFIG ============

ENV_KW_DEFAULT = dict(
    # network and timing
    inp_path="Net3.inp",
    episode_hours=12.0,
    hyd_step_s=300,
    report_step_s=600,
    min_pressure_m=10.0,
    req_pressure_m=20.0,
    # leak sizing
    leak_area_mode="scaled_by_D2",
    leak_area_m2=1e-4,
    leak_k_scale=0.05,
    leak_area_jitter=0.20,
    # multi-leak scheduling
    max_leaks=3,
    min_leaks=1,
    min_leak_start_h=0.25,
    min_gap_h=0.5,
    allow_same_pipe_multiple_events=False,
    # observations
    n_node_sensors=None,
    n_link_sensors=None,
    # boundary limits / guard-rails / gating / cumvol
    k_boundaries=None,
    cooldown_steps=2,
    allow_full_action=False,
    action_penalty_coef=0.20,
    rapid_reopen_penalty=0.50,
    k_max_closed=3,
    q_gate_m3s=0,
    wsa_gate_min=0.90,
    idle_gate_steps=1,
    cumvol_mode="saved_frac",
    cumvol_cap=1.5,
)


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(description="Training MaskablePPO su LeakSegEnv multi-leak")
    # RL/training
    parser.add_argument("--n-envs", type=int, default=6, help="num env paralleli")
    parser.add_argument("--total-steps", type=int, default=50_000, help="timesteps totali")
    parser.add_argument("--n-steps", type=int, default=256, help="passi di rollout per ENV")
    parser.add_argument("--batch-size", type=int, default=0, help="0 = auto (divisore di n_envs*n_steps)")
    parser.add_argument("--target-rollout", type=int, default=4096, help="solo display")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="runs_leakseg_multi")
    parser.add_argument("--vecnorm", action="store_true", help="VecNormalize su osservazioni")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "auto"], help="Dispositivo PyTorch per SB3")
    # live visual plot
    parser.add_argument("--plot-every-episodes", type=int, default=50, help="aggiorna PNG ogni N episodi completati")
    parser.add_argument("--smooth-window", type=int, default=0, help="smoothing del live plot (0=off)")

    # evaluation (final only by default)
    parser.add_argument("--eval-episodes", type=int, default=5, help="episodi per la valutazione FINALE")
    parser.add_argument("--eval-freq", type=int, default=0, help="IGNORATO se --eval-during-train non è attivo")
    parser.add_argument("--eval-during-train", action="store_true", help="se impostato, abilita eval periodica (sconsigliato)")

    # quick environment overrides
    parser.add_argument("--inp-path", type=str, default=ENV_KW_DEFAULT["inp_path"])
    parser.add_argument("--episode-hours", type=float, default=ENV_KW_DEFAULT["episode_hours"])
    parser.add_argument("--hyd-step-s", type=int, default=ENV_KW_DEFAULT["hyd_step_s"])
    parser.add_argument("--leak-k", type=float, default=ENV_KW_DEFAULT["leak_k_scale"])
    parser.add_argument("--leak-area-jitter", type=float, default=ENV_KW_DEFAULT["leak_area_jitter"])
    parser.add_argument("--min-leaks", type=int, default=ENV_KW_DEFAULT["min_leaks"])
    parser.add_argument("--max-leaks", type=int, default=ENV_KW_DEFAULT["max_leaks"])
    parser.add_argument("--min-leak-start-h", type=float, default=ENV_KW_DEFAULT["min_leak_start_h"])
    parser.add_argument("--min-gap-h", type=float, default=ENV_KW_DEFAULT["min_gap_h"])
    parser.add_argument("--allow-same-pipe-multiple", action="store_true", default=ENV_KW_DEFAULT["allow_same_pipe_multiple_events"])
    parser.add_argument("--k-boundaries", type=int, default=-1, help="-1 per None (nessun limite)")

    args = parser.parse_args()
    set_random_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ===== compute rollout and batch =====
    rollout_size = int(args.n_envs) * int(args.n_steps)

    def _auto_batch(rs: int) -> int:
        cap = min(4096, rs)
        cand = cap
        while cand > 0 and (rs % cand != 0):
            cand -= 1
        return max(64, cand)

    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else _auto_batch(rollout_size)
    if rollout_size % batch_size != 0:
        for b in range(batch_size, 63, -1):
            if rollout_size % b == 0:
                batch_size = b
                break

    print(f"[AUTO] n_envs={args.n_envs} n_steps={args.n_steps} rollout={rollout_size} batch_size={batch_size}")

    # ===== build env_kw with overrides =====
    env_kw = dict(ENV_KW_DEFAULT)
    env_kw.update(dict(
        inp_path=str(args.inp_path),
        episode_hours=float(args.episode_hours),
        report_step_s=int(args.report_step_s),
        hyd_step_s=int(args.hyd_step_s),
        leak_k_scale=float(args.leak_k),
        leak_area_jitter=float(args.leak_area_jitter),
        min_leaks=int(args.min_leaks),
        max_leaks=int(args.max_leaks),
        min_leak_start_h=float(args.min_leak_start_h),
        min_gap_h=float(args.min_gap_h),
        allow_same_pipe_multiple_events=bool(args.allow_same_pipe_multiple),
    ))
    env_kw["k_boundaries"] = None if int(args.k_boundaries) < 0 else int(args.k_boundaries)

    # ===== Vec env =====
    if args.n_envs > 1:
        vec_env = SubprocVecEnv([make_env_fn(env_kw, seed=args.seed + i) for i in range(args.n_envs)])
    else:
        vec_env = DummyVecEnv([make_env_fn(env_kw, seed=args.seed)])

    if args.vecnorm:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    def linear_schedule_with_floor(init_lr: float, min_lr: float):
        def f(progress_remaining: float) -> float:  # 1 -> 0
            return max(min_lr, init_lr * progress_remaining)
        return f
    # ===== Model =====
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        gamma=0.995,
        gae_lambda=0.95,
        n_steps=int(args.n_steps),
        batch_size=int(batch_size),
        learning_rate=linear_schedule_with_floor(5e-4, 3e-5),
        clip_range=0.2,
        clip_range_vf=0.3,
        ent_coef=0.01,
        vf_coef=0.6,
        n_epochs=12,
        target_kl=0.03,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            activation_fn=nn.ReLU,  # optional, but recommended
            net_arch=dict(pi=[128,128,64], vf=[128,128,64])
        ),
        verbose=1,
        seed=args.seed,
        tensorboard_log=os.path.join(args.save_dir, "tb"),
        device=args.device,
    )

    # ===== Callback: ONLY live plot during training =====
        # ===== Callback: live plot + live CSV + periodic checkpoints =====
    callbacks = [
        TrainRewardLivePNGCallback(
            save_dir=args.save_dir,
            plot_every_episodes=int(args.plot_every_episodes),
            smooth_window=int(args.smooth_window)
        ),
        EpisodeRewardCSVLoggerCallback(
            save_dir=args.save_dir,
            filename="train_rewards_live.csv"
        ),
        PeriodicCheckpointCallback(
            save_dir=args.save_dir,
            save_freq=100_000,            # <<<<<<<<<< save every 100k timesteps
            save_vecnorm=bool(args.vecnorm)
        )
    ]

    # (Optional) Periodic eval — disabled by default
    if args.eval_during_train and int(args.eval_freq) > 0:
        # Very light callback: just prints a “skip” for clarity
        class _NoEvalDuringTrain(BaseCallback):
            def _on_step(self) -> bool:
                return True
        callbacks.append(_NoEvalDuringTrain())

    # ===== Learn =====
    model.learn(total_timesteps=args.total_steps, callback=CallbackList(callbacks))

    # ===== Save VecNormalize =====
    if args.vecnorm:
        try:
            vec_env.save(os.path.join(args.save_dir, "vecnorm.pkl"))
        except Exception:
            pass

    # ===== FINAL Evaluation & plots =====
    finalize_and_eval(model, env_kw, args, save_dir=args.save_dir)


if __name__ == "__main__":
    main()