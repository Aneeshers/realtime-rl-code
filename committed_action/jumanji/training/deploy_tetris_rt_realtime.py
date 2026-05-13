"""
deploy_tetris_rt_realtime.py

True two-GPU real-time deployment at a configurable FPS.  Supports Tetris, Pacman, and Snake.

  GPU 0: env stepping + reflex policy (committed actions) + gating network
  GPU 1: MCTS planner (async, via background thread)

At each meta-step the gating policy chooses K ∈ {1,2,3,4}.  While MCTS runs K×32 sims
on GPU 1, the environment advances K-1 committed actions (argmax policy logits) on GPU 0 at
the target frame rate.  The MCTS result is applied on the K-th frame.

At 9 FPS: frame = 111ms, K=4 → budget = 444ms for 128 sims.

Batch size note: JAX requires total_batch_size % num_devices == 0.  With two GPUs visible,
the minimum valid batch size is 2.  We run B=2 envs simultaneously; K is driven by env 0's
gating choice; episode stats are tracked for env 0 only.

Usage:
  python -m jumanji.training.deploy_tetris_rt_realtime \\
      --game tetris --gpu_type h100 \\
      --az_checkpoint_path    /path/to/training_state_epoch_000050.pkl \\
      --gating_checkpoint_path /path/to/gating_state_best.pkl \\
      --fps 9 --n_episodes 100
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import wandb

import jumanji
from jumanji.wrappers import VmapAutoResetWrapper

from jumanji.training.eval_tetris_rt_gating_policy import load_gating_checkpoint

_DEFAULT_SIM_OPTIONS = [32, 64, 96, 128]

_GAME_MODULE = {
    "tetris": "jumanji.training.train_tetris_rt_gating_ppo",
    "pacman": "jumanji.training.train_pacman_gating_ppo",
    "snake":  "jumanji.training.train_snake_kt_gating_ppo",
}
_GAME_ENV_NAME = {
    "tetris": "TetrisRTKT-v0",
    "pacman": "PacManKT-v1",
    "snake":  "SnakeKT-v1",
}
_DEPLOY_ROOT = "./checkpoints/committed_action"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# B=2: minimum batch size that satisfies total_batch_size % num_devices == 0
# with two visible GPUs.  Only env 0 is tracked for episode stats.
_BATCH = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default="tetris", choices=list(_GAME_MODULE.keys()),
                   help="Environment family (selects training module and default env_name).")
    p.add_argument("--gpu_type", default="h100", choices=["h100", "a100", "a40"],
                   help="GPU label used for output path and WandB run name.")
    p.add_argument("--env_name", default=None,
                   help="Jumanji env ID. Defaults to the canonical env for --game.")
    p.add_argument("--fps", type=float, default=9.0,
                   help="Target game speed in frames/second (default: 9).")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--az_checkpoint_path", required=True)
    p.add_argument("--gating_checkpoint_path", required=True)
    p.add_argument("--output_dir", default=None,
                   help="Output directory. Auto-derived from --game/--gpu_type/--fps if omitted.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sim_options", type=int, nargs=4, default=_DEFAULT_SIM_OPTIONS,
                   metavar=("S1", "S2", "S3", "S4"),
                   help="MCTS sim counts for K=1,2,3,4 (default: 32 64 96 128).")
    # W&B
    p.add_argument("--wandb_project", default="rt_deploy_realtime")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


# ── Build per-K MCTS JIT functions (inputs expected on gpu1) ─────────────────

def _build_mcts_jits(agents, az_p1, az_s1, k_options) -> Dict[int, Any]:
    mcts_jits: Dict[int, Any] = {}
    for k_idx, K in enumerate(k_options):
        ag = agents[k_idx]

        def _make(a=ag):
            def _fn(root_state, root_obs_grid, root_time_vec, invalid, key):
                return a._mcts_policy(
                    raw_env=a.raw_env_train,
                    params=az_p1,
                    net_state=az_s1,
                    rng_key=key,
                    root_state=root_state,
                    root_obs_grid=root_obs_grid,
                    root_time_vec=root_time_vec,
                    invalid_actions=invalid,
                )
            return jax.jit(_fn)

        mcts_jits[K] = _make()
    return mcts_jits


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Resolve game-specific defaults ───────────────────────────────────────
    if args.env_name is None:
        args.env_name = _GAME_ENV_NAME[args.game]
    fps_int = int(args.fps) if args.fps == int(args.fps) else args.fps
    if args.output_dir is None:
        args.output_dir = os.path.join(
            _DEPLOY_ROOT, "rt_deploy_out", args.game, args.gpu_type, f"fps_{fps_int}"
        )

    # ── Import game-specific training module ─────────────────────────────────
    _mod = importlib.import_module(_GAME_MODULE[args.game])
    load_az_checkpoint = _mod.load_az_checkpoint
    make_agents        = _mod.make_agents
    make_gating_forward = _mod.make_gating_forward
    K_OPTIONS          = _mod.K_OPTIONS
    SIM_OPTIONS        = _mod.SIM_OPTIONS  # noqa: F841 (used as reference; args.sim_options used below)

    os.makedirs(args.output_dir, exist_ok=True)

    FRAME_S = 1.0 / args.fps
    logger.info(f"Game: {args.game}  GPU: {args.gpu_type}  Env: {args.env_name}")
    logger.info(f"Target: {args.fps} FPS  →  frame = {FRAME_S*1000:.1f} ms")
    logger.info(f"K=4 budget: {4*FRAME_S*1000:.1f} ms  (4 frames)")
    logger.info(f"Output dir: {args.output_dir}")

    # ── W&B ──────────────────────────────────────────────────────────────────
    run_name = f"{args.game}_{args.gpu_type}_fps{fps_int}_seed{args.seed}"
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "game":        args.game,
                "gpu_type":    args.gpu_type,
                "env_name":    args.env_name,
                "fps":         args.fps,
                "frame_ms":    FRAME_S * 1000.0,
                "n_episodes":  args.n_episodes,
                "sim_options": args.sim_options,
                "seed":        args.seed,
                "az_checkpoint":      args.az_checkpoint_path,
                "gating_checkpoint":  args.gating_checkpoint_path,
            },
            name=run_name,
        )

    # ── Device setup ──────────────────────────────────────────────────────────
    gpus = jax.devices("gpu")
    if len(gpus) < 2:
        raise RuntimeError(f"Two GPUs required, found {len(gpus)}: {gpus}")
    gpu0, gpu1 = gpus[0], gpus[1]
    logger.info(f"GPU 0 (env + reflex + gating): {gpu0}")
    logger.info(f"GPU 1 (MCTS planner):          {gpu1}")

    # ── Environment (B=_BATCH) ────────────────────────────────────────────────
    # B=2 satisfies total_batch_size % jax.local_device_count() == 0.
    # Both envs run independently; K is driven by env 0's gating choice.
    raw_env = jumanji.make(args.env_name)
    env     = VmapAutoResetWrapper(raw_env)
    agents  = make_agents(env, num_envs=_BATCH, sim_options=args.sim_options)

    # ── Checkpoints ───────────────────────────────────────────────────────────
    az_state     = load_az_checkpoint(args.az_checkpoint_path)
    gating_state = load_gating_checkpoint(args.gating_checkpoint_path)
    gating_fwd   = make_gating_forward()

    az_p0 = jax.device_put(az_state.params.net, gpu0)
    az_s0 = jax.device_put(az_state.net_state,  gpu0)
    az_p1 = jax.device_put(az_state.params.net, gpu1)
    az_s1 = jax.device_put(az_state.net_state,  gpu1)

    gating_params = jax.device_put(gating_state.params, gpu0)

    # ── GPU 0 JIT functions ───────────────────────────────────────────────────

    @jax.jit
    def env_step_jit(state, action):
        return env.step(state, action)

    @jax.jit
    def az_fwd_jit(obs_grid, time_vec):
        (logits, _), _ = agents[0].forward.apply(
            az_p0, az_s0, obs_grid, time_vec, is_eval=True
        )
        return logits  # (B, n_actions)

    @jax.jit
    def az_fwd_features_jit(obs_grid, time_vec):
        (logits, val, trunk), _ = agents[0].forward_with_features.apply(
            az_p0, az_s0, obs_grid, time_vec, is_eval=True
        )
        return logits, val, trunk  # (B,A), (B,), (B,128)

    @jax.jit
    def gating_jit(obs_grid, time_vec, az_trunk, az_val_2d):
        logits, _ = gating_fwd.apply(gating_params, obs_grid, time_vec, az_trunk, az_val_2d)
        return jnp.argmax(logits, axis=-1)  # (B,)

    # ── GPU 1 MCTS JIT functions ──────────────────────────────────────────────
    mcts_jits = _build_mcts_jits(agents, az_p1, az_s1, K_OPTIONS)

    # ── Warmup: force JIT compilation before timing ───────────────────────────
    logger.info("Compiling JIT functions (takes a few minutes)...")
    _k0 = jax.random.PRNGKey(0)
    _s0, _ts0 = env.reset(jax.random.split(_k0, _BATCH))
    _obs0 = _ts0.observation
    _og, _tv = agents[0]._get_grid_and_time(_s0, _obs0)
    _inv     = agents[0]._get_invalid_actions(agents[0].raw_env_train, _s0, _obs0)

    jax.block_until_ready(env_step_jit(_s0, jnp.zeros((_BATCH,), dtype=jnp.int32)))
    jax.block_until_ready(az_fwd_jit(_og, _tv))
    _l, _v, _tr = az_fwd_features_jit(_og, _tv)
    jax.block_until_ready((_l, _v, _tr))
    jax.block_until_ready(gating_jit(_og, _tv, _tr, _v[:, None]))

    for K in K_OPTIONS:
        s1  = jax.device_put(_s0,  gpu1)
        og1 = jax.device_put(_og,  gpu1)
        tv1 = jax.device_put(_tv,  gpu1)
        iv1 = jax.device_put(_inv, gpu1)
        k1  = jax.device_put(_k0,  gpu1)
        jax.block_until_ready(mcts_jits[K](s1, og1, tv1, iv1, k1))
        logger.info(f"  MCTS K={K} (sims={args.sim_options[K-1]}) compiled.")
    logger.info("All JITs compiled.")

    # ── Shared state for the MCTS background thread ───────────────────────────
    _mcts_result: Dict[str, Any] = {"action": None, "latency_ms": 0.0}
    _mcts_done   = threading.Event()

    def _run_mcts(K: int, s_g1, og_g1, tv_g1, inv_g1, key_g1) -> None:
        t0  = time.perf_counter()
        out = mcts_jits[K](s_g1, og_g1, tv_g1, inv_g1, key_g1)
        jax.block_until_ready(out)
        lat = (time.perf_counter() - t0) * 1000.0
        _mcts_result["action"]     = jax.device_put(out.action, gpu0)
        _mcts_result["latency_ms"] = lat
        _mcts_done.set()

    # ── Episode loop ──────────────────────────────────────────────────────────
    # Both envs run in parallel.  K is determined by env 0's gating choice.
    # Episode stats (return, K distribution, latency) are tracked for env 0 only.
    all_returns: List[float] = []
    k_counts = np.zeros(4, dtype=np.int64)
    mcts_latencies: List[float] = []
    deadline_misses = 0
    total_meta_steps = 0

    rng = jax.random.PRNGKey(args.seed)

    for ep in range(args.n_episodes):
        rng, reset_key = jax.random.split(rng)
        state, ts = env.reset(jax.random.split(reset_key, _BATCH))
        state  = jax.device_put(state, gpu0)
        obs    = ts.observation
        ep_ret = 0.0
        done   = False  # tracks env 0

        while not done:
            # ── Gating (env 0 drives K for both envs) ───────────────────────
            frame_deadline = time.perf_counter() + FRAME_S

            obs_grid, time_vec = agents[0]._get_grid_and_time(state, obs)
            logits, az_val, az_trunk = az_fwd_features_jit(obs_grid, time_vec)
            k_choices = gating_jit(obs_grid, time_vec, az_trunk, az_val[:, None])
            k_idx = int(k_choices[0])   # env 0's choice drives meta-step length
            K     = K_OPTIONS[k_idx]
            k_counts[k_idx] += 1

            # ── Launch MCTS on GPU 1 (B envs searched simultaneously) ───────
            rng, mcts_key = jax.random.split(rng)
            inv  = agents[0]._get_invalid_actions(agents[0].raw_env_train, state, obs)
            s1   = jax.device_put(state,    gpu1)
            og1  = jax.device_put(obs_grid, gpu1)
            tv1  = jax.device_put(time_vec, gpu1)
            iv1  = jax.device_put(inv,      gpu1)
            k1   = jax.device_put(mcts_key, gpu1)
            _mcts_done.clear()
            threading.Thread(
                target=_run_mcts,
                args=(K, s1, og1, tv1, iv1, k1),
                daemon=True,
            ).start()

            # ── K-1 committed actions at the target frame rate ───────────────
            for _ in range(K - 1):
                time.sleep(max(0.0, frame_deadline - time.perf_counter()))
                frame_deadline += FRAME_S
                committed = jnp.argmax(logits, axis=-1)   # (B,) per-env argmax
                state, ts  = env_step_jit(state, committed)
                ep_ret    += float(ts.reward[0])           # track env 0
                if bool(ts.last()[0]):
                    done = True
                    break
                obs = ts.observation
                obs_grid, time_vec = agents[0]._get_grid_and_time(state, obs)
                logits = az_fwd_jit(obs_grid, time_vec)

            # ── K-th frame: apply MCTS result ────────────────────────────────
            if not done:
                time.sleep(max(0.0, frame_deadline - time.perf_counter()))
                if not _mcts_done.wait(timeout=0.005):
                    deadline_misses += 1
                    logger.warning(
                        f"Ep {ep:3d}, step {total_meta_steps}: "
                        f"MCTS deadline miss (K={K})"
                    )
                    _mcts_done.wait()

                mcts_action = _mcts_result["action"]   # (B,) on gpu0
                mcts_latencies.append(_mcts_result["latency_ms"])

                state, ts = env_step_jit(state, mcts_action)
                ep_ret   += float(ts.reward[0])
                done      = done or bool(ts.last()[0])
                if not done:
                    obs = ts.observation

            total_meta_steps += 1

        all_returns.append(ep_ret)

        if not args.no_wandb:
            k_dist_so_far = k_counts / max(k_counts.sum(), 1)
            lat_mean = float(np.mean(mcts_latencies)) if mcts_latencies else 0.0
            wandb.log({
                "episode":              ep + 1,
                "episode_return":       ep_ret,
                "running_mean_return":  float(np.mean(all_returns)),
                "k1_frac":              float(k_dist_so_far[0]),
                "k2_frac":              float(k_dist_so_far[1]),
                "k3_frac":              float(k_dist_so_far[2]),
                "k4_frac":              float(k_dist_so_far[3]),
                "mcts_latency_mean_ms": lat_mean,
                "deadline_miss_rate":   deadline_misses / max(total_meta_steps, 1),
            })

        if (ep + 1) % 10 == 0 or ep == 0:
            logger.info(
                f"Episode {ep+1:3d}/{args.n_episodes}  "
                f"return={ep_ret:.1f}  "
                f"running_mean={np.mean(all_returns):.1f}"
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    returns_arr   = np.array(all_returns)
    latencies_arr = np.array(mcts_latencies) if mcts_latencies else np.array([0.0])
    k_dist        = k_counts / max(k_counts.sum(), 1)

    mean_ret = float(np.mean(returns_arr))
    se_ret   = float(np.std(returns_arr, ddof=1) / np.sqrt(len(returns_arr)))
    miss_pct = 100.0 * deadline_misses / max(total_meta_steps, 1)

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"Real-Time Deployment: {args.game} / {args.gpu_type}  ({args.fps} FPS, {args.n_episodes} episodes)")
    print(sep)
    print(f"Episode return:   {mean_ret:.1f} ± {se_ret:.1f}")
    print(
        f"K distribution:   K1={k_dist[0]*100:.1f}%  K2={k_dist[1]*100:.1f}%  "
        f"K3={k_dist[2]*100:.1f}%  K4={k_dist[3]*100:.1f}%"
    )
    print(
        f"MCTS latency:     mean={latencies_arr.mean():.1f} ms  "
        f"std={latencies_arr.std():.1f} ms  "
        f"p95={np.percentile(latencies_arr, 95):.1f} ms"
    )
    print(f"Deadline misses:  {deadline_misses}/{total_meta_steps} ({miss_pct:.2f}%)")
    print(sep + "\n")

    if not args.no_wandb:
        wandb.summary["mean_return"]        = mean_ret
        wandb.summary["se_return"]          = se_ret
        wandb.summary["mcts_latency_mean_ms"] = float(latencies_arr.mean())
        wandb.summary["mcts_latency_p95_ms"]  = float(np.percentile(latencies_arr, 95))
        wandb.summary["deadline_miss_pct"]  = miss_pct
        for i in range(4):
            wandb.summary[f"k{i+1}_frac"]  = float(k_dist[i])
        wandb.finish()

    result = {
        "game":            args.game,
        "gpu_type":        args.gpu_type,
        "env_name":        args.env_name,
        "fps":             args.fps,
        "frame_ms":        FRAME_S * 1000.0,
        "n_episodes":      args.n_episodes,
        "mean_return":     mean_ret,
        "se_return":       se_ret,
        "k_distribution":  {f"k{k+1}": float(k_dist[k]) for k in range(4)},
        "mcts_latency_ms": {
            "mean": float(latencies_arr.mean()),
            "std":  float(latencies_arr.std()),
            "p50":  float(np.percentile(latencies_arr, 50)),
            "p95":  float(np.percentile(latencies_arr, 95)),
            "p99":  float(np.percentile(latencies_arr, 99)),
            "all":  latencies_arr.tolist(),
        },
        "deadline_misses":  deadline_misses,
        "total_meta_steps": total_meta_steps,
        "all_returns":      returns_arr.tolist(),
    }

    out_path = os.path.join(args.output_dir, "deploy_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
