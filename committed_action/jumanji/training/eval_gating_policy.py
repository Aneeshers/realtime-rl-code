"""Standalone evaluation script for the PacMan adaptive gating policy.

Evaluates a trained gating checkpoint against fixed-K baselines, reporting:
  - Mean +/- SE raw episode return over N episodes
  - MCTS inference time and effective frames/second per (K, sims) pair
  - Per-step K/sims distribution over episode progression (mean + SE fill)
  - Ghost distance and pellet fraction conditioned on K choice
  - Frightened-state usage by K

Usage:
  python -m jumanji.training.eval_gating_policy \\
    --gating_checkpoint_path /path/to/gating_state_best.pkl \\
    --az_checkpoint_path /path/to/training_state_epoch_000050.pkl \\
    [--n_episodes 100] [--output_dir pac_man_eval_results/]
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import pickle
import time
from typing import Dict, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

import jumanji
from jumanji.training.train_pacman_gating_ppo import (
    _sel4,
    load_az_checkpoint,
    make_agents,
    make_gating_forward,
    meta_step_one_k,
)
from jumanji.training.agents.ppo_gating import GatingParamsState
from jumanji.wrappers import VmapAutoResetWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# -- Argument parsing ----------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gating_checkpoint_path", type=str, required=True,
                   help="Path to gating_state_best.pkl or a specific epoch pkl")
    p.add_argument("--az_checkpoint_path", type=str, required=True,
                   help="Path to AZNet training_state_epoch_*.pkl")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--eval_num_envs", type=int, default=100,
                   help="Parallel envs; set equal to n_episodes for a single-pass eval")
    p.add_argument("--eval_meta_steps", type=int, default=1200,
                   help="Max meta-steps per episode (1200 > time_limit=1000 for K=1)")
    p.add_argument("--timing_batch", type=int, default=1,
                   help="Batch size for per-call MCTS timing benchmark")
    p.add_argument("--timing_reps", type=int, default=50,
                   help="Timed repetitions per K (after warmup)")
    p.add_argument("--timing_warmup", type=int, default=10)
    p.add_argument("--sim_options", type=int, nargs=4, default=[32, 64, 96, 128],
                   metavar=("S1", "S2", "S3", "S4"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", type=str, default="pacman_gating_eval")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--output_dir", type=str, default="pac_man_eval_results")
    return p.parse_args()


# -- Helpers -------------------------------------------------------------------

def load_gating_checkpoint(path: str) -> GatingParamsState:
    with open(path, "rb") as f:
        state = pickle.load(f)
    logger.info(f"Loaded gating checkpoint: {path}")
    return state


def mean_se(arr: np.ndarray) -> Tuple[float, float]:
    n = len(arr)
    m = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return m, se


# -- Main ----------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Mutable key holder to avoid nonlocal boilerplate
    _key = [jax.random.PRNGKey(args.seed)]

    def next_key():
        _key[0], sub = jax.random.split(_key[0])
        return sub

    def next_keys(n):
        parts = jax.random.split(_key[0], n + 1)
        _key[0] = parts[0]
        return parts[1:]

    logger.info(f"JAX devices: {jax.devices()}")

    # -- W&B --
    run_name = os.path.basename(os.path.dirname(args.gating_checkpoint_path))
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"eval_{run_name}",
        )

    # -- Environment + agents --
    raw_env = jumanji.make("PacManKT-v1")
    eval_env = VmapAutoResetWrapper(raw_env)
    agents = make_agents(eval_env, args.eval_num_envs, args.sim_options)

    # -- Load checkpoints --
    az_params_state = jax.device_put(load_az_checkpoint(args.az_checkpoint_path))
    gating_state = load_gating_checkpoint(args.gating_checkpoint_path)
    gating_params = gating_state.params
    gating_fwd = make_gating_forward()

    B = args.eval_num_envs

    # -- JIT eval: mirrors jit_eval from training ------------------------------
    # Used for baselines (force_k) and gating summary (force_k=-1).
    # Always accumulates raw (undiscounted) episode return.
    @functools.partial(jax.jit, static_argnames=("greedy", "random_policy", "force_k"))
    def jit_eval(g_params, init_states, init_obs, key, *, greedy, random_policy, force_k=-1):
        az_net_params = jax.lax.stop_gradient(az_params_state.params.net)
        az_net_state  = jax.lax.stop_gradient(az_params_state.net_state)

        def scan_body(carry, step_key):
            states, obs, raw_ret, cum_disc, done, k_cnt = carry
            obs_grid, time_vec = agents[0]._get_grid_and_time(states, obs)
            invalid = agents[0]._get_invalid_actions(agents[0].raw_env_train, states, obs)
            (_, az_val_raw, az_trk_raw), _ = agents[0].forward_with_features.apply(
                az_net_params, az_net_state, obs_grid, time_vec, is_eval=True
            )
            az_trk = jax.lax.stop_gradient(az_trk_raw)
            az_val = jax.lax.stop_gradient(az_val_raw[:, None])

            if force_k >= 0:
                k_choices = jnp.full((B,), force_k, dtype=jnp.int32)
            elif random_policy:
                k_choices = jax.random.randint(step_key, (B,), 0, 4)
            else:
                logits, _ = gating_fwd.apply(g_params, obs_grid, time_vec, az_trk, az_val)
                if greedy:
                    k_choices = jnp.argmax(logits, axis=-1)
                else:
                    k_choices = jax.random.categorical(step_key, logits)

            keys_m = jax.random.split(step_key, 4)
            results = [
                meta_step_one_k(agents[i], az_net_params, az_net_state,
                                states, obs, obs_grid, time_vec, invalid, keys_m[i],
                                reward_mode="raw")
                for i in range(4)
            ]
            r_all  = jnp.stack([r["r_meta"]  for r in results], axis=0)  # (4, B)
            d_all  = jnp.stack([r["discount"] for r in results], axis=0)  # (4, B)
            dn_all = jnp.stack([r["done"]     for r in results], axis=0)  # (4, B)
            arange_B = jnp.arange(B)
            sel_r  = r_all[k_choices, arange_B]
            sel_d  = d_all[k_choices, arange_B]
            sel_dn = dn_all[k_choices, arange_B]

            next_states = jax.vmap(_sel4)(k_choices,
                results[0]["next_states"], results[1]["next_states"],
                results[2]["next_states"], results[3]["next_states"])
            next_obs = jax.vmap(_sel4)(k_choices,
                results[0]["next_obs"], results[1]["next_obs"],
                results[2]["next_obs"], results[3]["next_obs"])

            nd = (~done).astype(jnp.float32)
            raw_ret  = raw_ret + sel_r * nd
            cum_disc = cum_disc * sel_d * nd
            k_cnt    = k_cnt.at[arange_B, k_choices].add((~done).astype(jnp.int32))
            done     = done | sel_dn
            return (next_states, next_obs, raw_ret, cum_disc, done, k_cnt), None

        init_carry = (
            init_states, init_obs,
            jnp.zeros(B), jnp.ones(B),
            jnp.zeros(B, dtype=bool),
            jnp.zeros((B, 4), dtype=jnp.int32),
        )
        step_keys = jax.random.split(key, args.eval_meta_steps)
        (_, _, raw_ret, _, _, k_cnt), _ = jax.lax.scan(scan_body, init_carry, step_keys)
        return raw_ret, k_cnt

    # -- Detailed eval: per-step context tracking ------------------------------
    # Returns per-step arrays for K choice, ghost distance, pellet fraction, etc.
    # Always uses greedy K selection.
    @jax.jit
    def detailed_eval(g_params, init_states, init_obs, key):
        az_net_params = jax.lax.stop_gradient(az_params_state.params.net)
        az_net_state  = jax.lax.stop_gradient(az_params_state.net_state)

        def scan_body(carry, step_key):
            states, obs, raw_ret, done, k_cnt = carry
            active_mask = ~done  # (B,) - was episode alive at START of this step

            obs_grid, time_vec = agents[0]._get_grid_and_time(states, obs)
            invalid = agents[0]._get_invalid_actions(agents[0].raw_env_train, states, obs)
            (_, az_val_raw, az_trk_raw), _ = agents[0].forward_with_features.apply(
                az_net_params, az_net_state, obs_grid, time_vec, is_eval=True
            )
            az_trk = jax.lax.stop_gradient(az_trk_raw)
            az_val = jax.lax.stop_gradient(az_val_raw[:, None])

            logits, _ = gating_fwd.apply(g_params, obs_grid, time_vec, az_trk, az_val)
            k_choices = jnp.argmax(logits, axis=-1)  # greedy

            # Context at decision point from env State.
            # ghost_locations: (B, 4, 2) in (col=y, row=x) convention.
            ghost_col = states.ghost_locations[:, :, 0]  # (B, 4) - y coordinate
            ghost_row = states.ghost_locations[:, :, 1]  # (B, 4) - x coordinate
            player_col = states.player_locations.y[:, None]  # (B, 1)
            player_row = states.player_locations.x[:, None]  # (B, 1)
            ghost_dists = (jnp.abs(ghost_col - player_col) +
                           jnp.abs(ghost_row - player_row))   # (B, 4)
            min_ghost_dist = jnp.min(ghost_dists, axis=-1).astype(jnp.int32)  # (B,)

            # pellets: remaining pellet count (starts at ~316 at reset)
            pellet_frac = states.pellets.astype(jnp.float32) / 316.0  # (B,)
            frightened  = (states.frightened_state_time > 0).astype(jnp.int32)  # (B,)

            keys_m = jax.random.split(step_key, 4)
            results = [
                meta_step_one_k(agents[i], az_net_params, az_net_state,
                                states, obs, obs_grid, time_vec, invalid, keys_m[i],
                                reward_mode="raw")
                for i in range(4)
            ]
            r_all  = jnp.stack([r["r_meta"] for r in results], axis=0)
            dn_all = jnp.stack([r["done"]   for r in results], axis=0)
            arange_B = jnp.arange(B)
            sel_r  = r_all[k_choices, arange_B]
            sel_dn = dn_all[k_choices, arange_B]

            next_states = jax.vmap(_sel4)(k_choices,
                results[0]["next_states"], results[1]["next_states"],
                results[2]["next_states"], results[3]["next_states"])
            next_obs = jax.vmap(_sel4)(k_choices,
                results[0]["next_obs"], results[1]["next_obs"],
                results[2]["next_obs"], results[3]["next_obs"])

            nd = active_mask.astype(jnp.float32)
            raw_ret = raw_ret + sel_r * nd
            k_cnt   = k_cnt.at[arange_B, k_choices].add(active_mask.astype(jnp.int32))
            new_done = done | sel_dn

            step_info = {
                "k_choices":      k_choices,                      # (B,) int 0-3
                "sims_chosen":    (k_choices + 1) * 32,           # (B,) int 32/64/96/128
                "min_ghost_dist": min_ghost_dist,                  # (B,) int
                "pellet_frac":    pellet_frac,                     # (B,) float 0-1
                "frightened":     frightened,                      # (B,) int 0/1
                "active":         active_mask.astype(jnp.int32),  # (B,) 1=episode alive
                "reward":         sel_r,                           # (B,) float
            }
            return (next_states, next_obs, raw_ret, new_done, k_cnt), step_info

        init_carry = (
            init_states, init_obs,
            jnp.zeros(B), jnp.zeros(B, dtype=bool),
            jnp.zeros((B, 4), dtype=jnp.int32),
        )
        step_keys = jax.random.split(key, args.eval_meta_steps)
        (_, _, raw_ret, _, k_cnt), step_data = jax.lax.scan(
            scan_body, init_carry, step_keys
        )
        return raw_ret, k_cnt, step_data

    # -- Batch-run helpers -----------------------------------------------------

    def run_jit_eval(n_episodes, force_k=-1, random_policy=False, greedy=True):
        """Run n_episodes with jit_eval; returns (returns array, k_cnt array)."""
        all_returns, all_kcnt = [], []
        n_batches = int(np.ceil(n_episodes / B))
        for _ in range(n_batches):
            rk, ek = next_keys(2)
            states, ts = eval_env.reset(jax.random.split(rk, B))
            raw_ret, k_cnt = jit_eval(
                gating_params, states, ts.observation, ek,
                greedy=greedy, random_policy=random_policy, force_k=force_k,
            )
            jax.block_until_ready(raw_ret)
            all_returns.append(np.array(raw_ret))
            all_kcnt.append(np.array(k_cnt))
        returns = np.concatenate(all_returns)[:n_episodes]
        k_cnts  = np.concatenate(all_kcnt, axis=0)[:n_episodes]
        return returns, k_cnts

    def run_detailed(n_episodes):
        """Run n_episodes with detailed_eval; returns (returns, step_data)."""
        all_returns, all_step_data = [], []
        n_batches = int(np.ceil(n_episodes / B))
        for _ in range(n_batches):
            rk, ek = next_keys(2)
            states, ts = eval_env.reset(jax.random.split(rk, B))
            raw_ret, _, step_data = detailed_eval(
                gating_params, states, ts.observation, ek,
            )
            jax.block_until_ready((raw_ret, step_data))
            all_returns.append(np.array(raw_ret))
            all_step_data.append(jax.tree_util.tree_map(np.array, step_data))
        returns = np.concatenate(all_returns)[:n_episodes]
        # Concatenate episodes along axis=1 (T, B) -> (T, n_episodes)
        step_data_all = jax.tree_util.tree_map(
            lambda *xs: np.concatenate(xs, axis=1)[:, :n_episodes],
            *all_step_data,
        )
        return returns, step_data_all

    # ===========================================================================
    # 1. MCTS Timing Benchmark
    # ===========================================================================
    logger.info("=== MCTS Timing Benchmark ===")

    # Create a small batch of states for timing
    timing_env = VmapAutoResetWrapper(jumanji.make("PacManKT-v1"))
    t_states, t_ts = timing_env.reset(jax.random.split(jax.random.PRNGKey(0), args.timing_batch))
    t_obs      = t_ts.observation
    t_og, t_tv = agents[0]._get_grid_and_time(t_states, t_obs)
    t_inv      = agents[0]._get_invalid_actions(agents[0].raw_env_train, t_states, t_obs)

    def make_timed_fn(k_idx):
        """Create a JIT'd single-K meta_step fn capturing k_idx by value."""
        _agent = agents[k_idx]
        _az_p  = az_params_state.params.net
        _az_s  = az_params_state.net_state

        @jax.jit
        def fn(key):
            return meta_step_one_k(
                _agent, _az_p, _az_s,
                t_states, t_obs, t_og, t_tv, t_inv,
                key, reward_mode="raw",
            )
        return fn

    timing_results: Dict[int, dict] = {}
    for k_idx in range(4):
        K = k_idx + 1
        sims = args.sim_options[k_idx]
        fn = make_timed_fn(k_idx)

        for _ in range(args.timing_warmup):
            jax.block_until_ready(fn(jax.random.PRNGKey(0)))

        times = []
        for i in range(args.timing_reps):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(jax.random.PRNGKey(i + 1)))
            times.append(time.perf_counter() - t0)

        mean_t = float(np.mean(times))
        std_t  = float(np.std(times))
        fps    = K / mean_t   # K env frames processed per MCTS call
        timing_results[K] = {"sims": sims, "mean_ms": mean_t * 1000,
                              "std_ms": std_t * 1000, "fps": fps}
        logger.info(f"  K={K} sims={sims}: {mean_t*1000:.1f} ± {std_t*1000:.1f} ms  ->  {fps:.1f} frames/s")

    print("\n  K  sims  t_mean(ms)  t_std(ms)  eff_fps")
    print("  " + "-" * 44)
    for K, tr in timing_results.items():
        print(f"  {K}   {tr['sims']:3d}    {tr['mean_ms']:7.1f}    {tr['std_ms']:6.1f}   {tr['fps']:7.1f}")
    print()

    # ===========================================================================
    # 2. Fixed-K Baselines
    # ===========================================================================
    logger.info("=== Baselines ===")
    baseline_results: Dict[str, dict] = {}

    for k_idx in range(4):
        K = k_idx + 1
        returns, _ = run_jit_eval(args.n_episodes, force_k=k_idx)
        m, se = mean_se(returns)
        baseline_results[f"always_k{K}"] = {"mean": m, "se": se}
        logger.info(f"  always_k{K}: {m:.1f} ± {se:.1f}")

    returns_rnd, _ = run_jit_eval(args.n_episodes, random_policy=True, force_k=-1)
    m, se = mean_se(returns_rnd)
    baseline_results["random"] = {"mean": m, "se": se}
    logger.info(f"  random: {m:.1f} ± {se:.1f}")

    # ===========================================================================
    # 3. Gating Policy - Summary
    # ===========================================================================
    logger.info("=== Gating Policy (summary) ===")
    gating_returns, gating_kcnt = run_jit_eval(args.n_episodes, force_k=-1)
    gating_mean, gating_se = mean_se(gating_returns)
    k_dist_raw = gating_kcnt.mean(axis=0)
    k_dist_pct = k_dist_raw / (k_dist_raw.sum() + 1e-8) * 100
    logger.info(f"  return: {gating_mean:.1f} ± {gating_se:.1f}")
    logger.info(f"  K_pct:  {np.round(k_dist_pct, 1)}")

    # ===========================================================================
    # 4. Gating Policy - Detailed Per-Step
    # ===========================================================================
    logger.info("=== Gating Policy (detailed) ===")
    gating_det_returns, step_data = run_detailed(args.n_episodes)
    gating_det_mean, gating_det_se = mean_se(gating_det_returns)
    logger.info(f"  detailed return: {gating_det_mean:.1f} ± {gating_det_se:.1f}")

    # Unpack step_data - each array is (T, N) where T=eval_meta_steps, N=n_episodes
    T, N = step_data["active"].shape
    active       = step_data["active"].astype(bool)        # (T, N)
    k_choices    = step_data["k_choices"]                  # (T, N)
    sims_chosen  = step_data["sims_chosen"]                # (T, N)
    min_ghost    = step_data["min_ghost_dist"]             # (T, N)
    pellet_frac  = step_data["pellet_frac"]                # (T, N)
    frightened   = step_data["frightened"]                 # (T, N)

    # Global K distribution over all active steps
    total_active = active.sum()
    global_k_pct = np.array([
        (active & (k_choices == k)).sum() / total_active * 100
        for k in range(4)
    ])
    logger.info(f"  global K_pct (detailed): {np.round(global_k_pct, 1)}")

    # -- Per-step statistics (vectorised) --------------------------------------
    act_count = active.sum(axis=1)           # (T,) number of still-active episodes
    valid_t   = act_count > 1               # (T,) bool - enough episodes for SE

    # Mean sims chosen per step
    sims_sum    = (sims_chosen * active).sum(axis=1).astype(float)
    step_mean_sims = np.where(valid_t, sims_sum / np.maximum(act_count, 1), np.nan)

    sims_sq_sum = (sims_chosen**2 * active).sum(axis=1).astype(float)
    step_var    = np.where(valid_t,
                           sims_sq_sum / np.maximum(act_count, 1) - step_mean_sims**2,
                           np.nan)
    step_se_sims = np.where(valid_t,
                             np.sqrt(np.maximum(step_var, 0) / np.maximum(act_count, 1)),
                             np.nan)

    # K fraction per step (T, 4)
    step_k_frac = np.full((T, 4), np.nan)
    for k in range(4):
        k_count = (active & (k_choices == k)).sum(axis=1).astype(float)
        step_k_frac[:, k] = np.where(valid_t, k_count / np.maximum(act_count, 1), np.nan)

    # -- Context stats conditioned on K choice ---------------------------------
    flat_active   = active.flatten()
    flat_k        = k_choices.flatten()
    flat_ghost    = min_ghost.flatten().astype(float)
    flat_pellet   = pellet_frac.flatten()
    flat_fright   = frightened.flatten().astype(float)

    ghost_dist_by_k   = {}
    pellet_frac_by_k  = {}
    frightened_by_k   = {}

    for k in range(4):
        mask = flat_active & (flat_k == k)
        n_k  = mask.sum()
        if n_k > 1:
            gd   = flat_ghost[mask]
            pf   = flat_pellet[mask]
            fr   = flat_fright[mask]
            ghost_dist_by_k[k]  = (gd.mean(), gd.std(ddof=1) / np.sqrt(n_k))
            pellet_frac_by_k[k] = (pf.mean(), pf.std(ddof=1) / np.sqrt(n_k))
            frightened_by_k[k]  = (fr.mean(), fr.std(ddof=1) / np.sqrt(n_k))
        else:
            ghost_dist_by_k[k]  = (np.nan, np.nan)
            pellet_frac_by_k[k] = (np.nan, np.nan)
            frightened_by_k[k]  = (np.nan, np.nan)

    logger.info("Ghost dist by K (mean ± SE):")
    for k in range(4):
        m, se = ghost_dist_by_k[k]
        logger.info(f"  K={k+1}: {m:.2f} ± {se:.2f}")
    logger.info("Pellet frac by K:")
    for k in range(4):
        m, se = pellet_frac_by_k[k]
        logger.info(f"  K={k+1}: {m:.3f} ± {se:.3f}")
    logger.info("Frightened % by K:")
    for k in range(4):
        m, se = frightened_by_k[k]
        logger.info(f"  K={k+1}: {m*100:.1f}% ± {se*100:.1f}%")

    # ===========================================================================
    # 5. Plots
    # ===========================================================================
    COLORS   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#777777"]
    K_LABELS = ["K=1 (32)", "K=2 (64)", "K=3 (96)", "K=4 (128)"]
    k_short  = ["K=1", "K=2", "K=3", "K=4"]

    # Plot 1: Return comparison bar chart
    cond_names = [f"always_k{k+1}" for k in range(4)] + ["random", "gating"]
    cond_means = [baseline_results[f"always_k{k+1}"]["mean"] for k in range(4)]
    cond_means += [baseline_results["random"]["mean"], gating_mean]
    cond_ses   = [baseline_results[f"always_k{k+1}"]["se"] for k in range(4)]
    cond_ses   += [baseline_results["random"]["se"], gating_se]
    bar_colors = COLORS[:4] + [COLORS[5], "black"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(cond_names, cond_means, color=bar_colors, alpha=0.85)
    ax.errorbar(range(len(cond_names)), cond_means, yerr=cond_ses,
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.set_ylabel("Raw Episode Return")
    ax.set_title(f"Baseline vs Gating Policy  (n={args.n_episodes} episodes)")
    ax.set_xticks(range(len(cond_names)))
    ax.set_xticklabels(cond_names, rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "return_comparison.png"), dpi=150)
    plt.close()

    # Plot 2: Per-step mean sims (line + SE fill)
    valid_idx = np.where(valid_t)[0]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(valid_idx, step_mean_sims[valid_idx], color="steelblue", linewidth=1.5,
            label="Mean sims chosen")
    ax.fill_between(valid_idx,
                    step_mean_sims[valid_idx] - step_se_sims[valid_idx],
                    step_mean_sims[valid_idx] + step_se_sims[valid_idx],
                    alpha=0.3, color="steelblue", label="±1 SE")
    ax.set_yticks([32, 64, 96, 128])
    ax.set_ylim(20, 140)
    ax.set_xlabel("Meta-step index")
    ax.set_ylabel("Sims chosen")
    ax.set_title(f"Gating Policy: Mean Sims per Meta-Step  (n={N} episodes)")
    ax.legend()
    # Right-side count annotation
    ax2 = ax.twinx()
    ax2.plot(valid_idx, act_count[valid_idx], color="gray", linewidth=0.8,
             linestyle="--", alpha=0.5, label="Active episodes")
    ax2.set_ylabel("Active episodes", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "per_step_sims.png"), dpi=150)
    plt.close()

    # Plot 3: Stacked area - K fraction per step
    fig, ax = plt.subplots(figsize=(11, 4))
    bottoms = np.zeros(len(valid_idx))
    for k in range(4):
        frac_k = step_k_frac[valid_idx, k]
        frac_k = np.where(np.isnan(frac_k), 0, frac_k)
        ax.fill_between(valid_idx, bottoms, bottoms + frac_k,
                        label=K_LABELS[k], alpha=0.85, color=COLORS[k])
        bottoms += frac_k
    ax.set_ylim(0, 1)
    ax.set_xlabel("Meta-step index")
    ax.set_ylabel("Fraction of active episodes")
    ax.set_title("Gating Policy: K Choice Distribution over Episode")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "per_step_k_fractions.png"), dpi=150)
    plt.close()

    # Plot 4: Ghost distance by K
    gd_means = [ghost_dist_by_k[k][0] for k in range(4)]
    gd_ses   = [ghost_dist_by_k[k][1] for k in range(4)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(k_short, gd_means, color=COLORS[:4], alpha=0.85)
    ax.errorbar(range(4), gd_means, yerr=gd_ses,
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.set_ylabel("Mean min. Manhattan distance to nearest ghost")
    ax.set_title("Ghost Proximity when Each K is Chosen")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "ghost_dist_by_k.png"), dpi=150)
    plt.close()

    # Plot 5: Pellet fraction by K
    pf_means = [pellet_frac_by_k[k][0] for k in range(4)]
    pf_ses   = [pellet_frac_by_k[k][1] for k in range(4)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(k_short, pf_means, color=COLORS[:4], alpha=0.85)
    ax.errorbar(range(4), pf_means, yerr=pf_ses,
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.set_ylabel("Mean pellet fraction remaining (1.0 = full board)")
    ax.set_title("Pellet Density when Each K is Chosen")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "pellet_frac_by_k.png"), dpi=150)
    plt.close()

    # Plot 6: Effective FPS vs sims
    k_vals   = [1, 2, 3, 4]
    sims_str = [str(timing_results[K]["sims"]) for K in k_vals]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    raw_ms  = [timing_results[K]["mean_ms"] for K in k_vals]
    raw_std = [timing_results[K]["std_ms"]  for K in k_vals]
    axes[0].bar(sims_str, raw_ms, color=COLORS[:4], alpha=0.85)
    axes[0].errorbar(range(4), raw_ms, yerr=raw_std,
                     fmt="none", color="black", capsize=5)
    axes[0].set_xlabel("Num MCTS simulations")
    axes[0].set_ylabel("Inference time (ms / call)")
    axes[0].set_title(f"MCTS Call Latency  (batch={args.timing_batch})")

    fps_vals = [timing_results[K]["fps"] for K in k_vals]
    axes[1].bar(sims_str, fps_vals, color=COLORS[:4], alpha=0.85)
    for i, (K, fps) in enumerate(zip(k_vals, fps_vals)):
        axes[1].text(i, fps + 0.5, f"{fps:.1f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_xlabel("Num MCTS simulations")
    axes[1].set_ylabel("Effective frames / second")
    axes[1].set_title("Effective FPS  (K frames per MCTS call)")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "fps_vs_sims.png"), dpi=150)
    plt.close()

    # Plot 7: Frightened-state usage by K
    fr_means = [frightened_by_k[k][0] * 100 for k in range(4)]
    fr_ses   = [frightened_by_k[k][1] * 100 for k in range(4)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(k_short, fr_means, color=COLORS[:4], alpha=0.85)
    ax.errorbar(range(4), fr_means, yerr=fr_ses,
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.set_ylabel("% of steps where ghosts were frightened")
    ax.set_title("Frightened-Ghost Steps when Each K is Chosen")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "frightened_by_k.png"), dpi=150)
    plt.close()

    logger.info(f"All plots saved to {args.output_dir}/")

    # ===========================================================================
    # 6. Summary JSON
    # ===========================================================================
    summary = {
        "n_episodes": args.n_episodes,
        "gating_checkpoint": args.gating_checkpoint_path,
        "az_checkpoint":     args.az_checkpoint_path,
        "gating": {
            "mean": gating_mean, "se": gating_se,
            "k_pct": {f"k{k+1}": float(k_dist_pct[k]) for k in range(4)},
        },
        "gating_detailed": {
            "mean": gating_det_mean, "se": gating_det_se,
            "k_pct_global": {f"k{k+1}": float(global_k_pct[k]) for k in range(4)},
        },
        "baselines": {name: {"mean": float(v["mean"]), "se": float(v["se"])}
                      for name, v in baseline_results.items()},
        "timing": {
            f"k{K}": {kk: float(vv) for kk, vv in tr.items()}
            for K, tr in timing_results.items()
        },
        "context_by_k": {
            f"k{k+1}": {
                "ghost_dist_mean": float(ghost_dist_by_k[k][0]),
                "ghost_dist_se":   float(ghost_dist_by_k[k][1]),
                "pellet_frac_mean": float(pellet_frac_by_k[k][0]),
                "pellet_frac_se":   float(pellet_frac_by_k[k][1]),
                "frightened_mean": float(frightened_by_k[k][0]),
                "frightened_se":   float(frightened_by_k[k][1]),
            }
            for k in range(4)
        },
    }
    json_path = os.path.join(args.output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    # ===========================================================================
    # 7. W&B Logging
    # ===========================================================================
    if not args.no_wandb:
        log_dict = {
            "gating/mean_return":    gating_mean,
            "gating/se_return":      gating_se,
            **{f"baseline/{name}_mean": v["mean"] for name, v in baseline_results.items()},
            **{f"baseline/{name}_se":   v["se"]   for name, v in baseline_results.items()},
            **{f"gating/k{k+1}_pct": float(k_dist_pct[k]) for k in range(4)},
            **{f"timing/k{K}_fps":   tr["fps"]     for K, tr in timing_results.items()},
            **{f"timing/k{K}_ms":    tr["mean_ms"] for K, tr in timing_results.items()},
            **{f"context/ghost_dist_k{k+1}": ghost_dist_by_k[k][0] for k in range(4)},
            **{f"context/pellet_frac_k{k+1}": pellet_frac_by_k[k][0] for k in range(4)},
            **{f"context/frightened_k{k+1}": frightened_by_k[k][0] for k in range(4)},
        }
        wandb.log(log_dict)

        plot_files = [
            "return_comparison.png", "per_step_sims.png", "per_step_k_fractions.png",
            "ghost_dist_by_k.png", "pellet_frac_by_k.png",
            "fps_vs_sims.png", "frightened_by_k.png",
        ]
        for fname in plot_files:
            fpath = os.path.join(args.output_dir, fname)
            if os.path.exists(fpath):
                wandb.log({fname.replace(".png", ""): wandb.Image(fpath)})

        wandb.finish()

    logger.info("Eval complete.")


if __name__ == "__main__":
    main()
