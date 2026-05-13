"""Standalone evaluation script for the SnakeKT adaptive gating policy.

Evaluates a trained gating checkpoint against fixed-K baselines, reporting:
  - Mean ± SE raw episode return over N episodes
  - MCTS inference time and effective frames/second per (K, sims) pair
  - Per-step K/sims distribution over episode progression (mean + SE fill)
  - Snake length and board fill fraction conditioned on K choice
  - Valid move count by K (immediate constraint proxy)
  - Fruit distance by K (goal difficulty proxy)
  - Post-eating K distribution (response to body growth event)
  - Local body density near head by K (neighborhood constraint)
  - Flood-fill reachability from head by K (prospective constraint)

Snake-specific context analogues to PacMan/Tetris:
  snake_length  ↔  stack_height   (danger level — longer snake = harder to navigate)
  board_fill    ↔  board_fill     (overall board density = snake_length / total_cells)

Usage:
  python -m jumanji.training.eval_snake_kt_gating_policy \\
    --gating_checkpoint_path /path/to/gating_state_best.pkl \\
    --az_checkpoint_path /path/to/training_state_epoch_000100.pkl \\
    [--n_episodes 100] [--output_dir snake_kt_eval_results/]
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
from jumanji.training.train_snake_kt_gating_ppo import (
    _sel4,
    load_az_checkpoint,
    make_agents,
    make_gating_forward,
    meta_step_one_k,
)
from jumanji.training.agents.ppo_gating.gating_net_tetris import GatingParamsState
from jumanji.wrappers import VmapAutoResetWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gating_checkpoint_path", type=str, required=True)
    p.add_argument("--az_checkpoint_path", type=str, required=True)
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--eval_num_envs", type=int, default=100,
                   help="Parallel envs; set equal to n_episodes for a single-pass eval")
    p.add_argument("--eval_meta_steps", type=int, default=500,
                   help="Max meta-steps per episode (500 >> typical Snake episode length)")
    p.add_argument("--timing_batch", type=int, default=1)
    p.add_argument("--timing_reps", type=int, default=50)
    p.add_argument("--timing_warmup", type=int, default=10)
    p.add_argument("--sim_options", type=int, nargs=4, default=[32, 64, 96, 128],
                   metavar=("S1", "S2", "S3", "S4"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--wandb_project", type=str, default="snake_kt_gating_eval")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--output_dir", type=str, default="snake_kt_eval_results")
    p.add_argument("--env_name", type=str, default="SnakeKT-v1",
                   help="Jumanji env ID to use.")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def compute_local_density(
    body_grid: np.ndarray,
    head_row: np.ndarray,
    head_col: np.ndarray,
    radius: int = 1,
) -> np.ndarray:
    """Fraction of (2r+1)² neighborhood cells occupied by snake body.

    Uses a summed-area table so the per-step cost is O(H*W), not O(r²) per cell.

    body_grid : (T, B, H, W) bool
    head_row  : (T, B) int32
    head_col  : (T, B) int32
    Returns   : (T, B) float32 — density in [0, 1]
    """
    T, B, H, W = body_grid.shape
    side = 2 * radius + 1

    # Pad with 1 (wall = occupied) so neighborhood queries near borders are clean.
    padded = np.pad(
        body_grid.astype(np.float32),
        ((0, 0), (0, 0), (radius, radius), (radius, radius)),
        constant_values=1.0,
    )  # (T, B, H+2r, W+2r)

    # Summed-area table: integral[t, b, r, c] = sum of padded[t, b, :r, :c]
    cumsum = padded.cumsum(axis=2).cumsum(axis=3)
    Hp, Wp = H + 2 * radius, W + 2 * radius
    integral = np.zeros((T, B, Hp + 1, Wp + 1), dtype=np.float32)
    integral[:, :, 1:, 1:] = cumsum

    t_flat = np.repeat(np.arange(T), B)
    b_flat = np.tile(np.arange(B), T)
    hr_flat = head_row.flatten().astype(int)
    hc_flat = head_col.flatten().astype(int)

    # Box corners in padded (= integral offset by 1) coordinates.
    r0, r1 = hr_flat, hr_flat + side
    c0, c1 = hc_flat, hc_flat + side

    box_sum = (
        integral[t_flat, b_flat, r1, c1]
        - integral[t_flat, b_flat, r0, c1]
        - integral[t_flat, b_flat, r1, c0]
        + integral[t_flat, b_flat, r0, c0]
    )
    return (box_sum / (side * side)).reshape(T, B).astype(np.float32)


def compute_reachability(
    body_grid: np.ndarray,
    head_row: np.ndarray,
    head_col: np.ndarray,
) -> np.ndarray:
    """BFS flood-fill count of cells reachable from head without crossing body.

    Processes all (T, B) grids in a single vectorised BFS; terminates early
    when no new cells are discovered (at most H+W expansions needed).

    body_grid : (T, B, H, W) bool  — True = body cell (obstacle)
    head_row  : (T, B) int32
    head_col  : (T, B) int32
    Returns   : (T, B) int32 — reachable cell count (≥ 1, includes head)
    """
    T, B, H, W = body_grid.shape

    # Head cell is the starting point — force it free even if body overlaps.
    free = ~body_grid.astype(bool).copy()
    t_flat = np.repeat(np.arange(T), B)
    b_flat = np.tile(np.arange(B), T)
    hr_flat = head_row.flatten().astype(int)
    hc_flat = head_col.flatten().astype(int)
    free[t_flat, b_flat, hr_flat, hc_flat] = True

    visited = np.zeros((T, B, H, W), dtype=bool)
    visited[t_flat, b_flat, hr_flat, hc_flat] = True

    for _ in range(H + W):  # max BFS depth on an H×W grid
        # 4-connected expansion; boundaries stay zero (no wrap).
        new_cells = np.zeros((T, B, H, W), dtype=bool)
        new_cells[:, :, :-1, :] |= visited[:, :, 1:, :]    # neighbour from below
        new_cells[:, :, 1:,  :] |= visited[:, :, :-1, :]   # neighbour from above
        new_cells[:, :, :, :-1] |= visited[:, :, :, 1:]    # neighbour from right
        new_cells[:, :, :, 1:]  |= visited[:, :, :, :-1]   # neighbour from left
        new_cells &= free
        new_cells &= ~visited
        if not new_cells.any():
            break
        visited |= new_cells

    return visited.sum(axis=(2, 3)).astype(np.int32)  # (T, B)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    _key = [jax.random.PRNGKey(args.seed)]

    def next_keys(n):
        parts = jax.random.split(_key[0], n + 1)
        _key[0] = parts[0]
        return parts[1:]

    logger.info(f"JAX devices: {jax.devices()}")

    run_name = os.path.basename(os.path.dirname(args.gating_checkpoint_path))
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"eval_{run_name}",
        )

    # ── Environment + agents ──
    raw_env = jumanji.make(args.env_name)
    eval_env = VmapAutoResetWrapper(raw_env)
    agents = make_agents(eval_env, args.eval_num_envs, args.sim_options, gamma=args.gamma)

    # Infer grid dims from env spec
    _dummy_obs = raw_env.observation_spec.generate_value()
    H, W = _dummy_obs.grid.shape[0], _dummy_obs.grid.shape[1]
    board_total_cells = H * W
    logger.info(f"SnakeKT grid: {H}×{W}")

    # ── Load checkpoints ──
    az_params_state = jax.device_put(load_az_checkpoint(args.az_checkpoint_path))
    gating_state = load_gating_checkpoint(args.gating_checkpoint_path)
    gating_params = gating_state.params
    gating_fwd = make_gating_forward()

    B = args.eval_num_envs

    # ═══════════════════════════════════════════════════════════════════════════
    # JIT eval — summary (fast baseline + gating return collection)
    # Mirrors jit_eval from training exactly: lax.scan, no Python loop.
    # ═══════════════════════════════════════════════════════════════════════════
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
                k_choices = jnp.argmax(logits, axis=-1) if greedy else jax.random.categorical(step_key, logits)

            keys_m = jax.random.split(step_key, 4)
            results = [
                meta_step_one_k(agents[i], az_net_params, az_net_state,
                                states, obs, obs_grid, time_vec, invalid, keys_m[i],
                                reward_mode="raw")
                for i in range(4)
            ]
            r_all  = jnp.stack([r["r_meta"]  for r in results], axis=0)
            d_all  = jnp.stack([r["discount"] for r in results], axis=0)
            dn_all = jnp.stack([r["done"]     for r in results], axis=0)
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

    # ═══════════════════════════════════════════════════════════════════════════
    # Detailed eval — per-step Snake context tracking (greedy gating only)
    #
    # Tracks at each meta-step (all captured BEFORE the MCTS call):
    #   snake_length  — current snake length (longer = harder to navigate)
    #   board_fill    — snake_length / total_cells (board density)
    #   valid_moves   — number of available actions (1–4); proxy for immediate constraint
    #   fruit_dist    — Manhattan distance head→fruit; proxy for goal difficulty
    #   head_row/col  — head position (needed for post-hoc density + reachability)
    #   body_grid     — full body occupancy (B, H, W) for density + flood-fill
    # ═══════════════════════════════════════════════════════════════════════════
    @jax.jit
    def detailed_eval(g_params, init_states, init_obs, key):
        az_net_params = jax.lax.stop_gradient(az_params_state.params.net)
        az_net_state  = jax.lax.stop_gradient(az_params_state.net_state)

        def scan_body(carry, step_key):
            states, obs, raw_ret, done, k_cnt = carry
            active_mask = ~done  # (B,) — was episode alive at START of this step

            obs_grid, time_vec = agents[0]._get_grid_and_time(states, obs)
            invalid = agents[0]._get_invalid_actions(agents[0].raw_env_train, states, obs)
            (_, az_val_raw, az_trk_raw), _ = agents[0].forward_with_features.apply(
                az_net_params, az_net_state, obs_grid, time_vec, is_eval=True
            )
            az_trk = jax.lax.stop_gradient(az_trk_raw)
            az_val = jax.lax.stop_gradient(az_val_raw[:, None])

            logits, _ = gating_fwd.apply(g_params, obs_grid, time_vec, az_trk, az_val)
            k_choices = jnp.argmax(logits, axis=-1)  # greedy

            # ── Snake context at decision time ────────────────────────────────
            snake_length = states.length.astype(jnp.int32)         # (B,)
            board_fill   = states.length.astype(jnp.float32) / board_total_cells  # (B,)

            # Number of legal actions (1–4); 1 = cornered with only one exit.
            valid_moves = obs.action_mask.sum(axis=-1).astype(jnp.int32)  # (B,)

            # Manhattan distance from head to fruit — goal-reach difficulty.
            fruit_dist = (
                jnp.abs(states.head_position.row - states.fruit_position.row)
                + jnp.abs(states.head_position.col - states.fruit_position.col)
            ).astype(jnp.int32)  # (B,)

            # Raw positional data for post-hoc density + flood-fill.
            head_row = states.head_position.row.astype(jnp.int32)  # (B,)
            head_col = states.head_position.col.astype(jnp.int32)  # (B,)

            # Full body occupancy — used by compute_local_density / compute_reachability.
            body_grid_step = states.body  # (B, H, W) bool

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
            raw_ret  = raw_ret + sel_r * nd
            k_cnt    = k_cnt.at[arange_B, k_choices].add(active_mask.astype(jnp.int32))
            new_done = done | sel_dn

            step_info = {
                "k_choices":    k_choices,                       # (B,) int 0–3
                "sims_chosen":  (k_choices + 1) * 32,            # (B,) int 32/64/96/128
                "snake_length": snake_length,                    # (B,) int
                "board_fill":   board_fill,                      # (B,) float 0–1
                "valid_moves":  valid_moves,                     # (B,) int 1–4
                "fruit_dist":   fruit_dist,                      # (B,) int
                "head_row":     head_row,                        # (B,) int
                "head_col":     head_col,                        # (B,) int
                "body_grid":    body_grid_step,                  # (B, H, W) bool
                "active":       active_mask.astype(jnp.int32),  # (B,) 1=alive
                "reward":       sel_r,                           # (B,) float
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

    # ── Batch-run helpers ─────────────────────────────────────────────────────

    def run_jit_eval(n_episodes, force_k=-1, random_policy=False, greedy=True):
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
        step_data_all = jax.tree_util.tree_map(
            lambda *xs: np.concatenate(xs, axis=1)[:, :n_episodes],
            *all_step_data,
        )
        return returns, step_data_all

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. MCTS Timing Benchmark
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=== MCTS Timing Benchmark ===")

    timing_env = VmapAutoResetWrapper(jumanji.make(args.env_name))
    t_states, t_ts = timing_env.reset(jax.random.split(jax.random.PRNGKey(0), args.timing_batch))
    t_obs      = t_ts.observation
    t_og, t_tv = agents[0]._get_grid_and_time(t_states, t_obs)
    t_inv      = agents[0]._get_invalid_actions(agents[0].raw_env_train, t_states, t_obs)

    def make_timed_fn(k_idx):
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
        fps    = K / mean_t
        timing_results[K] = {"sims": sims, "mean_ms": mean_t * 1000,
                              "std_ms": std_t * 1000, "fps": fps}
        logger.info(f"  K={K} sims={sims}: {mean_t*1000:.1f} ± {std_t*1000:.1f} ms  →  {fps:.1f} frames/s")

    print("\n  K  sims  t_mean(ms)  t_std(ms)  eff_fps")
    print("  " + "-" * 44)
    for K, tr in timing_results.items():
        print(f"  {K}   {tr['sims']:3d}    {tr['mean_ms']:7.1f}    {tr['std_ms']:6.1f}   {tr['fps']:7.1f}")
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Fixed-K Baselines
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=== Baselines ===")
    baseline_results: Dict[str, dict] = {}
    for k_idx in range(4):
        K = k_idx + 1
        returns, _ = run_jit_eval(args.n_episodes, force_k=k_idx)
        m, se = mean_se(returns)
        baseline_results[f"always_k{K}"] = {"mean": m, "se": se}
        logger.info(f"  always_k{K}: {m:.3f} ± {se:.3f}")

    returns_rnd, _ = run_jit_eval(args.n_episodes, random_policy=True, force_k=-1)
    m, se = mean_se(returns_rnd)
    baseline_results["random"] = {"mean": m, "se": se}
    logger.info(f"  random: {m:.3f} ± {se:.3f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Gating Policy — Summary
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=== Gating Policy (summary) ===")
    gating_returns, gating_kcnt = run_jit_eval(args.n_episodes, force_k=-1)
    gating_mean, gating_se = mean_se(gating_returns)
    k_dist_raw = gating_kcnt.mean(axis=0)
    k_dist_pct = k_dist_raw / (k_dist_raw.sum() + 1e-8) * 100
    logger.info(f"  return: {gating_mean:.3f} ± {gating_se:.3f}")
    logger.info(f"  K_pct:  {np.round(k_dist_pct, 1)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Gating Policy — Detailed Per-Step Context
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=== Gating Policy (detailed) ===")
    gating_det_returns, step_data = run_detailed(args.n_episodes)
    gating_det_mean, gating_det_se = mean_se(gating_det_returns)
    logger.info(f"  detailed return: {gating_det_mean:.3f} ± {gating_det_se:.3f}")

    T, N = step_data["active"].shape
    active        = step_data["active"].astype(bool)   # (T, N)
    k_choices     = step_data["k_choices"]              # (T, N)
    sims_chosen   = step_data["sims_chosen"]            # (T, N)
    snake_length  = step_data["snake_length"]           # (T, N) int
    board_fill    = step_data["board_fill"]             # (T, N) float
    valid_moves   = step_data["valid_moves"]            # (T, N) int 1–4
    fruit_dist    = step_data["fruit_dist"]             # (T, N) int
    head_row      = step_data["head_row"]               # (T, N) int
    head_col      = step_data["head_col"]               # (T, N) int
    body_grid     = step_data["body_grid"]              # (T, N, H, W) bool
    reward        = step_data["reward"]                 # (T, N) float

    total_active = active.sum()
    global_k_pct = np.array([
        (active & (k_choices == k)).sum() / total_active * 100
        for k in range(4)
    ])
    logger.info(f"  global K_pct (detailed): {np.round(global_k_pct, 1)}")

    # ── Per-step statistics ───────────────────────────────────────────────────
    act_count = active.sum(axis=1)          # (T,)
    valid_t   = act_count > 1              # (T,)

    # Mean sims per step + SE
    sims_sum       = (sims_chosen * active).sum(axis=1).astype(float)
    step_mean_sims = np.where(valid_t, sims_sum / np.maximum(act_count, 1), np.nan)
    sims_sq_sum    = (sims_chosen**2 * active).sum(axis=1).astype(float)
    step_var       = np.where(valid_t,
                               sims_sq_sum / np.maximum(act_count, 1) - step_mean_sims**2,
                               np.nan)
    step_se_sims   = np.where(valid_t,
                               np.sqrt(np.maximum(step_var, 0) / np.maximum(act_count, 1)),
                               np.nan)

    # K fraction per step (T, 4)
    step_k_frac = np.full((T, 4), np.nan)
    for k in range(4):
        k_count = (active & (k_choices == k)).sum(axis=1).astype(float)
        step_k_frac[:, k] = np.where(valid_t, k_count / np.maximum(act_count, 1), np.nan)

    # ── Context statistics conditioned on K ──────────────────────────────────
    flat_active = active.flatten()
    flat_k      = k_choices.flatten()
    flat_length = snake_length.flatten().astype(float)
    flat_fill   = board_fill.flatten()
    flat_vmoves = valid_moves.flatten().astype(float)
    flat_fdist  = fruit_dist.flatten().astype(float)

    def _by_k(flat_vals):
        """Mean ± SE of flat_vals for each K choice, over active steps."""
        out = {}
        for k in range(4):
            mask = flat_active & (flat_k == k)
            n_k = mask.sum()
            if n_k > 1:
                v = flat_vals[mask]
                out[k] = (v.mean(), v.std(ddof=1) / np.sqrt(n_k))
            else:
                out[k] = (np.nan, np.nan)
        return out

    snake_length_by_k = _by_k(flat_length)
    board_fill_by_k   = _by_k(flat_fill)
    valid_moves_by_k  = _by_k(flat_vmoves)
    fruit_dist_by_k   = _by_k(flat_fdist)

    logger.info("Snake length by K (mean ± SE):")
    for k in range(4):
        m, se = snake_length_by_k[k]
        logger.info(f"  K={k+1}: {m:.2f} ± {se:.2f}")
    logger.info("Board fill by K:")
    for k in range(4):
        m, se = board_fill_by_k[k]
        logger.info(f"  K={k+1}: {m:.3f} ± {se:.3f}")
    logger.info("Valid moves by K (mean ± SE):")
    for k in range(4):
        m, se = valid_moves_by_k[k]
        logger.info(f"  K={k+1}: {m:.3f} ± {se:.3f}")
    logger.info("Fruit distance by K (mean ± SE):")
    for k in range(4):
        m, se = fruit_dist_by_k[k]
        logger.info(f"  K={k+1}: {m:.3f} ± {se:.3f}")

    # ── Post-eating K distribution ───────────────────────────────────────────
    # Collect the K chosen on the meta-step immediately after a fruit was eaten
    # (reward > 0 on step t  →  record k_choices[t+1] if still active).
    ate_mask = (reward > 0.5) & active   # (T, N) bool — fruit-eating events
    post_eat_ks = []
    for t in range(T - 1):
        follow_active = active[t + 1]
        triggered = ate_mask[t] & follow_active
        if triggered.any():
            post_eat_ks.extend(k_choices[t + 1][triggered].tolist())
    post_eat_ks = np.array(post_eat_ks, dtype=np.int32)
    if len(post_eat_ks) > 0:
        post_eat_k_pct = np.array([
            (post_eat_ks == k).sum() / len(post_eat_ks) * 100 for k in range(4)
        ])
    else:
        post_eat_k_pct = np.zeros(4)
    logger.info(f"Post-eating K distribution ({len(post_eat_ks)} events): "
                f"{np.round(post_eat_k_pct, 1)}")

    # ── Local body density and flood-fill reachability (post-hoc numpy) ──────
    # These are computed from the body_grid + head position saved per step.
    logger.info("Computing local body density (3×3 neighbourhood)…")
    local_density = compute_local_density(body_grid, head_row, head_col, radius=1)
    # (T, N) float32

    logger.info("Computing flood-fill reachability (BFS from head)…")
    reachability = compute_reachability(body_grid, head_row, head_col)
    # (T, N) int32

    flat_density  = local_density.flatten()
    flat_reach    = reachability.flatten().astype(float)
    local_density_by_k = _by_k(flat_density)
    reachability_by_k  = _by_k(flat_reach)

    logger.info("Local body density by K (mean ± SE):")
    for k in range(4):
        m, se = local_density_by_k[k]
        logger.info(f"  K={k+1}: {m:.4f} ± {se:.4f}")
    logger.info("Flood-fill reachability by K (mean ± SE):")
    for k in range(4):
        m, se = reachability_by_k[k]
        logger.info(f"  K={k+1}: {m:.2f} ± {se:.2f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. Plots
    # ═══════════════════════════════════════════════════════════════════════════
    COLORS   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#777777"]
    K_LABELS = ["K=1 (32)", "K=2 (64)", "K=3 (96)", "K=4 (128)"]
    k_short  = ["K=1", "K=2", "K=3", "K=4"]
    valid_idx = np.where(valid_t)[0]

    # Plot 1: Return comparison bar chart
    cond_names = [f"always_k{k+1}" for k in range(4)] + ["random", "gating"]
    cond_means = [baseline_results[f"always_k{k+1}"]["mean"] for k in range(4)]
    cond_means += [baseline_results["random"]["mean"], gating_mean]
    cond_ses   = [baseline_results[f"always_k{k+1}"]["se"]   for k in range(4)]
    cond_ses   += [baseline_results["random"]["se"], gating_se]
    bar_colors = COLORS[:4] + [COLORS[5], "black"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(cond_names, cond_means, color=bar_colors, alpha=0.85)
    ax.errorbar(range(len(cond_names)), cond_means, yerr=cond_ses,
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.set_ylabel("Raw Episode Return (fruits eaten)")
    ax.set_title(f"SnakeKT: Baselines vs Gating Policy  (n={args.n_episodes} episodes)")
    ax.set_xticks(range(len(cond_names)))
    ax.set_xticklabels(cond_names, rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "return_comparison.png"), dpi=150)
    plt.close()

    # Plot 2: Per-step mean sims (line + SE fill)
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
    ax.set_title(f"SnakeKT Gating: Mean Sims per Meta-Step  (n={N} episodes)")
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(valid_idx, act_count[valid_idx], color="gray", linewidth=0.8,
             linestyle="--", alpha=0.5, label="Active episodes")
    ax2.set_ylabel("Active episodes", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "per_step_sims.png"), dpi=150)
    plt.close()

    # Plot 3: Stacked area — K fraction per step
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
    ax.set_title("SnakeKT Gating: K Choice Distribution over Episode")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "per_step_k_fractions.png"), dpi=150)
    plt.close()

    # Plot 4: Snake length by K  (analogous to stack_height by K)
    sl_means = [snake_length_by_k[k][0] for k in range(4)]
    sl_ses   = [snake_length_by_k[k][1] for k in range(4)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(k_short, sl_means, color=COLORS[:4], alpha=0.85)
    ax.errorbar(range(4), sl_means, yerr=sl_ses,
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.set_ylabel(f"Mean snake length (cells, max={board_total_cells})")
    ax.set_title("Snake Length when Each K is Chosen")
    ax.set_ylim(0, board_total_cells)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "snake_length_by_k.png"), dpi=150)
    plt.close()

    # Plot 5: Board fill fraction by K
    bf_means = [board_fill_by_k[k][0] for k in range(4)]
    bf_ses   = [board_fill_by_k[k][1] for k in range(4)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(k_short, bf_means, color=COLORS[:4], alpha=0.85)
    ax.errorbar(range(4), bf_means, yerr=bf_ses,
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.set_ylabel("Mean board fill fraction (snake length / total cells)")
    ax.set_title("Board Fill when Each K is Chosen")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "board_fill_by_k.png"), dpi=150)
    plt.close()

    # Plot 6: MCTS latency + effective FPS
    k_vals   = [1, 2, 3, 4]
    sims_str_list = [str(timing_results[K]["sims"]) for K in k_vals]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    raw_ms  = [timing_results[K]["mean_ms"] for K in k_vals]
    raw_std = [timing_results[K]["std_ms"]  for K in k_vals]
    axes[0].bar(sims_str_list, raw_ms, color=COLORS[:4], alpha=0.85)
    axes[0].errorbar(range(4), raw_ms, yerr=raw_std, fmt="none", color="black", capsize=5)
    axes[0].set_xlabel("Num MCTS simulations")
    axes[0].set_ylabel("Inference time (ms / call)")
    axes[0].set_title(f"MCTS Call Latency  (batch={args.timing_batch})")
    fps_vals = [timing_results[K]["fps"] for K in k_vals]
    axes[1].bar(sims_str_list, fps_vals, color=COLORS[:4], alpha=0.85)
    for i, (K, fps) in enumerate(zip(k_vals, fps_vals)):
        axes[1].text(i, fps + 0.5, f"{fps:.1f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_xlabel("Num MCTS simulations")
    axes[1].set_ylabel("Effective frames / second")
    axes[1].set_title("Effective FPS  (K frames per MCTS call)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "fps_vs_sims.png"), dpi=150)
    plt.close()

    # Plot 7: Valid moves + fruit distance by K  (immediate vs goal-reach constraint)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    vm_means = [valid_moves_by_k[k][0] for k in range(4)]
    vm_ses   = [valid_moves_by_k[k][1] for k in range(4)]
    axes[0].bar(k_short, vm_means, color=COLORS[:4], alpha=0.85)
    axes[0].errorbar(range(4), vm_means, yerr=vm_ses,
                     fmt="none", color="black", capsize=5, linewidth=1.5)
    axes[0].set_ylabel("Mean valid moves (1–4)")
    axes[0].set_title("Valid Moves when Each K is Chosen")
    axes[0].set_ylim(0, 4.5)
    axes[0].axhline(y=4, color="gray", linestyle="--", linewidth=0.8, label="Max (4)")
    axes[0].legend(fontsize=9)

    fd_means = [fruit_dist_by_k[k][0] for k in range(4)]
    fd_ses   = [fruit_dist_by_k[k][1] for k in range(4)]
    axes[1].bar(k_short, fd_means, color=COLORS[:4], alpha=0.85)
    axes[1].errorbar(range(4), fd_means, yerr=fd_ses,
                     fmt="none", color="black", capsize=5, linewidth=1.5)
    axes[1].set_ylabel("Mean Manhattan distance (head → fruit)")
    axes[1].set_title("Fruit Distance when Each K is Chosen")
    axes[1].set_ylim(0, H + W)

    plt.suptitle("Constraint Proxies by K Choice", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "constraint_by_k.png"), dpi=150)
    plt.close()

    # Plot 8: Post-eating K distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(post_eat_ks) > 0:
        ax.bar(k_short, post_eat_k_pct, color=COLORS[:4], alpha=0.85)
        ax.set_ylabel("% of post-eating steps")
        ax.set_title(
            f"K Choice Immediately After Eating Fruit\n"
            f"({len(post_eat_ks)} post-eat transitions)"
        )
        ax.set_ylim(0, 105)
        for i, pct in enumerate(post_eat_k_pct):
            ax.text(i, pct + 1.5, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No post-eat events recorded", ha="center", va="center",
                transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "post_eat_k_dist.png"), dpi=150)
    plt.close()

    # Plot 9: Local body density + flood-fill reachability by K
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ld_means = [local_density_by_k[k][0] for k in range(4)]
    ld_ses   = [local_density_by_k[k][1] for k in range(4)]
    axes[0].bar(k_short, ld_means, color=COLORS[:4], alpha=0.85)
    axes[0].errorbar(range(4), ld_means, yerr=ld_ses,
                     fmt="none", color="black", capsize=5, linewidth=1.5)
    axes[0].set_ylabel("Mean body density in 3×3 neighbourhood")
    axes[0].set_title("Local Body Density when Each K is Chosen")
    axes[0].set_ylim(0, 1)

    rc_means = [reachability_by_k[k][0] for k in range(4)]
    rc_ses   = [reachability_by_k[k][1] for k in range(4)]
    axes[1].bar(k_short, rc_means, color=COLORS[:4], alpha=0.85)
    axes[1].errorbar(range(4), rc_means, yerr=rc_ses,
                     fmt="none", color="black", capsize=5, linewidth=1.5)
    axes[1].set_ylabel(f"Mean reachable cells (max={board_total_cells})")
    axes[1].set_title("Flood-Fill Reachability when Each K is Chosen")
    axes[1].set_ylim(0, board_total_cells * 1.05)

    plt.suptitle("Spatial Constraint Proxies by K Choice", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "spatial_constraint_by_k.png"), dpi=150)
    plt.close()

    logger.info(f"All plots saved to {args.output_dir}/")

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. Summary JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def _fmt_by_k(by_k_dict, suffix_mean="mean", suffix_se="se"):
        return {
            f"k{k+1}": {
                suffix_mean: float(by_k_dict[k][0]),
                suffix_se:   float(by_k_dict[k][1]),
            }
            for k in range(4)
        }

    summary = {
        "n_episodes":        args.n_episodes,
        "gating_checkpoint": args.gating_checkpoint_path,
        "az_checkpoint":     args.az_checkpoint_path,
        "grid_dims":         {"H": H, "W": W},
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
            "snake_length":    _fmt_by_k(snake_length_by_k),
            "board_fill":      _fmt_by_k(board_fill_by_k),
            "valid_moves":     _fmt_by_k(valid_moves_by_k),
            "fruit_dist":      _fmt_by_k(fruit_dist_by_k),
            "local_density":   _fmt_by_k(local_density_by_k),
            "reachability":    _fmt_by_k(reachability_by_k),
        },
        "post_eat_k_pct": {f"k{k+1}": float(post_eat_k_pct[k]) for k in range(4)},
        "post_eat_n_events": int(len(post_eat_ks)),
    }
    json_path = os.path.join(args.output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. W&B Logging
    # ═══════════════════════════════════════════════════════════════════════════
    if not args.no_wandb:
        log_dict = {
            "gating/mean_return":    gating_mean,
            "gating/se_return":      gating_se,
            **{f"baseline/{name}_mean": v["mean"] for name, v in baseline_results.items()},
            **{f"baseline/{name}_se":   v["se"]   for name, v in baseline_results.items()},
            **{f"gating/k{k+1}_pct":    float(k_dist_pct[k]) for k in range(4)},
            **{f"timing/k{K}_fps":       tr["fps"]     for K, tr in timing_results.items()},
            **{f"timing/k{K}_ms":        tr["mean_ms"] for K, tr in timing_results.items()},
            # context by K
            **{f"context/snake_length_k{k+1}":  snake_length_by_k[k][0]  for k in range(4)},
            **{f"context/board_fill_k{k+1}":    board_fill_by_k[k][0]    for k in range(4)},
            **{f"context/valid_moves_k{k+1}":   valid_moves_by_k[k][0]   for k in range(4)},
            **{f"context/fruit_dist_k{k+1}":    fruit_dist_by_k[k][0]    for k in range(4)},
            **{f"context/local_density_k{k+1}": local_density_by_k[k][0] for k in range(4)},
            **{f"context/reachability_k{k+1}":  reachability_by_k[k][0]  for k in range(4)},
            # post-eat
            **{f"post_eat/k{k+1}_pct": float(post_eat_k_pct[k]) for k in range(4)},
            "post_eat/n_events": int(len(post_eat_ks)),
        }
        wandb.log(log_dict)

        plot_files = [
            "return_comparison.png", "per_step_sims.png", "per_step_k_fractions.png",
            "snake_length_by_k.png", "board_fill_by_k.png", "fps_vs_sims.png",
            "constraint_by_k.png", "post_eat_k_dist.png", "spatial_constraint_by_k.png",
        ]
        for fname in plot_files:
            fpath = os.path.join(args.output_dir, fname)
            if os.path.exists(fpath):
                wandb.log({fname.replace(".png", ""): wandb.Image(fpath)})

        wandb.finish()

    logger.info("Eval complete.")


if __name__ == "__main__":
    main()
