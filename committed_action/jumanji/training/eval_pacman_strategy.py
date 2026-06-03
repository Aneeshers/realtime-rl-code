"""Strategy-bin evaluation for the PacMan adaptive gating policy.

Runs 50 episodes (vmap), computes per-episode K-choice frequency in 10
normalized episode-progress bins, and logs mean + SE to wandb as scalars.
No baselines, no MCTS timing, no context analysis.

Wandb keys logged:
  gating/mean_return, gating/se_return
  strategy/bin{00-09}_k{1-4}_mean
  strategy/bin{00-09}_k{1-4}_se

Usage:
  python -m jumanji.training.eval_pacman_strategy \\
    --gating_checkpoint_path /path/to/gating_state_best.pkl \\
    --az_checkpoint_path     /path/to/training_state_epoch_000050.pkl
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from typing import Tuple

import jax
import jax.numpy as jnp
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gating_checkpoint_path", type=str, required=True)
    p.add_argument("--az_checkpoint_path",     type=str, required=True)
    p.add_argument("--n_episodes",     type=int, default=50)
    p.add_argument("--eval_num_envs",  type=int, default=50,
                   help="Parallel envs; set equal to n_episodes for a single-pass eval")
    p.add_argument("--eval_meta_steps", type=int, default=1200,
                   help="Max meta-steps per episode")
    p.add_argument("--sim_options", type=int, nargs=4, default=[32, 64, 96, 128],
                   metavar=("S1", "S2", "S3", "S4"))
    p.add_argument("--n_bins",    type=int, default=10,
                   help="Number of normalized episode-progress bins")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--wandb_project", type=str, default="pacman_strategy_eval")
    p.add_argument("--wandb_entity",  type=str, default=None)
    p.add_argument("--no_wandb",      action="store_true")
    p.add_argument("--output_dir",    type=str, default="pacman_strategy_eval_results")
    return p.parse_args()


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


def compute_strategy_bins(
    k_choices: np.ndarray,
    active: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-episode K-frequency in normalized episode-progress bins.

    k_choices : (T, N) int, K choice index 0-3 at each meta-step
    active    : (T, N) int/bool, 1 = episode alive at start of step
    Returns (mean_freq, se_freq) each of shape (n_bins, 4).
    """
    T, N = active.shape
    active_b = active.astype(bool)
    bin_k_freq = np.zeros((N, n_bins, 4), dtype=np.float64)

    for e in range(N):
        active_steps = np.where(active_b[:, e])[0]
        L_e = len(active_steps)
        if L_e == 0:
            continue
        for rank, t in enumerate(active_steps):
            bin_idx = min(int(rank / L_e * n_bins), n_bins - 1)
            bin_k_freq[e, bin_idx, k_choices[t, e]] += 1
        for b in range(n_bins):
            total = bin_k_freq[e, b].sum()
            if total > 0:
                bin_k_freq[e, b] /= total

    mean_freq = bin_k_freq.mean(axis=0)                       # (n_bins, 4)
    se_freq   = bin_k_freq.std(axis=0) / np.sqrt(N)           # (n_bins, 4)
    return mean_freq, se_freq


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
            name=f"strategy_{run_name}",
        )

    raw_env  = jumanji.make("PacManKT-v1")
    eval_env = VmapAutoResetWrapper(raw_env)
    agents   = make_agents(eval_env, args.eval_num_envs, args.sim_options)

    az_params_state = jax.device_put(load_az_checkpoint(args.az_checkpoint_path))
    gating_state    = load_gating_checkpoint(args.gating_checkpoint_path)
    gating_params   = gating_state.params
    gating_fwd      = make_gating_forward()

    B = args.eval_num_envs

    @jax.jit
    def detailed_eval(g_params, init_states, init_obs, key):
        az_net_params = jax.lax.stop_gradient(az_params_state.params.net)
        az_net_state  = jax.lax.stop_gradient(az_params_state.net_state)

        def scan_body(carry, step_key):
            states, obs, raw_ret, done, k_cnt = carry
            active_mask = ~done

            obs_grid, time_vec = agents[0]._get_grid_and_time(states, obs)
            invalid = agents[0]._get_invalid_actions(agents[0].raw_env_train, states, obs)
            (_, az_val_raw, az_trk_raw), _ = agents[0].forward_with_features.apply(
                az_net_params, az_net_state, obs_grid, time_vec, is_eval=True
            )
            az_trk = jax.lax.stop_gradient(az_trk_raw)
            az_val = jax.lax.stop_gradient(az_val_raw[:, None])

            logits, _ = gating_fwd.apply(g_params, obs_grid, time_vec, az_trk, az_val)
            k_choices  = jnp.argmax(logits, axis=-1)

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

            nd       = active_mask.astype(jnp.float32)
            raw_ret  = raw_ret + sel_r * nd
            k_cnt    = k_cnt.at[arange_B, k_choices].add(active_mask.astype(jnp.int32))
            new_done = done | sel_dn

            step_info = {
                "k_choices": k_choices,                      # (B,) int 0-3
                "active":    active_mask.astype(jnp.int32),  # (B,) 1=alive
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

    logger.info("=== Strategy Eval ===")
    returns, step_data = run_detailed(args.n_episodes)
    gating_mean, gating_se = mean_se(returns)
    logger.info(f"  return: {gating_mean:.2f} ± {gating_se:.2f}")

    k_choices = step_data["k_choices"]  # (T, N)
    active    = step_data["active"]     # (T, N)

    mean_freq, se_freq = compute_strategy_bins(k_choices, active, args.n_bins)
    logger.info(f"  mean_freq shape: {mean_freq.shape}")
    logger.info(f"  global K pct: {(active.astype(bool)[:, :, None] * (k_choices[:, :, None] == np.arange(4))).sum(axis=(0,1)) / active.astype(bool).sum() * 100}")

    log_dict: dict = {
        "gating/mean_return": gating_mean,
        "gating/se_return":   gating_se,
    }
    for b in range(args.n_bins):
        for k in range(4):
            log_dict[f"strategy/bin{b:02d}_k{k+1}_mean"] = float(mean_freq[b, k])
            log_dict[f"strategy/bin{b:02d}_k{k+1}_se"]   = float(se_freq[b, k])

    if not args.no_wandb:
        wandb.log(log_dict)
        wandb.finish()

    logger.info("Done.")


if __name__ == "__main__":
    main()
