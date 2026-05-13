"""Input-ablation feature-importance analysis for the TetrisRT gating policy.

For each ablation condition (baseline, zero_obs, zero_time, zero_trunk, zero_value):
  1. Runs 50 full episodes where the gating policy receives the ablated input
     but MCTS/env use real observations — gives actual episode returns.
  2. Records per-step K choices and logits for distribution analysis.

Logged per condition:
  - Episode return (mean ± SE)
  - K distribution (% of steps choosing each K)
  - KL divergence of K distribution vs. baseline
  - Mean logit entropy
  - K agreement rate vs. baseline

Usage:
  python -m jumanji.training.eval_tetris_ablation \\
    --gating_checkpoint_path /path/to/gating_state_best.pkl \\
    --az_checkpoint_path     /path/to/training_state_epoch_000050.pkl
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import pickle
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import wandb

import jumanji
from jumanji.training.train_tetris_rt_gating_ppo import (
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

# Ablation modes (used as static_argnames in JIT)
ABL_NONE  = 0
ABL_OBS   = 1
ABL_TIME  = 2
ABL_TRUNK = 3
ABL_VALUE = 4

ABL_NAMES = {
    ABL_NONE:  "baseline",
    ABL_OBS:   "zero_obs",
    ABL_TIME:  "zero_time",
    ABL_TRUNK: "zero_trunk",
    ABL_VALUE: "zero_value",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gating_checkpoint_path", type=str, required=True)
    p.add_argument("--az_checkpoint_path",     type=str, required=True)
    p.add_argument("--n_episodes",     type=int, default=50)
    p.add_argument("--eval_num_envs",  type=int, default=50,
                   help="Parallel envs; set equal to n_episodes for single-pass eval")
    p.add_argument("--eval_meta_steps", type=int, default=2000,
                   help="Max meta-steps per episode")
    p.add_argument("--sim_options", type=int, nargs=4, default=[32, 64, 96, 128],
                   metavar=("S1", "S2", "S3", "S4"))
    p.add_argument("--env_name",  type=str, default="TetrisRTKT-v0")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--wandb_project", type=str, default="tetris_ablation_eval")
    p.add_argument("--wandb_entity",  type=str, default=None)
    p.add_argument("--no_wandb",      action="store_true")
    p.add_argument("--output_dir",    type=str, default="tetris_ablation_results")
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
            name=f"ablation_{run_name}",
        )

    raw_env  = jumanji.make(args.env_name)
    eval_env = VmapAutoResetWrapper(raw_env)
    agents   = make_agents(eval_env, args.eval_num_envs, args.sim_options)

    az_params_state = jax.device_put(load_az_checkpoint(args.az_checkpoint_path))
    gating_state    = load_gating_checkpoint(args.gating_checkpoint_path)
    gating_params   = gating_state.params
    gating_fwd      = make_gating_forward()

    B = args.eval_num_envs

    # ── JIT-compiled eval with ablation mode ──────────────────────────────────
    @functools.partial(jax.jit, static_argnames=("ablation_mode",))
    def eval_ablated(g_params, init_states, init_obs, key, *, ablation_mode):
        az_net_params = jax.lax.stop_gradient(az_params_state.params.net)
        az_net_state  = jax.lax.stop_gradient(az_params_state.net_state)

        def scan_body(carry, step_key):
            states, obs, raw_ret, done, k_cnt = carry
            active_mask = ~done

            # Real features for MCTS
            obs_grid, time_vec = agents[0]._get_grid_and_time(states, obs)
            invalid = agents[0]._get_invalid_actions(agents[0].raw_env_train, states, obs)
            (_, az_val_raw, az_trk_raw), _ = agents[0].forward_with_features.apply(
                az_net_params, az_net_state, obs_grid, time_vec, is_eval=True
            )
            az_trk = jax.lax.stop_gradient(az_trk_raw)
            az_val = jax.lax.stop_gradient(az_val_raw[:, None])

            # Ablated features for gating decision
            g_obs  = jnp.where(ablation_mode == ABL_OBS,   jnp.zeros_like(obs_grid), obs_grid)
            g_time = jnp.where(ablation_mode == ABL_TIME,  jnp.zeros_like(time_vec), time_vec)
            g_trk  = jnp.where(ablation_mode == ABL_TRUNK, jnp.zeros_like(az_trk),   az_trk)
            g_val  = jnp.where(ablation_mode == ABL_VALUE, jnp.zeros_like(az_val),   az_val)

            logits, _ = gating_fwd.apply(g_params, g_obs, g_time, g_trk, g_val)
            k_choices = jnp.argmax(logits, axis=-1)

            # Execute chosen K using real features
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
                "k_choices": k_choices,
                "logits":    logits,
                "active":    active_mask.astype(jnp.int32),
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

    # ── Run all ablation conditions ───────────────────────────────────────────
    results: Dict[str, dict] = {}
    baseline_k_flat = None
    baseline_k_dist = None

    for abl_mode in [ABL_NONE, ABL_OBS, ABL_TIME, ABL_TRUNK, ABL_VALUE]:
        abl_name = ABL_NAMES[abl_mode]
        logger.info(f"=== Running ablation: {abl_name} ===")

        # Use the same seed for env reset so episodes are comparable
        rng = jax.random.PRNGKey(args.seed + 1000)
        rk, ek = jax.random.split(rng)
        states, ts = eval_env.reset(jax.random.split(rk, B))

        raw_ret, k_cnt, step_data = eval_ablated(
            gating_params, states, ts.observation, ek,
            ablation_mode=abl_mode,
        )
        jax.block_until_ready((raw_ret, step_data))

        returns_np = np.array(raw_ret)
        ret_mean, ret_se = mean_se(returns_np)

        sd = jax.tree_util.tree_map(np.array, step_data)
        active = sd["active"].astype(bool)
        k_choices = sd["k_choices"]
        logits_all = sd["logits"]

        active_mask_flat = active.flatten()
        k_flat = k_choices.flatten()
        active_k = k_flat[active_mask_flat]
        k_dist = np.array([np.mean(active_k == k) for k in range(4)])

        # Logit entropy
        logits_flat = logits_all.reshape(-1, 4)
        probs_flat = np.exp(logits_flat - logits_flat.max(axis=-1, keepdims=True))
        probs_flat = probs_flat / probs_flat.sum(axis=-1, keepdims=True)
        entropy_flat = -np.sum(probs_flat * np.log(probs_flat + 1e-10), axis=-1)
        mean_entropy = float(np.mean(entropy_flat[active_mask_flat]))

        # Agreement and KL vs baseline
        if abl_mode == ABL_NONE:
            baseline_k_flat = k_flat.copy()
            baseline_k_dist = k_dist.copy()
            agreement = 1.0
            kl_div = 0.0
        else:
            agreement = float(np.mean(
                k_flat[active_mask_flat] == baseline_k_flat[active_mask_flat]
            ))
            kl_div = float(np.sum(
                baseline_k_dist * np.log(
                    (baseline_k_dist + 1e-10) / (k_dist + 1e-10)
                )
            ))

        results[abl_name] = {
            "return_mean": ret_mean,
            "return_se":   ret_se,
            "k_dist": {f"k{k+1}": float(k_dist[k]) for k in range(4)},
            "mean_entropy": mean_entropy,
            "k_agreement": agreement,
            "kl_from_baseline": kl_div,
        }

        logger.info(f"  Return: {ret_mean:.2f} ± {ret_se:.2f}")
        logger.info(f"  K dist: {np.round(k_dist * 100, 1)}%")
        logger.info(f"  Entropy: {mean_entropy:.3f}")
        logger.info(f"  Agreement: {agreement:.3f}")
        logger.info(f"  KL from baseline: {kl_div:.4f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    logger.info("\n=== Feature Importance Summary ===")
    logger.info(
        f"{'Condition':<15} {'Return':>12} {'K1%':>6} {'K2%':>6} "
        f"{'K3%':>6} {'K4%':>6} {'Entropy':>8} {'Agree%':>8} {'KL':>8}"
    )
    logger.info("-" * 95)
    for name, r in results.items():
        kd = r["k_dist"]
        logger.info(
            f"{name:<15} "
            f"{r['return_mean']:>6.1f}±{r['return_se']:<4.1f} "
            f"{kd['k1']*100:>5.1f}% {kd['k2']*100:>5.1f}% "
            f"{kd['k3']*100:>5.1f}% {kd['k4']*100:>5.1f}% "
            f"{r['mean_entropy']:>8.3f} {r['k_agreement']*100:>7.1f}% "
            f"{r['kl_from_baseline']:>8.4f}"
        )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    summary = {
        "n_episodes": args.n_episodes,
        "gating_checkpoint": args.gating_checkpoint_path,
        "az_checkpoint": args.az_checkpoint_path,
        "ablations": results,
    }
    json_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    # ── W&B ───────────────────────────────────────────────────────────────────
    if not args.no_wandb:
        log_dict = {}
        for name, r in results.items():
            log_dict[f"ablation/{name}_return_mean"] = r["return_mean"]
            log_dict[f"ablation/{name}_return_se"]   = r["return_se"]
            for k in range(4):
                log_dict[f"ablation/{name}_k{k+1}_pct"] = r["k_dist"][f"k{k+1}"] * 100
            log_dict[f"ablation/{name}_entropy"]   = r["mean_entropy"]
            log_dict[f"ablation/{name}_agreement"] = r["k_agreement"]
            log_dict[f"ablation/{name}_kl"]        = r["kl_from_baseline"]
        wandb.log(log_dict)
        wandb.finish()

    logger.info("Ablation eval complete.")


if __name__ == "__main__":
    main()
