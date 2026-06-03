"""Sweep one (K, sims) pair and record mean episode return.

Designed to be run as a SLURM array job. The array index maps to a
(K, sims) combination; results are written to a shared JSON file and
optionally logged to wandb.

Array layout (--array=0-47):
    SIM_VALUES = [2, 4, 8, 16, 32, 64, 128, 256]   (8 values)
    K_VALUES   = [1, 2, 3, 4, 5, 6]                 (6 values)
    job_id -> k_idx = job_id // 8,  sims_idx = job_id % 8

Uses eval_one_episode_kt (while_loop until done) for unbiased full-episode returns.
The gating training's jit_eval caps at eval_meta_steps=500 meta-steps which truncates
K=1 episodes (time_limit=1000 real steps -> 1000 meta-steps for K=1) and inflates
K=1 returns. This sweep uses full episodes so numbers are directly interpretable.
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import json
import logging
import os
import pickle
import time

import jax
import jax.numpy as jnp
import wandb

import jumanji
from jumanji.training.agents.gumbel_alphazero import GumbelAlphaZeroAgent
from jumanji.training.training_types import AlphaZeroParamsState, TrainingState
from jumanji.training.eval_pacman_kt_cross import eval_one_episode_kt
from jumanji.wrappers import VmapAutoResetWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SIM_VALUES = [2, 4, 8, 16, 32, 64, 128, 256]
K_VALUES   = [1, 2, 3, 4, 5, 6]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--array_id", type=int, default=None,
                   help="SLURM array task ID (0-47). Overrides --k and --sims.")
    p.add_argument("--k", type=int, default=None, help="Action delay K (1-6).")
    p.add_argument("--sims", type=int, default=None, help="MCTS simulation count.")
    p.add_argument("--eval_batch_size", type=int, default=64,
                   help="Number of episodes to average over.")
    p.add_argument("--az_checkpoint_dir", type=str,
                   default="./checkpoints/committed_action/pacman/base/k1")
    p.add_argument("--results_dir", type=str,
                   default="./eval_outputs/pac_man_sim_k_sweep")
    p.add_argument("--wandb_project", type=str, default="pacman_sim_k_sweep")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--num_channels", type=int, default=128)
    p.add_argument("--num_blocks", type=int, default=6)
    p.add_argument("--time_embed_dim", type=int, default=32)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_az_checkpoint(checkpoint_dir: str) -> AlphaZeroParamsState:
    best_path = os.path.join(checkpoint_dir, "training_state_best.pkl")

    def _fallback(reason: str) -> str:
        epoch_files = sorted(glob.glob(
            os.path.join(checkpoint_dir, "training_state_epoch_*.pkl")
        ))
        if not epoch_files:
            raise ValueError(f"{reason} and no epoch checkpoints in '{checkpoint_dir}'.")
        fb = epoch_files[-1]
        logger.warning(f"{reason}. Falling back to: {fb}")
        return fb

    try:
        path = best_path
        with open(path, "rb") as f:
            state: TrainingState = pickle.load(f)
    except FileNotFoundError:
        path = _fallback(f"'{best_path}' not found")
        with open(path, "rb") as f:
            state = pickle.load(f)

    if state is None or getattr(state, "params_state", None) is None:
        path = _fallback(f"'{path}' contains None")
        with open(path, "rb") as f:
            state = pickle.load(f)

    logger.info(f"Loaded AZNet from: {path}")
    params_state = jax.tree_util.tree_map(
        lambda x: x[0] if hasattr(x, "shape") and x.ndim > 0 else x,
        state.params_state,
    )
    return params_state


def append_result(results_file: str, record: dict) -> None:
    """Append one result record to a shared JSON-lines file (file-locked)."""
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(record) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def main():
    args = parse_args()

    # Resolve (K, sims) from array_id or explicit flags
    if args.array_id is not None:
        k_idx   = args.array_id // len(SIM_VALUES)
        sims_idx = args.array_id % len(SIM_VALUES)
        K    = K_VALUES[k_idx]
        sims = SIM_VALUES[sims_idx]
    elif args.k is not None and args.sims is not None:
        K    = args.k
        sims = args.sims
    else:
        raise ValueError("Provide --array_id, or both --k and --sims.")

    logger.info(f"Evaluating K={K}, sims={sims}")

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"K{K}_sims{sims}",
            config={"K": K, "sims": sims, "eval_batch_size": args.eval_batch_size,
                    "seed": args.seed},
        )

    # -- Environment --
    raw_env = jumanji.make("PacManKT-v1")

    # -- Agent for this (K, sims) pair --
    wrapped_env = VmapAutoResetWrapper(raw_env)
    agent = GumbelAlphaZeroAgent(
        env=wrapped_env,
        n_steps=1,
        total_batch_size=args.eval_batch_size,
        num_simulations=sims,
        gamma=args.gamma,
        learning_rate=3e-4,
        num_channels=args.num_channels,
        num_blocks=args.num_blocks,
        time_embed_dim=args.time_embed_dim,
        pacman_action_delay=K,
    )

    # -- Load frozen AZNet --
    params_state = load_az_checkpoint(args.az_checkpoint_dir)
    params_state = jax.device_put(params_state)

    # -- Batched eval via vmap over episodes --
    key = jax.random.PRNGKey(args.seed)
    eval_keys = jax.random.split(key, args.eval_batch_size)

    logger.info(f"Compiling eval (K={K}, sims={sims})...")
    t0 = time.time()

    # Warmup on 1 episode to trigger JIT
    warmup_fn = jax.vmap(lambda k: eval_one_episode_kt(agent, raw_env, params_state, k, K))
    _ = jax.block_until_ready(warmup_fn(jax.random.split(key, 1)))
    logger.info(f"  Compile done in {time.time()-t0:.1f}s")

    t1 = time.time()
    eval_fn = jax.vmap(lambda k: eval_one_episode_kt(agent, raw_env, params_state, k, K))
    metrics = jax.block_until_ready(eval_fn(eval_keys))
    elapsed = time.time() - t1

    returns = metrics["episode_return"]   # (batch,)
    lengths  = metrics["episode_length"]  # (batch,)
    mean_return = float(jnp.mean(returns))
    std_return  = float(jnp.std(returns))
    mean_length = float(jnp.mean(lengths))

    logger.info(
        f"K={K}, sims={sims}: "
        f"mean_return={mean_return:.1f} ± {std_return:.1f}, "
        f"mean_length={mean_length:.1f}, "
        f"elapsed={elapsed:.1f}s"
    )

    record = {
        "K": K,
        "sims": sims,
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_length": mean_length,
        "n_episodes": args.eval_batch_size,
        "seed": args.seed,
    }

    results_file = os.path.join(args.results_dir, "results.jsonl")
    append_result(results_file, record)
    logger.info(f"Result written to {results_file}")

    if not args.no_wandb:
        wandb.log(record)
        wandb.finish()


if __name__ == "__main__":
    main()
