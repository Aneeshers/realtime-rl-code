"""
Evaluate a trained Gumbel-AlphaZero checkpoint on Sokoban at varying MCTS sim budgets.

Usage (standalone):
    python -m jumanji.training.eval_mcts \
        --checkpoint /path/to/training_state_epoch_000090.pkl \
        --sims 2 4 8 16 32 64 \
        --eval_batch_size 128 \
        --wandb_project sokoban_gumbel_az_eval \
        --wandb_entity null \
        --seed 42

Each sim count creates a separate W&B run so curves are easy to compare.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import wandb

import jumanji
from jumanji.environments import Sokoban
from jumanji.training.agents.gumbel_alphazero import GumbelAlphaZeroAgent
from jumanji.training.training_types import (
    ActingState,
    AlphaZeroParamsState,
    TrainingState,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Evaluation logic (single-episode, vmapped, no pmap)
# ──────────────────────────────────────────────────────────


def eval_one_episode(
    agent: GumbelAlphaZeroAgent,
    eval_env: Sokoban,
    params_state: AlphaZeroParamsState,
    key,
):
    """Run a single greedy episode using MCTS and return metrics dict."""

    policy = agent.make_policy(
        params_state=params_state,
        stochastic=False,
        eval_env=eval_env,
    )

    def cond_fun(carry):
        acting_state, _ = carry
        return ~acting_state.timestep.last()

    def body_fun(carry):
        acting_state, return_ = carry
        key, action_key = jax.random.split(acting_state.key)

        action = policy(
            acting_state.state,
            acting_state.timestep.observation,
            action_key,
        )

        state, timestep = eval_env.step(acting_state.state, action)
        return_ += timestep.reward

        acting_state = ActingState(
            state=state,
            timestep=timestep,
            key=key,
            episode_count=jnp.array(0, jnp.int32),
            env_step_count=acting_state.env_step_count + 1,
        )
        return acting_state, return_

    reset_key, init_key = jax.random.split(key)
    state, timestep = eval_env.reset(reset_key)
    acting_state = ActingState(
        state=state,
        timestep=timestep,
        key=init_key,
        episode_count=jnp.array(0, jnp.int32),
        env_step_count=jnp.array(0, jnp.int32),
    )
    return_ = jnp.array(0.0, jnp.float32)

    final_acting_state, return_ = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (acting_state, return_),
    )

    eval_metrics = {
        "episode_return": return_,
        "episode_length": final_acting_state.env_step_count,
    }
    extras = final_acting_state.timestep.extras
    if extras:
        eval_metrics.update(extras)
    return eval_metrics


def run_batch_eval(
    agent: GumbelAlphaZeroAgent,
    eval_env: Sokoban,
    params_state: AlphaZeroParamsState,
    key,
    batch_size: int,
) -> Tuple[Dict[str, float], float]:
    """Evaluate `batch_size` episodes (vmapped) and return (mean_metrics, time_per_episode_sec)."""
    keys = jax.random.split(key, batch_size)

    eval_fn = jax.vmap(
        lambda k: eval_one_episode(agent, eval_env, params_state, k)
    )

    # Warmup: run once to trigger JIT compilation, then discard
    logger.info("  Warmup JIT compile (1 episode)...")
    warmup_key = jax.random.split(key, 1)
    warmup_fn = jax.vmap(
        lambda k: eval_one_episode(agent, eval_env, params_state, k)
    )
    _ = jax.block_until_ready(warmup_fn(warmup_key))
    logger.info("  Warmup done. Running timed evaluation...")

    # Timed run
    t0 = time.time()
    metrics = eval_fn(keys)
    jax.block_until_ready(metrics)
    elapsed = time.time() - t0

    time_per_episode = elapsed / batch_size

    return jax.tree_util.tree_map(lambda x: float(jnp.mean(x)), metrics), time_per_episode


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Eval Gumbel-AZ on Sokoban at varying sims")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a training_state_epoch_XXXXXX.pkl checkpoint.",
    )
    parser.add_argument(
        "--sims",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32, 64],
        help="List of MCTS simulation budgets to evaluate.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=128,
        help="Number of episodes per evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="sokoban_gumbel_az_eval")
    parser.add_argument("--wandb_entity", type=str, default=None)

    # Network architecture (must match training config)
    parser.add_argument("--num_channels", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--time_embed_dim", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)

    args = parser.parse_args()

    # ── Load checkpoint ──────────────────────────────────
    logger.info(f"Loading checkpoint from {args.checkpoint} …")
    with open(args.checkpoint, "rb") as f:
        training_state: TrainingState = pickle.load(f)

    # The checkpoint was saved under jax.pmap, so every leaf has an extra
    # leading (num_devices,) axis.  We take the first replica.
    params_state: AlphaZeroParamsState = jax.tree_util.tree_map(
        lambda x: x[0] if hasattr(x, "shape") and len(x.shape) > 0 else x,
        training_state.params_state,
    )

    checkpoint_name = args.checkpoint.rsplit("/", 1)[-1].replace(".pkl", "")
    logger.info(f"Checkpoint loaded: {checkpoint_name}")

    # ── Build env & agent ────────────────────────────────
    env = jumanji.make("Sokoban-v0")

    # We need a "dummy" agent just to get make_policy working.
    # The network architecture params must match training.
    from jumanji.wrappers import VmapAutoResetWrapper

    wrapped_env = VmapAutoResetWrapper(env)

    dummy_agent = GumbelAlphaZeroAgent(
        env=wrapped_env,
        n_steps=1,  # unused for eval
        total_batch_size=1,  # unused for eval
        num_simulations=2,  # will be overridden per run
        gamma=args.gamma,
        learning_rate=1e-4,  # unused for eval
        num_channels=args.num_channels,
        num_blocks=args.num_blocks,
        time_embed_dim=args.time_embed_dim,
    )

    # ── Evaluate at each sim budget ──────────────────────
    key = jax.random.PRNGKey(args.seed)

    # Collect results across all sims for the summary run
    all_results: List[Dict[str, float]] = []

    for num_sims in args.sims:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating with num_simulations = {num_sims}")
        logger.info(f"{'='*60}")

        # Patch the agent's sim count for this run
        dummy_agent.num_simulations = num_sims

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity if args.wandb_entity != "null" else None,
            name=f"{checkpoint_name}_sims{num_sims}",
            config={
                "checkpoint": args.checkpoint,
                "checkpoint_name": checkpoint_name,
                "num_simulations": num_sims,
                "eval_batch_size": args.eval_batch_size,
                "seed": args.seed,
                "num_channels": args.num_channels,
                "num_blocks": args.num_blocks,
                "time_embed_dim": args.time_embed_dim,
                "gamma": args.gamma,
            },
            reinit=True,
        )

        key, eval_key = jax.random.split(key)

        metrics, time_per_episode = run_batch_eval(
            agent=dummy_agent,
            eval_env=env,
            params_state=params_state,
            key=eval_key,
            batch_size=args.eval_batch_size,
        )

        # Log to W&B
        log_data = {
            "num_simulations": num_sims,
            "inference_time_per_episode_sec": time_per_episode,
            "total_inference_time_sec": time_per_episode * args.eval_batch_size,
        }
        for k, v in metrics.items():
            log_data[k] = v
            logger.info(f"  {k}: {v:.4f}")
        logger.info(f"  inference_time_per_episode_sec: {time_per_episode:.4f}")

        wandb.log(log_data)
        wandb.finish()

        # Save for summary
        all_results.append(log_data)

    # ── Summary W&B run: performance vs compute tradeoff ─
    logger.info(f"\n{'='*60}")
    logger.info("Creating summary tradeoff run...")
    logger.info(f"{'='*60}")

    summary_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity != "null" else None,
        name=f"{checkpoint_name}_tradeoff_summary",
        config={
            "checkpoint": args.checkpoint,
            "checkpoint_name": checkpoint_name,
            "sims_evaluated": args.sims,
            "eval_batch_size": args.eval_batch_size,
            "seed": args.seed,
        },
        reinit=True,
    )

    # Log each sim as a separate step so W&B can plot curves
    for result in all_results:
        wandb.log(result)

    # Also log a W&B Table for easy comparison
    columns = ["num_simulations", "episode_return", "solved", "prop_correct_boxes",
               "episode_length", "inference_time_per_episode_sec"]
    table = wandb.Table(columns=columns)
    for r in all_results:
        table.add_data(
            r.get("num_simulations", 0),
            r.get("episode_return", 0),
            r.get("solved", 0),
            r.get("prop_correct_boxes", 0),
            r.get("episode_length", 0),
            r.get("inference_time_per_episode_sec", 0),
        )
    wandb.log({"tradeoff_table": table})
    wandb.finish()

    logger.info("\nAll evaluations complete!")


if __name__ == "__main__":
    main()
