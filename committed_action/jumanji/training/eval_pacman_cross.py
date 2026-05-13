"""
Cross-model evaluation for PacMan Gumbel-AlphaZero.

For a model trained with K_model action delay, evaluate it at every combination of:
  - K_eval  ∈ args.k_eval_list   (test-time action delay; changes MCTS planning horizon
                                   AND real env execution)
  - num_sims ∈ args.sims_list    (MCTS simulation budget)

Reward metric: RAW (undiscounted) episode return — sum of every individual env-step
reward over the whole episode, with no gamma weighting.

Usage:
    python -m jumanji.training.eval_pacman_cross \
        --k_model 2 \
        --k_eval_list 1 2 3 \
        --sims_list 32 64 96 \
        --eval_batch_size 100 \
        --wandb_project pacman_cross_eval \
        --seed 42
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import pickle
import time
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import wandb

import jumanji
from jumanji.training.agents.gumbel_alphazero import GumbelAlphaZeroAgent
from jumanji.training.training_types import (
    ActingState,
    AlphaZeroParamsState,
    TrainingState,
)
from jumanji.wrappers import VmapAutoResetWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PACMAN_NOOP = 4  # "stay" action index in PacMan


# ──────────────────────────────────────────────────────────────────────────────
# Robust checkpoint loading
# ──────────────────────────────────────────────────────────────────────────────


def load_checkpoint(checkpoint_path: str) -> TrainingState:
    """Load a checkpoint, falling back to the latest epoch checkpoint when the
    target file is missing or contains None (e.g. crashed-run artifact).
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)

    def _find_fallback(reason: str) -> str:
        epoch_files = sorted(
            glob.glob(os.path.join(checkpoint_dir, "training_state_epoch_*.pkl"))
        )
        if not epoch_files:
            raise ValueError(
                f"{reason} and no epoch checkpoints were found in '{checkpoint_dir}'."
            )
        fallback = epoch_files[-1]
        logger.warning(f"{reason}. Falling back to latest epoch checkpoint: '{fallback}'")
        return fallback

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    try:
        with open(checkpoint_path, "rb") as f:
            training_state: TrainingState = pickle.load(f)
    except FileNotFoundError:
        fallback = _find_fallback(f"'{checkpoint_path}' not found")
        with open(fallback, "rb") as f:
            training_state = pickle.load(f)

    if training_state is None or getattr(training_state, "params_state", None) is None:
        fallback = _find_fallback(f"'{checkpoint_path}' contains None (crashed-run artifact)")
        with open(fallback, "rb") as f:
            training_state = pickle.load(f)
        if training_state is None or getattr(training_state, "params_state", None) is None:
            raise ValueError(
                f"Fallback checkpoint '{fallback}' also contains None or missing params_state."
            )

    logger.info("Checkpoint loaded.")
    return training_state


# ──────────────────────────────────────────────────────────────────────────────
# Core eval: single episode with K-step execution and raw reward accumulation
# ──────────────────────────────────────────────────────────────────────────────


def eval_one_episode_kstep(
    agent: GumbelAlphaZeroAgent,
    eval_env,
    params_state: AlphaZeroParamsState,
    key,
    k_eval: int,
):
    """
    Run one greedy MCTS episode with K_eval-step execution.

    Per agent decision:
      1. Execute K_eval-1 no-op steps in the real env.
      2. Execute 1 real action step in the real env.
      3. Accumulate ALL step rewards without any gamma discount.

    The episode terminates when any of the K_eval steps hits a terminal state.

    Args:
        agent:       GumbelAlphaZeroAgent with pacman_action_delay and
                     num_simulations already set for this run.
        eval_env:    Raw (unwrapped) PacMan environment.
        params_state: Checkpoint parameters (single-device, pmap axis stripped).
        key:         JAX PRNGKey.
        k_eval:      Python int — number of env steps per agent decision.

    Returns:
        Dict with "episode_return" (raw) and "episode_length" (in env steps).
    """
    policy = agent.make_policy(
        params_state=params_state,
        stochastic=False,
        eval_env=eval_env,
    )

    # ── while-loop condition: stop when any K-step sub-block terminates ──────
    def cond_fun(carry):
        _acting_state, _return, done = carry
        return ~done

    # ── while-loop body ───────────────────────────────────────────────────────
    def body_fun(carry):
        acting_state, return_, done = carry
        key, action_key = jax.random.split(acting_state.key)

        # 1. MCTS selects action from current state
        action = policy(
            acting_state.state,
            acting_state.timestep.observation,
            action_key,
        )

        # 2. Build the K_eval-length action sequence: [noop, …, noop, action]
        noops = jnp.full((k_eval - 1,), _PACMAN_NOOP, dtype=jnp.int32)
        real_action = jnp.reshape(action, (1,)).astype(jnp.int32)
        all_actions = jnp.concatenate([noops, real_action], axis=0)  # (k_eval,)

        # 3. Step through K_eval env steps, accumulating raw rewards.
        #    Once `early_done` triggers, we freeze the state and stop
        #    accumulating (but must still execute steps for JAX scan shape
        #    consistency — the results are masked out).
        def do_step(carry, a):
            s, cum_r, early_done = carry
            ns, ts = eval_env.step(s, a)

            # Mask reward after episode ends
            r = jnp.where(early_done, 0.0, ts.reward.astype(jnp.float32))
            new_done = early_done | ts.last()

            # Freeze state at first termination so subsequent steps are no-ops
            frozen_ns = jax.tree_util.tree_map(
                lambda n, o: jnp.where(early_done, o, n), ns, s
            )
            return (frozen_ns, cum_r + r, new_done), ts

        init_carry = (
            acting_state.state,
            jnp.array(0.0, jnp.float32),
            jnp.array(False),
        )
        (final_state, step_return, any_done), step_ts = jax.lax.scan(
            do_step, init_carry, all_actions
        )

        # The timestep after the last of the K steps (used for next observation)
        final_ts = jax.tree_util.tree_map(lambda x: x[-1], step_ts)

        new_acting_state = ActingState(
            state=final_state,
            timestep=final_ts,
            key=key,
            episode_count=acting_state.episode_count,
            env_step_count=acting_state.env_step_count + k_eval,
        )
        return new_acting_state, return_ + step_return, done | any_done

    # ── Initialise episode ────────────────────────────────────────────────────
    reset_key, init_key = jax.random.split(key)
    state, timestep = eval_env.reset(reset_key)
    acting_state = ActingState(
        state=state,
        timestep=timestep,
        key=init_key,
        episode_count=jnp.array(0, jnp.int32),
        env_step_count=jnp.array(0, jnp.int32),
    )

    final_acting_state, return_, _ = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (acting_state, jnp.array(0.0, jnp.float32), jnp.array(False)),
    )

    eval_metrics = {
        "episode_return": return_,
        "episode_length": final_acting_state.env_step_count,
    }
    extras = final_acting_state.timestep.extras
    if extras:
        eval_metrics.update(extras)
    return eval_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Batch evaluation
# ──────────────────────────────────────────────────────────────────────────────


def run_batch_eval(
    agent: GumbelAlphaZeroAgent,
    eval_env,
    params_state: AlphaZeroParamsState,
    key,
    batch_size: int,
    k_eval: int,
) -> Tuple[Dict[str, float], float]:
    """
    Vmap eval_one_episode_kstep over batch_size episodes.

    Returns (mean_metrics_dict, time_per_episode_sec).
    """
    keys = jax.random.split(key, batch_size)

    eval_fn = jax.vmap(
        lambda k: eval_one_episode_kstep(agent, eval_env, params_state, k, k_eval)
    )

    # Warmup: compile once before timing
    logger.info("  Warmup JIT compile (1 episode)...")
    warmup_fn = jax.vmap(
        lambda k: eval_one_episode_kstep(agent, eval_env, params_state, k, k_eval)
    )
    _ = jax.block_until_ready(warmup_fn(jax.random.split(key, 1)))
    logger.info("  Warmup done. Running timed evaluation...")

    t0 = time.time()
    metrics = eval_fn(keys)
    jax.block_until_ready(metrics)
    elapsed = time.time() - t0

    time_per_episode = elapsed / batch_size
    mean_metrics = jax.tree_util.tree_map(lambda x: float(jnp.mean(x)), metrics)
    return mean_metrics, time_per_episode


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model PacMan evaluation: load K_model checkpoint, "
                    "evaluate over K_eval × num_sims grid."
    )
    parser.add_argument(
        "--k_model",
        type=int,
        required=True,
        help="Action delay K used when training the checkpoint (1, 2, or 3).",
    )
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default="./checkpoints/committed_action/pacman/base/k",
        help="Base path for model directories; k_model is appended (e.g. …_2/).",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="training_state_best.pkl",
        help="Checkpoint filename inside the model directory.",
    )
    parser.add_argument(
        "--k_eval_list",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Action delays to test at evaluation time.",
    )
    parser.add_argument(
        "--sims_list",
        type=int,
        nargs="+",
        default=[32, 64, 96],
        help="MCTS simulation budgets to sweep.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=100,
        help="Number of episodes per (k_eval, num_sims) evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="pacman_cross_eval")
    parser.add_argument("--wandb_entity", type=str, default=None)

    # Network architecture — must match training config
    parser.add_argument("--num_channels", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--time_embed_dim", type=int, default=32)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor used *only* inside MCTS tree expansion. "
             "Does NOT affect the reported episode return.",
    )

    args = parser.parse_args()

    checkpoint_path = f"{args.base_model_dir}{args.k_model}/{args.checkpoint_name}"
    training_state = load_checkpoint(checkpoint_path)

    # Strip the leading (num_devices,) axis added by pmap during training
    params_state: AlphaZeroParamsState = jax.tree_util.tree_map(
        lambda x: x[0] if hasattr(x, "shape") and len(x.shape) > 0 else x,
        training_state.params_state,
    )

    # ── Build environment and agent ──────────────────────────────────────────
    env = jumanji.make("PacMan-v1")
    wrapped_env = VmapAutoResetWrapper(env)  # needed for agent __init__ specs

    agent = GumbelAlphaZeroAgent(
        env=wrapped_env,
        n_steps=1,           # unused at eval
        total_batch_size=1,  # unused at eval
        num_simulations=2,   # placeholder — overridden per run
        gamma=args.gamma,
        learning_rate=1e-4,  # unused at eval
        num_channels=args.num_channels,
        num_blocks=args.num_blocks,
        time_embed_dim=args.time_embed_dim,
        pacman_action_delay=args.k_model,  # initialise to training K
    )

    key = jax.random.PRNGKey(args.seed)

    # ── Sweep over (K_eval, num_sims) ────────────────────────────────────────
    all_results: List[Dict] = []

    for k_eval in args.k_eval_list:
        # Patch the agent's planning horizon to match the evaluation environment
        agent.pacman_action_delay = k_eval

        for num_sims in args.sims_list:
            agent.num_simulations = num_sims

            logger.info(
                f"\n{'='*60}\n"
                f"k_model={args.k_model}  k_eval={k_eval}  num_sims={num_sims}\n"
                f"{'='*60}"
            )

            run_name = f"kmodel{args.k_model}_keval{k_eval}_sims{num_sims}"
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity if args.wandb_entity != "null" else None,
                name=run_name,
                config={
                    "k_model": args.k_model,
                    "k_eval": k_eval,
                    "num_simulations": num_sims,
                    "eval_batch_size": args.eval_batch_size,
                    "seed": args.seed,
                    "checkpoint": checkpoint_path,
                    "num_channels": args.num_channels,
                    "num_blocks": args.num_blocks,
                    "time_embed_dim": args.time_embed_dim,
                    "gamma": args.gamma,
                    "reward_type": "raw_undiscounted",
                },
                reinit=True,
            )

            key, eval_key = jax.random.split(key)
            metrics, tpe = run_batch_eval(
                agent=agent,
                eval_env=env,
                params_state=params_state,
                key=eval_key,
                batch_size=args.eval_batch_size,
                k_eval=k_eval,
            )

            log_data = {
                "k_model": args.k_model,
                "k_eval": k_eval,
                "num_simulations": num_sims,
                "inference_time_per_episode_sec": tpe,
            }
            for metric_name, metric_val in metrics.items():
                log_data[metric_name] = metric_val
                logger.info(f"  {metric_name}: {metric_val:.4f}")
            logger.info(f"  inference_time_per_episode_sec: {tpe:.4f}")

            wandb.log(log_data)
            wandb.finish()

            all_results.append(log_data)

    # ── Summary W&B run: full 3×3 (or N×M) grid as a Table ─────────────────
    logger.info(f"\n{'='*60}\nCreating summary run for k_model={args.k_model}\n{'='*60}")

    summary_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity != "null" else None,
        name=f"summary_kmodel{args.k_model}",
        config={
            "k_model": args.k_model,
            "k_eval_list": args.k_eval_list,
            "sims_list": args.sims_list,
            "eval_batch_size": args.eval_batch_size,
            "seed": args.seed,
            "checkpoint": checkpoint_path,
        },
        reinit=True,
    )

    columns = [
        "k_model", "k_eval", "num_simulations",
        "episode_return", "episode_length",
        "inference_time_per_episode_sec",
    ]
    table = wandb.Table(columns=columns)
    for r in all_results:
        table.add_data(
            r.get("k_model", args.k_model),
            r.get("k_eval", 0),
            r.get("num_simulations", 0),
            r.get("episode_return", 0.0),
            r.get("episode_length", 0.0),
            r.get("inference_time_per_episode_sec", 0.0),
        )
    wandb.log({"cross_eval_table": table})

    # Also log each result as a step so W&B can draw lines
    for r in all_results:
        wandb.log(r)

    wandb.finish()
    logger.info("All evaluations complete.")


if __name__ == "__main__":
    main()
