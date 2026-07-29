"""Fit entropy thresholds for adaptive MCTS depth in Tetris.

This mirrors fit_pacman_entropy_thresholds.py:
  * Load a frozen AZNet checkpoint.
  * Build K=1..4 planners with simulation budgets 32,64,96,128.
  * Exhaustively evaluate ordered triples from a threshold grid.
  * Rank on coarse-search seeds.
  * Re-evaluate the top candidates on disjoint validation seeds.
  * Save the best thresholds and, optionally, all search results.

The fitting procedure performs no gradient updates. It selects thresholds by
mean raw undiscounted episode return.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import jumanji
from jumanji.training.train_tetris_rt_gating_ppo import (
    K_OPTIONS,
    _sel4,
    load_az_checkpoint,
    make_agents,
    meta_step_one_k,
)
from jumanji.wrappers import VmapAutoResetWrapper


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateResult:
    thresholds: Tuple[float, float, float]
    mean_return: float
    standard_error: float
    mean_k: float
    k_frequencies: Tuple[float, float, float, float]
    completed_fraction: float
    num_episodes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--az_checkpoint_dir",
        type=str,
        default="./checkpoints/committed_action/tetris_rt/base",
    )
    parser.add_argument(
        "--az_checkpoint_path",
        type=str,
        default=None,
        help="Direct checkpoint path; overrides --az_checkpoint_dir.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/committed_action/tetris_rt/entropy_threshold",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="entropy_thresholds_best.json",
    )
    parser.add_argument(
        "--threshold_values",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument("--search_num_episodes", type=int, default=32)
    parser.add_argument("--validation_num_episodes", type=int, default=64)
    parser.add_argument("--top_k_candidates", type=int, default=8)
    parser.add_argument("--eval_meta_steps", type=int, default=1200)
    parser.add_argument("--search_seed_base", type=int, default=0)
    parser.add_argument("--validation_seed_base", type=int, default=10_000)
    parser.add_argument("--mcts_seed_offset", type=int, default=1_000_000)
    parser.add_argument(
        "--sim_options",
        type=int,
        nargs=4,
        default=[32, 64, 96, 128],
        metavar=("S1", "S2", "S3", "S4"),
    )
    parser.add_argument("--save_all_results", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.search_num_episodes <= 0:
        raise ValueError("--search_num_episodes must be positive.")
    if args.validation_num_episodes <= 0:
        raise ValueError("--validation_num_episodes must be positive.")
    if args.top_k_candidates <= 0:
        raise ValueError("--top_k_candidates must be positive.")
    if args.eval_meta_steps <= 0:
        raise ValueError("--eval_meta_steps must be positive.")
    if len(args.sim_options) != 4:
        raise ValueError("--sim_options must contain exactly four integers.")

    values = [float(x) for x in args.threshold_values]
    if len(set(values)) < 3:
        raise ValueError("At least three distinct threshold values are required.")
    if any(not (0.0 <= x <= 1.0) for x in values):
        raise ValueError("All threshold values must lie in [0, 1].")


def masked_normalized_policy_entropy(
    policy_logits: jax.Array,
    invalid_actions: jax.Array,
) -> jax.Array:
    """Return policy entropy normalised by log(number of valid actions)."""
    valid = ~invalid_actions
    masked_logits = jnp.where(valid, policy_logits, -jnp.inf)
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
    probs = jnp.where(valid, jnp.exp(log_probs), 0.0)
    safe_log_probs = jnp.where(valid, log_probs, 0.0)
    entropy = -jnp.sum(probs * safe_log_probs, axis=-1)

    num_valid = jnp.sum(valid, axis=-1)
    normaliser = jnp.log(jnp.maximum(num_valid, 2).astype(jnp.float32))
    normalised = entropy / normaliser
    return jnp.where(num_valid <= 1, 0.0, jnp.clip(normalised, 0.0, 1.0))


def entropy_to_k_index(
    entropy: jax.Array,
    thresholds: jax.Array,
) -> jax.Array:
    """Map normalised entropy to branch index 0,1,2,3."""
    return jnp.sum(entropy[..., None] >= thresholds, axis=-1).astype(jnp.int32)


def generate_threshold_candidates(
    values: Sequence[float],
) -> List[Tuple[float, float, float]]:
    ordered = sorted(set(float(x) for x in values))
    return list(itertools.combinations(ordered, 3))


def resolve_checkpoint_source(args: argparse.Namespace) -> str:
    return (
        args.az_checkpoint_path
        if args.az_checkpoint_path is not None
        else args.az_checkpoint_dir
    )


def reset_episode_batch(
    env: VmapAutoResetWrapper,
    num_episodes: int,
    seed_base: int,
):
    reset_keys = jax.random.split(
        jax.random.PRNGKey(seed_base),
        num_episodes,
    )
    states, timesteps = env.reset(reset_keys)
    return states, timesteps.observation


def build_evaluator(
    agents,
    az_params_state,
    num_envs: int,
    eval_meta_steps: int,
):
    """Build the JIT-compiled entropy-policy evaluator."""
    az_net_params = jax.lax.stop_gradient(az_params_state.params.net)
    az_net_state = jax.lax.stop_gradient(az_params_state.net_state)
    env_indices = jnp.arange(num_envs)

    @jax.jit
    def evaluate(
        thresholds: jax.Array,
        init_states: Any,
        init_obs: Any,
        mcts_key: jax.Array,
    ):
        def scan_body(carry, step_key):
            states, obs, raw_return, done, k_counts = carry

            obs_grid, time_vec = agents[0]._get_grid_and_time(states, obs)
            invalid = agents[0]._get_invalid_actions(
                agents[0].raw_env_train,
                states,
                obs,
            )
            (policy_logits, _), _ = agents[0].forward.apply(
                az_net_params,
                az_net_state,
                obs_grid,
                time_vec,
                is_eval=True,
            )
            entropy = masked_normalized_policy_entropy(policy_logits, invalid)
            k_choices = entropy_to_k_index(entropy, thresholds)

            branch_keys = jax.random.split(step_key, 4)
            results = [
                meta_step_one_k(
                    agents[i],
                    az_net_params,
                    az_net_state,
                    states,
                    obs,
                    obs_grid,
                    time_vec,
                    invalid,
                    branch_keys[i],
                    reward_mode="raw",
                )
                for i in range(4)
            ]

            reward_all = jnp.stack(
                [result["r_meta"] for result in results],
                axis=0,
            )
            done_all = jnp.stack(
                [result["done"] for result in results],
                axis=0,
            )
            selected_reward = reward_all[k_choices, env_indices]
            selected_done = done_all[k_choices, env_indices]

            next_states = jax.vmap(_sel4)(
                k_choices,
                results[0]["next_states"],
                results[1]["next_states"],
                results[2]["next_states"],
                results[3]["next_states"],
            )
            next_obs = jax.vmap(_sel4)(
                k_choices,
                results[0]["next_obs"],
                results[1]["next_obs"],
                results[2]["next_obs"],
                results[3]["next_obs"],
            )

            active = ~done
            raw_return = raw_return + selected_reward * active.astype(jnp.float32)
            k_counts = k_counts.at[env_indices, k_choices].add(
                active.astype(jnp.int32)
            )
            done = done | selected_done

            return (
                next_states,
                next_obs,
                raw_return,
                done,
                k_counts,
            ), None

        initial_carry = (
            init_states,
            init_obs,
            jnp.zeros((num_envs,), dtype=jnp.float32),
            jnp.zeros((num_envs,), dtype=bool),
            jnp.zeros((num_envs, 4), dtype=jnp.int32),
        )
        step_keys = jax.random.split(mcts_key, eval_meta_steps)
        (_, _, raw_return, done, k_counts), _ = jax.lax.scan(
            scan_body,
            initial_carry,
            step_keys,
        )
        return raw_return, done, k_counts

    return evaluate


def summarise(
    thresholds: Tuple[float, float, float],
    returns: jax.Array,
    done: jax.Array,
    k_counts: jax.Array,
) -> CandidateResult:
    returns_np = np.asarray(jax.device_get(returns), dtype=np.float64)
    done_np = np.asarray(jax.device_get(done), dtype=bool)
    counts_np = np.asarray(jax.device_get(k_counts), dtype=np.float64)

    n = int(returns_np.shape[0])
    mean_return = float(np.mean(returns_np))
    standard_error = (
        float(np.std(returns_np, ddof=1) / math.sqrt(n))
        if n > 1
        else 0.0
    )
    total_choices = float(np.sum(counts_np))
    if total_choices > 0:
        frequencies = np.sum(counts_np, axis=0) / total_choices
        mean_k = float(
            np.sum(frequencies * np.asarray(K_OPTIONS, dtype=np.float64))
        )
    else:
        frequencies = np.zeros((4,), dtype=np.float64)
        mean_k = 0.0

    return CandidateResult(
        thresholds=thresholds,
        mean_return=mean_return,
        standard_error=standard_error,
        mean_k=mean_k,
        k_frequencies=tuple(float(x) for x in frequencies),
        completed_fraction=float(np.mean(done_np)),
        num_episodes=n,
    )


def result_to_dict(result: CandidateResult) -> Dict[str, Any]:
    data = asdict(result)
    data["thresholds"] = list(result.thresholds)
    data["k_frequencies"] = list(result.k_frequencies)
    return data


def log_result(prefix: str, result: CandidateResult) -> None:
    logger.info(
        "%s thresholds=%s return=%.3f +/- %.3f mean_K=%.3f "
        "K_freq=%s completed=%.1f%%",
        prefix,
        result.thresholds,
        result.mean_return,
        result.standard_error,
        result.mean_k,
        np.round(np.asarray(result.k_frequencies) * 100.0, 1).tolist(),
        100.0 * result.completed_fraction,
    )


def evaluate_candidates(
    *,
    env,
    evaluator,
    candidates: Sequence[Tuple[float, float, float]],
    num_episodes: int,
    reset_seed_base: int,
    mcts_seed: int,
    phase_name: str,
) -> List[CandidateResult]:
    results: List[CandidateResult] = []

    for index, candidate in enumerate(candidates, start=1):
        states, obs = reset_episode_batch(env, num_episodes, reset_seed_base)
        returns, done, k_counts = evaluator(
            jnp.asarray(candidate, dtype=jnp.float32),
            states,
            obs,
            jax.random.PRNGKey(mcts_seed),
        )
        result = summarise(candidate, returns, done, k_counts)
        results.append(result)
        logger.info(
            "%s %d/%d thresholds=%s return=%.3f mean_K=%.3f completed=%.1f%%",
            phase_name,
            index,
            len(candidates),
            candidate,
            result.mean_return,
            result.mean_k,
            100.0 * result.completed_fraction,
        )

    return results


def main() -> None:
    args = parse_args()
    validate_args(args)

    checkpoint_source = resolve_checkpoint_source(args)
    candidates = generate_threshold_candidates(args.threshold_values)

    logger.info("JAX devices: %s", jax.devices())
    logger.info("Loading frozen AZNet from: %s", checkpoint_source)
    az_params_state = jax.device_put(load_az_checkpoint(checkpoint_source))
    logger.info("Loaded AZNet checkpoint from: %s", checkpoint_source)

    logger.info(
        "Generated %d ordered threshold triples from %s.",
        len(candidates),
        sorted(set(float(x) for x in args.threshold_values)),
    )

    search_env = VmapAutoResetWrapper(jumanji.make("TetrisRTKT-v1"))
    search_agents = make_agents(
        search_env,
        args.search_num_episodes,
        args.sim_options,
    )
    search_evaluator = build_evaluator(
        search_agents,
        az_params_state,
        args.search_num_episodes,
        args.eval_meta_steps,
    )

    logger.info(
        "Starting coarse search: %d candidates x %d episodes.",
        len(candidates),
        args.search_num_episodes,
    )
    coarse_results = evaluate_candidates(
        env=search_env,
        evaluator=search_evaluator,
        candidates=candidates,
        num_episodes=args.search_num_episodes,
        reset_seed_base=args.search_seed_base,
        mcts_seed=args.search_seed_base + args.mcts_seed_offset,
        phase_name="Coarse",
    )
    coarse_ranking = sorted(
        coarse_results,
        key=lambda result: result.mean_return,
        reverse=True,
    )
    top_count = min(args.top_k_candidates, len(coarse_ranking))
    top_candidates = [r.thresholds for r in coarse_ranking[:top_count]]

    logger.info("Top %d coarse-search candidates:", top_count)
    for rank, result in enumerate(coarse_ranking[:top_count], start=1):
        log_result(f"coarse rank={rank}", result)

    validation_env = VmapAutoResetWrapper(jumanji.make("TetrisRTKT-v1"))
    validation_agents = make_agents(
        validation_env,
        args.validation_num_episodes,
        args.sim_options,
    )
    validation_evaluator = build_evaluator(
        validation_agents,
        az_params_state,
        args.validation_num_episodes,
        args.eval_meta_steps,
    )

    logger.info(
        "Starting validation: %d candidates x %d episodes.",
        len(top_candidates),
        args.validation_num_episodes,
    )
    validation_results = evaluate_candidates(
        env=validation_env,
        evaluator=validation_evaluator,
        candidates=top_candidates,
        num_episodes=args.validation_num_episodes,
        reset_seed_base=args.validation_seed_base,
        mcts_seed=args.validation_seed_base + args.mcts_seed_offset,
        phase_name="Validation",
    )
    validation_ranking = sorted(
        validation_results,
        key=lambda result: result.mean_return,
        reverse=True,
    )

    logger.info("Validation ranking:")
    for rank, result in enumerate(validation_ranking, start=1):
        log_result(f"validation rank={rank}", result)

    best = validation_ranking[0]
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, args.output_name)

    best_output = {
        "environment": "TetrisRTKT-v1",
        "thresholds": list(best.thresholds),
        "validation_result": result_to_dict(best),
        "az_checkpoint_source": checkpoint_source,
        "sim_options": [int(x) for x in args.sim_options],
        "threshold_values": sorted(
            set(float(x) for x in args.threshold_values)
        ),
        "search_num_episodes": args.search_num_episodes,
        "validation_num_episodes": args.validation_num_episodes,
        "top_k_candidates": top_count,
        "eval_meta_steps": args.eval_meta_steps,
        "search_seed_base": args.search_seed_base,
        "validation_seed_base": args.validation_seed_base,
        "mcts_seed_offset": args.mcts_seed_offset,
    }
    with open(best_path, "w", encoding="utf-8") as file:
        json.dump(best_output, file, indent=2)
    logger.info("Saved best thresholds to: %s", best_path)

    if args.save_all_results:
        all_path = os.path.join(
            args.output_dir,
            "entropy_threshold_search_results.json",
        )
        all_output = {
            **best_output,
            "coarse_results": [result_to_dict(r) for r in coarse_results],
            "coarse_ranking": [result_to_dict(r) for r in coarse_ranking],
            "validation_results": [
                result_to_dict(r) for r in validation_results
            ],
            "validation_ranking": [
                result_to_dict(r) for r in validation_ranking
            ],
        }
        with open(all_path, "w", encoding="utf-8") as file:
            json.dump(all_output, file, indent=2)
        logger.info("Saved complete search results to: %s", all_path)

    log_result("Best", best)


if __name__ == "__main__":
    main()
