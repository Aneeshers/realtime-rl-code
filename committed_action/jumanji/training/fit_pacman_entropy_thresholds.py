"""Fit entropy thresholds for adaptive MCTS depth in PacManKT.

This script fits a deterministic threshold policy over the frozen AZNet policy entropy:

    entropy < tau_1                 -> K=1
    tau_1 <= entropy < tau_2        -> K=2
    tau_2 <= entropy < tau_3        -> K=3
    entropy >= tau_3                -> K=4

The thresholds are selected with a two-stage black-box search:

1. Coarse search:
   Evaluate every ordered threshold triple from a fixed grid on a small set of
   complete PacMan episodes.

2. Validation:
   Re-evaluate the top candidates on a disjoint set of complete episodes and
   save the candidate with the highest validation mean return.

No gradients, PPO, or offline dataset are used. The frozen AZNet supplies the
policy logits, while the existing four GumbelAlphaZeroAgent instances provide
the K=1,2,3,4 MCTS/action-delay branches.

During each meta-step all four branches are evaluated and the threshold policy
selects one branch. This matches the simple implementation used by the PPO
gating trainer and deliberately prioritises correctness over compute efficiency.
"""

from __future__ import annotations

import argparse
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
from jumanji.training.train_pacman_gating_ppo import (
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
    seed_base: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--az_checkpoint_dir",
        type=str,
        default="./checkpoints/committed_action/pacman/base/k1",
        help="Directory containing training_state_best.pkl or epoch checkpoints.",
    )
    parser.add_argument(
        "--az_checkpoint_path",
        type=str,
        default=None,
        help="Direct AZNet checkpoint path. Overrides --az_checkpoint_dir.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/committed_action/pacman/entropy_threshold",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="entropy_thresholds_best.json",
    )

    parser.add_argument(
        "--sim_options",
        type=int,
        nargs=4,
        default=[32, 64, 96, 128],
        metavar=("S1", "S2", "S3", "S4"),
    )
    parser.add_argument(
        "--threshold_values",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Candidate scalar values used to form all strictly ordered triples.",
    )

    parser.add_argument(
        "--search_num_episodes",
        type=int,
        default=8,
        help="Episodes per threshold candidate during coarse search.",
    )
    parser.add_argument(
        "--validation_num_episodes",
        type=int,
        default=50,
        help="Episodes per retained candidate during validation.",
    )
    parser.add_argument(
        "--top_k_candidates",
        type=int,
        default=8,
        help="Number of coarse-search candidates retained for validation.",
    )
    parser.add_argument(
        "--eval_meta_steps",
        type=int,
        default=1200,
        help="Maximum meta-steps per episode batch.",
    )
    parser.add_argument(
        "--search_seed_base",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--validation_seed_base",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--mcts_seed_offset",
        type=int,
        default=1_000_000,
        help="Offset separating reset seeds from MCTS key seeds.",
    )
    parser.add_argument(
        "--save_all_results",
        action="store_true",
        help="Also save coarse and validation rankings to JSON.",
    )

    return parser.parse_args()


def resolve_checkpoint_path(args: argparse.Namespace) -> str:
    if args.az_checkpoint_path is not None:
        return args.az_checkpoint_path
    return args.az_checkpoint_dir


def validate_args(args: argparse.Namespace) -> None:
    if len(args.sim_options) != 4:
        raise ValueError("--sim_options must contain exactly four integers.")
    if args.search_num_episodes <= 0:
        raise ValueError("--search_num_episodes must be positive.")
    if args.validation_num_episodes <= 0:
        raise ValueError("--validation_num_episodes must be positive.")
    if args.eval_meta_steps <= 0:
        raise ValueError("--eval_meta_steps must be positive.")
    if args.top_k_candidates <= 0:
        raise ValueError("--top_k_candidates must be positive.")

    values = tuple(float(x) for x in args.threshold_values)
    if len(values) < 3:
        raise ValueError("--threshold_values must contain at least three values.")
    if any(not (0.0 <= x <= 1.0) for x in values):
        raise ValueError("Every threshold value must lie in [0, 1].")
    if tuple(sorted(set(values))) != values:
        raise ValueError(
            "--threshold_values must be strictly increasing with no duplicates."
        )


def masked_normalized_policy_entropy(
    logits: jax.Array,
    invalid_actions: jax.Array,
    eps: float = 1e-8,
) -> jax.Array:
    """Compute policy entropy over legal actions and normalise it to [0, 1].

    Args:
        logits: Shape (B, A).
        invalid_actions: Shape (B, A), where True means invalid.
        eps: Numerical lower bound for the entropy denominator.

    Returns:
        Shape (B,), with 0 for states having zero or one legal action.
    """
    legal_actions = ~invalid_actions
    num_legal = jnp.sum(legal_actions, axis=-1)

    masked_logits = jnp.where(
        legal_actions,
        logits,
        jnp.finfo(logits.dtype).min,
    )
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
    probs = jnp.where(legal_actions, jnp.exp(log_probs), 0.0)

    entropy = -jnp.sum(
        jnp.where(legal_actions, probs * log_probs, 0.0),
        axis=-1,
    )

    max_entropy = jnp.log(
        jnp.maximum(num_legal, 2).astype(logits.dtype)
    )
    normalised = entropy / jnp.maximum(max_entropy, eps)
    normalised = jnp.where(num_legal <= 1, 0.0, normalised)
    return jnp.clip(normalised, 0.0, 1.0)


def entropy_to_k_index(
    entropy: jax.Array,
    thresholds: jax.Array,
) -> jax.Array:
    """Map entropy values to branch indices 0,1,2,3."""
    return jnp.sum(
        entropy[:, None] >= thresholds[None, :],
        axis=-1,
        dtype=jnp.int32,
    )


def generate_threshold_candidates(
    values: Sequence[float],
) -> List[Tuple[float, float, float]]:
    return [
        tuple(float(x) for x in triple)
        for triple in itertools.combinations(values, 3)
    ]


def build_threshold_evaluator(
    agents,
    az_params_state,
    num_envs: int,
    eval_meta_steps: int,
):
    """Build one JIT evaluator reused by every threshold candidate."""
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
        (
            _,
            _,
            raw_return,
            done,
            k_counts,
        ), _ = jax.lax.scan(
            scan_body,
            initial_carry,
            step_keys,
        )
        return raw_return, done, k_counts

    return evaluate


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


def summarise_candidate(
    thresholds: Tuple[float, float, float],
    returns: jax.Array,
    done: jax.Array,
    k_counts: jax.Array,
    seed_base: int,
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
    if total_choices > 0.0:
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
        seed_base=int(seed_base),
    )


def evaluate_candidate(
    evaluator,
    env: VmapAutoResetWrapper,
    thresholds: Tuple[float, float, float],
    num_episodes: int,
    reset_seed_base: int,
    mcts_seed: int,
) -> CandidateResult:
    states, obs = reset_episode_batch(env, num_episodes, reset_seed_base)
    returns, done, k_counts = evaluator(
        jnp.asarray(thresholds, dtype=jnp.float32),
        states,
        obs,
        jax.random.PRNGKey(mcts_seed),
    )
    return summarise_candidate(
        thresholds,
        returns,
        done,
        k_counts,
        reset_seed_base,
    )


def result_to_dict(result: CandidateResult) -> Dict[str, Any]:
    data = asdict(result)
    data["thresholds"] = list(result.thresholds)
    data["k_frequencies"] = list(result.k_frequencies)
    return data


def log_result(prefix: str, rank: int, result: CandidateResult) -> None:
    logger.info(
        "%s rank=%d thresholds=%s return=%.3f +/- %.3f "
        "mean_K=%.3f K_freq=%s completed=%.1f%%",
        prefix,
        rank,
        result.thresholds,
        result.mean_return,
        result.standard_error,
        result.mean_k,
        np.round(np.asarray(result.k_frequencies) * 100.0, 1).tolist(),
        100.0 * result.completed_fraction,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    candidates = generate_threshold_candidates(args.threshold_values)
    if args.top_k_candidates > len(candidates):
        logger.warning(
            "top_k_candidates=%d exceeds candidate count=%d; retaining all.",
            args.top_k_candidates,
            len(candidates),
        )
    top_k = min(args.top_k_candidates, len(candidates))

    checkpoint_source = resolve_checkpoint_path(args)
    logger.info("JAX devices: %s", jax.devices())
    logger.info("Loading frozen AZNet from: %s", checkpoint_source)
    az_params_state = jax.device_put(load_az_checkpoint(checkpoint_source))

    logger.info(
        "Generated %d ordered threshold triples from %s.",
        len(candidates),
        list(args.threshold_values),
    )

    search_env = VmapAutoResetWrapper(jumanji.make("PacManKT-v1"))
    search_agents = make_agents(
        search_env,
        args.search_num_episodes,
        args.sim_options,
    )
    search_evaluator = build_threshold_evaluator(
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
    coarse_results: List[CandidateResult] = []
    for index, thresholds in enumerate(candidates):
        result = evaluate_candidate(
            evaluator=search_evaluator,
            env=search_env,
            thresholds=thresholds,
            num_episodes=args.search_num_episodes,
            reset_seed_base=args.search_seed_base,
            mcts_seed=(
                args.search_seed_base
                + args.mcts_seed_offset
            ),
        )
        coarse_results.append(result)

        logger.info(
            "Coarse %d/%d thresholds=%s return=%.3f mean_K=%.3f completed=%.1f%%",
            index + 1,
            len(candidates),
            thresholds,
            result.mean_return,
            result.mean_k,
            100.0 * result.completed_fraction,
        )

    coarse_results.sort(
        key=lambda item: item.mean_return,
        reverse=True,
    )
    retained = coarse_results[:top_k]

    logger.info("Top %d coarse-search candidates:", top_k)
    for rank, result in enumerate(retained, start=1):
        log_result("coarse", rank, result)

    validation_env = VmapAutoResetWrapper(jumanji.make("PacManKT-v1"))
    validation_agents = make_agents(
        validation_env,
        args.validation_num_episodes,
        args.sim_options,
    )
    validation_evaluator = build_threshold_evaluator(
        validation_agents,
        az_params_state,
        args.validation_num_episodes,
        args.eval_meta_steps,
    )

    logger.info(
        "Starting validation: %d candidates x %d episodes.",
        len(retained),
        args.validation_num_episodes,
    )
    validation_results: List[CandidateResult] = []
    for index, coarse_result in enumerate(retained):
        result = evaluate_candidate(
            evaluator=validation_evaluator,
            env=validation_env,
            thresholds=coarse_result.thresholds,
            num_episodes=args.validation_num_episodes,
            reset_seed_base=args.validation_seed_base,
            mcts_seed=(
                args.validation_seed_base
                + args.mcts_seed_offset
            ),
        )
        validation_results.append(result)

        logger.info(
            "Validation %d/%d thresholds=%s return=%.3f +/- %.3f "
            "mean_K=%.3f completed=%.1f%%",
            index + 1,
            len(retained),
            result.thresholds,
            result.mean_return,
            result.standard_error,
            result.mean_k,
            100.0 * result.completed_fraction,
        )

    validation_results.sort(
        key=lambda item: item.mean_return,
        reverse=True,
    )
    best = validation_results[0]

    logger.info("Validation ranking:")
    for rank, result in enumerate(validation_results, start=1):
        log_result("validation", rank, result)

    if best.completed_fraction < 1.0:
        logger.warning(
            "Only %.1f%% of validation episodes completed within %d meta-steps. "
            "Increase --eval_meta_steps before treating the result as final.",
            100.0 * best.completed_fraction,
            args.eval_meta_steps,
        )

    output = {
        "method": "masked_normalized_az_policy_entropy",
        "thresholds": list(best.thresholds),
        "k_options": list(K_OPTIONS),
        "sim_options": list(args.sim_options),
        "selection_metric": "validation_mean_raw_episode_return",
        "az_checkpoint_source": checkpoint_source,
        "threshold_values": list(args.threshold_values),
        "num_threshold_candidates": len(candidates),
        "search_num_episodes": args.search_num_episodes,
        "validation_num_episodes": args.validation_num_episodes,
        "top_k_candidates": top_k,
        "eval_meta_steps": args.eval_meta_steps,
        "search_seed_base": args.search_seed_base,
        "validation_seed_base": args.validation_seed_base,
        "mcts_seed_offset": args.mcts_seed_offset,
        "validation_mean_return": best.mean_return,
        "validation_standard_error": best.standard_error,
        "validation_mean_k": best.mean_k,
        "validation_k_frequencies": list(best.k_frequencies),
        "validation_completed_fraction": best.completed_fraction,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    logger.info("Saved best thresholds to: %s", output_path)

    if args.save_all_results:
        all_results_path = os.path.join(
            args.output_dir,
            "entropy_threshold_search_results.json",
        )
        with open(all_results_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "coarse_results": [
                        result_to_dict(result) for result in coarse_results
                    ],
                    "validation_results": [
                        result_to_dict(result) for result in validation_results
                    ],
                },
                file,
                indent=2,
            )
        logger.info("Saved complete search results to: %s", all_results_path)

    logger.info(
        "Best thresholds=%s validation_return=%.3f +/- %.3f "
        "mean_K=%.3f K_freq=%s",
        best.thresholds,
        best.mean_return,
        best.standard_error,
        best.mean_k,
        np.round(np.asarray(best.k_frequencies) * 100.0, 1).tolist(),
    )


if __name__ == "__main__":
    main()
