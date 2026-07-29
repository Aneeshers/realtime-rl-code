"""Evaluate a fitted PacMan entropy-threshold MCTS policy.

The script loads:
  1. A frozen AZNet checkpoint.
  2. A JSON file produced by fit_pacman_entropy_thresholds.py.

It evaluates the frozen entropy-threshold policy on untouched episode seeds and,
by default, also evaluates:
  - always K=1
  - always K=2
  - always K=3
  - always K=4
  - uniformly random K

All methods use raw undiscounted episode return. Each method receives the same
reset seeds and the same top-level MCTS key schedule. The evaluator runs all four
MCTS branches at every meta-step and selects the requested branch, matching the
simple implementation used during threshold fitting and PPO-gating evaluation.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import jumanji
from jumanji.training.fit_pacman_entropy_thresholds import (
    entropy_to_k_index,
    masked_normalized_policy_entropy,
)
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
class EvaluationResult:
    method: str
    mean_return: float
    standard_error: float
    mean_k: float
    k_frequencies: Tuple[float, float, float, float]
    completed_fraction: float
    num_episodes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--threshold_path",
        type=str,
        default=(
            "./checkpoints/committed_action/pacman/entropy_threshold/"
            "entropy_thresholds_best.json"
        ),
    )
    parser.add_argument(
        "--az_checkpoint_dir",
        type=str,
        default="./checkpoints/committed_action/pacman/base/k1",
    )
    parser.add_argument(
        "--az_checkpoint_path",
        type=str,
        default=None,
        help=(
            "Direct AZNet checkpoint path. Overrides both --az_checkpoint_dir "
            "and the checkpoint source stored in the threshold JSON."
        ),
    )
    parser.add_argument(
        "--use_checkpoint_from_threshold_json",
        action="store_true",
        help=(
            "Use az_checkpoint_source from the threshold JSON unless "
            "--az_checkpoint_path is supplied."
        ),
    )

    parser.add_argument(
        "--num_episodes",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--eval_meta_steps",
        type=int,
        default=1200,
    )
    parser.add_argument(
        "--evaluation_seed_base",
        type=int,
        default=20_000,
    )
    parser.add_argument(
        "--mcts_seed_offset",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--sim_options",
        type=int,
        nargs=4,
        default=None,
        metavar=("S1", "S2", "S3", "S4"),
        help=(
            "Optional override. By default sim_options are loaded from the "
            "threshold JSON."
        ),
    )
    parser.add_argument(
        "--entropy_only",
        action="store_true",
        help="Skip fixed-K and random-K baselines.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional JSON path for the final evaluation results.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_episodes <= 0:
        raise ValueError("--num_episodes must be positive.")
    if args.eval_meta_steps <= 0:
        raise ValueError("--eval_meta_steps must be positive.")
    if args.sim_options is not None and len(args.sim_options) != 4:
        raise ValueError("--sim_options must contain exactly four integers.")


def load_threshold_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)

    thresholds = config.get("thresholds")
    if thresholds is None or len(thresholds) != 3:
        raise ValueError(
            f"Threshold file '{path}' must contain exactly three thresholds."
        )

    thresholds = [float(x) for x in thresholds]
    if not (thresholds[0] < thresholds[1] < thresholds[2]):
        raise ValueError(
            f"Thresholds must be strictly increasing, received {thresholds}."
        )
    if any(not (0.0 <= x <= 1.0) for x in thresholds):
        raise ValueError(
            f"Thresholds must lie in [0, 1], received {thresholds}."
        )

    return config


def resolve_checkpoint_source(
    args: argparse.Namespace,
    threshold_config: Dict[str, Any],
) -> str:
    if args.az_checkpoint_path is not None:
        return args.az_checkpoint_path

    if args.use_checkpoint_from_threshold_json:
        source = threshold_config.get("az_checkpoint_source")
        if source:
            return str(source)
        logger.warning(
            "--use_checkpoint_from_threshold_json was set, but the JSON has no "
            "az_checkpoint_source. Falling back to --az_checkpoint_dir."
        )

    return args.az_checkpoint_dir


def resolve_sim_options(
    args: argparse.Namespace,
    threshold_config: Dict[str, Any],
) -> List[int]:
    if args.sim_options is not None:
        return [int(x) for x in args.sim_options]

    stored = threshold_config.get("sim_options")
    if stored is None:
        logger.warning(
            "Threshold JSON has no sim_options. Using [32, 64, 96, 128]."
        )
        return [32, 64, 96, 128]

    if len(stored) != 4:
        raise ValueError(
            f"Expected four sim_options in threshold JSON, received {stored}."
        )
    return [int(x) for x in stored]


def build_evaluator(
    agents,
    az_params_state,
    num_envs: int,
    eval_meta_steps: int,
):
    """Build a JIT evaluator supporting entropy, fixed-K, and random policies."""
    az_net_params = jax.lax.stop_gradient(az_params_state.params.net)
    az_net_state = jax.lax.stop_gradient(az_params_state.net_state)
    env_indices = jnp.arange(num_envs)

    @functools.partial(jax.jit, static_argnames=("policy_mode", "force_k"))
    def evaluate(
        thresholds: jax.Array,
        init_states: Any,
        init_obs: Any,
        mcts_key: jax.Array,
        *,
        policy_mode: str,
        force_k: int = -1,
    ):
        """Evaluate one policy.

        policy_mode:
            "entropy": thresholded AZNet policy entropy.
            "fixed": use force_k in {0,1,2,3}.
            "random": uniformly sample branch index in {0,1,2,3}.
        """
        if policy_mode not in ("entropy", "fixed", "random"):
            raise ValueError(f"Unknown policy_mode: {policy_mode}")
        if policy_mode == "fixed" and force_k not in (0, 1, 2, 3):
            raise ValueError("force_k must be 0,1,2,3 for fixed mode.")

        def scan_body(carry, step_key):
            states, obs, raw_return, done, k_counts = carry

            obs_grid, time_vec = agents[0]._get_grid_and_time(states, obs)
            invalid = agents[0]._get_invalid_actions(
                agents[0].raw_env_train,
                states,
                obs,
            )

            if policy_mode == "entropy":
                (policy_logits, _), _ = agents[0].forward.apply(
                    az_net_params,
                    az_net_state,
                    obs_grid,
                    time_vec,
                    is_eval=True,
                )
                entropy = masked_normalized_policy_entropy(
                    policy_logits,
                    invalid,
                )
                k_choices = entropy_to_k_index(entropy, thresholds)
            elif policy_mode == "fixed":
                k_choices = jnp.full(
                    (num_envs,),
                    force_k,
                    dtype=jnp.int32,
                )
            else:
                choice_key, _ = jax.random.split(step_key)
                k_choices = jax.random.randint(
                    choice_key,
                    (num_envs,),
                    minval=0,
                    maxval=4,
                    dtype=jnp.int32,
                )

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


def summarise(
    method: str,
    returns: jax.Array,
    done: jax.Array,
    k_counts: jax.Array,
) -> EvaluationResult:
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

    return EvaluationResult(
        method=method,
        mean_return=mean_return,
        standard_error=standard_error,
        mean_k=mean_k,
        k_frequencies=tuple(float(x) for x in frequencies),
        completed_fraction=float(np.mean(done_np)),
        num_episodes=n,
    )


def result_to_dict(result: EvaluationResult) -> Dict[str, Any]:
    data = asdict(result)
    data["k_frequencies"] = list(result.k_frequencies)
    return data


def log_result(result: EvaluationResult) -> None:
    logger.info(
        "%-14s return=%.3f +/- %.3f mean_K=%.3f "
        "K_freq=%s completed=%.1f%%",
        result.method,
        result.mean_return,
        result.standard_error,
        result.mean_k,
        np.round(np.asarray(result.k_frequencies) * 100.0, 1).tolist(),
        100.0 * result.completed_fraction,
    )


def run_policy(
    evaluator,
    env: VmapAutoResetWrapper,
    thresholds: jax.Array,
    num_episodes: int,
    reset_seed_base: int,
    mcts_seed: int,
    *,
    method: str,
    policy_mode: str,
    force_k: int = -1,
) -> EvaluationResult:
    states, obs = reset_episode_batch(
        env,
        num_episodes,
        reset_seed_base,
    )
    returns, done, k_counts = evaluator(
        thresholds,
        states,
        obs,
        jax.random.PRNGKey(mcts_seed),
        policy_mode=policy_mode,
        force_k=force_k,
    )
    return summarise(method, returns, done, k_counts)


def main() -> None:
    args = parse_args()
    validate_args(args)

    threshold_config = load_threshold_config(args.threshold_path)
    thresholds_list = [
        float(x) for x in threshold_config["thresholds"]
    ]
    thresholds = jnp.asarray(thresholds_list, dtype=jnp.float32)

    checkpoint_source = resolve_checkpoint_source(args, threshold_config)
    sim_options = resolve_sim_options(args, threshold_config)

    logger.info("JAX devices: %s", jax.devices())
    logger.info("Threshold file: %s", args.threshold_path)
    logger.info("Thresholds: %s", thresholds_list)
    logger.info("Simulation options: %s", sim_options)
    logger.info("Loading frozen AZNet from: %s", checkpoint_source)

    az_params_state = jax.device_put(
        load_az_checkpoint(checkpoint_source)
    )

    env = VmapAutoResetWrapper(jumanji.make("PacManKT-v1"))
    agents = make_agents(
        env,
        args.num_episodes,
        sim_options,
    )
    evaluator = build_evaluator(
        agents,
        az_params_state,
        args.num_episodes,
        args.eval_meta_steps,
    )

    reset_seed = args.evaluation_seed_base
    mcts_seed = (
        args.evaluation_seed_base
        + args.mcts_seed_offset
    )

    results: List[EvaluationResult] = []

    entropy_result = run_policy(
        evaluator,
        env,
        thresholds,
        args.num_episodes,
        reset_seed,
        mcts_seed,
        method="entropy",
        policy_mode="entropy",
    )
    results.append(entropy_result)
    log_result(entropy_result)

    if not args.entropy_only:
        for k_index, k_value in enumerate(K_OPTIONS):
            result = run_policy(
                evaluator,
                env,
                thresholds,
                args.num_episodes,
                reset_seed,
                mcts_seed,
                method=f"always_k{k_value}",
                policy_mode="fixed",
                force_k=k_index,
            )
            results.append(result)
            log_result(result)

        random_result = run_policy(
            evaluator,
            env,
            thresholds,
            args.num_episodes,
            reset_seed,
            mcts_seed,
            method="random_k",
            policy_mode="random",
        )
        results.append(random_result)
        log_result(random_result)

    incomplete = [
        result for result in results
        if result.completed_fraction < 1.0
    ]
    if incomplete:
        logger.warning(
            "%d method(s) had incomplete episodes within %d meta-steps. "
            "Increase --eval_meta_steps before reporting final results.",
            len(incomplete),
            args.eval_meta_steps,
        )

    output = {
        "threshold_path": args.threshold_path,
        "thresholds": thresholds_list,
        "az_checkpoint_source": checkpoint_source,
        "sim_options": sim_options,
        "num_episodes": args.num_episodes,
        "eval_meta_steps": args.eval_meta_steps,
        "evaluation_seed_base": args.evaluation_seed_base,
        "mcts_seed_offset": args.mcts_seed_offset,
        "results": [result_to_dict(result) for result in results],
    }

    if args.output_path is not None:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as file:
            json.dump(output, file, indent=2)
        logger.info("Saved evaluation results to: %s", args.output_path)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
