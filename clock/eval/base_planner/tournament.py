"""Tournament-style evaluation for the clock-environment AlphaZero base planners.

Plays trained checkpoints head-to-head across MCTS-sim budgets and seeds,
and records action-prefix diversity per game stage.
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import time
from typing import Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import mctx
import pgx
from pydantic import BaseModel
import wandb
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from collections import defaultdict, Counter

from network import AZNet  # same AZNet you used for training

tree_map = jtu.tree_map  # JAX v0.6+: jax.tree_map removed


# ---------------------------------------------------------------------
# Config stub for unpickling (matches training script)
# ---------------------------------------------------------------------
class Config(BaseModel):
    env_id: pgx.EnvId = "gardner_chess"
    seed: int = 0
    max_num_iters: int = 400
    # network params
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    max_num_steps: int = 256
    # training params
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    # eval params
    eval_interval: int = 5

    class Config:
        extra = "forbid"


# Map any pickled "Config" class to this one
class ConfigUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Config":
            return Config
        return super().find_class(module, name)


def load_checkpoint(path: str):
    """Load a checkpoint and return (env_id, cfg, model)."""
    with open(path, "rb") as f:
        data = ConfigUnpickler(f).load()
    model = data["model"]      # (params, state)
    cfg: Config = data["config"]
    env_id = data.get("env_id", cfg.env_id)
    return env_id, cfg, model


def build_forward(env: pgx.Env, cfg: Config):
    """Rebuild the same Haiku net as in training."""
    def forward_fn(x, is_eval: bool = False):
        net = AZNet(
            num_actions=env.num_actions,
            num_channels=cfg.num_channels,
            num_blocks=cfg.num_layers,
            resnet_v2=cfg.resnet_v2,
        )
        policy_out, value_out = net(
            x, is_training=not is_eval, test_local_stats=False
        )
        return policy_out, value_out

    return hk.without_apply_rng(hk.transform_with_state(forward_fn))


def make_recurrent_fn(env: pgx.Env, forward):
    """
    MuZero recurrent function, same structure as in your training script.
    """
    def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
        # model: (params, model_state)
        del rng_key
        model_params, model_state = model

        current_player = state.current_player
        state = jax.vmap(env.step)(state, action)

        (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )

        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(
            state.legal_action_mask,
            logits,
            jnp.finfo(logits.dtype).min,
        )

        batch_size = state.rewards.shape[0]
        reward = state.rewards[jnp.arange(batch_size), current_player]

        value = jnp.where(state.terminated, 0.0, value)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)

        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        ), state

    return recurrent_fn


# ---------------------------------------------------------------------
# Stats structures
# ---------------------------------------------------------------------
@dataclasses.dataclass
class ModelStats:
    name: str
    total_games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    steps_win_sum: float = 0.0
    steps_loss_sum: float = 0.0
    steps_draw_sum: float = 0.0
    moves_win: list[float] = dataclasses.field(default_factory=list)
    moves_loss: list[float] = dataclasses.field(default_factory=list)
    moves_draw: list[float] = dataclasses.field(default_factory=list)

    def describe(self):
        def ratio(n):
            return n / self.total_games if self.total_games else 0.0

        def avg(total, n):
            return total / n if n > 0 else float("nan")

        return {
            "games": self.total_games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": ratio(self.wins),
            "loss_rate": ratio(self.losses),
            "draw_rate": ratio(self.draws),
            "avg_moves_win": avg(self.steps_win_sum, self.wins),
            "avg_moves_loss": avg(self.steps_loss_sum, self.losses),
            "avg_moves_draw": avg(self.steps_draw_sum, self.draws),
        }


def discover_checkpoints(
    root: str,
    seeds: list[int] = [0, 1, 2, 3],
) -> Dict[Tuple[str, int], str]:
    """
    Find the latest checkpoints like:
      root/nsim_2/0/000800.ckpt (latest iteration)
      root/nsim_2/1/000800.ckpt
      ...
    Returns a dict { ("nsim_2", 0): path, ... }.
    """
    ckpts: Dict[Tuple[str, int], str] = {}
    if not os.path.isdir(root):
        raise ValueError(f"Checkpoint root '{root}' does not exist")

    for subdir in os.listdir(root):
        if not subdir.startswith("nsim_"):
            continue
        nsim = subdir.split("_", 1)[1]
        base = os.path.join(root, subdir)
        if not os.path.isdir(base):
            continue

        for seed in seeds:
            seed_dir = os.path.join(base, str(seed))
            if not os.path.isdir(seed_dir):
                continue

            ckpt_files = [f for f in os.listdir(seed_dir) if f.endswith(".ckpt")]
            if not ckpt_files:
                continue

            def get_iter_num(filename: str) -> int:
                try:
                    return int(filename.split(".")[0])
                except ValueError:
                    return -1

            latest_ckpt = max(ckpt_files, key=get_iter_num)
            ckpt_path = os.path.join(seed_dir, latest_ckpt)
            ckpts[(f"nsim_{nsim}", seed)] = ckpt_path
            print(f"Found checkpoint: {ckpt_path} (iteration {get_iter_num(latest_ckpt)})")

    return ckpts


def _get_step_count(final_state) -> np.ndarray:
    """Robustly extract step counts from pgx.State across pgx versions."""
    if hasattr(final_state, "_step_count"):
        return np.asarray(final_state._step_count)
    if hasattr(final_state, "step_count"):
        return np.asarray(final_state.step_count)
    raise AttributeError("Could not find step_count field on pgx.State (expected _step_count or step_count).")


# ---------------------------------------------------------------------
# Main tournament, with per-model MCTS budget == training num_simulations
# ---------------------------------------------------------------------
def main():
    # --- tweak these as needed ---
    CKPT_ROOT = "./checkpoints/clock/hex/base"
    ENV_ID = "gardner_chess"
    NUM_GAMES_PER_COLOR = 500           # games as White + games as Black per pairing
    SEEDS = [0, 1, 2, 3]                # seeds to compare
    TOURNAMENT_SEED = 2                 # seed for tournament randomness
    WANDB_PROJECT = "pgx-az-tournament"

    # For timing micro-benchmark
    TIMING_BATCH_SIZE = 64             # number of parallel boards for timing
    TIMING_REPS = 5                    # number of timed runs per model
    # ------------------------------

    print("Discovering latest checkpoints...")
    ckpt_paths = discover_checkpoints(CKPT_ROOT, SEEDS)
    if not ckpt_paths:
        raise RuntimeError("No checkpoints found; check CKPT_ROOT/SEEDS")

    # Use first checkpoint to recover env + architecture
    first_key = sorted(ckpt_paths.keys())[0]
    env_id0, cfg0, model0 = load_checkpoint(ckpt_paths[first_key])
    if env_id0 != ENV_ID:
        raise RuntimeError(f"Checkpoint env_id {env_id0} != expected {ENV_ID}")

    env = pgx.make(env_id0)
    forward = build_forward(env, cfg0)
    recurrent_fn = make_recurrent_fn(env, forward)

    max_steps = int(cfg0.max_num_steps)
    num_players = env.num_players

    # Load all models + their configs
    models: Dict[Tuple[str, int], Tuple[Tuple, Config]] = {first_key: (model0, cfg0)}
    for key, path in ckpt_paths.items():
        if key == first_key:
            continue
        env_id, cfg, model = load_checkpoint(path)
        if env_id != env_id0:
            raise RuntimeError(f"Env id mismatch for {key}: {env_id} != {env_id0}")
        if cfg.num_layers != cfg0.num_layers or cfg.num_channels != cfg0.num_channels:
            raise RuntimeError(
                f"Network arch mismatch for {key} "
                f"(layers {cfg.num_layers} vs {cfg0.num_layers}, "
                f"channels {cfg.num_channels} vs {cfg0.num_channels})"
            )
        if cfg.max_num_steps != cfg0.max_num_steps:
            raise RuntimeError(
                f"max_num_steps mismatch for {key} "
                f"({cfg.max_num_steps} vs {cfg0.max_num_steps})"
            )
        models[key] = (model, cfg)

    nsim_values = sorted(set(nsim for nsim, _ in models.keys()))
    print("Models found:")
    for nsim in nsim_values:
        seeds_for_nsim = [s for n, s in models.keys() if n == nsim]
        _, cfg = models[(nsim, seeds_for_nsim[0])]
        print(f"  {nsim}: seeds={sorted(seeds_for_nsim)}, num_simulations={cfg.num_simulations}")

    model_cfg_summary = {}
    for (nsim, seed), (_, cfg) in models.items():
        model_cfg_summary[f"{nsim}_seed{seed}"] = {
            "num_simulations": int(cfg.num_simulations),
            "num_layers": int(cfg.num_layers),
            "num_channels": int(cfg.num_channels),
        }

    iter_numbers = {}
    for (nsim, seed), path in ckpt_paths.items():
        filename = os.path.basename(path)
        iter_num = int(filename.split(".")[0])
        iter_numbers[f"{nsim}_seed{seed}"] = iter_num

    wandb_config = {
        "ckpt_root": CKPT_ROOT,
        "env_id": ENV_ID,
        "num_games_per_color": NUM_GAMES_PER_COLOR,
        "seeds": SEEDS,
        "tournament_seed": TOURNAMENT_SEED,
        "models": model_cfg_summary,
        "iterations": iter_numbers,
    }
    max_iter = max(iter_numbers.values()) if iter_numbers else 0
    WANDB_DIR = "./eval_outputs/clock/hex/wandb"
    os.makedirs(WANDB_DIR, exist_ok=True)

    wandb.init(
        project=WANDB_PROJECT,
        config=wandb_config,
        name=f"tournament_{ENV_ID}_latest_iter{max_iter}_seeds{SEEDS}",
        dir=WANDB_DIR,
    )

    # Stats: (nsim_name, seed) -> ModelStats
    stats: Dict[Tuple[str, int], ModelStats] = {}
    for (nsim, seed) in models.keys():
        stats[(nsim, seed)] = ModelStats(f"{nsim}_seed{seed}")

    print(f"Initialized stats for {len(stats)} model keys")

    pair_results: Dict[Tuple[Tuple[str, int], Tuple[str, int]], Dict[str, int]] = {}

    heatmap_data: Dict[Tuple[str, str], Dict[str, list]] = defaultdict(
        lambda: {"wins_p0": [], "losses_p0": [], "draws": []}
    )

    first_move_data: Dict[Tuple[str, str], list] = defaultdict(list)

    # Diversity tracking (sampled)
    DIVERSITY_SAMPLE_SIZE = min(100, NUM_GAMES_PER_COLOR)
    diversity_data: Dict[Tuple[str, str], Dict[str, object]] = defaultdict(
        lambda: {
            "opening_sequences": [],
            "game_sequences": [],
            "unique_openings": set(),
            "unique_sequences": set(),
        }
    )

    rng = jax.random.PRNGKey(TOURNAMENT_SEED)

    # -----------------------------------------------------------------
    # Shared JAX helpers
    # -----------------------------------------------------------------
    def select_actions_mcts(model, num_sims: int, state: pgx.State, rng_key: jnp.ndarray):
        """Single MCTS move selection with per-model num_simulations."""
        model_params, model_state = model
        (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )
        root = mctx.RootFnOutput(
            prior_logits=logits,
            value=value,
            embedding=state,
        )
        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=int(num_sims),
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        return policy_output.action  # (batch,)

    def update_stats(key: Tuple[str, int], outcome, steps):
        """
        outcome: +1 win, -1 loss, 0 draw (from that model's POV)
        steps:   number of moves (total step_count) for that game
        """
        if key not in stats:
            nsim, seed = key
            stats[key] = ModelStats(f"{nsim}_seed{seed}")

        s = stats[key]
        outcome = np.asarray(outcome)
        steps = np.asarray(steps)

        s.total_games += int(outcome.shape[0])

        win_mask = outcome > 0
        loss_mask = outcome < 0
        draw_mask = outcome == 0

        s.wins += int(win_mask.sum())
        s.losses += int(loss_mask.sum())
        s.draws += int(draw_mask.sum())

        s.steps_win_sum += float(steps[win_mask].sum())
        s.steps_loss_sum += float(steps[loss_mask].sum())
        s.steps_draw_sum += float(steps[draw_mask].sum())

        s.moves_win.extend(steps[win_mask].tolist())
        s.moves_loss.extend(steps[loss_mask].tolist())
        s.moves_draw.extend(steps[draw_mask].tolist())

    def compute_move_stats(outcomes, steps):
        outcomes = np.asarray(outcomes)
        steps = np.asarray(steps)

        mask_win = outcomes > 0
        mask_loss = outcomes < 0
        mask_draw = outcomes == 0

        def avg(msk):
            if not np.any(msk):
                return float("nan")
            return float(steps[msk].mean())

        return {
            "avg_moves_win": avg(mask_win),
            "avg_moves_loss": avg(mask_loss),
            "avg_moves_draw": avg(mask_draw),
        }

    # -----------------------------------------------------------------
    # Shared JIT-compiled match function (reused across all pairs)
    # -----------------------------------------------------------------
    def run_match(
        rng_key: jnp.ndarray,
        model_a,
        model_b,
        num_sims_a: int,
        num_sims_b: int,
        num_games_per_color: int,
        max_steps: int,
        num_players: int,
    ):
        """Play num_games_per_color games with A as player 0, B as player 1."""
        keys = jax.random.split(rng_key, num_games_per_color)
        state = jax.vmap(env.init)(keys)
        R = jnp.zeros((num_games_per_color, num_players), dtype=jnp.float32)
        step = jnp.array(0, dtype=jnp.int32)

        def cond_fn(carry):
            step, state, R, rng_key = carry
            done_all = jnp.all(state.terminated)
            return jnp.logical_and(step < max_steps, ~done_all)

        def body_fn(carry):
            step, state, R, rng_key = carry
            rng_key, key_a, key_b = jax.random.split(rng_key, 3)

            actions_a = select_actions_mcts(model_a, num_sims_a, state, key_a)
            actions_b = select_actions_mcts(model_b, num_sims_b, state, key_b)

            current_player = state.current_player
            actions = jnp.where(current_player == 0, actions_a, actions_b)

            state = jax.vmap(env.step)(state, actions)
            R = R + state.rewards
            step = step + 1
            return step, state, R, rng_key

        step, state, R, rng_key = jax.lax.while_loop(
            cond_fn, body_fn, (step, state, R, rng_key)
        )
        return state, R

    run_match_jit = jax.jit(
        run_match,
        static_argnums=(3, 4, 5, 6, 7),
    )

    # -----------------------------------------------------------------
    # batched + jit diversity sampler that records actions
    # -----------------------------------------------------------------
    def run_match_with_actions(
        rng_key: jnp.ndarray,
        model_a,
        model_b,
        num_sims_a: int,
        num_sims_b: int,
        batch_size: int,
        max_steps: int,
    ):
        """
        Play `batch_size` games (A as player0, B as player1) and record actions.

        Returns:
          final_state: pgx.State (batched)
          actions_hist: int32 array (max_steps, batch_size), with -1 for steps after termination
        """
        keys = jax.random.split(rng_key, batch_size)
        state = jax.vmap(env.init)(keys)

        actions_hist = -jnp.ones((max_steps, batch_size), dtype=jnp.int32)
        step = jnp.array(0, dtype=jnp.int32)

        def cond_fn(carry):
            step, state, actions_hist, rng_key = carry
            return jnp.logical_and(step < max_steps, ~jnp.all(state.terminated))

        def body_fn(carry):
            step, state, actions_hist, rng_key = carry
            rng_key, key_a, key_b = jax.random.split(rng_key, 3)

            actions_a = select_actions_mcts(model_a, num_sims_a, state, key_a)
            actions_b = select_actions_mcts(model_b, num_sims_b, state, key_b)
            actions = jnp.where(state.current_player == 0, actions_a, actions_b).astype(jnp.int32)

            done = state.terminated  # (B,)
            actions_hist = actions_hist.at[step].set(jnp.where(done, -1, actions))

            # Step only active games; freeze terminated ones.
            # (We still need a valid action for env.step; 0 is arbitrary but masked out by freezing.)
            actions_safe = jnp.where(done, 0, actions)
            next_state = jax.vmap(env.step)(state, actions_safe)

            B = done.shape[0]

            def _freeze_leaf(new_leaf, old_leaf):
                # Only freeze leaves with leading batch dim B
                if not hasattr(new_leaf, "shape") or new_leaf.shape is None:
                    return new_leaf
                if new_leaf.ndim == 0:
                    return new_leaf
                if new_leaf.shape[0] != B:
                    return new_leaf
                mask = done.reshape((B,) + (1,) * (new_leaf.ndim - 1))
                return jnp.where(mask, old_leaf, new_leaf)

            next_state = tree_map(_freeze_leaf, next_state, state)

            return step + 1, next_state, actions_hist, rng_key

        step, final_state, actions_hist, rng_key = jax.lax.while_loop(
            cond_fn, body_fn, (step, state, actions_hist, rng_key)
        )
        return final_state, actions_hist

    run_match_with_actions_jit = jax.jit(
        run_match_with_actions,
        static_argnums=(3, 4, 5, 6),  # num_sims_a, num_sims_b, batch_size, max_steps
    )

    # -----------------------------------------------------------------
    # Diversity metrics (NO observation hashing; uses action-prefix diversity)
    # -----------------------------------------------------------------
    def compute_diversity_metrics(move_sequences, opening_length=5):
        """
        move_sequences: list[list[int]] of length N games

        NOTE:
          - "state_diversity_by_stage" is computed from action-prefixes.
          - To match your previous hash-based indexing semantics:
              stage_k corresponds to diversity of the state *before move k*,
              i.e., prefix length = k-1.
        """
        if not move_sequences:
            return {}

        num_games = len(move_sequences)

        # Openings
        openings = []
        for seq in move_sequences:
            if len(seq) >= opening_length:
                openings.append(tuple(seq[:opening_length]))
            elif len(seq) > 0:
                openings.append(tuple(seq))
            else:
                openings.append(tuple())

        full_sequences = [tuple(seq) for seq in move_sequences if len(seq) > 0]

        unique_openings = len(set(openings))
        unique_sequences = len(set(full_sequences))

        opening_counter = Counter(openings)
        most_common = opening_counter.most_common(1)[0] if opening_counter else None

        # "State diversity by stage" via prefix sets (deterministic environments)
        state_diversity_by_stage = {}
        for stage in [1, 5, 10, 20]:
            prefix_len = max(stage - 1, 0)
            stage_prefixes = [
                tuple(seq[:prefix_len])
                for seq in move_sequences
                if len(seq) >= prefix_len
            ]
            if stage_prefixes:
                state_diversity_by_stage[f"stage_{stage}"] = len(set(stage_prefixes)) / len(stage_prefixes)
            else:
                state_diversity_by_stage[f"stage_{stage}"] = 0.0

        return {
            "num_games": num_games,
            "unique_openings": unique_openings,
            "opening_diversity": unique_openings / max(num_games, 1),
            "unique_sequences": unique_sequences,
            "sequence_diversity": unique_sequences / max(num_games, 1),
            "state_diversity_by_stage": state_diversity_by_stage,
            "avg_game_length": float(np.mean([len(seq) for seq in move_sequences])) if move_sequences else 0.0,
            "most_common_opening": most_common,
            "most_common_opening_freq": (most_common[1] / num_games) if most_common else 0.0,
            "opening_entropy": float(scipy_stats.entropy(list(opening_counter.values()))) if opening_counter else 0.0,
        }

    # -----------------------------------------------------------------
    # Round-robin tournament: compare nsim_X seed_Y vs nsim_Z seed_Y
    # -----------------------------------------------------------------
    for i, nsim_a in enumerate(nsim_values):
        for nsim_b in nsim_values[i:]:  # Include self-comparisons
            for seed in SEEDS:
                key_a = (nsim_a, seed)
                key_b = (nsim_b, seed)

                if key_a not in models or key_b not in models:
                    continue

                model_a, cfg_a = models[key_a]
                model_b, cfg_b = models[key_b]

                num_sims_a = int(cfg_a.num_simulations)
                num_sims_b = int(cfg_b.num_simulations)

                print(
                    f"\n=== {nsim_a} seed {seed} (sims={num_sims_a}) "
                    f"vs {nsim_b} seed {seed} (sims={num_sims_b}) ==="
                )

                # Orientation 1: nsim_a as player 0, nsim_b as player 1
                rng, key1 = jax.random.split(rng)
                final1, R1 = run_match_jit(
                    key1,
                    model_a,
                    model_b,
                    num_sims_a,
                    num_sims_b,
                    NUM_GAMES_PER_COLOR,
                    max_steps,
                    num_players,
                )
                final1 = jax.device_get(final1)
                R1 = np.asarray(jax.device_get(R1))
                steps1 = _get_step_count(final1)

                # Orientation 2: nsim_b as player 0, nsim_a as player 1
                rng, key2 = jax.random.split(rng)
                final2, R2 = run_match_jit(
                    key2,
                    model_b,
                    model_a,
                    num_sims_b,
                    num_sims_a,
                    NUM_GAMES_PER_COLOR,
                    max_steps,
                    num_players,
                )
                final2 = jax.device_get(final2)
                R2 = np.asarray(jax.device_get(R2))
                steps2 = _get_step_count(final2)

                # Outcomes from R[:, 0] (player-0's return)
                outcome_A1 = np.sign(R1[:, 0])  # nsim_a as player 0
                outcome_B1 = -outcome_A1        # nsim_b as player 1

                outcome_B2 = np.sign(R2[:, 0])  # nsim_b as player 0
                outcome_A2 = -outcome_B2        # nsim_a as player 1

                update_stats(key_a, outcome_A1, steps1)
                update_stats(key_b, outcome_B1, steps1)
                update_stats(key_b, outcome_B2, steps2)
                update_stats(key_a, outcome_A2, steps2)

                wins_A = int((outcome_A1 > 0).sum() + (outcome_A2 > 0).sum())
                wins_B = int((outcome_B1 > 0).sum() + (outcome_B2 > 0).sum())
                draws = int((outcome_A1 == 0).sum() + (outcome_A2 == 0).sum())
                total_games = 2 * NUM_GAMES_PER_COLOR

                pair_results[(key_a, key_b)] = {
                    "games": total_games,
                    f"{nsim_a}_wins": wins_A,
                    f"{nsim_b}_wins": wins_B,
                    "draws": draws,
                }
                print(
                    f"{nsim_a} wins: {wins_A}, {nsim_b} wins: {wins_B}, "
                    f"draws: {draws} (out of {total_games})"
                )

                # Heatmap aggregation
                wins_p0_1 = int((outcome_A1 > 0).sum())
                losses_p0_1 = int((outcome_A1 < 0).sum())
                draws_1 = int((outcome_A1 == 0).sum())
                heatmap_data[(nsim_a, nsim_b)]["wins_p0"].append(wins_p0_1)
                heatmap_data[(nsim_a, nsim_b)]["losses_p0"].append(losses_p0_1)
                heatmap_data[(nsim_a, nsim_b)]["draws"].append(draws_1)

                if nsim_a != nsim_b:
                    wins_p0_2 = int((outcome_B2 > 0).sum())
                    losses_p0_2 = int((outcome_B2 < 0).sum())
                    draws_2 = int((outcome_B2 == 0).sum())
                    heatmap_data[(nsim_b, nsim_a)]["wins_p0"].append(wins_p0_2)
                    heatmap_data[(nsim_b, nsim_a)]["losses_p0"].append(losses_p0_2)
                    heatmap_data[(nsim_b, nsim_a)]["draws"].append(draws_2)

                # First move advantage tracking
                pair_key = (nsim_a, nsim_b)
                for p0, p1 in zip(outcome_A1, outcome_B1):
                    first_move_data[pair_key].append((float(p0), float(p1)))
                for p0, p1 in zip(outcome_B2, outcome_A2):
                    first_move_data[pair_key].append((float(p0), float(p1)))

                na_outcomes = np.concatenate([outcome_A1, outcome_A2], axis=0)
                nb_outcomes = np.concatenate([outcome_B1, outcome_B2], axis=0)
                pair_steps = np.concatenate([steps1, steps2], axis=0)

                na_move_stats = compute_move_stats(na_outcomes, pair_steps)
                nb_move_stats = compute_move_stats(nb_outcomes, pair_steps)

                pair_log = {
                    f"pair/{nsim_a}_seed{seed}_vs_{nsim_b}_seed{seed}/games": total_games,
                    f"pair/{nsim_a}_seed{seed}_vs_{nsim_b}_seed{seed}/{nsim_a}_wins": wins_A,
                    f"pair/{nsim_a}_seed{seed}_vs_{nsim_b}_seed{seed}/{nsim_b}_wins": wins_B,
                    f"pair/{nsim_a}_seed{seed}_vs_{nsim_b}_seed{seed}/draws": draws,
                    f"pair/{nsim_a}_seed{seed}_vs_{nsim_b}_seed{seed}/{nsim_a}_win_rate": wins_A / total_games,
                    f"pair/{nsim_a}_seed{seed}_vs_{nsim_b}_seed{seed}/{nsim_b}_win_rate": wins_B / total_games,
                    f"pair/{nsim_a}_seed{seed}_vs_{nsim_b}_seed{seed}/{nsim_a}_avg_moves_win": na_move_stats["avg_moves_win"],
                    f"pair/{nsim_a}_seed{seed}_vs_{nsim_b}_seed{seed}/{nsim_a}_avg_moves_loss": na_move_stats["avg_moves_loss"],
                    f"pair/{nsim_a}_seed{seed}_vs_{nsim_b}_seed{seed}/{nsim_a}_avg_moves_draw": na_move_stats["avg_moves_draw"],
                    f"pair/{nsim_a}_seed{seed}_vs_{nsim_b}_seed{seed}/{nsim_b}_avg_moves_win": nb_move_stats["avg_moves_win"],
                    f"pair/{nsim_a}_seed{seed}_vs_{nsim_b}_seed{seed}/{nsim_b}_avg_moves_loss": nb_move_stats["avg_moves_loss"],
                    f"pair/{nsim_a}_seed{seed}_vs_{nsim_b}_seed{seed}/{nsim_b}_avg_moves_draw": nb_move_stats["avg_moves_draw"],
                }
                wandb.log(pair_log)

    # -----------------------------------------------------------------
    # Diversity tracking: sample games for each pair (batched+jitted)
    # -----------------------------------------------------------------
    print("\n=== Tracking game diversity (sampling games) ===")
    for i, nsim_a in enumerate(nsim_values):
        for nsim_b in nsim_values[i:]:
            pair_key = (nsim_a, nsim_b)

            model_a = None
            model_b = None
            num_sims_a = None
            num_sims_b = None

            for seed in SEEDS:
                key_a = (nsim_a, seed)
                key_b = (nsim_b, seed)
                if key_a in models and key_b in models:
                    model_a, cfg_a = models[key_a]
                    model_b, cfg_b = models[key_b]
                    num_sims_a = int(cfg_a.num_simulations)
                    num_sims_b = int(cfg_b.num_simulations)
                    break

            if model_a is None or model_b is None:
                continue

            print(f"  Sampling {DIVERSITY_SAMPLE_SIZE} games for {nsim_a} vs {nsim_b}...")

            rng, div_key = jax.random.split(rng)

            final_state, actions_hist = run_match_with_actions_jit(
                div_key,
                model_a,
                model_b,
                num_sims_a,
                num_sims_b,
                DIVERSITY_SAMPLE_SIZE,
                max_steps,
            )

            final_state = jax.device_get(final_state)
            actions_hist = np.asarray(jax.device_get(actions_hist))  # (T, B)
            steps = _get_step_count(final_state).astype(int)         # (B,)

            move_sequences = []
            for b in range(DIVERSITY_SAMPLE_SIZE):
                # Use step_count to bound sequence length, then drop -1s
                T = max(0, min(int(steps[b]), max_steps))
                seq = actions_hist[:T, b]
                seq = seq[seq >= 0]
                move_sequences.append(seq.tolist())

            diversity_data[pair_key]["game_sequences"] = move_sequences

            # Update opening/unique sets
            for seq in move_sequences:
                if len(seq) >= 5:
                    diversity_data[pair_key]["opening_sequences"].append(tuple(seq[:5]))
                    diversity_data[pair_key]["unique_openings"].add(tuple(seq[:5]))
                elif len(seq) > 0:
                    diversity_data[pair_key]["opening_sequences"].append(tuple(seq))
                    diversity_data[pair_key]["unique_openings"].add(tuple(seq))

                if len(seq) > 0:
                    diversity_data[pair_key]["unique_sequences"].add(tuple(seq))

    # -----------------------------------------------------------------
    # Final per-pair printout
    # -----------------------------------------------------------------
    print("\n=== Per-pair results ===")
    for (key_a, key_b), res in pair_results.items():
        print(f"{key_a} vs {key_b}: {res}")

    # -----------------------------------------------------------------
    # Per-model global summary + wandb logging
    # -----------------------------------------------------------------
    print("\n=== Per-model summary ===")
    for (nsim, seed) in sorted(stats.keys()):
        d = stats[(nsim, seed)].describe()
        name = f"{nsim}_seed{seed}"
        print(f"{name}:")
        print(
            f"  games={d['games']}  W/L/D = "
            f"{d['wins']}/{d['losses']}/{d['draws']} "
            f"(win_rate={d['win_rate']:.3f})"
        )
        print(
            f"  avg_moves (win/loss/draw) = "
            f"{d['avg_moves_win']:.2f} / "
            f"{d['avg_moves_loss']:.2f} / "
            f"{d['avg_moves_draw']:.2f}"
        )

        model_log = {
            f"model/{name}/games": d["games"],
            f"model/{name}/wins": d["wins"],
            f"model/{name}/losses": d["losses"],
            f"model/{name}/draws": d["draws"],
            f"model/{name}/win_rate": d["win_rate"],
            f"model/{name}/loss_rate": d["loss_rate"],
            f"model/{name}/draw_rate": d["draw_rate"],
            f"model/{name}/avg_moves_win": d["avg_moves_win"],
            f"model/{name}/avg_moves_loss": d["avg_moves_loss"],
            f"model/{name}/avg_moves_draw": d["avg_moves_draw"],
        }
        wandb.log(model_log)

    # -----------------------------------------------------------------
    # Inference timing micro-benchmark (per model, per decision)
    # -----------------------------------------------------------------
    print("\n=== Inference timing (approx, per decision) ===")
    timing_results = {}

    for idx, (key, (model, cfg)) in enumerate(sorted(models.items())):
        nsim, seed = key
        name = f"{nsim}_seed{seed}"
        num_sims = int(cfg.num_simulations)

        key_states = jax.random.PRNGKey(TOURNAMENT_SEED + 12345 + idx)
        keys = jax.random.split(key_states, TIMING_BATCH_SIZE)
        state = jax.vmap(env.init)(keys)

        def mcts_once(state, rng_key):
            return select_actions_mcts(model, num_sims, state, rng_key)

        mcts_once_jit = jax.jit(mcts_once)

        warm_key = jax.random.PRNGKey(TOURNAMENT_SEED + 9999 + idx)
        _ = jax.block_until_ready(mcts_once_jit(state, warm_key))

        total_time = 0.0
        for rep in range(TIMING_REPS):
            rep_key = jax.random.PRNGKey(TOURNAMENT_SEED + 100000 * (idx + 1) + rep)
            t0 = time.time()
            actions = mcts_once_jit(state, rep_key)
            _ = jax.block_until_ready(actions)
            t1 = time.time()
            total_time += (t1 - t0)

        avg_batch_time = total_time / TIMING_REPS
        avg_decision_time = avg_batch_time / float(TIMING_BATCH_SIZE)
        timing_results[name] = avg_decision_time

        print(
            f"{name} (sims={num_sims}): "
            f"{avg_batch_time:.4f}s per batch of {TIMING_BATCH_SIZE} "
            f"~ {avg_decision_time*1000:.3f} ms / decision"
        )

        wandb.log({
            f"timing/{name}/avg_batch_time_s": avg_batch_time,
            f"timing/{name}/avg_decision_time_s": avg_decision_time,
            f"timing/{name}/num_simulations": num_sims,
        })

    # -----------------------------------------------------------------
    # Distribution plots of game length for W / L / D per model
    # -----------------------------------------------------------------
    print("\n=== Plotting move-length distributions per model ===")

    for (nsim, seed) in sorted(stats.keys()):
        name = f"{nsim}_seed{seed}"
        s = stats[(nsim, seed)]

        win_moves = np.asarray(s.moves_win, dtype=float)
        loss_moves = np.asarray(s.moves_loss, dtype=float)
        draw_moves = np.asarray(s.moves_draw, dtype=float)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

        data_lists = [win_moves, loss_moves, draw_moves]
        titles = ["Wins", "Losses", "Draws"]

        for ax, data, title in zip(axes, data_lists, titles):
            if data.size > 0:
                ax.hist(data, bins=20, density=True)
                mean_val = float(data.mean())
                ax.axvline(mean_val, linestyle="--")
                ax.set_title(f"{title} (n={len(data)}, mean={mean_val:.1f})")
                ax.set_xlabel("Moves")
            else:
                ax.set_title(f"{title} (no games)")
                ax.set_xlabel("Moves")
            ax.grid(True, linestyle=":")

        axes[0].set_ylabel("Density")
        fig.suptitle(f"{name}: distribution of game length by outcome")
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))

        wandb.log({f"plots/{name}_move_length_distribution": wandb.Image(fig)})
        plt.close(fig)

    # -----------------------------------------------------------------
    # Heatmaps: W/L/D rates aggregated across seeds
    # -----------------------------------------------------------------
    print("\n=== Creating heatmaps (aggregated across seeds) ===")

    n_nsim = len(nsim_values)
    win_matrix = np.full((n_nsim, n_nsim), np.nan)
    loss_matrix = np.full((n_nsim, n_nsim), np.nan)
    draw_matrix = np.full((n_nsim, n_nsim), np.nan)
    win_std_matrix = np.full((n_nsim, n_nsim), np.nan)
    loss_std_matrix = np.full((n_nsim, n_nsim), np.nan)
    draw_std_matrix = np.full((n_nsim, n_nsim), np.nan)

    for i, nsim_a in enumerate(nsim_values):
        for j, nsim_b in enumerate(nsim_values):
            pair_key = (nsim_a, nsim_b)
            if pair_key not in heatmap_data:
                continue

            data = heatmap_data[pair_key]
            games_per_batch = NUM_GAMES_PER_COLOR

            win_rates = [w / games_per_batch for w in data["wins_p0"]]
            loss_rates = [l / games_per_batch for l in data["losses_p0"]]
            draw_rates = [d / games_per_batch for d in data["draws"]]

            if win_rates:
                win_matrix[i, j] = np.mean(win_rates)
                win_std_matrix[i, j] = np.std(win_rates)
            if loss_rates:
                loss_matrix[i, j] = np.mean(loss_rates)
                loss_std_matrix[i, j] = np.std(loss_rates)
            if draw_rates:
                draw_matrix[i, j] = np.mean(draw_rates)
                draw_std_matrix[i, j] = np.std(draw_rates)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    matrices = [win_matrix, loss_matrix, draw_matrix]
    std_matrices = [win_std_matrix, loss_std_matrix, draw_std_matrix]
    titles = ["Win Rate (Player 0)", "Loss Rate (Player 0)", "Draw Rate"]

    for ax, mat, std_mat, title in zip(axes, matrices, std_matrices, titles):
        im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=1)

        for ii in range(n_nsim):
            for jj in range(n_nsim):
                if not np.isnan(mat[ii, jj]):
                    mean_val = mat[ii, jj]
                    std_val = std_mat[ii, jj] if not np.isnan(std_mat[ii, jj]) else 0.0
                    text = f"{mean_val:.2f}\n±{std_val:.2f}"
                    ax.text(
                        jj,
                        ii,
                        text,
                        ha="center",
                        va="center",
                        color="white" if mean_val > 0.5 else "black",
                        fontsize=8,
                    )

        ax.set_xticks(range(n_nsim))
        ax.set_yticks(range(n_nsim))
        ax.set_xticklabels(nsim_values)
        ax.set_yticklabels(nsim_values)
        ax.set_xlabel("Player 1 (nsim)")
        ax.set_ylabel("Player 0 (nsim)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    fig.suptitle("Outcome Rates (aggregated across seeds, with std error bars)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    wandb.log({"plots/heatmaps_wld_rates": wandb.Image(fig)})
    plt.close(fig)

    # -----------------------------------------------------------------
    # First move advantage: correlation analysis
    # -----------------------------------------------------------------
    print("\n=== Analyzing first move advantage correlation ===")

    all_p0_outcomes = []
    all_p1_outcomes = []

    for (_, _), outcomes in first_move_data.items():
        for p0_out, p1_out in outcomes:
            all_p0_outcomes.append(p0_out)
            all_p1_outcomes.append(p1_out)

    all_p0_outcomes = np.array(all_p0_outcomes)
    all_p1_outcomes = np.array(all_p1_outcomes)

    if len(all_p0_outcomes) > 1:
        correlation, p_value = scipy_stats.pearsonr(all_p0_outcomes, all_p1_outcomes)
        print(f"Correlation between Player 0 and Player 1 outcomes: {correlation:.4f} (p={p_value:.4f})")

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.scatter(all_p0_outcomes, all_p1_outcomes, alpha=0.3, s=10)
        ax.set_xlabel("Player 0 Outcome (+1=win, 0=draw, -1=loss)")
        ax.set_ylabel("Player 1 Outcome (+1=win, 0=draw, -1=loss)")
        ax.set_title(f"First Move Advantage Correlation\nr={correlation:.4f}, p={p_value:.4f}")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.5)

        if not np.isnan(correlation):
            z = np.polyfit(all_p0_outcomes, all_p1_outcomes, 1)
            p = np.poly1d(z)
            x_line = np.linspace(all_p0_outcomes.min(), all_p0_outcomes.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f"y={z[0]:.3f}x+{z[1]:.3f}")
            ax.legend()

        fig.tight_layout()
        wandb.log({"plots/first_move_advantage_correlation": wandb.Image(fig)})
        plt.close(fig)

        print("\n=== Per-pair first move advantage ===")
        pair_correlations = {}
        for (nsim_a, nsim_b), outcomes in first_move_data.items():
            if len(outcomes) < 2:
                continue
            p0_vals = np.array([o[0] for o in outcomes])
            p1_vals = np.array([o[1] for o in outcomes])
            corr, p_val = scipy_stats.pearsonr(p0_vals, p1_vals)
            pair_correlations[f"{nsim_a}_vs_{nsim_b}"] = {
                "correlation": float(corr),
                "p_value": float(p_val),
                "n_games": len(outcomes),
            }
            print(f"{nsim_a} vs {nsim_b}: r={corr:.4f}, p={p_val:.4f}, n={len(outcomes)}")

        for pair_name, corr_data in pair_correlations.items():
            wandb.log({
                f"first_move/{pair_name}/correlation": corr_data["correlation"],
                f"first_move/{pair_name}/p_value": corr_data["p_value"],
                f"first_move/{pair_name}/n_games": corr_data["n_games"],
            })

    # -----------------------------------------------------------------
    # Game Diversity Analysis
    # -----------------------------------------------------------------
    print("\n=== Analyzing game diversity ===")

    diversity_metrics: Dict[Tuple[str, str], Dict] = {}

    for pair_key, data in diversity_data.items():
        nsim_a, nsim_b = pair_key
        if not data["game_sequences"]:
            continue

        metrics = compute_diversity_metrics(
            data["game_sequences"],
            opening_length=5,
        )
        diversity_metrics[pair_key] = metrics

        print(f"\n{nsim_a} vs {nsim_b}:")
        print(f"  Games analyzed: {metrics['num_games']}")
        print(f"  Opening diversity: {metrics['opening_diversity']:.3f} ({metrics['unique_openings']} unique openings)")
        print(f"  Sequence diversity: {metrics['sequence_diversity']:.3f} ({metrics['unique_sequences']} unique sequences)")
        print(f"  Most common opening frequency: {metrics['most_common_opening_freq']:.3f}")
        print(f"  Opening entropy: {metrics['opening_entropy']:.3f}")
        print(f"  State diversity by stage: {metrics['state_diversity_by_stage']}")

        pair_name = f"{nsim_a}_vs_{nsim_b}"
        wandb.log({
            f"diversity/{pair_name}/num_games": metrics["num_games"],
            f"diversity/{pair_name}/opening_diversity": metrics["opening_diversity"],
            f"diversity/{pair_name}/unique_openings": metrics["unique_openings"],
            f"diversity/{pair_name}/sequence_diversity": metrics["sequence_diversity"],
            f"diversity/{pair_name}/unique_sequences": metrics["unique_sequences"],
            f"diversity/{pair_name}/most_common_opening_freq": metrics["most_common_opening_freq"],
            f"diversity/{pair_name}/opening_entropy": metrics["opening_entropy"],
            f"diversity/{pair_name}/avg_game_length": metrics["avg_game_length"],
        })
        for stage, div in metrics["state_diversity_by_stage"].items():
            wandb.log({f"diversity/{pair_name}/state_diversity_{stage}": div})

    # Create diversity heatmaps
    if diversity_metrics:
        print("\n=== Creating diversity heatmaps ===")

        opening_div_matrix = np.full((n_nsim, n_nsim), np.nan)
        sequence_div_matrix = np.full((n_nsim, n_nsim), np.nan)
        opening_entropy_matrix = np.full((n_nsim, n_nsim), np.nan)

        for i, nsim_a in enumerate(nsim_values):
            for j, nsim_b in enumerate(nsim_values):
                pair_key = (nsim_a, nsim_b)
                if pair_key in diversity_metrics:
                    metrics = diversity_metrics[pair_key]
                    opening_div_matrix[i, j] = metrics["opening_diversity"]
                    sequence_div_matrix[i, j] = metrics["sequence_diversity"]
                    opening_entropy_matrix[i, j] = metrics["opening_entropy"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        matrices = [opening_div_matrix, sequence_div_matrix, opening_entropy_matrix]
        titles = ["Opening Diversity", "Sequence Diversity", "Opening Entropy"]
        vmaxs = [1.0, 1.0, None]

        for ax, mat, title, vmax in zip(axes, matrices, titles, vmaxs):
            im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)

            for ii in range(n_nsim):
                for jj in range(n_nsim):
                    if not np.isnan(mat[ii, jj]):
                        val = mat[ii, jj]
                        ax.text(
                            jj,
                            ii,
                            f"{val:.3f}",
                            ha="center",
                            va="center",
                            color="white" if (vmax is not None and val > vmax / 2) else "black",
                            fontsize=9,
                        )

            ax.set_xticks(range(n_nsim))
            ax.set_yticks(range(n_nsim))
            ax.set_xticklabels(nsim_values)
            ax.set_yticklabels(nsim_values)
            ax.set_xlabel("Player 1 (nsim)")
            ax.set_ylabel("Player 0 (nsim)")
            ax.set_title(title)
            plt.colorbar(im, ax=ax)

        fig.suptitle("Game Diversity Metrics (aggregated across seeds)")
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))

        wandb.log({"plots/diversity_heatmaps": wandb.Image(fig)})
        plt.close(fig)

        # Opening frequency distribution plot
        fig, axes = plt.subplots(len(diversity_metrics), 1, figsize=(12, 4 * len(diversity_metrics)))
        if len(diversity_metrics) == 1:
            axes = [axes]

        for idx, (pair_key, metrics) in enumerate(sorted(diversity_metrics.items())):
            nsim_a, nsim_b = pair_key
            pair_name = f"{nsim_a} vs {nsim_b}"

            opening_counter = Counter(diversity_data[pair_key]["opening_sequences"])
            top_openings = opening_counter.most_common(10)

            if top_openings:
                openings_str = [str(o) for o, _ in top_openings]
                freqs = [f for _, f in top_openings]

                axes[idx].barh(openings_str, freqs)
                axes[idx].set_xlabel("Frequency")
                axes[idx].set_title(f"{pair_name}: Top 10 Opening Sequences")
                axes[idx].invert_yaxis()
                axes[idx].grid(True, alpha=0.3)

        fig.suptitle("Most Common Opening Sequences")
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        wandb.log({"plots/opening_frequencies": wandb.Image(fig)})
        plt.close(fig)


if __name__ == "__main__":
    main()
