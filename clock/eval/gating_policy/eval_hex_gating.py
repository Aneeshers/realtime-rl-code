#!/usr/bin/env python3
"""
Gating policy evaluation (no-timeout variant).
Supports:
  - --env speed_gardner_chess  OR  --env speed_hex (and future speed_* envs)
  - --opponents 0 (policy-only opponent, no MCTS ever)

Key behavioral rule:
  - Neither side is allowed to lose by timeout in this eval harness.
    If the side-to-move cannot afford its intended MCTS cost (my_time <= intended_nsim),
    it plays AZ policy argmax and spends time_spent=0.

Metrics logged to wandb per opponent:
  - expected_score, fair_clock_expected_score (unique-game-deduplicated)
  - p0/p1 policy-only first step means (when fallback first triggered)
  - gate choice distribution (% of moves at each MCTS budget)

Optionally visualizes one win + one loss per (opponent x time) as annotated SVG animations.
"""

import os
import re
import pickle
import argparse
import uuid
import math
import hashlib
import importlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import pgx
from pydantic import BaseModel

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network_intermediate import AZNet  # must be compatible with pretrained checkpoints


# ================================================================
# Defaults
# ================================================================


ITER_FILE_DEFAULT = os.getenv(
    "ITER_FILE",
    "base_planner.ckpt"
)

# You will typically override these on the CLI anyway.
CKPT_ROOT_DEFAULT = os.getenv(
    "CKPT_ROOT",
    "./checkpoints/clock/hex/base",
)
GATE_ROOT_DEFAULT = os.getenv(
    "GATE_ROOT",
    "./checkpoints/clock/hex/gating",
)

SIM_OPTIONS_DEFAULT = [2, 8, 16, 32]


# ================================================================
# Speed-env loading
# ================================================================

@dataclass(frozen=True)
class SpeedEnvBundle:
    env_name: str
    module: Any
    env: Any
    base_env_id: str
    default_time: int
    max_steps: int
    step_board: Any
    observe: Any


def _try_make_env(module, hex_size: Optional[int] = None):
    """
    Heuristics to instantiate a speed env module.

    Expected in your speed env module (preferred):
      - make_env(...) -> env
    Otherwise common class names:
      - GardnerChess() for speed_gardner_chess
      - Hex(size=...) for speed_hex
      - Env() as a generic alias
    """
    if hasattr(module, "make_env"):
        try:
            return module.make_env(size=hex_size)  # optional kw
        except TypeError:
            return module.make_env()

    # common class names
    for cls_name in ("Env", "GardnerChess", "Hex", "SpeedHex", "SpeedHexEnv"):
        if hasattr(module, cls_name):
            cls = getattr(module, cls_name)
            try:
                # try with size if supported
                if hex_size is not None:
                    return cls(size=hex_size)
                return cls()
            except TypeError:
                return cls()

    raise RuntimeError(
        f"Could not construct env from module '{module.__name__}'. "
        "Expected make_env() or a known Env class."
    )


def load_speed_env(env_module_name: str, hex_size: Optional[int] = None) -> SpeedEnvBundle:
    """
    Load a speed env module and return needed hooks.

    Required exports in speed env module:
      - _step_board(state, action)   (pure board step; no clocks)
      - _observe(state, player_id)   (board-only observe)
      - DEFAULT_TIME (int)
      - MAX_TERMINATION_STEPS (int)
    """
    module = importlib.import_module(env_module_name)
    env = _try_make_env(module, hex_size=hex_size)

    # base env id: prefer module constant if provided, else env.id
    base_env_id = getattr(module, "BASE_ENV_ID", None)
    if base_env_id is None:
        base_env_id = getattr(env, "id", None)
    if base_env_id is None:
        raise RuntimeError(f"Could not infer base env id for {env_module_name}")

    step_board = getattr(module, "_step_board", None)
    observe = getattr(module, "_observe", None)
    if step_board is None or observe is None:
        raise RuntimeError(
            f"Module '{env_module_name}' must export _step_board and _observe"
        )

    default_time = int(getattr(module, "DEFAULT_TIME", 300))
    max_steps = int(getattr(module, "MAX_TERMINATION_STEPS", 256))

    return SpeedEnvBundle(
        env_name=env_module_name,
        module=module,
        env=env,
        base_env_id=str(base_env_id),
        default_time=default_time,
        max_steps=max_steps,
        step_board=step_board,
        observe=observe,
    )


def _set_clock(state, time_budget: int):
    # convention from your speed envs: private field _time_left exists
    if not hasattr(state, "_time_left"):
        raise RuntimeError("Speed env State must have _time_left for this eval script.")
    return state.replace(_time_left=jnp.int32([time_budget, time_budget]))


# ================================================================
# TrainConfig + checkpoint loading
# ================================================================

class TrainConfig(BaseModel):
    env_id: pgx.EnvId = "gardner_chess"
    seed: int = 0
    max_num_iters: int = 400
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    max_num_steps: int = 256
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    eval_interval: int = 5

    class Config:
        extra = "forbid"


class ConfigUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Config":
            return TrainConfig
        return super().find_class(module, name)


def load_checkpoint(path: str):
    with open(path, "rb") as f:
        data = ConfigUnpickler(f).load()
    model = data["model"]               # (params, state)
    cfg: TrainConfig = data["config"]
    env_id = data.get("env_id", cfg.env_id)
    return str(env_id), cfg, model


def discover_checkpoints(root: str, iter_filename: str, pretrained_seed: int = 1) -> Dict[str, str]:
    """
    Robust checkpoint discovery. Tries:
      1) root/nsim_32/<seed>/000600.ckpt         (your "seed dir" layout)
      2) root/nsim_32/<latest_run_dir>/000600.ckpt (older "run dir" layout)

    Returns: { "nsim_32": "/path/to/000600.ckpt", ... }
    """
    ckpts: Dict[str, str] = {}
    if not os.path.isdir(root):
        raise ValueError(f"Checkpoint root '{root}' does not exist")

    for subdir in os.listdir(root):
        if not subdir.startswith("nsim_"):
            continue
        nsim = subdir.split("_", 1)[1]
        base = os.path.join(root, subdir)
        if not os.path.isdir(base):
            continue

        # (1) seed layout
        seed_dir = os.path.join(base, str(pretrained_seed))
        ckpt_path = os.path.join(seed_dir, iter_filename)
        if os.path.isdir(seed_dir) and os.path.exists(ckpt_path):
            ckpts[f"nsim_{nsim}"] = ckpt_path
            continue

        # (2) run-dir layout (pick latest)
        run_dirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        run_dirs.sort()
        for d in reversed(run_dirs):
            cand = os.path.join(base, d, iter_filename)
            if os.path.exists(cand):
                ckpts[f"nsim_{nsim}"] = cand
                break

    return ckpts


def _resolve_flat_az_ckpt(root: str, iter_filename: str) -> Optional[str]:
    """Find an AZ checkpoint sitting directly in ``root``
    (e.g. ``checkpoints/clock/hex/base/base_planner.ckpt``) rather than under the
    ``nsim_*/<seed>/`` training layout that ``discover_checkpoints`` expects."""
    if not os.path.isdir(root):
        return None
    direct = os.path.join(root, iter_filename)
    if os.path.exists(direct):
        return direct
    cands = sorted(f for f in os.listdir(root) if f.endswith(".ckpt"))
    if cands:
        return os.path.join(root, cands[-1])
    return None


def _resolve_flat_gate_ckpt(gate_root: str, gate_iter: Optional[int] = None) -> Optional[str]:
    """Find a gate ``.pkl`` sitting directly in
    ``gate_root`` (e.g. ``checkpoints/clock/hex/gating/gate_001000.pkl``) rather
    than under the ``{env}/pre_{nsim}/opts..._seed{seed}/`` training layout."""
    if not os.path.isdir(gate_root):
        return None
    pkls = [f for f in os.listdir(gate_root) if f.endswith(".pkl")]
    if not pkls:
        return None

    def _iter_of(fn: str) -> int:
        m = re.search(r"(\d+)\.pkl$", fn)
        return int(m.group(1)) if m else -1

    if gate_iter is not None:
        for fn in pkls:
            if _iter_of(fn) == gate_iter:
                return os.path.join(gate_root, fn)
    pkls.sort(key=_iter_of)
    return os.path.join(gate_root, pkls[-1])


# ================================================================
# AZ forward + MCTS
# ================================================================

def build_forward(env, cfg: TrainConfig):
    def forward_fn(x, is_eval: bool = False):
        net = AZNet(
            num_actions=env.num_actions,
            num_channels=cfg.num_channels,
            num_blocks=cfg.num_layers,
            resnet_v2=cfg.resnet_v2,
        )
        policy_out, value_out, intermediate = net(
            x, is_training=not is_eval, test_local_stats=False
        )
        return policy_out, value_out, intermediate

    return hk.without_apply_rng(hk.transform_with_state(forward_fn))


def make_recurrent_fn_speed(forward, step_board_fn, observe_fn):
    """
    Recurrent fn for MCTS that ONLY uses board dynamics (no clocks),
    so searches don't spend time.
    """
    def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state):
        del rng_key
        model_params, model_state = model
        current_player = state.current_player

        state = jax.vmap(step_board_fn)(state, action)
        obs = jax.vmap(observe_fn)(state, state.current_player)
        state = state.replace(observation=obs)

        (logits, value, _intermediate), _ = forward.apply(
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

def allocate_midpeak_discrete_plan(
    time_budget: int,
    sim_options: List[int],
    n_moves: int = 24,
    peak_center: float = 0.35,
    peak_width: float = 0.3,
) -> np.ndarray:
    """
    Precompute discrete midpeak allocation plan.
    Returns (n_moves,) int32 array of sim_option values per P1 move.
    """
    opts = sorted(sim_options)

    # Bell curve weights
    x = np.linspace(0, 1, n_moves)
    weights = np.exp(-0.5 * ((x - peak_center) / peak_width) ** 2)
    weights = np.maximum(weights, 0.15)

    # Ideal continuous budget per move
    ideal = weights / weights.sum() * time_budget

    # Snap down to largest affordable option
    choices = np.array([
        max([s for s in opts if s <= ideal[i]], default=opts[0])
        for i in range(n_moves)
    ], dtype=np.int32)

    # Greedily upgrade moves, prioritizing peak region
    priority = np.argsort(-weights)
    budget_left = time_budget - int(choices.sum())
    changed = True
    while changed and budget_left > 0:
        changed = False
        for i in priority:
            current = int(choices[i])
            upgrades = [s for s in opts if s > current]
            if not upgrades:
                continue
            cost = min(upgrades) - current
            if cost <= budget_left:
                choices[i] = min(upgrades)
                budget_left -= cost
                changed = True

    return choices
def make_select_actions_mcts(forward, recurrent_fn, num_simulations: int):
    @jax.jit
    def select_actions_mcts(model, state, rng_key):
        model_params, model_state = model
        (logits, value, _intermediate), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )

        # mask invalid actions (important!)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(
            state.legal_action_mask,
            logits,
            jnp.finfo(logits.dtype).min,
        )

        root = mctx.RootFnOutput(
            prior_logits=logits,
            value=value,
            embedding=state,
        )
        out = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=int(num_simulations),
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,  # still fine; determinism comes from argmax below
        )

        # Deterministic action: take argmax over the improved policy distribution
        # out.action_weights: (B, num_actions)
        action = jnp.argmax(out.action_weights, axis=-1).astype(jnp.int32)
        return action  # (batch,)
    return select_actions_mcts


# ================================================================
# GateNet (GRU-based multi-class gate)
#
# Maintains a recurrent hidden state across steps within each episode.
# Single-step forward: (obs, time_feat, az_value, az_intermediate, hidden)
#   -> (logits, value, new_hidden)
# ================================================================

class GateNet(hk.Module):
    """
    Processes AZNet intermediate feature map and raw observation with
    conv + global avg/max pool, concatenates both with time_feat (5-dim)
    and AZ value to form a feature vector. This feature vector is fed
    through a GRU cell to maintain temporal context, then a 3-layer MLP
    produces logits (num_options-way) + scalar value.

    The GRU hidden state is carried externally so it can be explicitly
    managed: reset at the start of each game, then threaded through the
    scan carry for the duration of the game.
    """
    def __init__(self, num_options: int, gru_hidden_size: int = 128, name: str = "GateNetV2"):
        super().__init__(name=name)
        self.num_options = num_options
        self.gru_hidden_size = gru_hidden_size

    def _conv_pool(self, x, name_prefix):
        """Conv trunk -> global avg pool + global max pool -> (B, 128)."""
        x = hk.Conv2D(64, kernel_shape=3, padding="SAME", name=f"{name_prefix}_conv1")(x)
        x = jax.nn.relu(x)
        x = hk.Conv2D(64, kernel_shape=3, padding="SAME", name=f"{name_prefix}_conv2")(x)
        x = jax.nn.relu(x)
        avg = jnp.mean(x, axis=(1, 2))  # (B, 64)
        mx  = jnp.max(x, axis=(1, 2))   # (B, 64)
        return jnp.concatenate([avg, mx], axis=-1)  # (B, 128)

    def _extract_features(self, obs, time_feat, az_value, az_intermediate):
        """Extract feature vector from current-step inputs (pre-GRU)."""
        x_az = az_intermediate.astype(jnp.float32)
        z_az = self._conv_pool(x_az, "az") if x_az.ndim == 4 else hk.Flatten()(x_az)

        x_obs = obs.astype(jnp.float32)
        z_obs = self._conv_pool(x_obs, "obs") if x_obs.ndim == 4 else hk.Flatten()(x_obs)

        z = jnp.concatenate([z_az, z_obs], axis=-1)  # (B, 256)

        z = jnp.concatenate(
            [z, time_feat.astype(jnp.float32), az_value.astype(jnp.float32)[..., None]],
            axis=-1,
        )
        return z  # (B, feat_dim)

    def __call__(self, obs, time_feat, az_value, az_intermediate, hidden):
        """
        Single-step forward pass.

        Args:
            obs:              (B, *obs_shape) raw observation
            time_feat:        (B, 5) time features
            az_value:         (B,) AZ value prediction
            az_intermediate:  (B, *inter_shape) AZ intermediate features
            hidden:           (B, gru_hidden_size) GRU hidden state

        Returns:
            logits:     (B, num_options) gate action logits
            value:      (B,) value estimate
            new_hidden: (B, gru_hidden_size) updated GRU hidden state
        """
        z = self._extract_features(obs, time_feat, az_value, az_intermediate)

        gru_cell = hk.GRU(self.gru_hidden_size, name="gate_gru")
        gru_out, new_hidden = gru_cell(z, hidden)

        h = jax.nn.relu(hk.Linear(256)(gru_out))
        h = jax.nn.relu(hk.Linear(256)(h))
        h = jax.nn.relu(hk.Linear(128)(h))

        logits = hk.Linear(self.num_options)(h)  # (B, num_options)
        value  = hk.Linear(1)(h)[..., 0]         # (B,)
        return logits, value, new_hidden


def make_gate_forward(num_options: int, gru_hidden_size: int = 128):
    def gate_forward_fn(obs_batch, time_feat_batch, az_value_batch, az_inter_batch, hidden):
        net = GateNet(num_options=num_options, gru_hidden_size=gru_hidden_size)  # one logit per sim option
        return net(obs_batch, time_feat_batch, az_value_batch, az_inter_batch, hidden)
    return hk.without_apply_rng(hk.transform(gate_forward_fn))


# ================================================================
# Gate checkpoint discovery (GRU layout)
# ================================================================

def find_gate_ckpt_gru_style(
    gate_root: str,
    env_name: str,
    pretrained_nsim: int,
    sim_options: List[int],
    seed: int,
    prefer_best: bool = False,
    gate_iter: Optional[int] = None,
) -> Optional[str]:
    """
    Discovers GRU gate checkpoints saved by gru_gate_og.py:

      gate_root/{env_name}/pre_{pretrained_nsim}/
        opts{sim_options}_gru{hidden_size}_seed{seed}/
          gate_gru_{iter:06d}.pkl  or  gate_gru_best.pkl

    Matches any GRU hidden size via prefix/suffix matching.
    """
    base_root = os.path.join(gate_root, env_name, f"pre_{pretrained_nsim}")
    if not os.path.isdir(base_root):
        return None

    opts_prefix = f"opts{'-'.join(map(str, sim_options))}_gru"
    seed_suffix = f"_seed{seed}"
    matching = [
        d for d in os.listdir(base_root)
        if d.startswith(opts_prefix) and d.endswith(seed_suffix)
        and os.path.isdir(os.path.join(base_root, d))
    ]
    if not matching:
        return None

    base = os.path.join(base_root, matching[0])

    if gate_iter is not None:
        specific = os.path.join(base, f"gate_gru_{gate_iter:06d}.pkl")
        if os.path.exists(specific):
            return specific
        print(f"WARNING: requested gate_iter={gate_iter} not found at {specific}")
        return None

    best = os.path.join(base, "gate_gru_best.pkl")
    if prefer_best and os.path.exists(best):
        return best

    gates = []
    for fn in os.listdir(base):
        m = re.match(r"gate_gru_(\d+)\.pkl$", fn)
        if m:
            gates.append((int(m.group(1)), os.path.join(base, fn)))
    if gates:
        gates.sort()
        return gates[-1][1]

    if os.path.exists(best):
        return best
    return None


def load_gate_ckpt(path: str):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    if "gate_params" in ckpt:
        params = ckpt["gate_params"]
    elif "params" in ckpt:
        params = ckpt["params"]
    else:
        raise ValueError(f"No 'gate_params' or 'params' found in {path}")
    config = ckpt.get("config", {})
    gru_hidden_size = int(ckpt.get("gru_hidden_size", 128))
    return jax.device_put(params), config, ckpt, gru_hidden_size


# ================================================================
# Utility helpers
# ================================================================

def _extract_int_list(x) -> Optional[List[int]]:
    if isinstance(x, (list, tuple)) and all(isinstance(v, (int, np.integer)) for v in x):
        return [int(v) for v in x]
    return None


def _parse_int_list(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


# ================================================================
# Jitted play-many (no-timeout opponent variant)
#
# Opponent rules:
#   - fixed: always choose a specific budget from sim_options (must be present)
#   - random_gate: uniform random choice over sim_options
#   - policy_only: always AZ policy argmax, always time_spent=0
#
# "No-timeout" fallback (BOTH players):
#   - If the side-to-move cannot afford intended_nsim (my_time <= intended_nsim):
#       policy-only action, time_spent=0
#   - Opponent_kind == policy_only still forces P1 policy-only always.
#
#   - p1_timeout_avoided: True on a P1 move iff it hit the "cannot afford" fallback
#     (meaning it WOULD have lost if we enforced timeouts while still attempting that nsim).
# ================================================================

def make_play_many_jitted_multiclass(
    env_speed,
    model_pretrained,                 # (params,state) for AZNet
    forward,
    gate_forward,
    select_mcts_fns: List,            # list of fn per option index
    sim_options: List[int],           # e.g. [2,8,16,32]
    default_time: float,
    max_steps: int,
    step_board_fn,
    observe_fn,
    opponent_kind: str,               # "fixed" | "random_gate" | "policy_only"
    opponent_fixed_nsim: Optional[int] = None,  # required if fixed
    time_budget_init: int = 300,
    force_gate_choice_idx: Optional[int] = None,
    play_random_choice=False,
    use_gate:bool = True,
    use_pmap: bool = False,
    gru_hidden_size: int = 128,
):
    default_time_f32 = jnp.float32(default_time)

    num_options = len(sim_options)
    sim_options_arr = jnp.array(sim_options, dtype=jnp.int32)

    if opponent_kind == "fixed":
        if opponent_fixed_nsim is None:
            raise ValueError("opponent_fixed_nsim must be provided for opponent_kind='fixed'")
        if opponent_fixed_nsim not in sim_options:
            raise ValueError(f"opponent_fixed_nsim={opponent_fixed_nsim} not in sim_options={sim_options}")
        opp_idx = int(sim_options.index(opponent_fixed_nsim))
    elif opponent_kind == "random_gate":
        opp_idx = 0  # unused
    elif opponent_kind == "policy_only":
        opp_idx = 0  # unused
    elif opponent_kind == "proportional":
        opp_idx = 0  # computed dynamically per step
    elif opponent_kind == "midpeak":
        opp_idx = 0  # computed dynamically per step
    else:
        raise ValueError(f"Unknown opponent_kind='{opponent_kind}'")

    feat_params, feat_state = model_pretrained

    def policy_only_action_from_logits(logits_1d, legal_mask_1d):
        masked = jnp.where(legal_mask_1d, logits_1d, jnp.finfo(logits_1d.dtype).min)
        return jnp.argmax(masked).astype(jnp.int32)

    # constant (used inside jit) to distinguish "always policy-only opponent"
    P1_FORCED_POLICY_ONLY = jnp.bool_(opponent_kind == "policy_only")
    USE_GATE = jnp.bool_(use_gate)
    P0_RANDOM_CHOICE = jnp.bool_(play_random_choice)

    # Precompute midpeak plan if needed (captured by closure)
    if opponent_kind == "midpeak":
        _midpeak_plan_np = allocate_midpeak_discrete_plan(
            time_budget=time_budget_init,
            sim_options=sim_options,
            n_moves=25,
            peak_center=0.35,
            peak_width=0.3,
        )
        midpeak_plan_arr = jnp.array(_midpeak_plan_np, dtype=jnp.int32)
    def play_one_game(gate_params, rng_key):
        state0 = env_speed.init(rng_key)
        state0 = _set_clock(state0, time_budget_init)
        gru_hidden0 = jnp.zeros((gru_hidden_size,), dtype=jnp.float32)

        def step_fn(carry, step_idx):
            state, rng, gru_hidden = carry
            rng, rng_gate, rng_opp, rng_mcts = jax.random.split(rng, 4)

            alive = ~(state.terminated | state.truncated)
            cur = state.current_player
            time_before = state.time_left  # (2,)

            def decisions_when_alive(_):
                my_time  = jax.lax.select(cur == 0, time_before[0], time_before[1])
                opp_time = jax.lax.select(cur == 0, time_before[1], time_before[0])

                obs_b = state.observation[None, ...]
                time_norm_b = jnp.array(
                    [[
                        my_time / jnp.maximum(default_time_f32, 1.0),
                        opp_time / jnp.maximum(default_time_f32, 1.0),
                        jnp.log1p(my_time.astype(jnp.float32)),
                        jnp.log1p(opp_time.astype(jnp.float32)),
                        jnp.log1p(default_time_f32),
                    ]],
                    dtype=jnp.float32,
                )

                (az_logits_b, az_value_b, az_inter_b), _st = forward.apply(
                    feat_params, feat_state, obs_b, is_eval=True
                )

                # ---------- GRU gate forward (every alive step, both P0 and P1 turns) ----------
                # The GRU hidden state is always updated to match the training rollout behaviour.
                logits_gate_b, _, new_hidden_b = gate_forward.apply(
                    gate_params, obs_b, time_norm_b, az_value_b, az_inter_b, gru_hidden[None, ...]
                )
                new_hidden = new_hidden_b[0]  # (gru_hidden_size,)

                # ---------- Gate choice for P0 (gate / forced / random) ----------
                # Budget options are only legal while time remains; with no
                # time left the policy-only fallback below handles the move.
                has_time = my_time > 0
                _BIG_NEG = jnp.finfo(jnp.float32).min / 2
                masked_logits = jnp.where(
                    jnp.broadcast_to(has_time, (num_options,)),
                    logits_gate_b[0],
                    _BIG_NEG,
                )

                def _gate_argmax_choice(_):
                    return jnp.argmax(masked_logits).astype(jnp.int32)

                # If gate disabled, return a dummy index
                gate_choice_natural = jax.lax.cond(
                    USE_GATE,
                    _gate_argmax_choice,
                    lambda _: jnp.int32(0),
                    operand=None,
                )

                # Forced overrides everything
                if force_gate_choice_idx is not None:
                    gate_choice_p0 = jnp.int32(force_gate_choice_idx)
                else:
                    # Random-choice overrides natural (gate or dummy)
                    gate_choice_p0 = jax.lax.cond(
                        P0_RANDOM_CHOICE,
                        lambda _: jax.random.randint(rng_gate, (), 0, num_options).astype(jnp.int32),
                        lambda _: gate_choice_natural,
                        operand=None,
                    )

                # ---------- Opponent choice for P1 ----------
                p1_no_affordable = jnp.bool_(False)
                if opponent_kind == "fixed":
                    gate_choice_p1 = jnp.int32(opp_idx)
                elif opponent_kind == "random_gate":
                    gate_choice_p1 = jax.random.randint(rng_opp, (), 0, num_options).astype(jnp.int32)
                elif opponent_kind == "proportional":
                    # Pick largest sim_option P1 can afford (<= my_time)
                    affordable = (sim_options_arr <= my_time.astype(jnp.int32))
                    has_any = affordable.any()
                    best_idx = jnp.int32(num_options - 1) - jnp.argmax(affordable[::-1]).astype(jnp.int32)
                    gate_choice_p1 = jnp.where(has_any, best_idx, jnp.int32(0)).astype(jnp.int32)
                    p1_no_affordable = ~has_any
                elif opponent_kind == "midpeak":
                    # step_idx is global (both players), P1 move number = step_idx // 2
                    p1_move_num = step_idx // 2
                    plan_len = midpeak_plan_arr.shape[0]
                    # Clamp to plan length; if game goes longer, use smallest option
                    in_plan = (p1_move_num < plan_len)
                    planned_nsim = jnp.where(
                        in_plan,
                        midpeak_plan_arr[jnp.minimum(p1_move_num, plan_len - 1)],
                        sim_options_arr[0],  # fallback to smallest
                    )
                    # Find index in sim_options_arr
                    match = (sim_options_arr == planned_nsim)
                    has_any = match.any()
                    gate_choice_p1 = jnp.argmax(match).astype(jnp.int32)
                    # If remaining time can't cover planned_nsim, fall back
                    can_afford = (my_time >= planned_nsim.astype(jnp.float32))
                    p1_no_affordable = (~has_any) | (~can_afford & ~in_plan)
                    # If can't afford, try largest affordable from sim_options
                    affordable = (sim_options_arr <= my_time.astype(jnp.int32))
                    has_any_fallback = affordable.any()
                    fallback_idx = jnp.int32(num_options - 1) - jnp.argmax(affordable[::-1]).astype(jnp.int32)
                    gate_choice_p1 = jnp.where(
                        can_afford & has_any,
                        gate_choice_p1,
                        jnp.where(has_any_fallback, fallback_idx, jnp.int32(0)),
                    )
                    p1_no_affordable = ~(can_afford & has_any) & ~has_any_fallback
                
                else:
                    gate_choice_p1 = jnp.int32(0)  # unused when policy_only

                chosen_idx = jax.lax.select(cur == 0, gate_choice_p0, gate_choice_p1).astype(jnp.int32)

                intended_nsim_p0 = sim_options_arr[gate_choice_p0].astype(jnp.int32)
                intended_nsim_p1 = jax.lax.cond(
                    P1_FORCED_POLICY_ONLY,
                    lambda _: jnp.int32(0),
                    lambda _: sim_options_arr[gate_choice_p1].astype(jnp.int32),
                    operand=None,
                )
                intended_nsim = jax.lax.select(cur == 0, intended_nsim_p0, intended_nsim_p1).astype(jnp.int32)

                # ---------- P1 policy-only fallback rules ----------
                is_p1_turn = (cur == 1)
                p1_cannot_afford = (my_time <= 0) | p1_no_affordable
                p1_policy_only = is_p1_turn & (P1_FORCED_POLICY_ONLY | p1_cannot_afford)

                # timeout-avoided flag: only when NOT forced policy-only opponent
                p1_timeout_avoided = is_p1_turn & (~P1_FORCED_POLICY_ONLY) & p1_cannot_afford

                # ---------- P0 (gate) policy-only fallback rules (NEW) ----------
                is_p0_turn = (cur == 0)
                p0_cannot_afford = is_p0_turn & (my_time <= 0)
                p0_policy_only = p0_cannot_afford
                # Semantics: would have timed out if we forced it to spend intended_nsim_p0
                p0_timeout_avoided = p0_cannot_afford

                # ---------- Choose actual action ----------
                def do_policy_only(_unused):
                    a = policy_only_action_from_logits(az_logits_b[0], state.legal_action_mask)
                    nsim_spent = jnp.int32(0)
                    return a, nsim_spent

                def do_move_with_search(_unused):
                    state_b = jax.tree_util.tree_map(lambda x: x[None, ...], state)
                    branches = []
                    for i in range(num_options):
                        def _branch(_unused2, ii=i):
                            key = jax.random.fold_in(rng_mcts, ii)
                            return select_mcts_fns[ii](model_pretrained, state_b, key)[0].astype(jnp.int32)
                        branches.append(_branch)
                    a = jax.lax.switch(chosen_idx, branches, operand=None)
                    return a, intended_nsim

                policy_only_fallback = p1_policy_only | p0_policy_only
                action, nsim_spent = jax.lax.cond(
                    policy_only_fallback,
                    do_policy_only,
                    do_move_with_search,
                    operand=None,
                )

                return (
                    chosen_idx, intended_nsim, nsim_spent, action,
                    p1_policy_only, p1_timeout_avoided,
                    p0_policy_only, p0_timeout_avoided,
                    step_idx, new_hidden,
                )

            def decisions_when_dead(_):
                return (
                    jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.int32(0),
                    jnp.bool_(False),  # p1_policy_only
                    jnp.bool_(False),  # p1_timeout_avoided
                    jnp.bool_(False),  # p0_policy_only
                    jnp.bool_(False),  # p0_timeout_avoided
                    jnp.int32(-1),     # step_idx
                    gru_hidden,        # keep hidden unchanged when dead
                )

            chosen_idx, intended_nsim, nsim_spent, action, p1_policy_only, p1_timeout_avoided, p0_policy_only, p0_timeout_avoided, step_idx, new_hidden = jax.lax.cond(
                alive, decisions_when_alive, decisions_when_dead, operand=None
            )

            time_spent = nsim_spent  # 0 if dead or policy-only fallback

            def do_step(s):
                return env_speed.step(s, (action, time_spent))

            state_next = jax.lax.cond(alive, do_step, lambda s: s, state)
            done_after = state_next.terminated | state_next.truncated

            out = {
                "player": cur,
                "choice_idx": chosen_idx,
                "intended_nsim": intended_nsim,
                "nsim": nsim_spent,
                "action": action,
                "time_before": time_before,
                "time_after": state_next.time_left,
                "move_mask": alive,
                "done": done_after,

                # policy-only tracking: when did each player first use policy-only fallback?
                "p1_policy_only": p1_policy_only,
                "p1_policy_only_step": jax.lax.select(p1_policy_only, step_idx, jnp.int32(-1)),

                # timeout-avoided tracking (subset of policy-only)
                "p1_timeout_avoided": p1_timeout_avoided,
                "p1_timeout_avoided_step": jax.lax.select(p1_timeout_avoided, step_idx, jnp.int32(-1)),

                # P0 policy-only fallback tracking
                "p0_policy_only": p0_policy_only,
                "p0_policy_only_step": jax.lax.select(p0_policy_only, step_idx, jnp.int32(-1)),
                "p0_timeout_avoided": p0_timeout_avoided,
                "p0_timeout_avoided_step": jax.lax.select(p0_timeout_avoided, step_idx, jnp.int32(-1)),
            }
            return (state_next, rng, new_hidden), out

        (state_final, _, _), traj = jax.lax.scan(
            step_fn,
            (state0, rng_key, gru_hidden0),
            xs=jnp.arange(max_steps, dtype=jnp.int32),
            length=max_steps,
        )
        return traj, state_final

    import functools

    if use_pmap:
        @functools.partial(jax.pmap, in_axes=(None, 0))
        def play_many(gate_params, rng_keys_shard):
            trajs, finals = jax.vmap(play_one_game, in_axes=(None, 0))(gate_params, rng_keys_shard)
            return trajs, finals
    else:
        @jax.jit
        def play_many(gate_params, rng_keys):
            trajs, finals = jax.vmap(play_one_game, in_axes=(None, 0))(gate_params, rng_keys)
            return trajs, finals

    return play_many


# ================================================================
# Aggregation helpers
# ================================================================

def _mean_and_se_np(vals: List[float]) -> Tuple[float, float]:
    if len(vals) == 0:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    a = np.array(vals, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a)))


def _normalized_progress_bin(move_idx: int, total_moves: int, n_bins: int) -> int:
    if total_moves <= 0:
        return 0
    return min(n_bins - 1, (move_idx * n_bins) // total_moves)


def summarize_trajs_multiclass(
    trajs,
    final_states,
    sim_options: List[int],
    num_games: int,
    strategy_n_bins: int = 10,
) -> Dict[str, Any]:
    players_all = np.array(trajs["player"])          # (G,T)
    nsim_all = np.array(trajs["nsim"])               # (G,T)
    action_all = np.array(trajs["action"])           # (G,T)
    move_mask_all = np.array(trajs["move_mask"])     # (G,T)
    done_all = np.array(trajs["done"])               # (G,T)

    # policy-only
    p1_pol_all = np.array(trajs.get("p1_policy_only", np.zeros_like(move_mask_all, dtype=bool)))
    p1_pol_step_all = np.array(trajs.get("p1_policy_only_step", -np.ones_like(players_all, dtype=np.int32)))

    # timeout-avoided (subset of policy-only)
    p1_to_all = np.array(trajs.get("p1_timeout_avoided", np.zeros_like(move_mask_all, dtype=bool)))
    p1_to_step_all = np.array(trajs.get("p1_timeout_avoided_step", -np.ones_like(players_all, dtype=np.int32)))

    # P0 policy-only + timeout-avoided
    p0_pol_all = np.array(trajs.get("p0_policy_only", np.zeros_like(move_mask_all, dtype=bool)))
    p0_pol_step_all = np.array(trajs.get("p0_policy_only_step", -np.ones_like(players_all, dtype=np.int32)))

    p0_to_all = np.array(trajs.get("p0_timeout_avoided", np.zeros_like(move_mask_all, dtype=bool)))
    p0_to_step_all = np.array(trajs.get("p0_timeout_avoided_step", -np.ones_like(players_all, dtype=np.int32)))

    rewards_all = np.array(final_states.rewards, dtype=np.float32)      # (G,2)
    time_left_all = np.array(final_states.time_left, dtype=np.float32) # (G,2)

    total_wins_p0 = 0
    total_wins_p1 = 0
    total_draws = 0

    # Win/loss breakdown by timeout vs not-timeout
    p0_wins_p1_timeout = 0
    p0_wins_non_timeout = 0

    total_losses_p0 = 0
    total_losses_p0_timeout = 0
    total_losses_p0_non_timeout = 0

    # Total-games counters (ALL games including duplicates, for un-deduplicated reporting)
    raw_wins_p0 = 0
    raw_wins_p1 = 0
    raw_draws = 0
    raw_p1_to_l = 0   # P1 timeout-avoided losses (-> benefit P0 in fair clock)
    raw_p1_to_d = 0
    raw_p0_to_w = 0   # P0 timeout-avoided wins (-> penalise P0 in fair clock)
    raw_p0_to_d = 0

    # Gate usage stats (P0 only), counts over sim_options (ignore nsim==0)
    counts_all = {s: 0 for s in sim_options}
    counts_win = {s: 0 for s in sim_options}
    counts_loss = {s: 0 for s in sim_options}
    counts_draw = {s: 0 for s in sim_options}

    # policy-only outcome splits
    p1_policy_only_games = 0
    p1_policy_only_w = 0
    p1_policy_only_l = 0
    p1_policy_only_d = 0
    p1_no_policy_only_games = 0
    p1_no_policy_only_w = 0
    p1_no_policy_only_l = 0
    p1_no_policy_only_d = 0
    p1_policy_only_first_steps: List[int] = []

    # within policy-only games, when P1 wins (== P0 loss), why?
    p1_policy_only_p1wins_total = 0
    p1_policy_only_p1wins_p0_timeout = 0
    p1_policy_only_p1wins_p0_checkmate_else = 0

    # opponent policy-only diagnostics
    p1_policy_only_moves_total = 0
    p1_policy_only_games_anymove = 0
    p1_policy_only_moves_win = 0
    p1_policy_only_moves_loss = 0
    p1_policy_only_moves_draw = 0

    # timeout-avoided (would-timeout-if-enforced) game-level + move-level
    p1_timeout_avoided_games = 0
    p1_timeout_avoided_w = 0
    p1_timeout_avoided_l = 0
    p1_timeout_avoided_d = 0
    p1_timeout_avoided_first_steps: List[int] = []
    p1_timeout_avoided_moves_total = 0
    p1_timeout_avoided_games_anymove = 0
    
    # P0 policy-only + timeout-avoided (game-level + move-level)
    p0_policy_only_games = 0
    p0_policy_only_w = 0
    p0_policy_only_l = 0
    p0_policy_only_d = 0
    p0_policy_only_first_steps: List[int] = []
    p0_policy_only_moves_total = 0
    p0_policy_only_games_anymove = 0

    p0_timeout_avoided_games = 0
    p0_timeout_avoided_w = 0
    p0_timeout_avoided_l = 0
    p0_timeout_avoided_d = 0
    p0_timeout_avoided_first_steps: List[int] = []
    p0_timeout_avoided_moves_total = 0
    p0_timeout_avoided_games_anymove = 0

    # within timeout-avoided games, when P1 wins (== P0 loss), why?
    p1_timeout_avoided_p1wins_total = 0
    p1_timeout_avoided_p1wins_p0_timeout = 0
    p1_timeout_avoided_p1wins_p0_checkmate_else = 0

    all_game_lengths = []
    all_final_times_p0 = []
    all_final_times_p1 = []

    # Unique-game deduplication: track first-seen signature to skip duplicate move sequences
    sig_seen: set = set()
    # Also map sig -> game_idx for the first occurrence (used by visualization)
    sig_to_first_game: Dict[str, int] = {}

    # Strategy: per-player-turn average nsim accumulators (unique games only)
    T_steps = move_mask_all.shape[1]
    max_pturn = T_steps // 2 + 2          # upper bound on per-player turns
    _z  = lambda: np.zeros(max_pturn, dtype=np.float64)
    _zi = lambda: np.zeros(max_pturn, dtype=np.int64)
    p0_nsim_sum = {"all": _z(), "win": _z(), "loss": _z()}
    p0_nsim_cnt = {"all": _zi(), "win": _zi(), "loss": _zi()}
    p1_nsim_sum = {"all": _z(), "win": _z(), "loss": _z()}
    p1_nsim_cnt = {"all": _zi(), "win": _zi(), "loss": _zi()}
    strategy_bin_game_values = {
        (bin_idx, sim): [] for bin_idx in range(strategy_n_bins) for sim in sim_options
    }

    for g in range(num_games):
        players_g = players_all[g]
        nsim_g = nsim_all[g]
        action_g = action_all[g]
        move_mask_g = move_mask_all[g].astype(bool)
        done_g = done_all[g]
        p1pol_g = p1_pol_all[g].astype(bool)
        p1polstep_g = p1_pol_step_all[g].astype(np.int32)

        p1to_g = p1_to_all[g].astype(bool)
        p1tostep_g = p1_to_step_all[g].astype(np.int32)
        p0pol_g = p0_pol_all[g].astype(bool)
        p0polstep_g = p0_pol_step_all[g].astype(np.int32)

        p0to_g = p0_to_all[g].astype(bool)
        p0tostep_g = p0_to_step_all[g].astype(np.int32)

        T = move_mask_g.shape[0]
        if done_g.any():
            last_step = int(np.argmax(done_g))
        else:
            last_step = T - 1

        valid_mask = move_mask_g & (np.arange(T) <= last_step)
        move_indices = np.where(valid_mask)[0]

        # ---- Total-game accounting (ALL games, including duplicates) ----
        r0_raw = float(rewards_all[g, 0])
        r1_raw = float(rewards_all[g, 1])
        p0_tl_raw = float(time_left_all[g, 0])
        p1_tl_raw = float(time_left_all[g, 1])
        if r0_raw > 0:
            raw_wins_p0 += 1
        elif r1_raw > 0:
            raw_wins_p1 += 1
        else:
            raw_draws += 1
        # Fair-clock subsets for total games
        if len(move_indices) > 0:
            p1_to_moves_raw = p1_to_all[g][move_indices].astype(bool)
            p0_to_moves_raw = p0_to_all[g][move_indices].astype(bool)
            if p1_to_moves_raw.any():
                if r1_raw > 0:   # P1 won a game where P1 avoided timeout -> count
                    raw_p1_to_l += 1
                elif r0_raw == 0 and r1_raw == 0:
                    raw_p1_to_d += 1
            if p0_to_moves_raw.any():
                if r0_raw > 0:   # P0 won a game where P0 avoided timeout -> penalise
                    raw_p0_to_w += 1
                elif r0_raw == 0 and r1_raw == 0:
                    raw_p0_to_d += 1

        # Compute signature and skip duplicate move sequences
        players_moves = players_g[move_indices].astype(np.int8, copy=False)
        actions_moves = action_g[move_indices].astype(np.int32, copy=False)
        packed = np.stack([players_moves.astype(np.int32), actions_moves], axis=1) if len(move_indices) > 0 else np.zeros((0, 2), dtype=np.int32)
        sig = hashlib.sha1(packed.tobytes()).hexdigest()
        if sig in sig_seen:
            continue
        sig_seen.add(sig)
        sig_to_first_game[sig] = g

        r0 = float(rewards_all[g, 0])
        r1 = float(rewards_all[g, 1])

        # useful always
        p0_time_left = float(time_left_all[g, 0])
        p1_time_left = float(time_left_all[g, 1])

        if r0 > 0:
            total_wins_p0 += 1
            outcome = "win"
        elif r1 > 0:
            total_wins_p1 += 1
            outcome = "loss"
        else:
            total_draws += 1
            outcome = "draw"

        # timeout vs non-timeout breakdown
        if outcome == "win":
            if p1_time_left <= 0.0:
                p0_wins_p1_timeout += 1
            else:
                p0_wins_non_timeout += 1

        if r0 < 0.0:
            total_losses_p0 += 1
            if p0_time_left <= 0.0:
                total_losses_p0_timeout += 1
            else:
                total_losses_p0_non_timeout += 1

        # policy-only ever? (P1 only)
        if len(move_indices) > 0:
            p1_moves_mask = (players_g[move_indices] == 1)
            p1_pol_moves = p1pol_g[move_indices][p1_moves_mask]
            pol_ever = bool(np.any(p1_pol_moves))
        else:
            pol_ever = False

        if pol_ever:
            p1_policy_only_games += 1
            if outcome == "win":
                p0_res = 1
                p1_policy_only_w += 1
            elif outcome == "loss":
                p1_policy_only_l += 1
            else:
                p1_policy_only_d += 1

            # if P1 wins within policy-only games, why did P0 lose?
            if outcome == "loss":
                p1_policy_only_p1wins_total += 1
                if p0_time_left <= 0.0:
                    p1_policy_only_p1wins_p0_timeout += 1
                else:
                    p1_policy_only_p1wins_p0_checkmate_else += 1

            steps = p1polstep_g[move_indices]
            steps = steps[steps >= 0]
            if steps.size > 0:
                p1_policy_only_first_steps.append(int(steps.min()))
        else:
            p1_no_policy_only_games += 1
            if outcome == "win":
                p1_no_policy_only_w += 1
            elif outcome == "loss":
                p1_no_policy_only_l += 1
            else:
                p1_no_policy_only_d += 1

        # policy-only move totals (P1)
        if len(move_indices) > 0:
            p1_moves_mask = (players_g[move_indices] == 1)
            p1_nsim = nsim_g[move_indices][p1_moves_mask]
            p1_pol = p1pol_g[move_indices][p1_moves_mask]
            p1_policy_only = (p1_nsim == 0) | p1_pol
            c_pol = int(p1_policy_only.sum())
            p1_policy_only_moves_total += c_pol
            if c_pol > 0:
                p1_policy_only_games_anymove += 1
            if outcome == "win":
                p1_policy_only_moves_win += c_pol
            elif outcome == "loss":
                p1_policy_only_moves_loss += c_pol
            else:
                p1_policy_only_moves_draw += c_pol

        # timeout-avoided (would timeout if enforced) ever? (P1 only)
        if len(move_indices) > 0:
            p1_moves_mask = (players_g[move_indices] == 1)
            p1_to_moves = p1to_g[move_indices][p1_moves_mask]
            to_ever = bool(np.any(p1_to_moves))
            c_to = int(p1_to_moves.sum())
        else:
            to_ever = False
            c_to = 0
        # ----------------------------
        # P0 policy-only ever? + move totals
        # ----------------------------
        if len(move_indices) > 0:
            p0_moves_mask = (players_g[move_indices] == 0)

            # policy-only (P0)
            p0_pol_moves = p0pol_g[move_indices][p0_moves_mask]
            p0_pol_ever = bool(np.any(p0_pol_moves))
            c_p0pol = int(p0_pol_moves.sum())

            # timeout-avoided (P0) (subset of policy-only in this harness)
            p0_to_moves = p0to_g[move_indices][p0_moves_mask]
            p0_to_ever = bool(np.any(p0_to_moves))
            c_p0to = int(p0_to_moves.sum())
        else:
            p0_pol_ever = False
            c_p0pol = 0
            p0_to_ever = False
            c_p0to = 0

        # policy-only move totals (P0)
        if c_p0pol > 0:
            p0_policy_only_moves_total += c_p0pol
            p0_policy_only_games_anymove += 1

        if p0_pol_ever:
            p0_policy_only_games += 1
            if outcome == "win":
                p0_policy_only_w += 1
            elif outcome == "loss":
                p0_policy_only_l += 1
            else:
                p0_policy_only_d += 1

            steps = p0polstep_g[move_indices]
            steps = steps[steps >= 0]
            if steps.size > 0:
                p0_policy_only_first_steps.append(int(steps.min()))

        # timeout-avoided move totals (P0)
        if c_p0to > 0:
            p0_timeout_avoided_moves_total += c_p0to
            p0_timeout_avoided_games_anymove += 1

        if p0_to_ever:
            p0_timeout_avoided_games += 1
            if outcome == "win":
                p0_timeout_avoided_w += 1
            elif outcome == "loss":
                p0_timeout_avoided_l += 1
            else:
                p0_timeout_avoided_d += 1

            steps = p0tostep_g[move_indices]
            steps = steps[steps >= 0]
            if steps.size > 0:
                p0_timeout_avoided_first_steps.append(int(steps.min()))

        if c_to > 0:
            p1_timeout_avoided_moves_total += c_to
            p1_timeout_avoided_games_anymove += 1

        if to_ever:
            p1_timeout_avoided_games += 1
            if outcome == "win":
                p1_timeout_avoided_w += 1
            elif outcome == "loss":
                p1_timeout_avoided_l += 1
            else:
                p1_timeout_avoided_d += 1

            # first timeout-avoided step index
            steps = p1tostep_g[move_indices]
            steps = steps[steps >= 0]
            if steps.size > 0:
                p1_timeout_avoided_first_steps.append(int(steps.min()))

            # if P1 wins in timeout-avoided games, why did P0 lose?
            if outcome == "loss":
                p1_timeout_avoided_p1wins_total += 1
                if p0_time_left <= 0.0:
                    p1_timeout_avoided_p1wins_p0_timeout += 1
                else:
                    p1_timeout_avoided_p1wins_p0_checkmate_else += 1

        # Gate usage stats (P0 only), count only sim_options (ignore nsim==0)
        nsim_moves = nsim_g[move_indices]
        players_moves_full = players_g[move_indices]
        p0_mask = (players_moves_full == 0)
        p0_nsims = nsim_moves[p0_mask]

        for s in sim_options:
            c = int((p0_nsims == s).sum())
            counts_all[s] += c
            if outcome == "win":
                counts_win[s] += c
            elif outcome == "loss":
                counts_loss[s] += c
            else:
                counts_draw[s] += c

        # Strategy: accumulate per-player-turn nsim (unique games only)
        p0_turns = [int(t) for t in move_indices if players_g[t] == 0]
        p1_turns = [int(t) for t in move_indices if players_g[t] == 1]
        for k, t in enumerate(p0_turns):
            v = float(nsim_g[t])
            p0_nsim_sum["all"][k] += v;  p0_nsim_cnt["all"][k] += 1
            p0_nsim_sum[outcome][k] += v;  p0_nsim_cnt[outcome][k] += 1
        for k, t in enumerate(p1_turns):
            v = float(nsim_g[t])
            p1_nsim_sum["all"][k] += v;  p1_nsim_cnt["all"][k] += 1
            p1_nsim_sum[outcome][k] += v;  p1_nsim_cnt[outcome][k] += 1

        if p0_turns:
            p0_bin_totals = np.zeros(strategy_n_bins, dtype=np.int64)
            p0_bin_counts = {
                sim: np.zeros(strategy_n_bins, dtype=np.int64) for sim in sim_options
            }
            total_p0_turns = len(p0_turns)
            for pturn_idx, t in enumerate(p0_turns):
                bin_idx = _normalized_progress_bin(pturn_idx, total_p0_turns, strategy_n_bins)
                p0_bin_totals[bin_idx] += 1
                nsim_val = int(nsim_g[t])
                if nsim_val in p0_bin_counts:
                    p0_bin_counts[nsim_val][bin_idx] += 1
            for bin_idx in range(strategy_n_bins):
                denom = int(p0_bin_totals[bin_idx])
                if denom <= 0:
                    continue
                for sim in sim_options:
                    strategy_bin_game_values[(bin_idx, sim)].append(
                        float(p0_bin_counts[sim][bin_idx] / denom)
                    )

        all_game_lengths.append(len(move_indices))
        all_final_times_p0.append(p0_time_left)
        all_final_times_p1.append(p1_time_left)

    def pct_dict(counts: Dict[int, int]) -> Dict[int, float]:
        total = sum(counts.values())
        if total <= 0:
            return {k: 0.0 for k in counts}
        return {k: 100.0 * v / total for k, v in counts.items()}

    def _avg_series(sums, cnts):
        """Mean nsim per player-turn, trimmed after the last turn with any data."""
        out = []
        for k in range(max_pturn):
            if cnts[k] > 0:
                out.append(float(sums[k] / cnts[k]))
            else:
                break
        return out

    strategy_bin_stats: Dict[str, float] = {}
    for bin_idx in range(strategy_n_bins):
        for k_idx, sim in enumerate(sim_options, start=1):
            mean, se = _mean_and_se_np(strategy_bin_game_values[(bin_idx, sim)])
            strategy_bin_stats[f"strategy/bin{bin_idx:02d}_k{k_idx}_mean"] = mean
            strategy_bin_stats[f"strategy/bin{bin_idx:02d}_k{k_idx}_se"] = se

    # All counts are already over unique games only (duplicates skipped above)
    num_unique_games = len(sig_seen)
    expected_score = (total_wins_p0 + 0.5 * total_draws) / max(1, num_unique_games)

    p1_pol_first_mean = float(np.mean(p1_policy_only_first_steps)) if len(p1_policy_only_first_steps) else float("nan")
    p1_to_first_mean = float(np.mean(p1_timeout_avoided_first_steps)) if len(p1_timeout_avoided_first_steps) else float("nan")

    pol_denom = float(p1_policy_only_w + p1_policy_only_d + p1_policy_only_l)
    p1_policy_only_expected_score = float("nan") if pol_denom <= 0 else float((p1_policy_only_w + 0.5 * p1_policy_only_d) / pol_denom)

    to_denom = float(p1_timeout_avoided_w + p1_timeout_avoided_d + p1_timeout_avoided_l)
    p1_timeout_avoided_expected_score = float("nan") if to_denom <= 0 else float((p1_timeout_avoided_w + 0.5 * p1_timeout_avoided_d) / to_denom)

    p0_pol_denom = float(p0_policy_only_w + p0_policy_only_d + p0_policy_only_l)
    p0_policy_only_expected_score = float("nan") if p0_pol_denom <= 0 else float(
        (p0_policy_only_w + 0.5 * p0_policy_only_d) / p0_pol_denom
    )

    p0_to_denom = float(p0_timeout_avoided_w + p0_timeout_avoided_d + p0_timeout_avoided_l)
    p0_timeout_avoided_expected_score = float("nan") if p0_to_denom <= 0 else float(
        (p0_timeout_avoided_w + 0.5 * p0_timeout_avoided_d) / p0_to_denom
    )

    p0_pol_first_mean = float(np.mean(p0_policy_only_first_steps)) if len(p0_policy_only_first_steps) else float("nan")
    p0_to_first_mean = float(np.mean(p0_timeout_avoided_first_steps)) if len(p0_timeout_avoided_first_steps) else float("nan")

    # Unique-game outcome breakdown (same-sequence games always have same outcome)
    unique_total = num_unique_games
    unique_wins = total_wins_p0
    unique_draws = total_draws
    unique_losses = total_wins_p1
    unique_mixed = 0  # impossible: deterministic once action sequence is fixed

    summary = {
        "num_games": int(num_unique_games),  # unique games only
        "total_games_raw": int(num_games),   # total (including duplicates)

        "p0_wins": int(total_wins_p0),
        "p0_losses": int(total_wins_p1),
        "draws": int(total_draws),

        "p0_win_rate": float(total_wins_p0 / max(1, num_unique_games)),
        "p1_win_rate": float(total_wins_p1 / max(1, num_unique_games)),
        "draw_rate": float(total_draws / max(1, num_unique_games)),
        "expected_score": float(expected_score),

        # timeout vs non-timeout (general)
        "p0_wins_p1_timeout": int(p0_wins_p1_timeout),
        "p0_wins_non_timeout": int(p0_wins_non_timeout),

        "p0_losses_total": int(total_losses_p0),
        "p0_losses_timeout": int(total_losses_p0_timeout),
        "p0_losses_non_timeout": int(total_losses_p0_non_timeout),

        # NEW aliases (for plotters that expect checkmate_else naming)
        "p0_wins_checkmate_else": int(p0_wins_non_timeout),
        "p0_losses_checkmate_else": int(total_losses_p0_non_timeout),

        # gate usage (P0 only, sim_options only)
        "p0_nsim_counts_all": counts_all,
        "p0_nsim_pct_all": pct_dict(counts_all),
        "p0_nsim_counts_win": counts_win,
        "p0_nsim_counts_loss": counts_loss,
        "p0_nsim_counts_draw": counts_draw,

        "avg_game_len": float(np.mean(all_game_lengths)) if all_game_lengths else 0.0,
        "avg_final_time_p0": float(np.mean(all_final_times_p0)) if all_final_times_p0 else 0.0,
        "avg_final_time_p1": float(np.mean(all_final_times_p1)) if all_final_times_p1 else 0.0,

        "unique_games_total": int(unique_total),
        "unique_games_wins": int(unique_wins),
        "unique_games_draws": int(unique_draws),
        "unique_games_losses": int(unique_losses),
        "unique_games_mixed": int(unique_mixed),

        # policy-only effects (game-level)
        "p1_policy_only_games": int(p1_policy_only_games),
        "p1_policy_only_w": int(p1_policy_only_w),
        "p1_policy_only_l": int(p1_policy_only_l),
        "p1_policy_only_d": int(p1_policy_only_d),
        "p1_policy_only_expected_score": float(p1_policy_only_expected_score),
        "p1_policy_only_first_step_mean": p1_pol_first_mean,

        # in policy-only games, P1 wins -> why did P0 lose?
        "p1_policy_only_p1wins_total": int(p1_policy_only_p1wins_total),
        "p1_policy_only_p1wins_p0_timeout": int(p1_policy_only_p1wins_p0_timeout),
        "p1_policy_only_p1wins_p0_checkmate_else": int(p1_policy_only_p1wins_p0_checkmate_else),

        "p1_no_policy_only_games": int(p1_no_policy_only_games),
        "p1_no_policy_only_w": int(p1_no_policy_only_w),
        "p1_no_policy_only_l": int(p1_no_policy_only_l),
        "p1_no_policy_only_d": int(p1_no_policy_only_d),

        # policy-only diagnostics (move-level)
        "p1_policy_only_moves_total": int(p1_policy_only_moves_total),
        "p1_policy_only_games_anymove": int(p1_policy_only_games_anymove),
        "p1_policy_only_moves_win": int(p1_policy_only_moves_win),
        "p1_policy_only_moves_loss": int(p1_policy_only_moves_loss),
        "p1_policy_only_moves_draw": int(p1_policy_only_moves_draw),

        # timeout-avoided subset (games where P1 would have lost if timeouts enforced)
        "p1_timeout_avoided_games": int(p1_timeout_avoided_games),
        "p1_timeout_avoided_w": int(p1_timeout_avoided_w),
        "p1_timeout_avoided_l": int(p1_timeout_avoided_l),
        "p1_timeout_avoided_d": int(p1_timeout_avoided_d),
        "p1_timeout_avoided_expected_score": float(p1_timeout_avoided_expected_score),
        "p1_timeout_avoided_first_step_mean": p1_to_first_mean,
        "p1_timeout_avoided_moves_total": int(p1_timeout_avoided_moves_total),
        "p1_timeout_avoided_games_anymove": int(p1_timeout_avoided_games_anymove),

        # within timeout-avoided games, when P1 wins (== P0 loss), why?
        "p1_timeout_avoided_p1wins_total": int(p1_timeout_avoided_p1wins_total),
        "p1_timeout_avoided_p1wins_p0_timeout": int(p1_timeout_avoided_p1wins_p0_timeout),
        "p1_timeout_avoided_p1wins_p0_checkmate_else": int(p1_timeout_avoided_p1wins_p0_checkmate_else),

        # P0 policy-only subset
        "p0_policy_only_games": int(p0_policy_only_games),
        "p0_policy_only_w": int(p0_policy_only_w),
        "p0_policy_only_d": int(p0_policy_only_d),
        "p0_policy_only_l": int(p0_policy_only_l),
        "p0_policy_only_expected_score": float(p0_policy_only_expected_score),
        "p0_policy_only_first_step_mean": float(p0_pol_first_mean),
        "p0_policy_only_moves_total": int(p0_policy_only_moves_total),
        "p0_policy_only_games_anymove": int(p0_policy_only_games_anymove),

        # P0 timeout-avoided subset (P0 would have lost if strict timeouts enforced)
        "p0_timeout_avoided_games": int(p0_timeout_avoided_games),
        "p0_timeout_avoided_w": int(p0_timeout_avoided_w),
        "p0_timeout_avoided_d": int(p0_timeout_avoided_d),
        "p0_timeout_avoided_l": int(p0_timeout_avoided_l),
        "p0_timeout_avoided_expected_score": float(p0_timeout_avoided_expected_score),
        "p0_timeout_avoided_first_step_mean": float(p0_to_first_mean),
        "p0_timeout_avoided_moves_total": int(p0_timeout_avoided_moves_total),
        "p0_timeout_avoided_games_anymove": int(p0_timeout_avoided_games_anymove),

        # Strategy: mean nsim per player-turn-number, split by outcome (unique games)
        "strategy_p0_nsim_all":  _avg_series(p0_nsim_sum["all"],  p0_nsim_cnt["all"]),
        "strategy_p0_nsim_win":  _avg_series(p0_nsim_sum["win"],  p0_nsim_cnt["win"]),
        "strategy_p0_nsim_loss": _avg_series(p0_nsim_sum["loss"], p0_nsim_cnt["loss"]),
        "strategy_p1_nsim_all":  _avg_series(p1_nsim_sum["all"],  p1_nsim_cnt["all"]),
        "strategy_p1_nsim_win":  _avg_series(p1_nsim_sum["win"],  p1_nsim_cnt["win"]),
        "strategy_p1_nsim_loss": _avg_series(p1_nsim_sum["loss"], p1_nsim_cnt["loss"]),
        "strategy_bin_stats": strategy_bin_stats,

        # Total-games stats (ALL games including duplicates)
        "total_raw_wins_p0": int(raw_wins_p0),
        "total_raw_wins_p1": int(raw_wins_p1),
        "total_raw_draws":   int(raw_draws),
        "total_expected_score": float((raw_wins_p0 + 0.5 * raw_draws) / max(1, num_games)),
        "total_fair_clock_expected_score": float(
            (raw_wins_p0 + raw_p1_to_l + raw_p1_to_d - raw_p0_to_w - raw_p0_to_d
             + 0.5 * (raw_draws - raw_p1_to_d - raw_p0_to_d))
            / max(1, num_games)
        ),
    }
    return summary


# ================================================================
# End-of-run printing + plotting helpers
# ================================================================

def _mean_and_se(x: List[float]) -> Tuple[float, float]:
    if not x:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), 0.0
    arr = np.array(x, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(len(arr)))


def print_wld_summary(results: Dict[str, Any], times: List[int], opponent_labels: List[str]) -> None:
    by_time = results.get("by_time", {})
    print("\n\n==================== FINAL W/L/D SUMMARY ====================")
    for T in times:
        tkey = str(T)
        if tkey not in by_time:
            continue
        print(f"\n--- Time T={T} ---")
        for opp_label in opponent_labels:
            total_games = 0
            W = 0
            L = 0
            D = 0
            seed_scores = []

            seeds_dict = by_time[tkey]
            for seed_str, opps_dict in seeds_dict.items():
                if opp_label not in opps_dict:
                    continue
                s = opps_dict[opp_label]
                ng = int(s.get("num_games", 0))
                total_games += ng
                W += int(s.get("p0_wins", 0))
                L += int(s.get("p0_losses", 0))
                D += int(s.get("draws", 0))
                seed_scores.append(float(s.get("expected_score", 0.0)))

            if total_games <= 0:
                print(f"{opp_label:>12} | (no data)")
                continue

            win_pct = 100.0 * W / total_games
            loss_pct = 100.0 * L / total_games
            draw_pct = 100.0 * D / total_games
            exp_score = (W + 0.5 * D) / total_games

            mean_seed, se_seed = _mean_and_se(seed_scores)

            print(
                f"{opp_label:>12} | games={total_games:5d} | "
                f"W/L/D = {W:4d}/{L:4d}/{D:4d} | "
                f"win/draw/loss = {win_pct:5.1f}%/{draw_pct:5.1f}%/{loss_pct:5.1f}% | "
                f"expected={exp_score:0.3f} | "
                f"seed-mean±SE={mean_seed:0.3f}±{se_seed:0.3f} (n_seeds={len(seed_scores)})"
            )


def plot_expected_scores_per_time(
    results: Dict[str, Any],
    times: List[int],
    seeds: List[int],
    output_dir: str,
    include_random_gate_bar_if_present: bool = True,
) -> None:
    by_time = results.get("by_time", {})

    for T in times:
        tkey = str(T)
        if tkey not in by_time:
            continue

        opponent_order = list(results["meta"]["opponents"])
        labels = []
        means = []
        ses = []
        nseed_per_opp = []

        for opp_label in opponent_order:
            seed_scores = []
            for seed in seeds:
                sdict = by_time[tkey].get(str(seed), {})
                if opp_label not in sdict:
                    continue
                summ = sdict[opp_label]
                seed_scores.append(float(summ.get("expected_score", 0.0)))
            if not seed_scores:
                continue
            m, se = _mean_and_se(seed_scores)
            labels.append(opp_label)
            means.append(m)
            ses.append(se)
            nseed_per_opp.append(len(seed_scores))

        if not labels:
            continue

        x = np.arange(len(labels), dtype=np.int32)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(x, means, yerr=ses, capsize=4)
        ax.axhline(0.5, color="red", linestyle=":", linewidth=2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Expected score (win=1, draw=0.5, loss=0)")
        ax.set_title(f"Gate expected score vs opponents | T={T} | mean ± SE over seeds")

        for xi, m, nseeds in zip(x, means, nseed_per_opp):
            ax.text(xi, min(0.98, m + 0.03), f"n={nseeds}", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        out_path = os.path.join(output_dir, f"expected_score_T{T}.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved plot: {out_path}")


# ================================================================
# Opponent parsing
# ================================================================

OpponentSpec = Tuple[str, str, Optional[int]]  # (label, kind, fixed_nsim or None)

def _parse_opponent_specs(s: str) -> List[OpponentSpec]:
    """
    Supports tokens:
      - "random" / "random_gate" -> ("random_gate", "random_gate", None)
      - "0" -> ("always0", "policy_only", 0)
      - integer N>0 -> ("alwaysN", "fixed", N)
    """
    tokens = [t.strip() for t in s.split(",") if t.strip()]
    specs: List[OpponentSpec] = []
    seen = set()
    for tok in tokens:
        low = tok.lower()
        if low in ("random", "rand", "random_gate", "randomgate", "rgate"):
            spec = ("random_gate", "random_gate", None)
        elif low in ("policy", "policy_only", "nomcts", "no_mcts", "0"):
            spec = ("always0", "policy_only", 0)
        elif low in ("proportional", "prop", "greedy"):
            spec = ("proportional", "proportional", None)
        elif low in ("midpeak", "mid", "midp"):
            spec = ("midpeak", "midpeak", None)
        else:
            n = int(tok)
            if n == 0:
                spec = ("always0", "policy_only", 0)
            else:
                spec = (f"always{n}", "fixed", n)

        if spec[0] not in seen:
            specs.append(spec)
            seen.add(spec[0])
    return specs


# ================================================================
# Core: run one eval for one (T, seed, opponent)
# ================================================================

def run_eval_once_multiclass(
    env_bundle: SpeedEnvBundle,
    gate_ckpt_path: Optional[str],
    time_budget: int,
    seed: int,
    num_games: int,
    ckpt_root: str,
    iter_file: str,
    pretrained_nsim: int,
    pretrained_seed: int,
    sim_options: List[int],
    opponent_kind: str,
    opponent_fixed_nsim: Optional[int],
    opponent_label: str,
    force_gate_choice: Optional[int] = None,
    play_random_choice: bool = False,
    use_pmap: bool = False,
) -> Tuple[Dict[str, Any], Any, Any]:
    """Returns (summary, trajs, rng_keys)."""
    chosen_default_time = int(time_budget) if time_budget is not None else int(env_bundle.default_time)

    use_gate = (force_gate_choice is None) and (not play_random_choice)
    print(
        f"\n>>> Eval env={env_bundle.env_name} | P0=GRU-gate vs P1={opponent_label}, "
        f"time={chosen_default_time}, num_games={num_games}, seed={seed}"
    )
    print(f"AZ ckpt root: {ckpt_root} | iter_file: {iter_file} | pretrained_nsim={pretrained_nsim} | pretrained_seed={pretrained_seed}")
    print(f"Gate ckpt: {gate_ckpt_path if gate_ckpt_path is not None else '(DISABLED)'}")
    print(f"SIM_OPTIONS (gate choices): {sim_options}")
    print("Timeout is DISABLED for BOTH sides: if the side-to-move can't afford intended MCTS cost, it goes policy-only with time_spent=0.")

    ckpt_paths = discover_checkpoints(ckpt_root, iter_file, pretrained_seed=pretrained_seed)
    key = f"nsim_{pretrained_nsim}"
    if key in ckpt_paths:
        az_ckpt_path = ckpt_paths[key]
    else:
        # fall back to a checkpoint sitting directly in ckpt_root
        az_ckpt_path = _resolve_flat_az_ckpt(ckpt_root, iter_file)
        if az_ckpt_path is None:
            raise RuntimeError(f"Missing pretrained AZ checkpoint {key} under {ckpt_root}")
        print(f"Using AZ checkpoint: {az_ckpt_path}")

    env_id, cfg, model_pretrained = load_checkpoint(az_ckpt_path)
    expected_env_id = env_bundle.base_env_id
    if env_id != expected_env_id:
        raise RuntimeError(f"AZ env_id mismatch: got {env_id}, expected {expected_env_id}")

    gate_cfg = {}
    gate_ckpt = {}
    gate_params = None
    gate_forward = None
    gru_hidden_size = 128  # default; overridden from checkpoint
    gate_update = None

    if use_gate:
        if not gate_ckpt_path:
            raise RuntimeError("use_gate=True but no gate_ckpt_path was provided/found.")

        gate_params, gate_cfg, gate_ckpt, gru_hidden_size = load_gate_ckpt(gate_ckpt_path)
        gate_update = gate_ckpt.get("update", gate_ckpt.get("best_update", "unknown"))
        print(f"GRU hidden size: {gru_hidden_size} (from checkpoint)")

        sim_from_ckpt = _extract_int_list(gate_cfg.get("sim_options", None))
        if sim_from_ckpt is not None and sim_from_ckpt != sim_options:
            print(f"NOTE: checkpoint sim_options={sim_from_ckpt} differs from CLI sim_options={sim_options}. Using CLI.")

        gate_forward = make_gate_forward(
            num_options=len(sim_options),
            gru_hidden_size=gru_hidden_size,
        )
        print(f"GRU gate forward built (num_options={len(sim_options)}, gru_hidden_size={gru_hidden_size})")
    else:
        print("Gate is DISABLED (forced/random P0 budget selection).")

    env_speed = env_bundle.env
    rng = jax.random.PRNGKey(seed)

    # Build networks
    forward = build_forward(env_speed, cfg)
    recurrent_fn_speed = make_recurrent_fn_speed(forward, env_bundle.step_board, env_bundle.observe)
    select_mcts_fns = [make_select_actions_mcts(forward, recurrent_fn_speed, n) for n in sim_options]

    default_time = float(chosen_default_time)
    max_steps = int(env_bundle.max_steps)
    force_gate_choice_idx = None
    if force_gate_choice is not None:
        if force_gate_choice not in sim_options:
            raise ValueError(f"force_gate_choice={force_gate_choice} not in sim_options={sim_options}")
        force_gate_choice_idx = sim_options.index(force_gate_choice)
        print(f"FORCING gate to always choose budget={force_gate_choice} (index {force_gate_choice_idx})")

    if not use_gate:
        # Dummy gate forward: accepts hidden state, returns zeros + passthrough hidden.
        def _dummy_apply(params, obs_b, time_norm_b, az_value_b, az_inter_b, hidden):
            B = obs_b.shape[0]
            logits = jnp.zeros((B, len(sim_options)), dtype=jnp.float32)
            value = jnp.zeros((B,), dtype=jnp.float32)
            return logits, value, hidden

        class _DummyGateForward:
            apply = staticmethod(_dummy_apply)

        gate_forward = _DummyGateForward()
        gate_params = jnp.int32(0)   # dummy pytree
        gate_ckpt_path = None
        gru_hidden_size = 1  # minimal hidden state for dummy

    play_many = make_play_many_jitted_multiclass(
        env_speed=env_speed,
        model_pretrained=model_pretrained,
        forward=forward,
        gate_forward=gate_forward,
        select_mcts_fns=select_mcts_fns,
        sim_options=sim_options,
        default_time=default_time,
        max_steps=max_steps,
        step_board_fn=env_bundle.step_board,
        observe_fn=env_bundle.observe,
        opponent_kind=opponent_kind,
        opponent_fixed_nsim=opponent_fixed_nsim,
        time_budget_init=chosen_default_time,
        force_gate_choice_idx=force_gate_choice_idx,
        play_random_choice=play_random_choice,
        use_gate=use_gate,
        use_pmap=use_pmap,
        gru_hidden_size=gru_hidden_size,
    )

    rng, key_games = jax.random.split(rng)
    rng_keys = jax.random.split(key_games, num_games)

    n_devices = jax.local_device_count()
    if use_pmap and n_devices > 1:
        print(f"Compiling & running JAX simulation (pmap over {n_devices} devices)...")
        # Pad num_games to be divisible by n_devices
        pad = (-num_games) % n_devices
        if pad > 0:
            extra = jax.random.split(rng_keys[0], pad)
            rng_keys_run = jnp.concatenate([rng_keys, extra], axis=0)
        else:
            rng_keys_run = rng_keys
        # Shape: (n_devices, games_per_device)
        gpd = rng_keys_run.shape[0] // n_devices
        rng_keys_sharded = rng_keys_run.reshape(n_devices, gpd, *rng_keys_run.shape[1:])
        trajs, final_states = play_many(gate_params, rng_keys_sharded)
        # Flatten (n_devices, gpd, ...) -> (num_games + pad, ...)
        trajs = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), trajs)
        final_states = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), final_states)
        # Trim padding
        if pad > 0:
            trajs = jax.tree_util.tree_map(lambda x: x[:num_games], trajs)
            final_states = jax.tree_util.tree_map(lambda x: x[:num_games], final_states)
    else:
        if use_pmap and n_devices == 1:
            print("Compiling & running JAX simulation (pmap requested but only 1 device; using jit)...")
        else:
            print("Compiling & running JAX simulation...")
        trajs, final_states = play_many(gate_params, rng_keys)

    trajs = jax.device_get(trajs)
    final_states = jax.device_get(final_states)

    summary = summarize_trajs_multiclass(
        trajs=trajs,
        final_states=final_states,
        sim_options=sim_options,
        num_games=num_games,
    )

    summary.update(
        {
            "env": env_bundle.env_name,
            "base_env_id": env_bundle.base_env_id,
            "time_budget": int(chosen_default_time),
            "seed": int(seed),
            "opponent": opponent_label,
            "opponent_kind": opponent_kind,
            "opponent_fixed_nsim": (int(opponent_fixed_nsim) if opponent_fixed_nsim is not None else None),
            "gate_ckpt": gate_ckpt_path if use_gate else None,
            "gate_update": gate_update if use_gate else None,
            "p0_budget_mode": ("gate" if use_gate else ("forced" if force_gate_choice is not None else "random")),
        }
    )
    return summary, trajs, rng_keys


# ================================================================
# Trend plots + visualization helpers
# ================================================================

def _log_trend_plots_to_wandb(results, times, seeds, opponent_labels, sim_options, env_bundle, args):
    """Create a summary wandb run with trend line plots (expected_score and fair_clock_score vs time)."""
    import wandb
    by_time = results.get("by_time", {})

    run_uid = os.getenv("RUN_UID", "") or uuid.uuid4().hex[:8]
    wandb.init(
        project=args.wandb_project or f"gate_eval_{env_bundle.base_env_id}",
        entity=(args.wandb_entity or None),
        group=(args.wandb_group or None),
        name=f"eval_summary_{run_uid}",
        tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()] or ["eval", "summary", env_bundle.base_env_id],
        config={
            "mode": "eval_summary",
            "env": args.env,
            "times": times,
            "opponents": opponent_labels,
            "sim_options": sim_options,
        },
        settings=wandb.Settings(_disable_stats=True),
        mode=args.wandb_mode,
    )

    plot_specs = [
        # (summary_key, fallback_fn_or_None, wandb_key, title_suffix, group_label)
        ("expected_score",             None,  "trend/unique/expected_score",             "Unique games",             "Unique games expected score"),
        ("fair_clock_expected_score",  None,  "trend/unique/fair_clock_expected_score",  "Unique games (fair clock)", "Unique games fair clock expected score"),
        ("total_expected_score",       None,  "trend/total/expected_score",              "Total games (incl. dupes)", "Total games expected score"),
        ("total_fair_clock_expected_score", None, "trend/total/fair_clock_expected_score", "Total games (incl. dupes, fair clock)", "Total games fair clock expected score"),
    ]

    for metric_key, _fallback, wandb_key, subtitle, ylabel_label in plot_specs:
        fig, ax = plt.subplots(figsize=(11.0, 4.8))
        x = np.arange(len(times), dtype=np.int32)
        for opp_label in opponent_labels:
            means, ses = [], []
            for T in times:
                vals = []
                for seed in seeds:
                    s = by_time.get(str(T), {}).get(str(seed), {}).get(opp_label, {})
                    if not s:
                        continue
                    v = s.get(metric_key, None)
                    if v is None:
                        # Fallback recomputation for unique fair_clock if key not stored
                        if metric_key == "fair_clock_expected_score":
                            W = s.get("p0_wins", 0); D = s.get("draws", 0); L = s.get("p0_losses", 0)
                            p1_to_l = s.get("p1_timeout_avoided_l", 0); p1_to_d = s.get("p1_timeout_avoided_d", 0)
                            p0_to_w = s.get("p0_timeout_avoided_w", 0); p0_to_d = s.get("p0_timeout_avoided_d", 0)
                            fair_W = W + p1_to_l + p1_to_d - p0_to_w - p0_to_d
                            fair_D = D - p1_to_d - p0_to_d
                            total = max(1, W + D + L)
                            v = (fair_W + 0.5 * fair_D) / total
                        else:
                            continue
                    vals.append(float(v))
                if vals:
                    arr = np.array(vals, dtype=np.float64)
                    m = float(arr.mean())
                    se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
                else:
                    m, se = float("nan"), 0.0
                means.append(m)
                ses.append(se)
            means_arr = np.array(means, dtype=np.float64)
            ses_arr = np.array(ses, dtype=np.float64)
            valid = ~np.isnan(means_arr)
            if valid.any():
                ax.plot(x[valid], means_arr[valid], marker="o", label=opp_label)
                ax.fill_between(x[valid], means_arr[valid] - ses_arr[valid], means_arr[valid] + ses_arr[valid], alpha=0.15)
        ax.axhline(0.5, linestyle=":", linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in times], rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(ylabel_label)
        ax.set_title(f"{ylabel_label} vs time budget ({subtitle}, mean ± SE over seeds)")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=9)
        fig.tight_layout()
        wandb.log({wandb_key: wandb.Image(fig)})
        plt.close(fig)
        print(f"Logged trend plot for {wandb_key} to wandb.")

    wandb.finish()


def _make_hex_frame_annotated(dwg, state, config, annotations: dict):
    """Render one hex board frame and overlay clock/choice text using pure Python floats.

    We deliberately do NOT call _make_hex_dwg_annotated from hex.py because its
    board_w computation uses the module-level `r3 = jnp.sqrt(3)` (a 0-d JAX array),
    which causes svgwrite to raise "iteration over a 0-d array" when building the
    insert tuple.  Instead we call the plain _make_hex_dwg and add text ourselves.
    """
    import math as _math
    from pgx._src.dwg.hex import _make_hex_dwg

    board_g = _make_hex_dwg(dwg, state, config)

    # All arithmetic uses plain Python floats - no JAX scalars.
    GS     = float(config["GRID_SIZE"]) / 2.0          # effective half-size (same as _make_hex_dwg)
    r3     = _math.sqrt(3)
    size   = int(_math.sqrt(int(state._x.board.shape[-1])))
    fsize  = float(max(10, int(GS * 0.9)))

    # _make_hex_dwg translates the whole group by (3*GS, 2*GS).
    # Board cells span y in [2*GS .. (size-1)*GS*1.5 + 2*GS].
    board_h = float((size - 1) * GS * 1.5 + 2.0 * GS)
    text_y1 = board_h + fsize * 1.5                    # first annotation line
    text_y2 = text_y1 + fsize * 1.4                    # second annotation line
    x_left  = float(3.0 * GS)
    board_w = float((size + (size - 1) / 2.0) * GS * r3 + 3.0 * GS)
    x_right = float(board_w - 9.0 * fsize)

    outer_g = dwg.g()
    outer_g.add(board_g)

    p0_time     = annotations.get("p0_time", "")
    p1_time     = annotations.get("p1_time", "")
    move_num    = annotations.get("move_num", "")
    choice_desc = annotations.get("choice_desc", "")

    outer_g.add(dwg.text(
        f"P0: {p0_time}s",
        insert=(x_left, text_y1),
        font_size=f"{int(fsize)}px",
        fill="black",
        font_family="monospace",
    ))
    outer_g.add(dwg.text(
        f"P1: {p1_time}s",
        insert=(x_right, text_y1),
        font_size=f"{int(fsize)}px",
        fill="black",
        font_family="monospace",
    ))
    outer_g.add(dwg.text(
        f"Move {move_num}: {choice_desc}",
        insert=(x_left, text_y2),
        font_size=f"{int(fsize)}px",
        fill="#333333",
        font_family="monospace",
    ))
    return outer_g


def _save_hex_svg_animation_annotated(states, annotations_list, filename, frame_duration_seconds=0.5):
    """Save an animated SVG for a Hex game with per-frame clock/choice annotations."""
    import svgwrite
    import math as _math
    from pgx._src.visualizer import ColorSet

    if not states:
        return

    state0  = states[0]
    GRID_SIZE = 30                                       # full config value; _make_hex_dwg halves it
    size    = int(_math.sqrt(int(state0._x.board.shape[-1])))
    GS      = float(GRID_SIZE) / 2.0                    # effective half-size
    r3      = _math.sqrt(3)
    fsize   = float(max(10, int(GS * 0.9)))

    config = {
        "GRID_SIZE": GRID_SIZE,
        "COLOR_THEME": "light",
        "COLOR_SET": ColorSet(
            p1_color="black", p2_color="white",
            p1_outline="black", p2_outline="black",
            background_color="white", grid_color="black", text_color="lightgray",
        ),
        "SCALE": 1.0,
    }

    board_h = float((size - 1) * GS * 1.5 + 2.0 * GS)
    board_w = float((size + (size - 1) / 2.0) * GS * r3 + 3.0 * GS)
    svg_w   = board_w + GS
    svg_h   = board_h + fsize * 3.5

    frame_groups = []
    dwg = svgwrite.Drawing(str(filename), (svg_w, svg_h))
    for i, (state, ann) in enumerate(zip(states, annotations_list)):
        g = _make_hex_frame_annotated(dwg, state, config, ann)
        g["id"] = f"_fr{i:x}"
        g["class"] = "frame"
        frame_groups.append(g)

    total_seconds = frame_duration_seconds * len(frame_groups)
    style = f".frame{{visibility:hidden; animation:{total_seconds}s linear _k infinite;}}"
    style += f"@keyframes _k{{0%,{100/len(frame_groups)}%{{visibility:visible}}{100/len(frame_groups) * 1.000001}%,100%{{visibility:hidden}}}}"
    for i, group in enumerate(frame_groups):
        dwg.add(group)
        style += f"#{group['id']}{{animation-delay:{i * frame_duration_seconds}s}}"
    dwg.defs.add(svgwrite.container.Style(content=style))
    dwg.saveas(str(filename))


def _visualize_eval_games(env_bundle, trajs, rng_keys, sim_options, opponent_label, T, output_dir, run):
    """Find one win and one loss game from trajs and save annotated SVG animations."""
    import jax

    players_all = np.array(trajs["player"])
    action_all = np.array(trajs["action"])
    move_mask_all = np.array(trajs["move_mask"])
    done_all = np.array(trajs["done"])
    nsim_all = np.array(trajs["nsim"])
    choice_idx_all = np.array(trajs.get("choice_idx", np.zeros_like(players_all, dtype=np.int32)))
    time_before_all = np.array(trajs["time_before"])  # (G, T_steps, 2)

    rewards_all = np.array(trajs.get("rewards", None) if "rewards" in trajs else None)
    # Fall back to final_states rewards - stored separately; we use the sign of actions to determine outcome
    # Actually we need to get outcomes from the done flags + player perspective
    # Use a simple scan: last done step, check if p0 won or p1 won

    # Get outcomes per game from the trajectory
    G = players_all.shape[0]
    game_outcomes = []  # "win", "loss", "draw" from P0 perspective
    for g in range(G):
        done_g = done_all[g]
        if done_g.any():
            last_step = int(np.argmax(done_g))
        else:
            last_step = done_g.shape[0] - 1
        # Use the nsim and player at last step: the player who has no legal moves loses.
        # We don't have rewards stored in trajs directly. Use the "p0_wins/losses" from summarize.
        # Instead, we'll reconstruct by replaying briefly - just need the final state rewards.
        # For simplicity, store outcome as unknown and use the trajs to infer.
        game_outcomes.append(None)

    # We need to replay to know the outcome. Instead, use time_before to track.
    # Better: re-run the game to get rewards. For now, identify by replaying last move.
    # Since we need actual state.rewards, replay each candidate game.

    # Find first win game and first loss game by replaying
    env = env_bundle.env
    target_outcomes = {"win": None, "loss": None}

    for g in range(G):
        if all(v is not None for v in target_outcomes.values()):
            break
        done_g = done_all[g]
        move_mask_g = move_mask_all[g].astype(bool)
        if done_g.any():
            last_step = int(np.argmax(done_g))
        else:
            last_step = done_g.shape[0] - 1
        valid_mask = move_mask_g & (np.arange(done_g.shape[0]) <= last_step)
        move_indices = np.where(valid_mask)[0]

        # Quick replay to get final rewards
        import jax.numpy as jnp_local
        state = env.init(rng_keys[g])
        for t in move_indices:
            action = int(action_all[g, t])
            nsim_spent = int(nsim_all[g, t])
            state = env.step(state, (jnp_local.int32(action), jnp_local.int32(nsim_spent)))
        state = jax.device_get(state)
        r0 = float(np.array(state.rewards)[0])
        r1 = float(np.array(state.rewards)[1])
        if r0 > 0:
            outcome = "win"
        elif r1 > 0:
            outcome = "loss"
        else:
            outcome = "draw"

        if outcome in target_outcomes and target_outcomes[outcome] is None:
            target_outcomes[outcome] = g

    # Render each found game
    for outcome, g in target_outcomes.items():
        if g is None:
            print(f"[viz] No {outcome} game found for opp={opponent_label} T={T}")
            continue
        svg_path = os.path.join(output_dir, f"viz_{opponent_label}_T{T}_{outcome}.svg")
        _replay_and_save_hex_svg(
            env=env,
            rng_key=rng_keys[g],
            action_seq=action_all[g],
            nsim_seq=nsim_all[g],
            player_seq=players_all[g],
            move_mask=move_mask_all[g].astype(bool),
            done_seq=done_all[g],
            time_before=time_before_all[g],
            sim_options=sim_options,
            opponent_label=opponent_label,
            outcome=outcome,
            svg_path=svg_path,
        )
        print(f"[viz] Saved {outcome} game: {svg_path}")
        if run is not None:
            try:
                import wandb
                with open(svg_path, "r") as f:
                    svg_content = f.read()
                wandb.log({f"viz/{opponent_label}_T{T}_{outcome}": wandb.Html(f"<div>{svg_content}</div>")})
            except Exception as e:
                print(f"[viz] Failed to log SVG to wandb: {e}")


def _replay_and_save_hex_svg(
    env, rng_key, action_seq, nsim_seq, player_seq, move_mask, done_seq,
    time_before, sim_options, opponent_label, outcome, svg_path
):
    """Replay a game from rng_key and save an annotated SVG animation."""
    import jax.numpy as jnp_local

    if done_seq.any():
        last_step = int(np.argmax(done_seq))
    else:
        last_step = done_seq.shape[0] - 1
    valid_mask = move_mask & (np.arange(done_seq.shape[0]) <= last_step)
    move_indices = np.where(valid_mask)[0]

    state = env.init(rng_key)
    states = [state]
    annotations = [{
        "p0_time": int(np.array(state.time_left)[0]),
        "p1_time": int(np.array(state.time_left)[1]),
        "move_num": 0,
        "choice_desc": f"Start | outcome: {outcome}",
    }]

    for t in move_indices:
        player = int(player_seq[t])
        action = int(action_seq[t])
        nsim_spent = int(nsim_seq[t])

        if nsim_spent == 0:
            choice_desc = f"P{player}: policy-only (fallback)"
        else:
            if player == 0:
                choice_desc = f"Gate: sim={nsim_spent}"
            else:
                choice_desc = f"{opponent_label}: sim={nsim_spent}"

        state = env.step(state, (jnp_local.int32(action), jnp_local.int32(nsim_spent)))
        t_left = np.array(state.time_left)
        annotations.append({
            "p0_time": int(t_left[0]),
            "p1_time": int(t_left[1]),
            "move_num": int(t) + 1,
            "choice_desc": choice_desc,
        })
        states.append(state)

    _save_hex_svg_animation_annotated(states, annotations, svg_path)


# ================================================================
# Main: sweep times x seeds x opponents
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained multi-class gate policy (P0) against opponents (P1) "
            "across time budgets and seeds.\n"
            "NO-TIMEOUT VARIANT: if the side-to-move can't afford intended MCTS cost, it switches to policy-only "
            "(AZ policy argmax) with time_spent=0.\n"
        )
    )

    parser.add_argument("--env", type=str, default="speed_gardner_chess",
                        help="Speed env module name, e.g. speed_gardner_chess or speed_hex")
    parser.add_argument("--hex_size", type=int, default=11, help="Hex board size (only used for speed_hex-style envs).")

    parser.add_argument("--gate_root", type=str, default=GATE_ROOT_DEFAULT,
                        help="Root directory for GRU gate checkpoints (e.g. .../gru_og)")
    parser.add_argument("--gate_ckpt", type=str, default="",
                        help="Direct path to a gate .pkl checkpoint (overrides auto-discovery)")
    parser.add_argument("--pretrained_nsim", type=int, default=32)
    parser.add_argument("--pretrained_seed", type=int, default=1)
    
    parser.add_argument("--times", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", type=str, required=True)

    parser.add_argument("--sim_options", type=str, default="2,8,16,32")
    parser.add_argument("--opponents", type=str, default="0,2,8,16,32,random")

    parser.add_argument("--num_games", type=int, default=400)

    parser.add_argument("--ckpt_root", type=str, default=CKPT_ROOT_DEFAULT)
    parser.add_argument("--iter_file", type=str, default=ITER_FILE_DEFAULT)

    parser.add_argument("--output_dir", type=str, default="gate_eval_multiclass_results_no_timeout_cf")
    parser.add_argument("--prefer_best", action="store_true")
    parser.add_argument("--gate_iter", type=int, default=None,
                    help="Specific gate checkpoint iteration (e.g. 100 for gate_000100.pkl). If None, use latest.")
    parser.add_argument("--force_gate_choice", type=int, default=None,
                    help="Force gate to always choose this budget (must be in sim_options). If None, use actual gate.")
    parser.add_argument("--play_random_choice", action="store_true", help="Have the gating policy (P0) choose a random sim_option each move. Useful for debugging.")
    parser.add_argument("--pmap", action="store_true", help="Use jax.pmap to run games in parallel across all local devices.")

    # Visualization
    parser.add_argument("--visualize_opponents", type=str, default="",
                        help="Comma-separated opponent labels to visualize (e.g. 'always32,midpeak'). Must be a subset of --opponents.")
    parser.add_argument("--visualize_times", type=str, default="",
                        help="Comma-separated time budgets to visualize (e.g. '600,900,1200'). Must be a subset of --times.")
    parser.add_argument("--visualize_dir", type=str, default="viz_output",
                        help="Output directory for SVG game visualizations.")

    # WandB
    parser.add_argument("--wandb", action="store_true", default=True)
    parser.add_argument("--wandb_project", type=str, default="gating_policy_eval")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument("--wandb_tags", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument(
        "--strategy_plot_opponent",
        type=str,
        default="",
        help="Opponent label whose strategy/binXX_kY metrics should also be logged without an opponent prefix for plot_strategy.py. Defaults to always0 if present, else the first opponent.",
    )

    args = parser.parse_args()

    # Load env bundle
    env_bundle = load_speed_env(args.env, hex_size=args.hex_size if "hex" in args.env else None)
    print(f"[eval] Loaded env module={env_bundle.env_name} base_env_id={env_bundle.base_env_id} default_time={env_bundle.default_time} max_steps={env_bundle.max_steps}")

    times = sorted([int(t) for t in args.times])
    seeds = _parse_int_list(args.seeds)
    sim_options = [int(x) for x in _parse_int_list(args.sim_options)]
    opponent_specs = _parse_opponent_specs(args.opponents)
    opponent_labels = [spec[0] for spec in opponent_specs]
    if args.strategy_plot_opponent:
        strategy_plot_opponent = args.strategy_plot_opponent
    elif "always0" in opponent_labels:
        strategy_plot_opponent = "always0"
    else:
        strategy_plot_opponent = opponent_labels[0]
    if strategy_plot_opponent not in opponent_labels:
        raise ValueError(
            f"--strategy_plot_opponent={strategy_plot_opponent} is not in opponents={opponent_labels}"
        )

    # Parse visualization targets
    viz_opponents = set(t.strip() for t in args.visualize_opponents.split(",") if t.strip())
    viz_times = set(int(t.strip()) for t in args.visualize_times.split(",") if t.strip())
    do_viz = bool(viz_opponents) and bool(viz_times)
    if do_viz:
        os.makedirs(args.visualize_dir, exist_ok=True)

    results: Dict[str, Any] = {
        "meta": {
            "env": args.env,
            "hex_size": int(args.hex_size),
            "base_env_id": env_bundle.base_env_id,
            "times": times,
            "seeds": seeds,
            "sim_options": sim_options,
            "opponents": opponent_labels,
            "num_games": int(args.num_games),
            "pretrained_nsim": int(args.pretrained_nsim),
            "pretrained_seed": int(args.pretrained_seed),
            "ckpt_root": args.ckpt_root,
            "iter_file": args.iter_file,
            "gate_root": args.gate_root,
            "gate_ckpt": args.gate_ckpt or None,
            "arch": "gru",
            "variant": "opponent_timeout_disabled_policy_only_fallback",
            "tracks_policy_only_subset_wdl": True,
            "tracks_timeout_avoided_subset": True,
            "tracks_unique_games_only": True,
            "strategy_plot_opponent": strategy_plot_opponent,
        },
        "by_time": {},
    }
    if args.force_gate_choice is not None and args.play_random_choice:
        raise ValueError("--force_gate_choice and --play_random_choice are mutually exclusive.")


    for T in times:
        results["by_time"][str(T)] = {}
        for seed in seeds:
            results["by_time"][str(T)][str(seed)] = {}

            run = None
            if args.wandb and args.wandb_mode != "disabled":
                import wandb

                project = args.wandb_project or f"gate_eval_no_timeout_cf_{env_bundle.base_env_id}"
                tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()] or [
                    "eval", "gate", "multiclass", "no_timeout", "cf", env_bundle.base_env_id
                ]

                run_uid = os.getenv("RUN_UID", "") or uuid.uuid4().hex[:8]
                run_name = (
                    f"eval_universal{env_bundle.base_env_id}_"
                    f"T{T}_pre{args.pretrained_nsim}_"
                    f"opts{'-'.join(map(str, sim_options))}_"
                    f"seed{seed}"
                )

                wandb.init(
                    project=project,
                    entity=(args.wandb_entity or None),
                    group=(args.wandb_group or None),
                    name=run_name,
                    tags=tags,
                    config={
                        "mode": "eval",
                        "env": args.env,
                        "base_env_id": env_bundle.base_env_id,
                        "time_budget": int(T),
                        "seed": int(seed),
                        "num_games": int(args.num_games),
                        "pretrained_nsim": int(args.pretrained_nsim),
                        "pretrained_seed": int(args.pretrained_seed),
                        "sim_options": sim_options,
                        "opponents": opponent_labels,
                        "ckpt_root": args.ckpt_root,
                        "iter_file": args.iter_file,
                        "gate_root": args.gate_root,
                        "gate_ckpt": args.gate_ckpt or None,
                        "arch": "gru",
                        "variant": "opponent_timeout_disabled_policy_only_fallback",
                        "strategy_plot_opponent": strategy_plot_opponent,
                    },
                    settings=wandb.Settings(_disable_stats=True),
                    mode=args.wandb_mode,
                )
                run = wandb.run

            need_gate = (not args.play_random_choice) and (args.force_gate_choice is None)
            if not need_gate:
                gate_ckpt = None
            else:
                # pick gate ckpt for this (T, seed)
                if args.gate_ckpt:
                    gate_ckpt = args.gate_ckpt
                else:
                    
                    gate_ckpt = find_gate_ckpt_gru_style(
                        gate_root=args.gate_root,
                        env_name=args.env,
                        pretrained_nsim=args.pretrained_nsim,
                        sim_options=sim_options,
                        seed=seed,
                        prefer_best=args.prefer_best,
                        gate_iter=args.gate_iter,
                    )
                    if gate_ckpt is None:
                        # fall back to a gate .pkl sitting directly in gate_root
                        gate_ckpt = _resolve_flat_gate_ckpt(args.gate_root, args.gate_iter)
                        if gate_ckpt is not None:
                            print(f"Using gate checkpoint: {gate_ckpt}")
                    if gate_ckpt is None:
                        print(f"\nWARNING: No gate ckpt found for T={T}, seed={seed}. Skipping.")
                        if run is not None:
                            import wandb
                            wandb.finish()
                        continue

            summaries_by_opp: Dict[str, Dict[str, Any]] = {}

            for opp_label, opp_kind, opp_nsim in opponent_specs:
                print("\n------------------------------------------------------------")
                print(f"Running: env={args.env} | T={T}, seed={seed}, opponent={opp_label}")
                print("------------------------------------------------------------")

                summary, trajs, rng_keys = run_eval_once_multiclass(
                    env_bundle=env_bundle,
                    gate_ckpt_path=gate_ckpt,
                    time_budget=T,
                    seed=seed,
                    num_games=args.num_games,
                    ckpt_root=args.ckpt_root,
                    iter_file=args.iter_file,
                    pretrained_nsim=args.pretrained_nsim,
                    pretrained_seed=seed,
                    sim_options=sim_options,
                    opponent_kind=opp_kind,
                    opponent_fixed_nsim=opp_nsim,
                    opponent_label=opp_label,
                    force_gate_choice=args.force_gate_choice,
                    play_random_choice=args.play_random_choice,
                    use_pmap=args.pmap,
                )
                results["by_time"][str(T)][str(seed)][opp_label] = summary
                summaries_by_opp[opp_label] = summary

                # Visualization: render one win + one loss for specified (opp, T) pairs
                if do_viz and opp_label in viz_opponents and T in viz_times:
                    _visualize_eval_games(
                        env_bundle=env_bundle,
                        trajs=trajs,
                        rng_keys=rng_keys,
                        sim_options=sim_options,
                        opponent_label=opp_label,
                        T=T,
                        output_dir=args.visualize_dir,
                        run=run,
                    )

                if run is not None:
                    import wandb

                    W = summary["p0_wins"]
                    D = summary["draws"]
                    L = summary["p0_losses"]
                    total = max(1, W + D + L)

                    # Fair-clock score: reassign games where P1 or P0 used timeout-avoided fallback
                    p1_to_l = summary.get("p1_timeout_avoided_l", 0)
                    p1_to_d = summary.get("p1_timeout_avoided_d", 0)
                    p0_to_w = summary.get("p0_timeout_avoided_w", 0)
                    p0_to_d = summary.get("p0_timeout_avoided_d", 0)
                    fair_W = W + p1_to_l + p1_to_d - p0_to_w - p0_to_d
                    fair_D = D - p1_to_d - p0_to_d
                    fair_expected_score = (fair_W + 0.5 * fair_D) / total

                    log_dict = {
                        f"{opp_label}/num_games": summary["num_games"],
                        f"{opp_label}/p0_wins": W,
                        f"{opp_label}/p0_losses": L,
                        f"{opp_label}/draws": D,

                        f"{opp_label}/p0_win_rate": summary["p0_win_rate"],
                        f"{opp_label}/p1_win_rate": summary["p1_win_rate"],
                        f"{opp_label}/draw_rate": summary["draw_rate"],
                        f"{opp_label}/expected_score": summary["expected_score"],
                        f"{opp_label}/fair_clock_expected_score": float(fair_expected_score),

                        # timeout vs non-timeout
                        f"{opp_label}/p0_wins_p1_timeout": summary.get("p0_wins_p1_timeout", 0),
                        f"{opp_label}/p0_wins_non_timeout": summary.get("p0_wins_non_timeout", 0),
                        f"{opp_label}/p0_losses_timeout": summary.get("p0_losses_timeout", 0),
                        f"{opp_label}/p0_losses_checkmate_else": summary.get("p0_losses_non_timeout", 0),

                        # policy-only fallback step tracking
                        f"{opp_label}/p0_policy_only_first_step_mean": summary.get("p0_policy_only_first_step_mean", float("nan")),
                        f"{opp_label}/p1_policy_only_first_step_mean": summary.get("p1_policy_only_first_step_mean", float("nan")),
                        f"{opp_label}/p1_timeout_avoided_first_step_mean": summary.get("p1_timeout_avoided_first_step_mean", float("nan")),
                        f"{opp_label}/p0_timeout_avoided_first_step_mean": summary.get("p0_timeout_avoided_first_step_mean", float("nan")),

                        # P0 policy-only / timeout-avoided
                        f"{opp_label}/p0_policy_only_games": summary.get("p0_policy_only_games", 0),
                        f"{opp_label}/p0_policy_only_w": summary.get("p0_policy_only_w", 0),
                        f"{opp_label}/p0_policy_only_d": summary.get("p0_policy_only_d", 0),
                        f"{opp_label}/p0_policy_only_l": summary.get("p0_policy_only_l", 0),
                        f"{opp_label}/p0_policy_only_expected_score": summary.get("p0_policy_only_expected_score", float("nan")),
                        f"{opp_label}/p0_timeout_avoided_games": summary.get("p0_timeout_avoided_games", 0),
                        f"{opp_label}/p0_timeout_avoided_expected_score": summary.get("p0_timeout_avoided_expected_score", float("nan")),

                        # P1 policy-only
                        f"{opp_label}/p1_policy_only_games": summary.get("p1_policy_only_games", 0),
                        f"{opp_label}/p1_policy_only_w": summary.get("p1_policy_only_w", 0),
                        f"{opp_label}/p1_policy_only_d": summary.get("p1_policy_only_d", 0),
                        f"{opp_label}/p1_policy_only_l": summary.get("p1_policy_only_l", 0),
                        f"{opp_label}/p1_policy_only_expected_score": summary.get("p1_policy_only_expected_score", float("nan")),
                        f"{opp_label}/p1_timeout_avoided_games": summary.get("p1_timeout_avoided_games", 0),
                        f"{opp_label}/p1_timeout_avoided_w": summary.get("p1_timeout_avoided_w", 0),
                        f"{opp_label}/p1_timeout_avoided_d": summary.get("p1_timeout_avoided_d", 0),
                        f"{opp_label}/p1_timeout_avoided_l": summary.get("p1_timeout_avoided_l", 0),
                        f"{opp_label}/p1_timeout_avoided_expected_score": summary.get("p1_timeout_avoided_expected_score", float("nan")),
                    }

                    # Gate choice distribution (P0 moves)
                    pct = summary.get("p0_nsim_pct_all", {})
                    cnt = summary.get("p0_nsim_counts_all", {})
                    for sopt in sim_options:
                        log_dict[f"{opp_label}/gate_choice_pct_nsim{sopt}"] = float(pct.get(sopt, 0.0))
                        log_dict[f"{opp_label}/gate_choice_count_nsim{sopt}"] = int(cnt.get(sopt, 0))
                    strategy_bin_stats = summary.get("strategy_bin_stats", {})
                    for key, value in strategy_bin_stats.items():
                        log_dict[f"{opp_label}/{key}"] = float(value)
                    if opp_label == strategy_plot_opponent:
                        for key, value in strategy_bin_stats.items():
                            log_dict[key] = float(value)
                        log_dict["strategy/source_opponent"] = opp_label

                    # Unique-game dedup stats
                    log_dict[f"{opp_label}/total_games_raw"]   = summary.get("total_games_raw", 0)
                    log_dict[f"{opp_label}/unique_wins"]        = summary.get("unique_games_wins", 0)
                    log_dict[f"{opp_label}/unique_draws"]       = summary.get("unique_games_draws", 0)
                    log_dict[f"{opp_label}/unique_losses"]      = summary.get("unique_games_losses", 0)
                    log_dict[f"{opp_label}/avg_game_len"]       = summary.get("avg_game_len", 0.0)
                    log_dict[f"{opp_label}/avg_final_time_p0"]  = summary.get("avg_final_time_p0", 0.0)
                    log_dict[f"{opp_label}/avg_final_time_p1"]  = summary.get("avg_final_time_p1", 0.0)

                    # Strategy per turn: mean nsim for P0 and P1 at each player-turn-number
                    try:
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

                        panels = [
                            ("All games",  "strategy_p0_nsim_all",  "strategy_p1_nsim_all"),
                            ("Win games",  "strategy_p0_nsim_win",  "strategy_p1_nsim_win"),
                            ("Loss games", "strategy_p0_nsim_loss", "strategy_p1_nsim_loss"),
                        ]
                        y_min = min(sim_options)
                        y_max = max(sim_options)
                        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
                        for ax, (label, p0_key, p1_key) in zip(axes, panels):
                            p0_d = summary.get(p0_key, [])
                            p1_d = summary.get(p1_key, [])
                            if p0_d:
                                ax.plot(range(1, len(p0_d) + 1), p0_d, label="P0 (gate)", color="tab:blue")
                            if p1_d:
                                ax.plot(range(1, len(p1_d) + 1), p1_d, label=f"P1 ({opp_label})", color="tab:orange")
                            ax.set_xlim(1, 70)
                            ax.set_ylim(y_min, y_max)
                            ax.set_xlabel("Player turn #")
                            ax.set_ylabel("Avg nsim")
                            ax.set_title(label)
                            ax.legend(fontsize=8)
                        fig.suptitle(f"Strategy per turn | {opp_label} T={T} seed={seed}")
                        plt.tight_layout()
                        log_dict[f"{opp_label}/strategy_per_turn"] = wandb.Image(fig)
                        plt.close(fig)
                    except Exception as _e:
                        print(f"[strategy plot] {opp_label} T={T}: {_e}")

                    wandb.log(log_dict)

            if run is not None:
                import wandb
                wandb.finish()

    print_wld_summary(results, times=times, opponent_labels=opponent_labels)

    # Log trend plots (expected_score and fair_clock_score vs time) to a summary wandb run
    if args.wandb and args.wandb_mode != "disabled":
        _log_trend_plots_to_wandb(
            results=results,
            times=times,
            seeds=seeds,
            opponent_labels=opponent_labels,
            sim_options=sim_options,
            env_bundle=env_bundle,
            args=args,
        )


if __name__ == "__main__":
    main()
