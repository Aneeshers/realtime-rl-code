#!/usr/bin/env python3
"""
gating_gru_go.py

Single-file PPO training script for a **gate policy** (choose MCTS simulation budget per move)
for speed-go 9x9.

This fork keeps the GRU/PPO/gating stack from gating_gru.py, but is set up
for pure self-play training against a frozen Go 9x9 AlphaZero base model.
"""

import os
import re
import json
import pickle
import uuid
import argparse
import importlib
from dataclasses import dataclass
from typing import Dict, Any, List, Sequence, Tuple, Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax
import pgx
from pydantic import BaseModel
from tqdm import trange
import wandb

from network_intermediate import AZNet  # AZNet returns (policy, value, intermediate)


# ================================================================
# 0) CLI + dynamic env loading
# ================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--env",
        type=str,
        default=os.getenv("SPEED_ENV", "speed_go"),
        help="Speed env module name, e.g. speed_go or speed_hex",
    )
    p.add_argument(
        "--env_kwargs",
        type=str,
        default=os.getenv("ENV_KWARGS", ""),
        help='Optional JSON dict of env kwargs, e.g. \'{"size":11, "default_time":900}\'',
    )
    p.add_argument("--seed", type=int, required=True, help="Seed for both pretrained ckpt and training")
    return p.parse_args()


def _parse_env_kwargs(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError("ENV_KWARGS must decode to a JSON object/dict")
        return obj
    except json.JSONDecodeError as e:
        raise ValueError(
            f"ENV_KWARGS must be JSON (parse error: {e}). Example: "
            '\'{"size": 11, "default_time": 900}\''
        )


def _parse_int_list_envvar(name: str, default: str, min_len: int = 1) -> List[int]:
    raw = os.getenv(name, default)
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    out = [int(p) for p in parts]
    if len(out) < min_len:
        raise ValueError(f"{name} must contain at least {min_len} comma-separated ints (got {raw!r})")
    if any(v < 0 for v in out):
        raise ValueError(f"All {name} values must be non-negative integers (got {out})")
    return out


def load_speed_env_module(env_module_name: str) -> Tuple[Callable[..., Any], Callable, Callable]:
    """
    Returns:
      make_env(**kwargs) -> env instance
      step_board_fn(state, action)
      observe_fn(state, player_id)
    """
    mod = importlib.import_module(env_module_name)

    if not hasattr(mod, "_step_board") or not hasattr(mod, "_observe"):
        raise ValueError(f"{env_module_name} must export _step_board and _observe")

    step_board_fn = getattr(mod, "_step_board")
    observe_fn = getattr(mod, "_observe")

    if hasattr(mod, "make_env"):
        make_env = getattr(mod, "make_env")
        return make_env, step_board_fn, observe_fn

    if hasattr(mod, "ENV_CLS"):
        EnvCls = getattr(mod, "ENV_CLS")
    elif hasattr(mod, "GardnerChess"):
        EnvCls = getattr(mod, "GardnerChess")
    elif hasattr(mod, "Hex"):
        EnvCls = getattr(mod, "Hex")
    elif hasattr(mod, "Env"):
        EnvCls = getattr(mod, "Env")
    else:
        raise ValueError(
            f"{env_module_name} must export make_env OR ENV_CLS OR a class named GardnerChess/Hex/Env"
        )

    def make_env(**kwargs):
        return EnvCls(**kwargs)

    return make_env, step_board_fn, observe_fn


# ================================================================
# 1) Hyperparams / envvars
# ================================================================

CKPT_ROOT = os.getenv("CKPT_ROOT", "./checkpoints/clock/go/base")
ITER_FILE = os.getenv("ITER_FILE", "000000.ckpt")
PRETRAINED_NSIM = int(os.getenv("PRETRAINED_NSIM", "16"))

SIM_OPTIONS = _parse_int_list_envvar("SIM_OPTIONS", "16,32,64,96", min_len=2)
NUM_OPTIONS = len(SIM_OPTIONS)

NUM_UPDATES = int(os.getenv("NUM_UPDATES", "500"))
ROLLOUT_STEPS = int(os.getenv("ROLLOUT_STEPS", "32768"))

GAMMA = float(os.getenv("GAMMA", "0.99"))
LAMBDA = float(os.getenv("LAMBDA", "0.95"))

PPO_EPOCHS = int(os.getenv("PPO_EPOCHS", "4"))
PPO_CLIP_EPS = float(os.getenv("PPO_CLIP_EPS", "0.2"))
PPO_LR = float(os.getenv("PPO_LR", "3e-4"))
PPO_VF_COEF = float(os.getenv("PPO_VF_COEF", "0.5"))
PPO_ENT_COEF = float(os.getenv("PPO_ENT_COEF", "0.01"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "512"))
CLIP_GRAD_NORM = float(os.getenv("CLIP_GRAD_NORM", "1.0"))

TRAJ_LOG_INTERVAL = int(os.getenv("TRAJ_LOG_INTERVAL", "50"))

GATE_CKPT_ROOT_ENV = os.getenv("GATE_CKPT_ROOT", "./checkpoints/clock/go/gating")

GATE_SETUP = int(os.getenv("GATE_SETUP", "2"))
assert GATE_SETUP in (1, 2)

TIME_FEAT_DIM = 5

# GRU history
GRU_HIDDEN_DIM = int(os.getenv("GRU_HIDDEN_DIM", "128"))
SEQ_LEN = int(os.getenv("SEQ_LEN", "64"))  # chunk length for recurrent PPO

NUM_ENVS = int(os.getenv("NUM_ENVS", "256"))
assert ROLLOUT_STEPS % NUM_ENVS == 0, "ROLLOUT_STEPS must be divisible by NUM_ENVS"
STEPS_PER_ENV = ROLLOUT_STEPS // NUM_ENVS
assert STEPS_PER_ENV % SEQ_LEN == 0, f"STEPS_PER_ENV ({STEPS_PER_ENV}) must be divisible by SEQ_LEN ({SEQ_LEN})"

# UPDATED: Domain randomization pool of time budgets to sample from per-episode.
DEFAULT_TIMES_SWEEP = _parse_int_list_envvar(
    "DEFAULT_TIMES_SWEEP",
    "100,300,600,900,1200,1800,2300,2900,3500,4100,4800,5700",
    min_len=1
)

TIME_FEAT_DIM = 5

# Scheduling
EVAL_INTERVAL = int(os.getenv("EVAL_INTERVAL", "100"))
WARMUP_SELFPLAY_ITERS = int(os.getenv("WARMUP_SELFPLAY_ITERS", "100"))
# Periodic checkpoint interval (0 = disable periodic checkpoints)
CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", "25"))

# Eval config
EVAL_NUM_GAMES = int(os.getenv("EVAL_NUM_GAMES", "200"))
EVAL_FIXED_OPPONENTS = _parse_int_list_envvar("EVAL_FIXED_OPPONENTS", "16,32,64,96", min_len=1)

# Training mixture proportions (post-warmup)
TRAIN_P_SELFPLAY = float(os.getenv("TRAIN_P_SELFPLAY", "0.30"))
TRAIN_P_ALLCOMBO = float(os.getenv("TRAIN_P_ALLCOMBO", "0.20"))
TRAIN_P_BOTTOM15 = float(os.getenv("TRAIN_P_BOTTOM15", "0.50"))
# Safety check
if abs((TRAIN_P_SELFPLAY + TRAIN_P_ALLCOMBO + TRAIN_P_BOTTOM15) - 1.0) > 1e-6:
    raise ValueError("TRAIN_P_SELFPLAY + TRAIN_P_ALLCOMBO + TRAIN_P_BOTTOM15 must sum to 1.0")

BOTTOM_K = int(os.getenv("BOTTOM_K", "15"))

# WandB
WANDB_PROJECT = os.getenv("WANDB_PROJECT", f"gru_go9x9_selfplay_psim{PRETRAINED_NSIM}")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "")
WANDB_MODE = os.getenv("WANDB_MODE", "online")  # online/offline/disabled


# ================================================================
# 2) Checkpoint loading utilities
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
    model = data["model"]  # (params, state)
    cfg: TrainConfig = data["config"]
    env_id = data.get("env_id", cfg.env_id)
    return env_id, cfg, model


def discover_checkpoints(root: str, iter_filename: str, seed: int = 1) -> Dict[str, str]:
    """
    Find checkpoints like:
      root/nsim_32/{seed}/000600.ckpt
    Returns dict { "nsim_32": path, ... }.
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
        seed_dir = os.path.join(base, str(seed))
        if not os.path.isdir(seed_dir):
            continue
        ckpt_path = os.path.join(seed_dir, iter_filename)
        if os.path.exists(ckpt_path):
            ckpts[f"nsim_{nsim}"] = ckpt_path
    return ckpts


# ================================================================
# 3) AZ forward + MCTS selectors (speed env)
# ================================================================

def build_forward(env, cfg: TrainConfig):
    def forward_fn(x, is_eval: bool = False):
        x = x.astype(jnp.float32)
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
    Recurrent fn for MCTS that only uses board dynamics (no clocks).
    Uses discount=-1 per ply to match "current player" value convention.
    """
    def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state):
        del rng_key
        model_params, model_state = model
        current_player = state.current_player

        state = jax.vmap(step_board_fn)(state, action)
        obs = jax.vmap(observe_fn)(state, state.current_player).astype(jnp.float32)
        state = state.replace(observation=obs)

        (logits, value, _), _ = forward.apply(
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


def make_select_actions_mcts(forward, recurrent_fn, num_simulations: int):
    def select_actions_mcts(model, state, rng_key):
        model_params, model_state = model
        (logits, value, _), _ = forward.apply(
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
            num_simulations=int(num_simulations),
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        return policy_output.action  # (batch,)
    return select_actions_mcts


# ================================================================
# 4) Gate network (same as your universal training: 5D time)
# ================================================================

class GateNetV2(hk.Module):
    """
    mode=2:
      - process AZNet intermediate feature map with conv + global avg/max pool
      - process raw observation with conv + global avg/max pool
      - concat both + time_feat (5-dim) + AZ value
      - 3-layer MLP → logits (NUM_OPTIONS-way) + scalar value
    """
    def __init__(self, num_options: int, mode: int = 2, name: str = "GateNetV2"):
        super().__init__(name=name)
        self.num_options = num_options
        assert mode in (1, 2)
        self.mode = mode

    def _conv_pool(self, x, name_prefix):
        x = hk.Conv2D(64, kernel_shape=3, padding="SAME", name=f"{name_prefix}_conv1")(x)
        x = jax.nn.relu(x)
        x = hk.Conv2D(64, kernel_shape=3, padding="SAME", name=f"{name_prefix}_conv2")(x)
        x = jax.nn.relu(x)
        avg = jnp.mean(x, axis=(1, 2))  # (B, 64)
        mx  = jnp.max(x, axis=(1, 2))   # (B, 64)
        return jnp.concatenate([avg, mx], axis=-1)  # (B, 128)

    def __call__(self, obs, time_feat, az_value, az_intermediate, h_in):
        """
        Args:
          obs: (B, ...) observation
          time_feat: (B, 5)
          az_value: (B,)
          az_intermediate: (B, ...) AZNet intermediate
          h_in: (B, GRU_HIDDEN_DIM) GRU hidden state input

        Returns:
          logits: (B, num_options)
          value: (B,)
          h_out: (B, GRU_HIDDEN_DIM) updated GRU hidden state
        """
        x_az = az_intermediate.astype(jnp.float32)
        if x_az.ndim == 4:
            z_az = self._conv_pool(x_az, "az")  # (B, 128)
        else:
            z_az = hk.Flatten()(x_az)

        if self.mode == 2:
            x_obs = obs.astype(jnp.float32)
            if x_obs.ndim == 4:
                z_obs = self._conv_pool(x_obs, "obs")  # (B, 128)
            else:
                z_obs = hk.Flatten()(x_obs)
            z = jnp.concatenate([z_az, z_obs], axis=-1)  # (B, 256)
        else:
            z = z_az

        time_feat = time_feat.astype(jnp.float32)        # (B, 5)
        az_value = az_value.astype(jnp.float32)          # (B,)
        z = jnp.concatenate([z, time_feat, az_value[..., None]], axis=-1)

        # GRU: integrate history
        gru_cell = hk.GRU(GRU_HIDDEN_DIM, name="gate_gru")
        h_out, h_out_state = gru_cell(z, h_in)  # h_out == h_out_state for GRU

        h = jax.nn.relu(hk.Linear(256)(h_out))
        h = jax.nn.relu(hk.Linear(256)(h))
        h = jax.nn.relu(hk.Linear(128)(h))

        logits = hk.Linear(self.num_options)(h)
        value = hk.Linear(1)(h)[..., 0]
        return logits, value, h_out_state


def gate_forward_fn(obs_batch, time_batch, az_value_batch, az_inter_batch, h_in):
    net = GateNetV2(num_options=NUM_OPTIONS, mode=GATE_SETUP)
    return net(obs_batch, time_batch, az_value_batch, az_inter_batch, h_in)


gate_forward = hk.without_apply_rng(hk.transform(gate_forward_fn))


# ================================================================
# 5) PPO + correct turn-based GAE (sign flip + bootstrap)
# ================================================================

@jax.tree_util.register_pytree_node_class
@dataclass
class PPOBatch:
    """
    Recurrent PPO batch: all fields are (num_seqs, SEQ_LEN, ...) except h_init which is (num_seqs, GRU_HIDDEN_DIM).
    """
    obs: jnp.ndarray           # (S, L, ...)
    time: jnp.ndarray          # (S, L, 5)
    actions: jnp.ndarray       # (S, L)
    logp_old: jnp.ndarray      # (S, L)
    values_old: jnp.ndarray    # (S, L)
    returns: jnp.ndarray       # (S, L)
    advantages: jnp.ndarray    # (S, L)
    az_value: jnp.ndarray      # (S, L)
    az_inter: jnp.ndarray      # (S, L, ...)
    gate_mask: jnp.ndarray     # (S, L)
    h_init: jnp.ndarray        # (S, GRU_HIDDEN_DIM)

    def tree_flatten(self):
        children = (
            self.obs, self.time, self.actions, self.logp_old,
            self.values_old, self.returns, self.advantages,
            self.az_value, self.az_inter, self.gate_mask, self.h_init,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.jit
def compute_gae_turn_based(
    rewards: jnp.ndarray,     # (T,B)
    values: jnp.ndarray,      # (T,B)
    dones: jnp.ndarray,       # (T,B) float32
    last_value: jnp.ndarray,  # (B,)
    gamma: float,
    lam: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    neg_gamma = jnp.array(-gamma, dtype=jnp.float32)
    neg_gamma_lam = jnp.array(-gamma * lam, dtype=jnp.float32)

    rev_r = rewards[::-1]
    rev_v = values[::-1]
    rev_d = dones[::-1]

    def body(carry, inp):
        next_adv, next_value = carry
        r, v, d = inp
        mask = 1.0 - d
        delta = r + neg_gamma * next_value * mask - v
        adv = delta + neg_gamma_lam * next_adv * mask
        ret = adv + v
        return (adv, v), (ret, adv)

    init = (jnp.zeros_like(last_value), last_value)
    (_, _), (rev_returns, rev_advs) = jax.lax.scan(body, init, (rev_r, rev_v, rev_d))
    returns = rev_returns[::-1]
    advantages = rev_advs[::-1]
    return returns, advantages


def make_ppo_loss_fn():
    def loss_fn(params, batch: PPOBatch):
        # batch fields: (S, L, ...) except h_init: (S, H)
        # We need to scan the GRU over L for each sequence.
        # Merge S into batch dim, scan over L.

        S, L = batch.actions.shape[:2]

        def scan_seq(h0, obs_seq, time_seq, az_val_seq, az_inter_seq):
            """Scan one sequence of length L. All inputs: (L, ...)."""
            def step(h, inp):
                o, t, av, ai = inp
                logits_t, value_t, h_new = gate_forward.apply(
                    params, o[None], t[None], av[None], ai[None], h[None]
                )
                return h_new[0], (logits_t[0], value_t[0])
            h_final, (logits_all, values_all) = jax.lax.scan(step, h0, (obs_seq, time_seq, az_val_seq, az_inter_seq))
            return logits_all, values_all  # (L, num_options), (L,)

        # vmap over S sequences
        all_logits, all_values = jax.vmap(scan_seq)(
            batch.h_init,       # (S, H)
            batch.obs,          # (S, L, ...)
            batch.time,         # (S, L, 5)
            batch.az_value,     # (S, L)
            batch.az_inter,     # (S, L, ...)
        )  # all_logits: (S, L, num_options), all_values: (S, L)

        # Flatten to (S*L,) for standard PPO math
        logits = all_logits.reshape(S * L, -1)
        values = all_values.reshape(S * L)
        actions_flat = batch.actions.reshape(S * L)
        logp_old_flat = batch.logp_old.reshape(S * L)
        advantages_flat = batch.advantages.reshape(S * L)
        returns_flat = batch.returns.reshape(S * L)
        gate_mask_flat = batch.gate_mask.reshape(S * L)

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        logp = jnp.take_along_axis(log_probs, actions_flat[..., None], axis=-1)[..., 0]

        ratios = jnp.exp(logp - logp_old_flat)
        adv = advantages_flat

        unclipped = ratios * adv
        clipped = jnp.clip(ratios, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * adv
        per_step_policy = jnp.minimum(unclipped, clipped)

        num_gated = jnp.maximum(gate_mask_flat.sum(), 1.0)
        policy_loss = -jnp.sum(per_step_policy * gate_mask_flat) / num_gated

        value_loss = jnp.mean((returns_flat - values) ** 2)

        probs = jnp.exp(log_probs)
        per_step_entropy = -jnp.sum(probs * log_probs, axis=-1)
        entropy = jnp.sum(per_step_entropy * gate_mask_flat) / num_gated

        loss = policy_loss + PPO_VF_COEF * value_loss - PPO_ENT_COEF * entropy

        approx_kl = jnp.sum((logp_old_flat - logp) * gate_mask_flat) / num_gated
        clipfrac_per = (jnp.abs(ratios - 1.0) > PPO_CLIP_EPS).astype(jnp.float32)
        clipfrac = jnp.sum(clipfrac_per * gate_mask_flat) / num_gated

        var_y = jnp.var(returns_flat)
        explained_var = 1.0 - (jnp.var(returns_flat - values) / (var_y + 1e-8))

        fallback_rate = 1.0 - jnp.mean(gate_mask_flat)

        metrics = {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "explained_var": explained_var,
            "value_pred_mean": jnp.mean(values),
            "ratio_mean": jnp.mean(ratios),
            "fallback_rate": fallback_rate,
        }
        return loss, metrics
    return loss_fn


# ================================================================
# 6) Rollout stats + PPO batch build
# ================================================================

def _safe_mean(x: np.ndarray) -> float:
    return float(x.mean()) if x.size > 0 else 0.0


def _safe_std(x: np.ndarray) -> float:
    return float(x.std()) if x.size > 0 else 0.0


def build_batch_and_stats(
    traj: Dict[str, jnp.ndarray],
    last_value: jnp.ndarray,  # (B,)
) -> Tuple[PPOBatch, Dict[str, Any]]:
    obs_arr = traj["obs"]
    time_arr = traj["time"]
    actions_arr = traj["action"]
    logp_arr = traj["logp"]
    values_arr = traj["value"]
    rewards_arr = traj["reward"]
    dones_arr = traj["done"]
    players_arr = traj["player"]
    az_value_arr = traj["az_value"]
    az_inter_arr = traj["az_inter"]
    gate_mask_arr = traj["gate_mask"]
    h_in_arr = traj["h_in"]  # (T, B, GRU_HIDDEN_DIM)

    T, B = actions_arr.shape

    returns, advantages_raw = compute_gae_turn_based(
        rewards_arr,
        values_arr,
        dones_arr.astype(jnp.float32),
        last_value.astype(jnp.float32),
        GAMMA,
        LAMBDA,
    )

    adv_mean = jnp.mean(advantages_raw)
    adv_std = jnp.std(advantages_raw) + 1e-8
    advantages = (advantages_raw - adv_mean) / adv_std

    # Chunk into contiguous sequences of length SEQ_LEN along T.
    # Truncate T to be divisible by SEQ_LEN.
    L = SEQ_LEN
    num_chunks = T // L
    T_used = num_chunks * L  # drop trailing steps if T not divisible

    def chunk(arr):
        """(T, B, ...) -> (num_chunks, L, B, ...) -> (num_chunks * B, L, ...)"""
        a = arr[:T_used]  # (T_used, B, ...)
        tail = a.shape[2:]
        a = a.reshape(num_chunks, L, B, *tail)        # (C, L, B, ...)
        a = a.transpose(0, 2, 1, *range(3, 3 + len(tail)))  # (C, B, L, ...)
        a = a.reshape(num_chunks * B, L, *tail)        # (C*B, L, ...)
        return a

    # h_init: take the hidden state at the START of each chunk = every L-th timestep
    # h_in_arr is (T, B, H); we want h at indices [0, L, 2L, ...]
    h_init_arr = h_in_arr[:T_used:L]  # (num_chunks, B, H)
    h_init_flat = h_init_arr.reshape(num_chunks * B, GRU_HIDDEN_DIM)  # (C*B, H)

    batch = PPOBatch(
        obs=chunk(obs_arr),
        time=chunk(time_arr),
        actions=chunk(actions_arr),
        logp_old=chunk(logp_arr),
        values_old=chunk(values_arr),
        returns=chunk(returns),
        advantages=chunk(advantages),
        az_value=chunk(az_value_arr),
        az_inter=chunk(az_inter_arr),
        gate_mask=chunk(gate_mask_arr),
        h_init=h_init_flat,
    )

    # Stats (computed on full T, B as before)
    dones_np = np.asarray(dones_arr > 0.5, dtype=bool)
    rewards_np = np.asarray(rewards_arr, dtype=np.float32)
    players_np = np.asarray(players_arr, dtype=np.int32)
    gate_mask_np = np.asarray(gate_mask_arr, dtype=np.float32)

    ep_lens: List[int] = []
    ep_p0_returns: List[float] = []
    episodes_completed = 0
    for b in range(B):
        done_idx = np.where(dones_np[:, b])[0]
        if len(done_idx) == 0:
            continue
        start = 0
        for idx in done_idx:
            episodes_completed += 1
            ep_lens.append(int(idx - start + 1))
            r_mover = float(rewards_np[idx, b])
            mover = int(players_np[idx, b])
            r_p0 = r_mover if mover == 0 else -r_mover
            ep_p0_returns.append(r_p0)
            start = idx + 1

    ep_lens_np = np.asarray(ep_lens, dtype=np.float32)
    ep_p0_np = np.asarray(ep_p0_returns, dtype=np.float32)

    T_star = T_used * B
    stats: Dict[str, Any] = {
        "steps_total": int(T_star),
        "episodes_completed": int(episodes_completed),
        "done_rate": float(dones_np.mean()),
        "ep_len_mean": _safe_mean(ep_lens_np),
        "ep_len_std": _safe_std(ep_lens_np),
        "p0_return_mean": _safe_mean(ep_p0_np),
        "p0_win_rate": float(np.mean(ep_p0_np > 0)) if ep_p0_np.size else 0.0,
        "p0_loss_rate": float(np.mean(ep_p0_np < 0)) if ep_p0_np.size else 0.0,
        "fallback_rate": float(1.0 - gate_mask_np.mean()),
    }
    return batch, stats


# ================================================================
# 7) Opponent definitions (training + eval)
# ================================================================

OPP_SELFPLAY = 0
OPP_FIXED = 1
OPP_BLUEPRINT = 2      # policy-only
OPP_RANDOM = 3
OPP_GREEDY = 4
OPP_MIDPEAK = 5

OPP_KIND_NAMES = {
    OPP_SELFPLAY: "selfplay",
    OPP_FIXED: "fixed",
    OPP_BLUEPRINT: "blueprint0",
    OPP_RANDOM: "random",
    OPP_GREEDY: "greedy",
    OPP_MIDPEAK: "midpeak",
}

def _sim_to_opt(sim: int) -> int:
    if sim not in SIM_OPTIONS:
        raise ValueError(f"Requested sim={sim} but SIM_OPTIONS={SIM_OPTIONS}")
    return SIM_OPTIONS.index(sim)

FIXED_EVAL_SIMS = [s for s in EVAL_FIXED_OPPONENTS if s in SIM_OPTIONS]
if len(FIXED_EVAL_SIMS) != len(EVAL_FIXED_OPPONENTS):
    missing = [s for s in EVAL_FIXED_OPPONENTS if s not in SIM_OPTIONS]
    if missing:
        raise ValueError(f"EVAL_FIXED_OPPONENTS contains sims not in SIM_OPTIONS: {missing}. Fix SIM_OPTIONS.")


def opponent_label_to_spec(label: str) -> Tuple[int, Optional[int]]:
    """Map eval label -> (kind, fixed_sim_or_None)."""
    if label.startswith("always"):
        n = int(label.replace("always", ""))
        return (OPP_FIXED, n)
    if label == "blueprint0":
        return (OPP_BLUEPRINT, None)
    if label == "random":
        return (OPP_RANDOM, None)
    if label == "greedy":
        return (OPP_GREEDY, None)
    if label == "midpeak":
        return (OPP_MIDPEAK, None)
    raise ValueError(f"Unknown opponent label: {label!r}")


def build_all_combo_tables() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, str]]]:
    """
    Build ALL (budget × opponent) tuple tables.

    Returns:
      all_kinds_np: (N,) int32
      all_fixed_opt_np: (N,) int32 (0 if not fixed)
      all_budgets_np: (N,) int32
      meta: list of (budget, label) aligned to rows (for debugging/logging)
    """
    budgets = list(DEFAULT_TIMES_SWEEP)
    opponent_labels: List[str] = []
    for s in FIXED_EVAL_SIMS:
        opponent_labels.append(f"always{s}")
    opponent_labels += ["blueprint0", "random", "greedy", "midpeak"]

    kinds = []
    fixed_opts = []
    bds = []
    meta: List[Tuple[int, str]] = []

    for T in budgets:
        for lab in opponent_labels:
            kind, fixed_sim = opponent_label_to_spec(lab)
            kinds.append(int(kind))
            fixed_opts.append(int(_sim_to_opt(fixed_sim) if fixed_sim is not None else 0))
            bds.append(int(T))
            meta.append((int(T), lab))

    all_kinds_np = np.array(kinds, dtype=np.int32)
    all_fixed_opt_np = np.array(fixed_opts, dtype=np.int32)
    all_budgets_np = np.array(bds, dtype=np.int32)
    return all_kinds_np, all_fixed_opt_np, all_budgets_np, meta


# ================================================================
# 8) Midpeak allocation plan (for midpeak opponent)
# ================================================================

def allocate_midpeak_discrete_plan_np(
    time_budget: int,
    sim_options: List[int],
    n_moves: int = 25,
    peak_center: float = 0.35,
    peak_width: float = 0.3,
) -> np.ndarray:
    opts = sorted(sim_options)
    x = np.linspace(0, 1, n_moves)
    weights = np.exp(-0.5 * ((x - peak_center) / peak_width) ** 2)
    weights = np.maximum(weights, 0.15)
    ideal = weights / weights.sum() * time_budget
    choices = np.array([
        max([s for s in opts if s <= ideal[i]], default=opts[0])
        for i in range(n_moves)
    ], dtype=np.int32)

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


# ================================================================
# 9) Elo-like scoring + bottom-K tuple selection
# ================================================================

def elo_from_expected_score(p: float, eps: float = 1e-4) -> float:
    """
    Convert expected score p (win=1, draw=0.5, loss=0) to an Elo-difference-like scalar.
    We clamp p away from 0 and 1 to avoid infinities.
    """
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(400.0 * np.log10(p / (1.0 - p)))


def rank_tuples_by_elo(eval_grid: Dict[Tuple[int, str], float]) -> List[Tuple[int, str, float, float]]:
    """
    Returns list sorted by ascending Elo (worst first):
      (budget, label, elo, expected_score)
    """
    rows = []
    for (T, lab), exp in eval_grid.items():
        e = elo_from_expected_score(exp)
        rows.append((int(T), str(lab), float(e), float(exp)))
    rows.sort(key=lambda x: x[2])  # by elo asc
    return rows


def bottom_k_specs_from_eval_grid(
    eval_grid: Dict[Tuple[int, str], float],
    k: int = 15,
) -> List[Tuple[int, Optional[int], int, str, float, float]]:
    """
    From eval grid, compute Elo per tuple and return bottom-k tuples as specs.

    Returns list of:
      (kind, fixed_sim_or_None, budget, label, elo, expected_score)
    """
    ranked = rank_tuples_by_elo(eval_grid)
    bottom = ranked[: max(0, min(k, len(ranked)))]
    out = []
    for T, lab, elo, exp in bottom:
        kind, fixed_sim = opponent_label_to_spec(lab)
        out.append((int(kind), fixed_sim, int(T), lab, float(elo), float(exp)))
    return out


def specs_to_fixed_shape_tables(
    specs: List[Tuple[int, Optional[int], int]],
    k: int,
    default_kind: int,
    default_fixed_opt: int,
    default_budget: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make fixed-shape (k,) tables from a variable-length list of (kind, fixed_sim_or_None, budget).
    Pads with defaults if needed.
    """
    kinds = []
    fixed_opts = []
    budgets = []
    for (kind, fixed_sim, bd) in specs:
        kinds.append(int(kind))
        fixed_opts.append(int(_sim_to_opt(int(fixed_sim)) if fixed_sim is not None else 0))
        budgets.append(int(bd))
    while len(kinds) < k:
        kinds.append(int(default_kind))
        fixed_opts.append(int(default_fixed_opt))
        budgets.append(int(default_budget))
    kinds = kinds[:k]
    fixed_opts = fixed_opts[:k]
    budgets = budgets[:k]
    return (
        np.array(kinds, dtype=np.int32),
        np.array(fixed_opts, dtype=np.int32),
        np.array(budgets, dtype=np.int32),
    )


# ================================================================
# 10) Rollout core (training) with per-episode tuple mixing
# ================================================================

def init_envs_with_random_budgets(env_speed, rng, num_envs):
    """
    Initial envs: uniform budgets from DEFAULT_TIMES_SWEEP, opponent defaults to selfplay.
    """
    time_budgets_arr = jnp.array(DEFAULT_TIMES_SWEEP, dtype=jnp.int32)
    num_budget_choices = len(DEFAULT_TIMES_SWEEP)

    rng, key_init, key_budgets = jax.random.split(rng, 3)
    init_keys = jax.random.split(key_init, num_envs)
    state = jax.vmap(env_speed.init)(init_keys)

    budget_indices = jax.random.randint(key_budgets, (num_envs,), 0, num_budget_choices)
    default_times = time_budgets_arr[budget_indices]

    time_left_init = jnp.stack([default_times, default_times], axis=1)
    state = state.replace(_time_left=time_left_init)

    return state, default_times, rng


def make_rollout_core_tuplemix(
    env_speed,
    forward,
    select_mcts_fns: Sequence,
    model_pretrained,
):
    """
    Batched rollout over NUM_ENVS envs with:
      - per-episode assignment of (starting_budget, opponent_kind, fixed_opt if needed)
      - assignment is sampled at RESET time according to:
          warmup: 100% self-play (budget uniform over DEFAULT_TIMES_SWEEP)
          main: 30% self-play, 20% ALL combos, 50% bottom-15 combos

    Self-play learning signal:
      - Both players sample gate, both contribute PPO when has_time.

    Non-self opponents:
      - Only P0 contributes PPO (gate_mask = has_time & is_p0).

    Timeout/fallback (TRAINING):
      - if has_time (time_left > 0): MCTS with chosen sims, spend that time
      - else: fallback to AZ policy sample, spend 0
    """

    feat_params, feat_state = model_pretrained
    sim_options_arr = jnp.array(SIM_OPTIONS, dtype=jnp.int32)
    budgets_sweep_arr = jnp.array(DEFAULT_TIMES_SWEEP, dtype=jnp.int32)
    num_budget_choices = len(DEFAULT_TIMES_SWEEP)

    MIDPEAK_NMOVES = 25

    # Midpeak plans for every budget in DEFAULT_TIMES_SWEEP (host-side once, then device arrays).
    midpeak_plans_np = np.stack(
        [allocate_midpeak_discrete_plan_np(int(T), SIM_OPTIONS, n_moves=MIDPEAK_NMOVES) for T in DEFAULT_TIMES_SWEEP],
        axis=0
    )  # (num_budgets, n_moves)
    midpeak_plans = jnp.array(midpeak_plans_np, dtype=jnp.int32)
    budgets_table = jnp.array(DEFAULT_TIMES_SWEEP, dtype=jnp.int32)

    def lookup_midpeak_nsim(default_time_scalar: jnp.int32, p1_move_num: jnp.int32) -> jnp.int32:
        idx = jnp.argmax(budgets_table == default_time_scalar).astype(jnp.int32)
        p1m = jnp.minimum(p1_move_num, jnp.int32(MIDPEAK_NMOVES - 1))
        return midpeak_plans[idx, p1m]

    def compute_time_feat(state, default_times):
        time_left = state.time_left  # (B,2)
        cur = state.current_player   # (B,)
        cur_idx = cur[:, None]
        opp_idx = (1 - cur)[:, None]
        my_time = jnp.take_along_axis(time_left, cur_idx, axis=1)[:, 0].astype(jnp.float32)
        opp_time = jnp.take_along_axis(time_left, opp_idx, axis=1)[:, 0].astype(jnp.float32)
        dt_f = default_times.astype(jnp.float32)
        return jnp.stack([
            my_time / jnp.maximum(dt_f, 1.0),
            opp_time / jnp.maximum(dt_f, 1.0),
            jnp.log1p(my_time),
            jnp.log1p(opp_time),
            jnp.log1p(dt_f),
        ], axis=1)

    def reset_env_with_budget(key, budget_scalar: jnp.int32):
        s = env_speed.init(key)
        s = s.replace(_time_left=jnp.int32([budget_scalar, budget_scalar]))
        return s

    def rollout_core(
        state,
        default_times,                # (B,) int32
        opponent_kind,                # (B,) int32
        opponent_fixed_opt,           # (B,) int32
        p1_move_counts,               # (B,) int32
        gate_params,
        rng_key,
        num_steps: int,
        *,
        gate_hidden,                  # (B, GRU_HIDDEN_DIM)
        phase_selfplay_only: bool,    # static for the chunk
        all_kinds: jnp.ndarray,       # (NALL,)
        all_fixed_opts: jnp.ndarray,  # (NALL,)
        all_budgets: jnp.ndarray,     # (NALL,)
        bot_kinds: jnp.ndarray,       # (BOTTOM_K,)
        bot_fixed_opts: jnp.ndarray,  # (BOTTOM_K,)
        bot_budgets: jnp.ndarray,     # (BOTTOM_K,)
    ):
        num_envs = NUM_ENVS
        n_all = all_kinds.shape[0]

        def step_fn(carry, _):
            state, default_times, opponent_kind, opponent_fixed_opt, p1_move_counts, gate_h, rng = carry
            rng, key_reset, key_gate_p0, key_gate_p1, key_opp, key_fallback, *keys_mcts = jax.random.split(
                rng, 6 + NUM_OPTIONS
            )

            done_prev = state.terminated | state.truncated
            keys_reset = jax.random.split(key_reset, num_envs)

            # Reset GRU hidden on episode boundary
            gate_h = jnp.where(done_prev[:, None], jnp.zeros_like(gate_h), gate_h)

            def sample_episode_tuple(k: jnp.ndarray):
                """
                Sample (kind, fixed_opt, budget) for a NEW episode.
                Implements:
                  warmup: always selfplay (budget uniform over sweep)
                  main: 30% selfplay, 20% all-combo, 50% bottom15
                """
                k, k_bucket, k_a, k_b, k_budget = jax.random.split(k, 5)

                # uniform budget for selfplay bucket
                bd_idx = jax.random.randint(k_budget, (), 0, num_budget_choices).astype(jnp.int32)
                bd_self = budgets_sweep_arr[bd_idx]
                return jnp.int32(OPP_SELFPLAY), jnp.int32(0), jnp.int32(bd_self)
            
            def reset_one(s, dt, ok, ofo, pmc, k, d):
                def do_reset(_):
                    # sample tuple spec
                    kind_new, fixed_new, bd_new = sample_episode_tuple(k)
                    ns = reset_env_with_budget(k, bd_new)
                    return ns, bd_new, kind_new, fixed_new, jnp.int32(0)
                def no_reset(_):
                    return s, dt, ok, ofo, pmc
                return jax.lax.cond(d, do_reset, no_reset, operand=None)

            state, default_times, opponent_kind, opponent_fixed_opt, p1_move_counts = jax.vmap(reset_one)(
                state, default_times, opponent_kind, opponent_fixed_opt, p1_move_counts, keys_reset, done_prev
            )

            obs = state.observation.astype(jnp.float32)
            time_feat = compute_time_feat(state, default_times)

            (az_logits_b, az_value_b, az_inter_b), _ = forward.apply(
                feat_params, feat_state, obs, is_eval=True
            )

            cur_player = state.current_player  # (B,)
            is_p0 = (cur_player == 0)
            kind = opponent_kind
            is_self_env = (kind == OPP_SELFPLAY)

            my_time = state.time_left[jnp.arange(num_envs), cur_player].astype(jnp.int32)
            has_time = my_time > 0  # training semantics

            # Gate logits always computed (with GRU hidden state)
            h_in_snapshot = gate_h  # save for trajectory before update
            logits_gate, value_gate, gate_h_new = gate_forward.apply(
                gate_params, obs, time_feat, az_value_b, az_inter_b, gate_h
            )
            log_probs_gate = jax.nn.log_softmax(logits_gate, axis=-1)
            batch_idx = jnp.arange(num_envs)

            # P0 gate sample (always)
            gate_action_p0 = jax.random.categorical(key_gate_p0, logits_gate, axis=-1).astype(jnp.int32)
            logp_p0 = log_probs_gate[batch_idx, gate_action_p0]

            # P1 self-play gate sample (only meaningful in self-play envs)
            gate_action_p1_self = jax.random.categorical(key_gate_p1, logits_gate, axis=-1).astype(jnp.int32)
            logp_p1_self = log_probs_gate[batch_idx, gate_action_p1_self]

            # --- P1 opponent choices (for non-selfplay envs) ---
            rand_idx = jax.random.randint(key_opp, (num_envs,), 0, NUM_OPTIONS).astype(jnp.int32)

            affordable = (sim_options_arr[None, :] <= my_time[:, None])
            has_any = affordable.any(axis=1)
            best_from_end = jnp.argmax(affordable[:, ::-1], axis=1).astype(jnp.int32)
            greedy_idx = (NUM_OPTIONS - 1 - best_from_end).astype(jnp.int32)
            greedy_idx = jnp.where(has_any, greedy_idx, jnp.int32(0))

            planned_nsim = jax.vmap(lookup_midpeak_nsim)(
                default_times.astype(jnp.int32),
                p1_move_counts.astype(jnp.int32),
            )
            match = (sim_options_arr[None, :] == planned_nsim[:, None])
            planned_idx = jnp.argmax(match, axis=1).astype(jnp.int32)
            planned_can_afford = (planned_nsim <= my_time)
            midpeak_idx = jnp.where(planned_can_afford, planned_idx, greedy_idx)

            fixed_idx = opponent_fixed_opt
            blueprint_idx = jnp.int32(0)  # placeholder (blueprint ignores gate budget)

            p1_idx = rand_idx
            p1_idx = jnp.where(kind == OPP_FIXED, fixed_idx, p1_idx)
            p1_idx = jnp.where(kind == OPP_BLUEPRINT, blueprint_idx, p1_idx)
            p1_idx = jnp.where(kind == OPP_RANDOM, rand_idx, p1_idx)
            p1_idx = jnp.where(kind == OPP_GREEDY, greedy_idx, p1_idx)
            p1_idx = jnp.where(kind == OPP_MIDPEAK, midpeak_idx, p1_idx)

            # Final gate action
            gate_action = jnp.where(
                is_p0,
                gate_action_p0,
                jnp.where(is_self_env, gate_action_p1_self, p1_idx),
            ).astype(jnp.int32)

            # Logp
            logp = jnp.where(
                is_p0,
                logp_p0,
                jnp.where(is_self_env, logp_p1_self, jnp.zeros((num_envs,), dtype=jnp.float32)),
            )

            # Candidate MCTS actions for each sim option
            cand_actions = []
            for i in range(NUM_OPTIONS):
                a_i = select_mcts_fns[i](model_pretrained, state, keys_mcts[i]).astype(jnp.int32)
                cand_actions.append(a_i)
            cand_actions = jnp.stack(cand_actions, axis=0)  # (O,B)

            mcts_action = cand_actions[gate_action, batch_idx]
            intended_cost = sim_options_arr[gate_action].astype(jnp.int32)
            mcts_time_spent = intended_cost

            fallback_action = jax.random.categorical(key_fallback, az_logits_b, axis=-1).astype(jnp.int32)

            actions = jnp.where(has_time, mcts_action, fallback_action)
            time_spent = jnp.where(has_time, mcts_time_spent, jnp.int32(0))

            # Gate mask:
            gate_mask = jnp.where(is_self_env, has_time, (has_time & is_p0)).astype(jnp.float32)

            prev_player = cur_player

            def step_one(env_state, a, t):
                return env_speed.step(env_state, (a, t))

            state_next = jax.vmap(step_one, in_axes=(0, 0, 0))(state, actions, time_spent)

            alive = ~(state.terminated | state.truncated)
            inc_p1 = (alive & (cur_player == 1)).astype(jnp.int32)
            p1_move_counts_next = p1_move_counts + inc_p1

            done = (state_next.terminated | state_next.truncated).astype(jnp.float32)
            rewards_all = state_next.rewards
            prev_idx = prev_player[:, None]
            reward_prev = jnp.take_along_axis(rewards_all, prev_idx, axis=1)[:, 0]
            reward = jnp.where(done > 0.0, reward_prev, jnp.float32(0.0))

            carry_next = (state_next, default_times, opponent_kind, opponent_fixed_opt, p1_move_counts_next, gate_h_new, rng)
            out = {
                "obs": obs,
                "time": time_feat,
                "action": gate_action,
                "logp": logp,
                "value": value_gate,
                "reward": reward,
                "done": done,
                "player": prev_player,
                "az_value": az_value_b,
                "az_inter": az_inter_b,
                "gate_mask": gate_mask,
                "h_in": h_in_snapshot,
            }
            return carry_next, out

        init_carry = (state, default_times, opponent_kind, opponent_fixed_opt, p1_move_counts, gate_hidden, rng_key)
        (state_final, default_times_final, opponent_kind_final, opponent_fixed_opt_final, p1_move_counts_final, gate_h_final, rng_final), traj = jax.lax.scan(
            step_fn, init_carry, xs=None, length=num_steps
        )

        # Bootstrap last_value
        obs_f = state_final.observation.astype(jnp.float32)
        time_feat_f = compute_time_feat(state_final, default_times_final)
        (_, az_value_f, az_inter_f), _ = forward.apply(
            feat_params, feat_state, obs_f, is_eval=True
        )
        _, value_f, _ = gate_forward.apply(
            gate_params, obs_f, time_feat_f, az_value_f, az_inter_f, gate_h_final
        )
        done_f = state_final.terminated | state_final.truncated
        last_value = jnp.where(done_f, 0.0, value_f)

        return (
            state_final,
            default_times_final,
            opponent_kind_final,
            opponent_fixed_opt_final,
            p1_move_counts_final,
            gate_h_final,
            rng_final,
            traj,
            last_value,
        )

    # num_steps is static; phase_selfplay_only is static (passed as python bool in main).
    return jax.jit(rollout_core, static_argnums=(7,), static_argnames=("phase_selfplay_only",))


# ================================================================
# 11) Evaluation (NO-TIMEOUT) vs opponent suite, then grid wrapper
# ================================================================

def make_eval_play_many(
    env_speed,
    forward,
    select_mcts_fns: Sequence,
    model_pretrained,
):
    feat_params, feat_state = model_pretrained
    sim_options_arr = jnp.array(SIM_OPTIONS, dtype=jnp.int32)

    def policy_only_action_from_logits(logits_1d, legal_mask_1d):
        masked = jnp.where(legal_mask_1d, logits_1d, jnp.finfo(logits_1d.dtype).min)
        return jnp.argmax(masked).astype(jnp.int32)

    @jax.jit
    def play_many(
        gate_params,
        rng_keys,                 # (G,2) keys
        time_budget: jnp.int32,   # scalar
        opponent_kind: jnp.int32, # scalar
        opponent_fixed_opt: jnp.int32,  # scalar
        midpeak_plan: jnp.ndarray,      # (MIDPEAK_NMOVES,) int32 sims
    ):
        num_games = rng_keys.shape[0]
        max_steps = int(getattr(env_speed, "max_termination_steps", 256))

        def play_one(rng_key):
            s = env_speed.init(rng_key)
            s = s.replace(_time_left=jnp.int32([time_budget, time_budget]))
            h0 = jnp.zeros((GRU_HIDDEN_DIM,), dtype=jnp.float32)

            def step_fn(carry, step_idx):
                state, rng, p1_move, gate_h = carry
                rng, k_gate, k_opp, k_mcts = jax.random.split(rng, 4)

                alive = ~(state.terminated | state.truncated)
                cur = state.current_player
                my_time = state.time_left[cur].astype(jnp.int32)

                obs = state.observation.astype(jnp.float32)
                (az_logits_b, az_value_b, az_inter_b), _ = forward.apply(
                    feat_params, feat_state, obs[None, ...], is_eval=True
                )
                az_logits = az_logits_b[0]
                az_value = az_value_b[0]
                az_inter = az_inter_b[0]

                opp_time = state.time_left[1 - cur].astype(jnp.float32)
                time_feat = jnp.array([
                    my_time.astype(jnp.float32) / jnp.maximum(time_budget.astype(jnp.float32), 1.0),
                    opp_time / jnp.maximum(time_budget.astype(jnp.float32), 1.0),
                    jnp.log1p(my_time.astype(jnp.float32)),
                    jnp.log1p(opp_time),
                    jnp.log1p(time_budget.astype(jnp.float32)),
                ], dtype=jnp.float32)

                def p0_choice():
                    logits_gate, _, h_new = gate_forward.apply(
                        gate_params,
                        obs[None, ...],
                        time_feat[None, ...],
                        jnp.array([az_value], dtype=jnp.float32),
                        az_inter[None, ...],
                        gate_h[None, ...],
                    )
                    return jnp.argmax(logits_gate[0]).astype(jnp.int32), h_new[0]

                def p1_choice():
                    fixed_idx = opponent_fixed_opt
                    rand_idx = jax.random.randint(k_opp, (), 0, NUM_OPTIONS).astype(jnp.int32)

                    affordable = (sim_options_arr <= my_time)
                    has_any = affordable.any()
                    best_from_end = jnp.argmax(affordable[::-1]).astype(jnp.int32)
                    greedy_idx = (NUM_OPTIONS - 1 - best_from_end).astype(jnp.int32)
                    greedy_idx = jnp.where(has_any, greedy_idx, jnp.int32(0))

                    planned_sim = midpeak_plan[jnp.minimum(p1_move, midpeak_plan.shape[0] - 1)]
                    match = (sim_options_arr == planned_sim)
                    planned_idx = jnp.argmax(match).astype(jnp.int32)
                    can_afford = planned_sim <= my_time
                    mid_idx = jnp.where(can_afford, planned_idx, greedy_idx)

                    idx = rand_idx
                    idx = jnp.where(opponent_kind == OPP_FIXED, fixed_idx, idx)
                    idx = jnp.where(opponent_kind == OPP_BLUEPRINT, jnp.int32(0), idx)
                    idx = jnp.where(opponent_kind == OPP_RANDOM, rand_idx, idx)
                    idx = jnp.where(opponent_kind == OPP_GREEDY, greedy_idx, idx)
                    idx = jnp.where(opponent_kind == OPP_MIDPEAK, mid_idx, idx)
                    return idx.astype(jnp.int32), gate_h  # opponent doesn't update P0's hidden state

                idx, gate_h_new = jax.lax.cond(
                    cur == 0,
                    lambda _: p0_choice(),
                    lambda _: p1_choice(),
                    operand=None,
                )
                intended_cost = sim_options_arr[idx].astype(jnp.int32)

                # NO-TIMEOUT fallback: if my_time <= intended_cost -> policy-only, spend 0
                can_afford = my_time > intended_cost
                pol_action = policy_only_action_from_logits(az_logits, state.legal_action_mask)

                def mcts_action():
                    state_b = jax.tree_util.tree_map(lambda x: x[None, ...], state)
                    acts = []
                    for i in range(NUM_OPTIONS):
                        key_i = jax.random.fold_in(k_mcts, i)
                        a_i = select_mcts_fns[i](model_pretrained, state_b, key_i)[0].astype(jnp.int32)
                        acts.append(a_i)
                    acts = jnp.stack(acts, axis=0)  # (NUM_OPTIONS,)
                    return acts[idx]

                use_mcts = can_afford & ((cur == 0) | (opponent_kind != OPP_BLUEPRINT))
                time_spent = jnp.where(use_mcts, intended_cost, jnp.int32(0))
                a = jax.lax.cond(use_mcts, lambda _: mcts_action(), lambda _: pol_action, operand=None)

                def do_step(_):
                    return env_speed.step(state, (a, time_spent))
                state_next = jax.lax.cond(alive, do_step, lambda _: state, operand=None)

                p1_move_next = p1_move + jnp.where(alive & (cur == 1), jnp.int32(1), jnp.int32(0))
                return (state_next, rng, p1_move_next, gate_h_new), None

            (sf, _, _, _), _ = jax.lax.scan(step_fn, (s, rng_key, jnp.int32(0), h0), xs=jnp.arange(max_steps), length=max_steps)
            return sf.rewards, sf.time_left

        rews, times = jax.vmap(play_one)(rng_keys[:, 0])
        return rews, times

    return play_many


def eval_suite_one_budget(
    env_speed,
    forward,
    select_mcts_fns,
    model_pretrained,
    gate_params,
    seed: int,
    time_budget: int,
) -> Dict[str, float]:
    """
    Runs eval vs a suite of opponents at ONE time budget, returns expected scores per opponent label.
    expected_score = (wins + 0.5*draws) / games, from P0 perspective.
    """
    play_many = make_eval_play_many(env_speed, forward, select_mcts_fns, model_pretrained)

    rng = jax.random.PRNGKey(seed ^ 0xABCDEF)
    rng, key_games = jax.random.split(rng)
    rng_keys = jax.random.split(key_games, EVAL_NUM_GAMES)
    rng_keys = jnp.stack([rng_keys, rng_keys], axis=1)

    midpeak_plan = allocate_midpeak_discrete_plan_np(time_budget, SIM_OPTIONS, n_moves=25)
    midpeak_plan = jnp.array(midpeak_plan, dtype=jnp.int32)

    opponents: List[Tuple[str, int, Optional[int]]] = []
    for s in FIXED_EVAL_SIMS:
        opponents.append((f"always{s}", OPP_FIXED, s))
    opponents += [
        ("blueprint0", OPP_BLUEPRINT, None),
        ("random", OPP_RANDOM, None),
        ("greedy", OPP_GREEDY, None),
        ("midpeak", OPP_MIDPEAK, None),
    ]

    out: Dict[str, float] = {}
    for label, kind, fixed_sim in opponents:
        fixed_opt = jnp.int32(_sim_to_opt(fixed_sim) if fixed_sim is not None else 0)
        rews, _times = play_many(
            gate_params,
            rng_keys,
            jnp.int32(time_budget),
            jnp.int32(kind),
            fixed_opt,
            midpeak_plan,
        )
        rews_np = np.asarray(jax.device_get(rews), dtype=np.float32)  # (G,2)
        r0 = rews_np[:, 0]
        r1 = rews_np[:, 1]
        wins = int(np.sum(r0 > 0))
        losses = int(np.sum(r1 > 0))
        draws = int(EVAL_NUM_GAMES - wins - losses)
        expected = (wins + 0.5 * draws) / max(1, EVAL_NUM_GAMES)
        out[label] = float(expected)
    return out


def eval_suite_grid(
    env_speed,
    forward,
    select_mcts_fns,
    model_pretrained,
    gate_params,
    seed: int,
) -> Dict[Tuple[int, str], float]:
    """
    UPDATED: evaluate across ALL budgets in DEFAULT_TIMES_SWEEP.

    Returns:
      dict[(time_budget, opponent_label)] = expected_score
    """
    out: Dict[Tuple[int, str], float] = {}
    for i, T in enumerate(DEFAULT_TIMES_SWEEP):
        scores_T = eval_suite_one_budget(
            env_speed=env_speed,
            forward=forward,
            select_mcts_fns=select_mcts_fns,
            model_pretrained=model_pretrained,
            gate_params=gate_params,
            seed=seed + 9973 * (i + 1),
            time_budget=int(T),
        )
        for lab, exp in scores_T.items():
            out[(int(T), str(lab))] = float(exp)
    return out


# ================================================================
# 12) Main training loop
# ================================================================

def main():
    args = parse_args()
    env_module_name = args.env
    env_kwargs = _parse_env_kwargs(args.env_kwargs)
    seed = args.seed
    PRETRAINED_SEED = seed

    make_env, step_board_fn, observe_fn = load_speed_env_module(env_module_name)
    env_speed = make_env(**env_kwargs)

    base_env_id_envvar = os.getenv("BASE_ENV_ID")
    expected_env_id = base_env_id_envvar if base_env_id_envvar is not None else env_speed.id

    print(f"[train_gate_adaptive] Using env module: {env_module_name}")
    print(f"[train_gate_adaptive] Env kwargs: {env_kwargs}")
    print(f"[train_gate_adaptive] Expected base env id: {expected_env_id}")
    print(f"[train_gate_adaptive] SIM_OPTIONS: {SIM_OPTIONS}  NUM_ENVS={NUM_ENVS}  STEPS_PER_ENV={STEPS_PER_ENV}")
    print(f"[train_gate_adaptive] Domain randomization budgets: {DEFAULT_TIMES_SWEEP}")
    print(f"[train_gate_adaptive] Warmup selfplay iters: {WARMUP_SELFPLAY_ITERS} | Eval interval: {EVAL_INTERVAL}")
    print(f"[train_gate_adaptive] Train mixture post-warmup: self={TRAIN_P_SELFPLAY} all={TRAIN_P_ALLCOMBO} bottom={TRAIN_P_BOTTOM15} bottomK={BOTTOM_K}")

    ckpt_paths = discover_checkpoints(CKPT_ROOT, ITER_FILE, seed=PRETRAINED_SEED)
    print("[train_gate_adaptive] Found checkpoints:", ckpt_paths)

    key = f"nsim_{PRETRAINED_NSIM}"
    if key not in ckpt_paths:
        raise RuntimeError(f"Missing checkpoint for {key} in {CKPT_ROOT}")

    env_id_from_ckpt, cfg, model_pretrained = load_checkpoint(ckpt_paths[key])
    print("[train_gate_adaptive] Loaded checkpoint:", env_id_from_ckpt, ckpt_paths[key])

    if env_id_from_ckpt != expected_env_id:
        raise RuntimeError(
            f"Pretrained model env_id mismatch: checkpoint env_id={env_id_from_ckpt} "
            f"but expected {expected_env_id}. (Set BASE_ENV_ID or choose a matching ckpt.)"
        )

    forward = build_forward(env_speed, cfg)
    recurrent_fn_speed = make_recurrent_fn_speed(forward, step_board_fn, observe_fn)
    select_mcts_fns = [make_select_actions_mcts(forward, recurrent_fn_speed, n) for n in SIM_OPTIONS]

    feat_params, feat_state = model_pretrained
    dummy_state = env_speed.init(jax.random.PRNGKey(0))
    dummy_obs_single = dummy_state.observation.astype(jnp.float32)
    (_, _, dummy_inter), _ = forward.apply(
        feat_params, feat_state, dummy_obs_single[None, ...], is_eval=True
    )
    obs_shape = tuple(dummy_obs_single.shape)
    inter_shape = tuple(dummy_inter.shape[1:])

    print(f"[train_gate_adaptive] Obs shape: {obs_shape}")
    print(f"[train_gate_adaptive] AZ intermediate shape: {inter_shape}")

    rng = jax.random.PRNGKey(seed)
    state, default_times, rng = init_envs_with_random_budgets(env_speed, rng, NUM_ENVS)

    # Per-env episode assignment state (will be re-sampled on episode resets inside rollout)
    opponent_kind = jnp.full((NUM_ENVS,), OPP_SELFPLAY, dtype=jnp.int32)
    opponent_fixed_opt = jnp.zeros((NUM_ENVS,), dtype=jnp.int32)
    p1_move_counts = jnp.zeros((NUM_ENVS,), dtype=jnp.int32)
    gate_hidden = jnp.zeros((NUM_ENVS, GRU_HIDDEN_DIM), dtype=jnp.float32)

    dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)
    dummy_time = jnp.zeros((1, TIME_FEAT_DIM), dtype=jnp.float32)
    dummy_az_value = jnp.zeros((1,), dtype=jnp.float32)
    dummy_az_inter = jnp.zeros((1,) + inter_shape, dtype=jnp.float32)
    dummy_h_in = jnp.zeros((1, GRU_HIDDEN_DIM), dtype=jnp.float32)

    gate_params = gate_forward.init(
        jax.random.PRNGKey(seed + 12345),
        dummy_obs,
        dummy_time,
        dummy_az_value,
        dummy_az_inter,
        dummy_h_in,
    )

    # Optimizer
    if CLIP_GRAD_NORM > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(CLIP_GRAD_NORM),
            optax.adam(PPO_LR),
        )
    else:
        optimizer = optax.adam(PPO_LR)
    opt_state = optimizer.init(gate_params)

    loss_fn = make_ppo_loss_fn()
    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def ppo_update_many(params, opt_state, batches: PPOBatch):
        def body(carry, mb: PPOBatch):
            params, opt_state = carry
            (loss, metrics), grads = value_and_grad(params, mb)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), metrics
        (params_final, opt_state_final), metrics_seq = jax.lax.scan(
            body, (params, opt_state), batches
        )
        return params_final, opt_state_final, metrics_seq

    rollout_jit = make_rollout_core_tuplemix(
        env_speed, forward, select_mcts_fns, model_pretrained
    )

    # WandB init
    run_uid = os.getenv("RUN_UID", "") or uuid.uuid4().hex[:8]
    run_name = (
        f"go_selfplay_{env_module_name}_pre{PRETRAINED_NSIM}_"
        f"opts{'-'.join(map(str, SIM_OPTIONS))}_seed{seed}_{run_uid}"
    )

    if WANDB_MODE == "disabled":
        os.environ["WANDB_MODE"] = "disabled"

    wandb.init(
        project=WANDB_PROJECT,
        entity=(WANDB_ENTITY or None),
        config={
            "speed_env_module": env_module_name,
            "env_kwargs": env_kwargs,
            "expected_env_id": expected_env_id,
            "ckpt_root": CKPT_ROOT,
            "iter_file": ITER_FILE,
            "pretrained_seed": PRETRAINED_SEED,
            "pretrained_nsim": PRETRAINED_NSIM,
            "sim_options": SIM_OPTIONS,
            "seed": seed,
            "num_updates": NUM_UPDATES,
            "rollout_steps": ROLLOUT_STEPS,
            "steps_per_env": STEPS_PER_ENV,
            "gamma": GAMMA,
            "lambda": LAMBDA,
            "ppo_epochs": PPO_EPOCHS,
            "ppo_clip_eps": PPO_CLIP_EPS,
            "ppo_lr": PPO_LR,
            "ppo_vf_coef": PPO_VF_COEF,
            "ppo_ent_coef": PPO_ENT_COEF,
            "batch_size": BATCH_SIZE,
            "clip_grad_norm": CLIP_GRAD_NORM,
            "gate_setup": GATE_SETUP,
            "domain_randomization_budgets": DEFAULT_TIMES_SWEEP,
            "time_feat_dim": TIME_FEAT_DIM,
            "num_envs": NUM_ENVS,
            "gru_hidden_dim": GRU_HIDDEN_DIM,
            "seq_len": SEQ_LEN,
            "obs_shape": obs_shape,
            "az_inter_shape": inter_shape,
            "eval_interval": EVAL_INTERVAL,
            "warmup_selfplay_iters": WARMUP_SELFPLAY_ITERS,
            "eval_num_games": EVAL_NUM_GAMES,
            "eval_fixed_opponents": EVAL_FIXED_OPPONENTS,
            "train_p_selfplay": TRAIN_P_SELFPLAY,
            "train_p_allcombo": TRAIN_P_ALLCOMBO,
            "train_p_bottom": TRAIN_P_BOTTOM15,
            "bottom_k": BOTTOM_K,
        },
        name=run_name,
    )

    # Save dir
    save_root = os.path.join(GATE_CKPT_ROOT_ENV, env_module_name)
    save_root = os.path.join(save_root, f"pre{PRETRAINED_NSIM}")
    save_dir = os.path.join(save_root, f"opts{'-'.join(map(str, SIM_OPTIONS))}_seed{seed}")
    os.makedirs(save_dir, exist_ok=True)

    # Build ALL combo tables once (host + device)
    all_kinds_np, all_fixed_np, all_budgets_np, all_meta = build_all_combo_tables()
    all_kinds = jnp.array(all_kinds_np, dtype=jnp.int32)
    all_fixed_opts = jnp.array(all_fixed_np, dtype=jnp.int32)
    all_budgets = jnp.array(all_budgets_np, dtype=jnp.int32)

    # Initialize bottom-K tables to a deterministic default (first K all-combos) until first eval.
    # This keeps JIT shapes stable.
    default_budget = int(DEFAULT_TIMES_SWEEP[len(DEFAULT_TIMES_SWEEP) // 2])
    bot_kinds_np = all_kinds_np[:BOTTOM_K].copy()
    bot_fixed_np = all_fixed_np[:BOTTOM_K].copy()
    bot_budgets_np = all_budgets_np[:BOTTOM_K].copy()
    if bot_kinds_np.shape[0] < BOTTOM_K:
        # pad if all-combos smaller than bottom_k (unlikely)
        pad_k = BOTTOM_K - bot_kinds_np.shape[0]
        bot_kinds_np = np.pad(bot_kinds_np, (0, pad_k), constant_values=OPP_SELFPLAY)
        bot_fixed_np = np.pad(bot_fixed_np, (0, pad_k), constant_values=0)
        bot_budgets_np = np.pad(bot_budgets_np, (0, pad_k), constant_values=default_budget)

    bot_kinds = jnp.array(bot_kinds_np, dtype=jnp.int32)
    bot_fixed_opts = jnp.array(bot_fixed_np, dtype=jnp.int32)
    bot_budgets = jnp.array(bot_budgets_np, dtype=jnp.int32)

    best_score = -1e9  # unchanged; you said ignore changing best-ckpt criterion
    bottom_specs_verbose: List[Tuple[int, Optional[int], int, str, float, float]] = []

    for update in trange(1, NUM_UPDATES + 1, desc="PPO updates"):
        #phase_selfplay_only = (update <= WARMUP_SELFPLAY_ITERS)
        phase_selfplay_only = True
        # Rollout (note: per-episode tuple assignment is done inside rollout resets)
        state, default_times, opponent_kind, opponent_fixed_opt, p1_move_counts, gate_hidden, rng, traj, last_value = rollout_jit(
            state,
            default_times,
            opponent_kind,
            opponent_fixed_opt,
            p1_move_counts,
            gate_params,
            rng,
            STEPS_PER_ENV,
            gate_hidden=gate_hidden,
            phase_selfplay_only=phase_selfplay_only,
            all_kinds=all_kinds,
            all_fixed_opts=all_fixed_opts,
            all_budgets=all_budgets,
            bot_kinds=bot_kinds,
            bot_fixed_opts=bot_fixed_opts,
            bot_budgets=bot_budgets,
        )
        batch, roll_stats = build_batch_and_stats(traj, last_value)

        # PPO minibatches (shuffle SEQUENCES, not individual timesteps)
        N_seqs = batch.obs.shape[0]  # num_seqs = (T // SEQ_LEN) * B
        seqs_per_mb = max(1, BATCH_SIZE // SEQ_LEN)  # sequences per minibatch
        num_mbs = N_seqs // seqs_per_mb
        if num_mbs <= 0:
            raise RuntimeError(f"Not enough sequences for minibatch: N_seqs={N_seqs}, seqs_per_mb={seqs_per_mb}")

        U = PPO_EPOCHS * num_mbs

        rng, key_perm = jax.random.split(rng)
        keys_epochs = jax.random.split(key_perm, PPO_EPOCHS)

        def make_epoch_idxs(k):
            idxs = jax.random.permutation(k, N_seqs)
            idxs = idxs[: num_mbs * seqs_per_mb]
            return idxs.reshape(num_mbs, seqs_per_mb)

        all_mb_indices = jax.vmap(make_epoch_idxs)(keys_epochs).reshape(U, seqs_per_mb)

        batches_seq = PPOBatch(
            obs=batch.obs[all_mb_indices],
            time=batch.time[all_mb_indices],
            actions=batch.actions[all_mb_indices],
            logp_old=batch.logp_old[all_mb_indices],
            values_old=batch.values_old[all_mb_indices],
            returns=batch.returns[all_mb_indices],
            advantages=batch.advantages[all_mb_indices],
            az_value=batch.az_value[all_mb_indices],
            az_inter=batch.az_inter[all_mb_indices],
            gate_mask=batch.gate_mask[all_mb_indices],
            h_init=batch.h_init[all_mb_indices],
        )

        gate_params, opt_state, metrics_seq = ppo_update_many(gate_params, opt_state, batches_seq)
        metrics_avg = {k: float(jnp.mean(v)) for k, v in metrics_seq.items()}

        # Periodic evaluation (GRID) + bottom-K update
        eval_grid = None
        ranking_rows = None
        f = False
        if f:
            eval_grid = eval_suite_grid(
                env_speed=env_speed,
                forward=forward,
                select_mcts_fns=select_mcts_fns,
                model_pretrained=model_pretrained,
                gate_params=gate_params,
                seed=seed + update,
            )

            # compute bottom-K tuples
            bottom_specs_verbose = bottom_k_specs_from_eval_grid(eval_grid, k=BOTTOM_K)

            # build device tables for rollout
            bottom_specs_simple = [(k, v, bd) for (k, v, bd, _lab, _elo, _exp) in bottom_specs_verbose]
            bot_kinds_np2, bot_fixed_np2, bot_budgets_np2 = specs_to_fixed_shape_tables(
                bottom_specs_simple,
                k=BOTTOM_K,
                default_kind=OPP_SELFPLAY,
                default_fixed_opt=0,
                default_budget=default_budget,
            )
            bot_kinds = jnp.array(bot_kinds_np2, dtype=jnp.int32)
            bot_fixed_opts = jnp.array(bot_fixed_np2, dtype=jnp.int32)
            bot_budgets = jnp.array(bot_budgets_np2, dtype=jnp.int32)

            ranking_rows = rank_tuples_by_elo(eval_grid)

            # Save checkpoint
            ckpt_path = os.path.join(save_dir, f"gate_{update:06d}.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump(
                    {
                        "update": update,
                        "gate_params": gate_params,
                        "opt_state": opt_state,
                        "config": dict(wandb.config),
                        "rollout_stats": roll_stats,
                        "eval_grid": { (int(T), str(lab)): float(v) for (T, lab), v in eval_grid.items() },
                        "bottom_specs": [
                            {
                                "kind": int(k),
                                "fixed_sim": (int(v) if v is not None else None),
                                "budget": int(bd),
                                "label": str(lab),
                                "elo": float(elo),
                                "expected_score": float(exp),
                            }
                            for (k, v, bd, lab, elo, exp) in bottom_specs_verbose
                        ],
                    },
                    f,
                )

        # Periodic checkpointing based only on training progress (no eval required)
        if CHECKPOINT_INTERVAL > 0 and (update % CHECKPOINT_INTERVAL) == 0:
            ckpt_path = os.path.join(save_dir, f"gate_periodic_{update:06d}.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump(
                    {
                        "update": update,
                        "gate_params": gate_params,
                        "opt_state": opt_state,
                        "config": dict(wandb.config),
                        "rollout_stats": roll_stats,
                    },
                    f,
                )

        # WandB logging
        log_dict: Dict[str, Any] = {
            "train/loss": metrics_avg["loss"],
            "train/policy_loss": metrics_avg["policy_loss"],
            "train/value_loss": metrics_avg["value_loss"],
            "train/entropy": metrics_avg["entropy"],
            "train/approx_kl": metrics_avg["approx_kl"],
            "train/clipfrac": metrics_avg["clipfrac"],
            "train/explained_var": metrics_avg["explained_var"],
            "train/value_pred_mean": metrics_avg["value_pred_mean"],
            "train/ratio_mean": metrics_avg["ratio_mean"],
            "train/fallback_rate": metrics_avg["fallback_rate"],

            "rollout/done_rate": roll_stats["done_rate"],
            "rollout/episodes_completed": roll_stats["episodes_completed"],
            "rollout/ep_len_mean": roll_stats["ep_len_mean"],
            "rollout/ep_len_std": roll_stats["ep_len_std"],
            "rollout/p0_return_mean": roll_stats["p0_return_mean"],
            "rollout/p0_win_rate": roll_stats["p0_win_rate"],
            "rollout/p0_loss_rate": roll_stats["p0_loss_rate"],
            "rollout/fallback_rate": roll_stats["fallback_rate"],

            "schedule/phase_selfplay_only": float(phase_selfplay_only),
        }

        # Log bottom tuples (string summary) if available
        if bottom_specs_verbose:
            worst_strs = []
            for (k, v, bd, lab, elo, exp) in bottom_specs_verbose:
                worst_strs.append(f"T{bd}:{lab}@elo{elo:.1f}(p={exp:.3f})")
            log_dict["schedule/bottom15_tuples"] = " | ".join(worst_strs[:BOTTOM_K])
        else:
            log_dict["schedule/bottom15_tuples"] = ""

        # Log eval summaries if present
        if eval_grid is not None and ranking_rows is not None:
            elos = [row[2] for row in ranking_rows]
            exps = [row[3] for row in ranking_rows]
            log_dict["eval/grid_mean_expected"] = float(np.mean(exps)) if exps else 0.0
            log_dict["eval/grid_mean_elo"] = float(np.mean(elos)) if elos else 0.0
            log_dict["eval/grid_min_elo"] = float(np.min(elos)) if elos else 0.0

            # Also log per-opponent average across budgets (keeps WandB sane)
            # Compute average expected per label across budgets.
            by_lab: Dict[str, List[float]] = {}
            for (T, lab), exp in eval_grid.items():
                by_lab.setdefault(lab, []).append(exp)
            for lab, vals in by_lab.items():
                log_dict[f"eval/{lab}_mean_expected_over_budgets"] = float(np.mean(vals))

        wandb.log(log_dict, step=update)

        if update % 10 == 0:
            print(
                f"[{env_module_name} seed={seed}] upd={update} "
                f"loss={metrics_avg['loss']:.4f} "
                f"p0_ret={roll_stats['p0_return_mean']:.3f} "
                f"ep_len={roll_stats['ep_len_mean']:.1f} "
                f"fallback={roll_stats['fallback_rate']:.3f} "
                f"phase_selfonly={phase_selfplay_only} "
                f"bottomK_ready={int((update % EVAL_INTERVAL)==0)}"
            )

    # Save final checkpoint after training loop (even if eval/curriculum is disabled)
    final_ckpt_path = os.path.join(save_dir, f"gate_final_{NUM_UPDATES:06d}.pkl")
    with open(final_ckpt_path, "wb") as f:
        pickle.dump(
            {
                "update": NUM_UPDATES,
                "gate_params": gate_params,
                "opt_state": opt_state,
                "config": dict(wandb.config),
                "rollout_stats": roll_stats,
            },
            f,
        )

    wandb.finish()


if __name__ == "__main__":
    main()
