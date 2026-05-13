"""PPO Gating Policy for Adaptive MCTS Depth (TetrisRT KStep).

Trains a meta-level PPO gating policy that chooses between 4 MCTS depth options:
  K=1 (sim=32), K=2 (sim=64), K=3 (sim=96), K=4 (sim=128)

The frozen pretrained AZNet (from tetris_rt_kstep_models_4) provides:
  - MCTS recurrent function for tree search
  - Intermediate spatial features (trunk) for the gating network input

The gating policy is trained with PPO and SMDP discounting (γ^K per meta-step).

Each meta-step:
  1. Gating policy samples K ∈ {1,2,3,4}
  2. K-1 noop steps (action=5) execute in the environment
  3. MCTS with K*32 sims selects the final action
  4. Meta-reward = sum of K raw env rewards, discount = γ^K

During training, all 4 MCTS options run for every env every meta-step
(compute overhead: 4x), with the gating choice used to select the result.
"""

from __future__ import annotations

import argparse
import functools
import glob
import logging
import os
import pickle
from typing import Any, Dict, List, NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

import jumanji
from jumanji.training.agents.gumbel_alphazero import GumbelAlphaZeroAgent
from jumanji.training.agents.ppo_gating.gating_net_tetris import GatingNetTetris, GatingParamsState
from jumanji.training.training_types import AlphaZeroParamsState, TrainingState
from jumanji.wrappers import VmapAutoResetWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# K options and their MCTS sim counts
K_OPTIONS = [1, 2, 3, 4]
SIM_OPTIONS = [32, 64, 96, 128]


def _sel4(g, o0, o1, o2, o3):
    """Select one of four pytree branches with lax.switch (avoids closure capture bugs)."""
    return jax.lax.switch(g, [lambda: o0, lambda: o1, lambda: o2, lambda: o3])


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # Environment / rollout
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--meta_steps", type=int, default=256,
                   help="Meta-steps collected per rollout (each step = K real env steps)")
    p.add_argument("--eval_meta_steps", type=int, default=2000,
                   help="Max meta-steps per eval episode. Tetris time_limit=2000 gravity ticks.")
    p.add_argument("--eval_num_envs", type=int, default=10,
                   help="Number of envs for eval/baselines (each runs one complete episode via done-masking)")

    # Training
    p.add_argument("--num_epochs", type=int, default=3000)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--num_minibatches", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epsilon_clip", type=float, default=0.2)
    p.add_argument("--value_loss_coef", type=float, default=0.5)
    p.add_argument("--entropy_coef", type=float, default=0.05)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument(
        "--reward_mode",
        type=str,
        default="gamma",
        choices=["gamma", "raw", "gamma_norm", "raw_norm"],
        help=(
            "How to compute r_meta for the gating policy:\n"
            "  gamma      — γ-discounted sum across K steps (current default)\n"
            "  raw        — unweighted sum across K steps (no within-option γ)\n"
            "  gamma_norm — γ-discounted sum / K (per-step, γ-weighted)\n"
            "  raw_norm   — raw sum / K (average reward per env step; "
            "most interpretable speed-accuracy trade-off)"
        ),
    )

    # Checkpointing
    p.add_argument("--az_checkpoint_dir", type=str,
                   default="./checkpoints/committed_action/tetris_rt/base/k4")
    p.add_argument("--az_checkpoint_path", type=str, default=None,
                   help="Direct path to a specific AZNet .pkl file. Overrides az_checkpoint_dir.")
    p.add_argument("--gating_checkpoint_dir", type=str,
                   default="./checkpoints/committed_action/tetris_rt/gating")

    # Logging
    p.add_argument("--wandb_project", type=str, default="tetris_rt_gating_ppo")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true")

    p.add_argument(
        "--sim_options", type=int, nargs=4, default=[32, 64, 96, 128],
        metavar=("S1", "S2", "S3", "S4"),
        help="MCTS simulation counts for K=1,2,3,4 (e.g. --sim_options 32 64 96 128)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env_name", type=str, default="TetrisRTKStep-v0",
                   help="Jumanji env ID to use (e.g. TetrisRTKStep-v0 or TetrisRTKT-v0).")
    return p.parse_args()


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_az_checkpoint(checkpoint_dir_or_path: str) -> AlphaZeroParamsState:
    """Load an AZNet checkpoint from a directory (best/latest) or a direct .pkl path."""
    if checkpoint_dir_or_path.endswith(".pkl"):
        with open(checkpoint_dir_or_path, "rb") as f:
            state: TrainingState = pickle.load(f)
        if state is None or getattr(state, "params_state", None) is None:
            raise ValueError(f"Checkpoint '{checkpoint_dir_or_path}' is empty or missing params_state.")
        logger.info(f"Loaded AZNet checkpoint from: {checkpoint_dir_or_path}")
        return jax.tree_util.tree_map(
            lambda x: x[0] if hasattr(x, "shape") and x.ndim > 0 else x,
            state.params_state,
        )

    checkpoint_dir = checkpoint_dir_or_path
    best_path = os.path.join(checkpoint_dir, "training_state_best.pkl")

    def _find_fallback(reason: str) -> str:
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
        path = _find_fallback(f"'{best_path}' not found")
        with open(path, "rb") as f:
            state = pickle.load(f)

    if state is None or getattr(state, "params_state", None) is None:
        path = _find_fallback(f"'{path}' contains None")
        with open(path, "rb") as f:
            state = pickle.load(f)

    logger.info(f"Loaded AZNet checkpoint from: {path}")
    params_state = jax.tree_util.tree_map(lambda x: x[0] if x.ndim > 0 else x,
                                           state.params_state)
    return params_state


def save_gating_checkpoint(gating_state: GatingParamsState, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(gating_state, f)
    logger.info(f"Saved gating checkpoint: {path}")


# ── Agent setup ───────────────────────────────────────────────────────────────

def make_agents(env: VmapAutoResetWrapper, num_envs: int,
                sim_options: List[int] = None) -> List[GumbelAlphaZeroAgent]:
    """Create 4 GumbelAlphaZeroAgent instances, one per (K, sims) pair."""
    if sim_options is None:
        sim_options = SIM_OPTIONS
    agents = []
    for K, sims in zip(K_OPTIONS, sim_options):
        agent = GumbelAlphaZeroAgent(
            env=env,
            n_steps=1,               # unused in gating training (we call _k_step directly)
            total_batch_size=num_envs,
            num_simulations=sims,
            gamma=0.99,
            learning_rate=3e-4,      # unused (gating PPO has its own optimizer)
            num_channels=128,
            num_blocks=6,
            time_embed_dim=32,
            pacman_action_delay=K,
        )
        agents.append(agent)
        logger.info(f"Created agent: K={K}, sims={sims}")
    return agents


# ── Gating network forward function (Haiku transform) ────────────────────────

def make_gating_forward() -> hk.Transformed:
    """Return hk.transform'd gating forward function (no state — LayerNorm)."""
    def forward_fn(obs_grid, time_vec, az_trunk_feats, az_value):
        net = GatingNetTetris(
            num_channels=64,
            num_blocks=3,
            time_embed_dim=32,
            az_feature_dim=128,
        )
        return net(obs_grid, time_vec, az_trunk_feats, az_value)

    return hk.without_apply_rng(hk.transform(forward_fn))


def init_gating(gating_fwd: hk.Transformed, key, dummy_obs_grid, dummy_time_vec,
                dummy_az_trunk, dummy_az_value, lr: float) -> GatingParamsState:
    params = gating_fwd.init(key, dummy_obs_grid, dummy_time_vec, dummy_az_trunk, dummy_az_value)
    opt = optax.adam(lr)
    opt_state = opt.init(params)
    return GatingParamsState(params=params, opt_state=opt_state)


# ── Single meta-step for one agent (one K value) ─────────────────────────────

def meta_step_one_k(
    agent: GumbelAlphaZeroAgent,
    az_net_params: hk.Params,
    az_net_state: hk.State,
    states: Any,
    obs: Any,
    obs_grid: jnp.ndarray,    # (B, H, W, 2) — pre-computed
    time_vec: jnp.ndarray,    # (B, 2) — pre-computed
    invalid: jnp.ndarray,     # (B, A) bool — pre-computed
    key: jnp.ndarray,
    reward_mode: str = "gamma",
) -> Dict[str, jnp.ndarray]:
    """Run MCTS search + K real env steps for one K-value configuration.

    reward_mode controls how r_meta is computed:
      "gamma"      — γ-discounted sum (default, standard SMDP)
      "raw"        — unweighted sum (no within-option γ)
      "gamma_norm" — γ-discounted sum / K
      "raw_norm"   — raw sum / K  (average reward per env step)

    total_discount (γ^K) is always computed with self.gamma regardless of mode.

    Returns:
        next_states, next_obs, r_meta, discount (γ^K), done
    """
    mcts_out = agent._mcts_policy(
        raw_env=agent.raw_env_train,
        params=az_net_params,
        net_state=az_net_state,
        rng_key=key,
        root_state=states,
        root_obs_grid=obs_grid,
        root_time_vec=time_vec,
        invalid_actions=invalid,
    )
    mcts_action = jax.lax.stop_gradient(mcts_out.action)  # (B,)

    # Select within-option discount: 1.0 for raw modes, agent.gamma for gamma modes.
    within_option_discount = None if reward_mode in ("gamma", "gamma_norm") else 1.0

    next_state, next_ts, r_meta, discount, done = agent._k_step(
        agent.env.step,
        states,
        mcts_action,
        params=az_net_params,
        net_state=az_net_state,
        initial_obs=obs,
        discount_within_option=within_option_discount,
    )

    # Normalize by K for _norm modes.
    if reward_mode in ("gamma_norm", "raw_norm"):
        r_meta = r_meta / float(agent.pacman_action_delay)

    return {
        "next_states": next_state,
        "next_obs": next_ts.observation,
        "r_meta": r_meta,      # (B,)
        "discount": discount,  # (B,) — γ^K * product of env discounts
        "done": done,          # (B,) bool — True if any K step ended episode
    }


# ── Full meta-step: run all 4 K options, gate, select ───────────────────────

def single_meta_step_collect(
    agents: List[GumbelAlphaZeroAgent],
    az_params_state: AlphaZeroParamsState,
    gating_fwd: hk.Transformed,
    gating_params: hk.Params,
    states: Any,
    obs: Any,
    key: jnp.ndarray,
    reward_mode: str = "gamma",
) -> Tuple[Any, Any, Dict]:
    """One meta-step for all num_envs envs."""
    az_net_params = jax.lax.stop_gradient(az_params_state.params.net)
    az_net_state = jax.lax.stop_gradient(az_params_state.net_state)

    key, key_az, key_gating, key_mcts = jax.random.split(key, 4)

    # -- Encode observation once (shared across all K options) --
    obs_grid, time_vec = agents[0]._get_grid_and_time(states, obs)
    invalid = agents[0]._get_invalid_actions(agents[0].raw_env_train, states, obs)

    # -- Extract frozen AZNet features for gating input --
    (_, az_value_raw, az_trunk_raw), _ = agents[0].forward_with_features.apply(
        az_net_params, az_net_state, obs_grid, time_vec, is_eval=True
    )
    az_trunk = jax.lax.stop_gradient(az_trunk_raw)      # (B, 128)
    az_value = jax.lax.stop_gradient(az_value_raw[:, None])  # (B, 1)

    # -- Gating policy: sample K choice --
    gating_logits, gating_value = gating_fwd.apply(
        gating_params, obs_grid, time_vec, az_trunk, az_value
    )  # logits: (B,4), value: (B,)

    k_choices = jax.random.categorical(key_gating, gating_logits)   # (B,) int in {0,1,2,3}
    log_prob_old = jax.nn.log_softmax(gating_logits)[
        jnp.arange(obs_grid.shape[0]), k_choices
    ]  # (B,)

    # -- Run all 4 MCTS options (all B envs, all 4 K values) --
    keys_mcts = jax.random.split(key_mcts, 4)
    results = [
        meta_step_one_k(agents[i], az_net_params, az_net_state,
                        states, obs, obs_grid, time_vec, invalid, keys_mcts[i],
                        reward_mode=reward_mode)
        for i in range(4)
    ]

    # Stack scalar results: (4, B) → index by k_choices
    r_meta_all   = jnp.stack([r["r_meta"]   for r in results], axis=0)  # (4, B)
    discount_all = jnp.stack([r["discount"] for r in results], axis=0)  # (4, B)
    done_all     = jnp.stack([r["done"]     for r in results], axis=0)  # (4, B)
    B = k_choices.shape[0]
    env_idx = jnp.arange(B)
    selected_r_meta   = r_meta_all[k_choices, env_idx]   # (B,)
    selected_discount = discount_all[k_choices, env_idx]  # (B,)
    selected_done     = done_all[k_choices, env_idx]      # (B,)

    next_states = jax.vmap(_sel4)(
        k_choices,
        results[0]["next_states"], results[1]["next_states"],
        results[2]["next_states"], results[3]["next_states"],
    )
    next_obs = jax.vmap(_sel4)(
        k_choices,
        results[0]["next_obs"], results[1]["next_obs"],
        results[2]["next_obs"], results[3]["next_obs"],
    )

    transition = {
        "obs_grid": obs_grid,          # (B, H, W, 2)
        "time_vec": time_vec,           # (B, 2)
        "az_trunk": az_trunk,           # (B, 128)
        "az_value": az_value,           # (B, 1)
        "k_choices": k_choices,         # (B,) int
        "log_prob_old": log_prob_old,   # (B,)
        "r_meta": selected_r_meta,      # (B,)
        "discount": selected_discount,  # (B,) — γ^K
        "done": selected_done,          # (B,) bool
        "gating_value": gating_value,   # (B,) — V(s_t) for GAE
    }
    return next_states, next_obs, transition


# ── SMDP GAE ─────────────────────────────────────────────────────────────────

@jax.jit
def compute_smdp_gae(
    r_meta: jnp.ndarray,          # (T, B)
    discounts: jnp.ndarray,        # (T, B) — γ^{K_t} per meta-step
    dones: jnp.ndarray,            # (T, B) bool
    values: jnp.ndarray,           # (T, B) — V(s_t) from gating critic
    bootstrap_value: jnp.ndarray,  # (B,)   — V(s_T) for final bootstrap
    gae_lambda: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """SMDP GAE:
        δ_t = r_t + γ^{K_t} · V(s_{t+1}) · (1 - done_t) - V(s_t)
        A_t = δ_t + γ^{K_t} · λ · (1 - done_t) · A_{t+1}

    Returns:
        advantages   (T, B)
        value_targets (T, B)
    """
    V_next = jnp.concatenate([values[1:], bootstrap_value[None, :]], axis=0)  # (T, B)
    not_done = 1.0 - dones.astype(jnp.float32)  # (T, B)

    def scan_fn(gae_carry, t_inputs):
        gamma_k, r, V_cur, V_n, nd = t_inputs
        delta = r + gamma_k * V_n * nd - V_cur
        new_gae = delta + gamma_k * gae_lambda * nd * gae_carry
        return new_gae, new_gae

    inputs = (discounts, r_meta, values, V_next, not_done)  # each (T, B)
    inputs_rev = jax.tree_util.tree_map(lambda x: x[::-1], inputs)
    init_gae = jnp.zeros_like(bootstrap_value)
    _, advantages_rev = jax.lax.scan(scan_fn, init_gae, inputs_rev)
    advantages = advantages_rev[::-1]  # (T, B)

    value_targets = advantages + values  # (T, B)
    return advantages, value_targets


# ── PPO loss and update ───────────────────────────────────────────────────────

def ppo_loss_fn(
    gating_params: hk.Params,
    gating_fwd: hk.Transformed,
    batch: Dict,
    epsilon_clip: float,
    value_loss_coef: float,
    entropy_coef: float,
) -> Tuple[jnp.ndarray, Dict]:
    """PPO loss for a mini-batch."""
    obs_grid    = batch["obs_grid"]      # (MB, H, W, 2)
    time_vec    = batch["time_vec"]      # (MB, 2)
    az_trunk    = batch["az_trunk"]      # (MB, 128)
    az_value    = batch["az_value"]      # (MB, 1)
    k_choices   = batch["k_choices"]     # (MB,)
    log_prob_old = batch["log_prob_old"] # (MB,)
    advantages  = batch["advantages"]    # (MB,)
    v_targets   = batch["v_targets"]     # (MB,)

    gating_logits, values_pred = gating_fwd.apply(
        gating_params, obs_grid, time_vec, az_trunk, az_value
    )  # (MB, 4), (MB,)

    log_probs_all = jax.nn.log_softmax(gating_logits, axis=-1)  # (MB, 4)
    log_prob_new  = log_probs_all[jnp.arange(k_choices.shape[0]), k_choices]  # (MB,)

    ratio = jnp.exp(log_prob_new - log_prob_old)  # (MB,)

    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    clip_ratio = jnp.clip(ratio, 1.0 - epsilon_clip, 1.0 + epsilon_clip)
    policy_loss = -jnp.mean(jnp.minimum(ratio * adv, clip_ratio * adv))

    value_loss = 0.5 * jnp.mean((values_pred - v_targets) ** 2)

    probs = jax.nn.softmax(gating_logits, axis=-1)  # (MB, 4)
    entropy = -jnp.mean(jnp.sum(probs * log_probs_all, axis=-1))

    total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "total_loss": total_loss,
        "mean_ratio": jnp.mean(ratio),
    }
    return total_loss, metrics


def make_ppo_update_step(
    optimizer: optax.GradientTransformation,
    gating_fwd: hk.Transformed,
    epsilon_clip: float,
    value_loss_coef: float,
    entropy_coef: float,
):
    """Return a JIT-compiled single-step PPO update."""
    @jax.jit
    def _step(
        gating_params: hk.Params,
        opt_state: optax.OptState,
        batch: Dict,
    ) -> Tuple[hk.Params, optax.OptState, Dict]:
        loss_fn = lambda p: ppo_loss_fn(
            p, gating_fwd, batch, epsilon_clip, value_loss_coef, entropy_coef
        )
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(gating_params)
        updates, new_opt_state = optimizer.update(grads, opt_state, gating_params)
        new_params = optax.apply_updates(gating_params, updates)
        return new_params, new_opt_state, metrics
    return _step


def ppo_epoch_update(
    gating_state: GatingParamsState,
    update_step_fn,
    data: Dict,
    advantages: jnp.ndarray,
    value_targets: jnp.ndarray,
    ppo_epochs: int,
    num_minibatches: int,
    key: jnp.ndarray,
) -> Tuple[GatingParamsState, Dict]:
    """Run ppo_epochs × num_minibatches gradient steps."""
    T, B = advantages.shape
    total_n = T * B
    mb_size = total_n // num_minibatches

    flat_data = jax.tree_util.tree_map(lambda x: x.reshape((total_n,) + x.shape[2:]), data)
    flat_data["advantages"] = advantages.reshape((total_n,))
    flat_data["v_targets"]  = value_targets.reshape((total_n,))

    params = gating_state.params
    opt_state = gating_state.opt_state
    all_metrics: List[Dict] = []

    for _ in range(ppo_epochs):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, total_n)
        shuffled = jax.tree_util.tree_map(lambda x: x[perm], flat_data)

        for mb_idx in range(num_minibatches):
            sl = slice(mb_idx * mb_size, (mb_idx + 1) * mb_size)
            mb = jax.tree_util.tree_map(lambda x: x[sl], shuffled)
            params, opt_state, metrics = update_step_fn(params, opt_state, mb)
            all_metrics.append(metrics)

    mean_metrics = {
        k: float(np.mean([m[k] for m in all_metrics]))
        for k in all_metrics[0]
    }
    new_state = GatingParamsState(params=params, opt_state=opt_state)
    return new_state, mean_metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    key = jax.random.PRNGKey(args.seed)

    logger.info(f"JAX devices: {jax.devices()}")

    # ── Resolve AZNet checkpoint path and epoch tag ──
    import re as _re
    if args.az_checkpoint_path:
        _ckpt_path = args.az_checkpoint_path
    else:
        _ckpt_path = os.path.join(args.az_checkpoint_dir, "training_state_best.pkl")
        if not os.path.exists(_ckpt_path):
            _epoch_files = sorted(glob.glob(
                os.path.join(args.az_checkpoint_dir, "training_state_epoch_*.pkl")
            ))
            _ckpt_path = _epoch_files[-1] if _epoch_files else _ckpt_path
    _m = _re.search(r"epoch_(\d+)", _ckpt_path)
    az_epoch_tag = f"ep{int(_m.group(1)):03d}" if _m else "epBest"

    sims_str = "-".join(str(s) for s in args.sim_options)
    args.gating_checkpoint_dir = os.path.join(
        args.gating_checkpoint_dir, args.reward_mode,
        f"sims_{sims_str}", az_epoch_tag, f"seed_{args.seed}"
    )
    os.makedirs(args.gating_checkpoint_dir, exist_ok=True)

    # ── W&B ──
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"gating_{az_epoch_tag}_sims{sims_str}_seed{args.seed}",
        )

    # ── Environment ──
    raw_env = jumanji.make(args.env_name)
    env = VmapAutoResetWrapper(raw_env)

    # ── Agents (4 K values) ──
    agents = make_agents(env, args.num_envs, args.sim_options)

    # ── Load frozen AZNet ──
    logger.info(f"Loading frozen AZNet from: {_ckpt_path}")
    az_params_state = load_az_checkpoint(_ckpt_path)
    az_params_state = jax.device_put(az_params_state)
    logger.info("AZNet loaded and frozen.")

    # ── Infer Tetris board dims from env spec ──
    _dummy_obs = raw_env.observation_spec.generate_value()
    _board_shape = _dummy_obs.board.shape  # (num_rows, num_cols)
    H, W = _board_shape[0], _board_shape[1]
    logger.info(f"TetrisRT board: H={H}, W={W}, obs_grid shape: (B, {H}, {W}, 2)")

    # ── Gating network init ──
    gating_fwd = make_gating_forward()

    key, init_key = jax.random.split(key)
    dummy_grid  = jnp.zeros((1, H, W, 2))
    dummy_time  = jnp.zeros((1, 2))
    dummy_trunk = jnp.zeros((1, 128))
    dummy_azval = jnp.zeros((1, 1))

    gating_state = init_gating(gating_fwd, init_key, dummy_grid, dummy_time,
                               dummy_trunk, dummy_azval, args.lr)
    optimizer = optax.adam(args.lr)

    ppo_update_step_fn = make_ppo_update_step(
        optimizer=optimizer,
        gating_fwd=gating_fwd,
        epsilon_clip=args.epsilon_clip,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
    )
    logger.info("Gating network initialized.")

    # ── JIT the entire rollout as a single lax.scan call ──
    @jax.jit
    def collect_rollout(gating_params, init_states, init_obs, key):
        step_keys = jax.random.split(key, args.meta_steps)

        def scan_body(carry, step_key):
            states, obs = carry
            next_states, next_obs, trans = single_meta_step_collect(
                agents, az_params_state, gating_fwd, gating_params, states, obs, step_key,
                reward_mode=args.reward_mode,
            )
            return (next_states, next_obs), trans

        (final_states, final_obs), data = jax.lax.scan(
            scan_body, (init_states, init_obs), step_keys
        )
        return final_states, final_obs, data

    # ── JIT-compiled eval ──
    @functools.partial(jax.jit, static_argnames=("greedy", "random_policy", "force_k"))
    def jit_eval(gating_params, init_states, init_obs, key, *, greedy, random_policy, force_k=-1):
        """Run eval_meta_steps meta-steps; accumulate per-env returns until first done.

        force_k: if >= 0, always use that K option (0-indexed). Used for fixed-K baselines.
        Always accumulates raw (undiscounted) env rewards so eval metrics are comparable
        across different training runs with different reward_mode values.
        """
        az_net_params = jax.lax.stop_gradient(az_params_state.params.net)
        az_net_state  = jax.lax.stop_gradient(az_params_state.net_state)
        B = args.eval_num_envs

        def scan_body(carry, step_key):
            states, obs, raw_ret, disc_ret, cum_disc, done, k_cnt = carry

            obs_grid, time_vec = agents[0]._get_grid_and_time(states, obs)
            invalid = agents[0]._get_invalid_actions(agents[0].raw_env_train, states, obs)

            (_, az_val_raw, az_trk_raw), _ = agents[0].forward_with_features.apply(
                az_net_params, az_net_state, obs_grid, time_vec, is_eval=True
            )
            az_trk = jax.lax.stop_gradient(az_trk_raw)
            az_val = jax.lax.stop_gradient(az_val_raw[:, None])

            if force_k >= 0:
                k_choices = jnp.full((B,), force_k, dtype=jnp.int32)
            elif random_policy:
                k_choices = jax.random.randint(step_key, (B,), 0, 4)
            else:
                logits, _ = gating_fwd.apply(gating_params, obs_grid, time_vec, az_trk, az_val)
                if greedy:
                    k_choices = jnp.argmax(logits, axis=-1)
                else:
                    k_choices = jax.random.categorical(step_key, logits)

            keys_m = jax.random.split(step_key, 4)
            results = [
                meta_step_one_k(agents[i], az_net_params, az_net_state,
                                states, obs, obs_grid, time_vec, invalid, keys_m[i],
                                reward_mode="raw")
                for i in range(4)
            ]

            r_all  = jnp.stack([r["r_meta"]   for r in results], axis=0)  # (4, B)
            d_all  = jnp.stack([r["discount"]  for r in results], axis=0)  # (4, B)
            dn_all = jnp.stack([r["done"]      for r in results], axis=0)  # (4, B)
            arange_B = jnp.arange(B)
            sel_r  = r_all[k_choices, arange_B]
            sel_d  = d_all[k_choices, arange_B]
            sel_dn = dn_all[k_choices, arange_B]

            next_states = jax.vmap(_sel4)(
                k_choices,
                results[0]["next_states"], results[1]["next_states"],
                results[2]["next_states"], results[3]["next_states"],
            )
            next_obs = jax.vmap(_sel4)(
                k_choices,
                results[0]["next_obs"], results[1]["next_obs"],
                results[2]["next_obs"], results[3]["next_obs"],
            )

            nd = (~done).astype(jnp.float32)
            raw_ret  = raw_ret  + sel_r * nd
            disc_ret = disc_ret + sel_r * cum_disc * nd
            cum_disc = cum_disc * sel_d * nd
            k_cnt    = k_cnt.at[arange_B, k_choices].add((~done).astype(jnp.int32))
            done     = done | sel_dn

            return (next_states, next_obs, raw_ret, disc_ret, cum_disc, done, k_cnt), None

        init_carry = (
            init_states, init_obs,
            jnp.zeros(B), jnp.zeros(B), jnp.ones(B),
            jnp.zeros(B, dtype=bool),
            jnp.zeros((B, 4), dtype=jnp.int32),
        )
        step_keys = jax.random.split(key, args.eval_meta_steps)
        (_, _, raw_ret, disc_ret, _, _, k_cnt), _ = jax.lax.scan(
            scan_body, init_carry, step_keys
        )
        return raw_ret, disc_ret, k_cnt

    # ── Compute fixed baselines ONCE (pre-training) ──────────────────────────
    logger.info("Computing fixed baselines (once, pre-training)...")
    _bkey = jax.random.PRNGKey(args.seed + 9999)
    _benv = VmapAutoResetWrapper(jumanji.make("TetrisRTKStep-v0"))
    _bstates, _bts = _benv.reset(jax.random.split(_bkey, args.eval_num_envs))
    _bobs = _bts.observation

    baseline_metrics = {}
    for _k_idx in range(4):
        _raw_b, _, _ = jit_eval(
            gating_state.params, _bstates, _bobs, _bkey,
            greedy=True, random_policy=False, force_k=_k_idx,
        )
        baseline_metrics[f"baseline/always_k{_k_idx + 1}_raw"] = float(jnp.mean(_raw_b))

    _raw_rnd, _, _ = jit_eval(
        gating_state.params, _bstates, _bobs, _bkey,
        greedy=True, random_policy=True, force_k=-1,
    )
    baseline_metrics["baseline/random_raw"] = float(jnp.mean(_raw_rnd))

    logger.info(f"Baselines: { {k: f'{v:.1f}' for k, v in baseline_metrics.items()} }")
    if not args.no_wandb:
        for k, v in baseline_metrics.items():
            wandb.run.summary[k] = v

    # ── Initial env state ──
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, args.num_envs)
    states, timesteps = env.reset(reset_keys)
    obs = timesteps.observation

    best_raw_return = float("-inf")
    cur_states, cur_obs = states, obs
    logger.info("Starting PPO gating training...")

    for epoch in range(args.num_epochs):
        # ── Collect T meta-steps via a single lax.scan call ──
        key, collect_key = jax.random.split(key)
        cur_states, cur_obs, data = collect_rollout(
            gating_state.params, cur_states, cur_obs, collect_key
        )

        # Bootstrap value for final state
        key, feat_key = jax.random.split(key)
        obs_grid_final, time_vec_final = agents[0]._get_grid_and_time(cur_states, cur_obs)
        az_net_params = jax.lax.stop_gradient(az_params_state.params.net)
        az_net_state  = jax.lax.stop_gradient(az_params_state.net_state)
        (_, az_val_final, az_trunk_final), _ = agents[0].forward_with_features.apply(
            az_net_params, az_net_state, obs_grid_final, time_vec_final, is_eval=True
        )
        _, bootstrap_value = gating_fwd.apply(
            gating_state.params,
            obs_grid_final, time_vec_final,
            jax.lax.stop_gradient(az_trunk_final),
            jax.lax.stop_gradient(az_val_final[:, None]),
        )  # (num_envs,)

        # ── SMDP GAE ──
        advantages, value_targets = compute_smdp_gae(
            r_meta=data["r_meta"],
            discounts=data["discount"],
            dones=data["done"],
            values=data["gating_value"],
            bootstrap_value=jax.lax.stop_gradient(bootstrap_value),
            gae_lambda=args.gae_lambda,
        )

        # ── PPO update ──
        key, ppo_key = jax.random.split(key)
        gating_state, ppo_metrics = ppo_epoch_update(
            gating_state=gating_state,
            update_step_fn=ppo_update_step_fn,
            data=data,
            advantages=advantages,
            value_targets=value_targets,
            ppo_epochs=args.ppo_epochs,
            num_minibatches=args.num_minibatches,
            key=ppo_key,
        )

        # ── Training metrics ──
        train_metrics = {
            "train/mean_raw_reward":   float(jnp.mean(data["r_meta"])),
            "train/mean_discount":     float(jnp.mean(data["discount"])),
            "train/done_rate":         float(jnp.mean(data["done"].astype(jnp.float32))),
            "train/mean_k_choice":     float(jnp.mean(data["k_choices"].astype(jnp.float32)) + 1),
            **{f"train/{k}": v for k, v in ppo_metrics.items()},
        }
        k_hist = np.array(jnp.mean(
            jax.nn.one_hot(data["k_choices"], 4).mean(axis=0), axis=0
        ))
        for k_idx, freq in enumerate(k_hist):
            train_metrics[f"train/k{k_idx+1}_freq"] = float(freq)

        # ── Eval ──
        eval_metrics = {}
        if epoch % args.eval_every == 0:
            key, eval_key = jax.random.split(key)
            eval_env = VmapAutoResetWrapper(jumanji.make("TetrisRTKStep-v0"))
            es, ets = eval_env.reset(jax.random.split(eval_key, args.eval_num_envs))

            raw_g, disc_g, k_cnt_g = jit_eval(
                gating_state.params, es, ets.observation, eval_key,
                greedy=True, random_policy=False, force_k=-1,
            )
            raw_g_mean  = float(jnp.mean(raw_g))
            disc_g_mean = float(jnp.mean(disc_g))
            k_dist_raw  = jnp.mean(k_cnt_g.astype(jnp.float32), axis=0)
            k_dist_pct  = np.array(k_dist_raw / (k_dist_raw.sum() + 1e-8))
            eval_metrics.update({
                "eval_gating/raw_episode_return":        raw_g_mean,
                "eval_gating/discounted_episode_return": disc_g_mean,
                **{f"eval_gating/k{i+1}_pct": float(k_dist_pct[i]) for i in range(4)},
            })

            logger.info(
                f"Epoch {epoch:4d} | "
                f"gating_raw={raw_g_mean:.2f} | "
                f"K_pct={np.round(k_dist_pct * 100, 1)}"
            )

            if raw_g_mean > best_raw_return:
                best_raw_return = raw_g_mean
                save_gating_checkpoint(
                    gating_state,
                    os.path.join(args.gating_checkpoint_dir, "gating_state_best.pkl"),
                )
            save_gating_checkpoint(
                gating_state,
                os.path.join(args.gating_checkpoint_dir, f"gating_state_epoch_{epoch:04d}.pkl"),
            )

        # ── Log to wandb ──
        if not args.no_wandb:
            wandb.log({**train_metrics, **eval_metrics, "epoch": epoch})

    # Final checkpoint
    save_gating_checkpoint(
        gating_state,
        os.path.join(args.gating_checkpoint_dir, "gating_state_final.pkl"),
    )
    if not args.no_wandb:
        wandb.finish()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
