# Copyright 2026
# (use repo header format / license header)

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import optax

from jumanji.env import Environment
from jumanji.training.agents.base import Agent
from jumanji.training.training_types import (
    ActingState,
    TrainingState,
    AlphaZeroParams,
    AlphaZeroParamsState,
)

# ----------------------------
# AlphaZero-style network
# ----------------------------

class _ResBlockV2(hk.Module):
    def __init__(self, num_channels: int, name: str = "resblock_v2"):
        super().__init__(name=name)
        self.num_channels = num_channels

    def __call__(self, x: jnp.ndarray, is_training: bool, test_local_stats: bool) -> jnp.ndarray:
        i = x
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        return x + i


class AZNet(hk.Module):
    """AlphaZero policy/value heads for grid input (B,H,W,C) + time features (B,2)."""

    def __init__(
        self,
        num_actions: int,
        num_channels: int = 128,
        num_blocks: int = 6,
        time_embed_dim: int = 32,
        name: str = "az_net",
    ):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.time_embed_dim = time_embed_dim

    def __call__(
        self,
        x: jnp.ndarray,
        time_vec: jnp.ndarray,
        is_training: bool,
        test_local_stats: bool,
        return_features: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            x: (B,H,W,C)
            time_vec: (B,2) where [:,0]=time_seen_frac and [:,1]=time_left_frac
            return_features: if True, also return global-avg-pooled trunk features (B, num_channels)
        """
        x = x.astype(jnp.float32)
        time_vec = time_vec.astype(jnp.float32)

        # Spatial trunk
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)

        for i in range(self.num_blocks):
            x = _ResBlockV2(self.num_channels, name=f"block_{i}")(
                x, is_training, test_local_stats
            )

        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)

        # Time embedding: (B,2) -> (B,num_channels)
        t = hk.Linear(self.time_embed_dim)(time_vec)
        t = jax.nn.relu(t)
        t = hk.Linear(self.num_channels)(t)
        t = jax.nn.relu(t)

        # Broadcast time embedding across spatial dimensions and fuse into trunk
        t = t[:, None, None, :]  # (B,1,1,num_channels)
        x = x + t
        # x is now (B,H,W,num_channels) — the full trunk output after time fusion.
        # Intercepted here for return_features before the policy/value heads.

        # Policy head
        p = hk.Conv2D(output_channels=2, kernel_shape=1)(x)
        p = hk.BatchNorm(True, True, 0.9)(p, is_training, test_local_stats)
        p = jax.nn.relu(p)
        p = hk.Flatten()(p)
        logits = hk.Linear(self.num_actions)(p)

        # Value head
        v = hk.Conv2D(output_channels=1, kernel_shape=1)(x)
        v = hk.BatchNorm(True, True, 0.9)(v, is_training, test_local_stats)
        v = jax.nn.relu(v)
        v = hk.Flatten()(v)
        v = hk.Linear(self.num_channels)(v)
        v = jax.nn.relu(v)
        v = hk.Linear(1)(v)
        v = jnp.tanh(v).reshape((-1,))

        if return_features:
            trunk_pooled = x.mean(axis=(1, 2))  # (B, num_channels)
            return logits, v, trunk_pooled
        return logits, v


# ----------------------------
# Types
# ----------------------------

class AZRollout(NamedTuple):
    obs_grid: jnp.ndarray        # (T,B,H,W,C)
    time_vec: jnp.ndarray        # (T,B,2)
    reward: jnp.ndarray          # (T,B)
    discount: jnp.ndarray        # (T,B)
    terminated: jnp.ndarray      # (T,B)
    policy_tgt: jnp.ndarray      # (T,B,A)


class AZBatch(NamedTuple):
    obs_grid: jnp.ndarray        # (T,B,H,W,C)
    time_vec: jnp.ndarray        # (T,B,2)
    policy_tgt: jnp.ndarray      # (T,B,A)
    value_tgt: jnp.ndarray       # (T,B)
    value_mask: jnp.ndarray      # (T,B)


# ----------------------------
# Agent
# ----------------------------

class GumbelAlphaZeroAgent(Agent):
    """
    Gumbel AlphaZero using mctx.gumbel_muzero_policy.

    Supports Sokoban, PacMan, Tetris, and TetrisRT environments. Env-specific observation
    encoding and invalid-action detection are dispatched via self._is_pacman /
    self._is_tetris / self._is_tetris_rt flags.

    Sokoban assumptions:
      - observation has .grid (H,W,C) and .step_count
      - invalid actions detected via env.detect_noop_action

    PacMan assumptions:
      - observation has scattered fields; encoded as (31,28,6) multi-channel grid
      - step_count read from state.step_count (not in observation)
      - invalid actions read directly from observation.action_mask

    Tetris assumptions:
      - action space is MultiDiscreteArray([4, num_cols]); flattened to 4*num_cols ints
        inside MCTS, unflattened to [rotation, x_pos] before env.step
      - observation encoded as (num_rows, num_cols, 3): board, piece, action heatmap
      - step_count read from state.step_count (obs.step_count may always be 0)
      - invalid actions from obs.action_mask (4, num_cols), flattened to (4*num_cols,)
    """

    def __init__(
        self,
        env: Environment,
        n_steps: int,
        total_batch_size: int,
        num_simulations: int = 32,
        gamma: float = 0.997,
        learning_rate: float = 1e-3,
        num_channels: int = 128,
        num_blocks: int = 6,
        time_embed_dim: int = 32,
        policy_loss_weight: float = 1.0,
        value_loss_weight: float = 1.0,
        gumbel_scale: float = 1.0,
        pacman_action_delay: int = 1,
    ):
        super().__init__(total_batch_size=total_batch_size)
        self.env = env
        self.raw_env_train = env.unwrapped  # used for training-time MCTS rollouts
        self.observation_spec = env.observation_spec
        self.time_limit = float(self.raw_env_train.time_limit)

        self.n_steps = int(n_steps)
        self.num_simulations = int(num_simulations)
        self.gamma = float(gamma)
        self.policy_loss_weight = float(policy_loss_weight)
        self.value_loss_weight = float(value_loss_weight)
        self.gumbel_scale = float(gumbel_scale)
        self.time_embed_dim = int(time_embed_dim)
        self.pacman_action_delay = int(pacman_action_delay)

        # Detect environment type FIRST (needed before num_actions for Tetris)
        from jumanji.environments.routing.pac_man import PacMan as _PacMan
        from jumanji.environments.routing.pac_man import PacManKT as _PacManKT
        from jumanji.environments.packing.tetris import Tetris as _Tetris
        from jumanji.environments.packing.tetris_rt import TetrisRT as _TetrisRT
        from jumanji.environments.packing.tetris_rt import TetrisRTKStep as _TetrisRTKStep
        from jumanji.environments.packing.tetris_rt import TetrisRTKT as _TetrisRTKT
        from jumanji.environments.routing.snake.env import Snake as _Snake
        from jumanji.environments.routing.snake.env import SnakeKT as _SnakeKT
        from jumanji.environments.packing.flat_pack_clock import FlatPackClock as _FlatPackClock
        # PacManKT subclasses PacMan — check it first and exclude from plain check.
        self._is_pacman_kt = isinstance(self.raw_env_train, _PacManKT)
        self._is_pacman    = isinstance(self.raw_env_train, _PacMan) and not self._is_pacman_kt
        self._is_snake_kt = isinstance(self.raw_env_train, _SnakeKT)
        self._is_snake    = isinstance(self.raw_env_train, _Snake) and not self._is_snake_kt
        self._is_tetris = isinstance(self.raw_env_train, _Tetris)
        # Check subclasses first; plain TetrisRT must exclude them.
        # TetrisRTKT subclasses TetrisRT (not TetrisRTKStep), so check independently.
        self._is_tetris_rt_kt    = isinstance(self.raw_env_train, _TetrisRTKT)
        self._is_tetris_rt_kstep = isinstance(self.raw_env_train, _TetrisRTKStep) and not self._is_tetris_rt_kt
        self._is_tetris_rt = (
            isinstance(self.raw_env_train, _TetrisRT)
            and not self._is_tetris_rt_kstep
            and not self._is_tetris_rt_kt
        )
        self._is_flat_pack_clock = isinstance(self.raw_env_train, _FlatPackClock)

        # Tetris uses MultiDiscreteArray([4, num_cols]) — flatten to single int action space.
        # TetrisRT uses DiscreteArray(6) — 6 flat actions.
        # FlatPackClock uses MultiDiscreteArray([num_blocks,4,inner,inner]) — flattened.
        # int(getattr(..., "num_values", 4)) would fail on [4, num_cols], so handle explicitly.
        if self._is_tetris:
            self.num_cols = int(self.raw_env_train.num_cols)
            self.num_actions = 4 * self.num_cols       # e.g. 40 for default 10-col grid
        elif self._is_tetris_rt or self._is_tetris_rt_kstep or self._is_tetris_rt_kt:
            self.num_actions = 6
        elif self._is_flat_pack_clock:
            _fp = self.raw_env_train
            self._fp_num_blocks = int(_fp.num_blocks)   # 9 for 3×3 board
            self._fp_inner = int(_fp.num_rows) - 2      # 5 for 7×7 grid
            self.num_actions = self._fp_num_blocks * 4 * self._fp_inner * self._fp_inner  # 900
        else:
            self.num_actions = int(getattr(env.action_spec, "num_values", 4))

        def forward_fn(
            obs_grid: jnp.ndarray,
            time_vec: jnp.ndarray,
            is_eval: bool = False,
        ):
            net = AZNet(
                num_actions=self.num_actions,
                num_channels=num_channels,
                num_blocks=num_blocks,
                time_embed_dim=time_embed_dim,
            )
            return net(
                obs_grid,
                time_vec,
                is_training=not is_eval,
                test_local_stats=False,
            )

        self.forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

        # Same network, same name="az_net" (default), but also returns global-avg-pooled
        # trunk features (B, num_channels) before the policy/value heads.
        # forward_with_features.apply(pretrained_params.net, pretrained_net_state, ...)
        # works with the same checkpoint weights as self.forward.
        def forward_with_features_fn(
            obs_grid: jnp.ndarray,
            time_vec: jnp.ndarray,
            is_eval: bool = False,
        ):
            net = AZNet(
                num_actions=self.num_actions,
                num_channels=num_channels,
                num_blocks=num_blocks,
                time_embed_dim=time_embed_dim,
            )
            return net(
                obs_grid,
                time_vec,
                is_training=not is_eval,
                test_local_stats=False,
                return_features=True,
            )

        self.forward_with_features = hk.without_apply_rng(
            hk.transform_with_state(forward_with_features_fn)
        )
        self.optimizer = optax.adam(learning_rate)

    # ---- Agent API ----

    def init_params(self, key: chex.PRNGKey) -> AlphaZeroParamsState:
        dummy_obs = self.observation_spec.generate_value()

        if self._is_pacman or self._is_pacman_kt:
            dummy_grid = self._encode_obs_pacman_unbatched(dummy_obs)[None, ...]  # (1,31,28,6)
            dummy_step_count = jnp.array([0], jnp.int32)                          # (1,)
        elif self._is_tetris:
            dummy_grid = self._encode_obs_tetris_unbatched(dummy_obs)[None, ...]  # (1,H,W,3)
            dummy_step_count = jnp.array([0], jnp.int32)                          # (1,)
        elif self._is_tetris_rt or self._is_tetris_rt_kstep or self._is_tetris_rt_kt:
            dummy_grid = self._encode_obs_tetris_rt_unbatched(dummy_obs)[None, ...]  # (1,H,W,2)
            dummy_step_count = jnp.array([0], jnp.int32)                             # (1,)
        elif self._is_flat_pack_clock:
            _fp = self.raw_env_train
            dummy_grid = jnp.zeros((1, _fp.num_rows, _fp.num_cols, 1), jnp.float32)  # (1,7,7,1)
            dummy_step_count = jnp.array([0], jnp.int32)
        else:
            dummy_grid = getattr(dummy_obs, "grid", dummy_obs)[None, ...]         # (1,H,W,C)
            dummy_step_count = getattr(dummy_obs, "step_count", jnp.array(0, jnp.int32))[None]

        if self._is_flat_pack_clock:
            dummy_time_left = jnp.array([self.time_limit], jnp.float32)
            dummy_time_vec = jnp.stack(
                [jnp.zeros(1), dummy_time_left / self.time_limit], axis=-1
            )  # (1,2)
        else:
            dummy_time_vec = self._time_features(dummy_step_count)  # (1,2)

        params_key, _ = jax.random.split(key)
        params, net_state = self.forward.init(
            params_key,
            dummy_grid,
            dummy_time_vec,
            is_eval=False,
        )

        opt_state = self.optimizer.init(params)

        return AlphaZeroParamsState(
            params=AlphaZeroParams(net=params),
            net_state=net_state,
            opt_state=opt_state,
            update_count=jnp.array(0, jnp.int32),
        )

    def run_epoch(self, training_state: TrainingState) -> Tuple[TrainingState, Dict]:
        if not isinstance(training_state.params_state, AlphaZeroParamsState):
            raise TypeError(
                "Expected params_state to be AlphaZeroParamsState, got "
                f"{type(training_state.params_state)}."
            )

        ps: AlphaZeroParamsState = training_state.params_state

        grads, (new_acting_state, metrics, new_net_state) = jax.grad(self._loss, has_aux=True)(
            ps.params.net,
            ps.net_state,
            training_state.acting_state,
        )

        grads, metrics = jax.lax.pmean((grads, metrics), axis_name="devices")
        updates, new_opt_state = self.optimizer.update(grads, ps.opt_state)
        new_net_params = optax.apply_updates(ps.params.net, updates)

        new_ps = AlphaZeroParamsState(
            params=AlphaZeroParams(net=new_net_params),
            net_state=new_net_state,
            opt_state=new_opt_state,
            update_count=ps.update_count + 1,
        )

        new_ts = TrainingState(params_state=new_ps, acting_state=new_acting_state)
        return new_ts, metrics

    def make_policy(
        self,
        params_state: AlphaZeroParamsState,
        stochastic: bool = True,
        eval_env: Optional[Environment] = None,
    ) -> Callable:
        """
        Evaluator-compatible policy factory.

        params_state must be AlphaZeroParamsState so we can access:
          - params_state.params.net  (hk.Params)
          - params_state.net_state   (hk.State)  [BatchNorm running stats]
        """
        if not isinstance(params_state, AlphaZeroParamsState):
            raise TypeError(f"Expected AlphaZeroParamsState, got {type(params_state)}")

        policy_params = params_state.params.net
        net_state = params_state.net_state

        # -------------------------
        # Evaluation with MCTS
        # -------------------------
        if eval_env is not None:
            raw_eval_env = eval_env.unwrapped

            def policy(state: Any, observation: Any, key: chex.PRNGKey) -> chex.Array:
                # Batch the single observation and state
                obs_b = jax.tree_util.tree_map(lambda x: x[None, ...], observation)
                state_b = jax.tree_util.tree_map(lambda x: x[None, ...], state)

                obs_grid_b, time_vec_b = self._get_grid_and_time(state_b, obs_b)
                invalid = self._get_invalid_actions(raw_eval_env, state_b, obs_b)

                out = self._mcts_policy(
                    raw_env=raw_eval_env,
                    params=policy_params,
                    net_state=net_state,
                    rng_key=key,
                    root_state=state_b,
                    root_obs_grid=obs_grid_b,
                    root_time_vec=time_vec_b,
                    invalid_actions=invalid,
                )
                return self._prepare_action(out.action[0])

            return policy

        # -------------------------
        # Net-only fallback (no MCTS)
        # -------------------------
        def net_policy(observation: Any, key: chex.PRNGKey):
            if self._is_pacman or self._is_pacman_kt:
                if observation.grid.ndim == 2:
                    grid_b = self._encode_obs_pacman_unbatched(observation)[None, ...]
                    time_b = self._time_features(jnp.array([0], jnp.int32))
                    unbatched = True
                else:
                    grid_b = self._encode_obs_pacman(observation)
                    time_b = self._time_features(jnp.zeros(observation.grid.shape[0], jnp.int32))
                    unbatched = False
            elif self._is_tetris:
                if observation.grid.ndim == 2:
                    grid_b = self._encode_obs_tetris_unbatched(observation)[None, ...]
                    time_b = self._time_features(jnp.array([0], jnp.int32))
                    unbatched = True
                else:
                    grid_b = self._encode_obs_tetris(observation)
                    time_b = self._time_features(jnp.zeros(observation.grid.shape[0], jnp.int32))
                    unbatched = False
            elif self._is_tetris_rt or self._is_tetris_rt_kstep or self._is_tetris_rt_kt:
                if observation.board.ndim == 2:
                    grid_b = self._encode_obs_tetris_rt_unbatched(observation)[None, ...]
                    time_b = self._time_features(jnp.array([0], jnp.int32))
                    unbatched = True
                else:
                    grid_b = self._encode_obs_tetris_rt(observation)
                    time_b = self._time_features(observation.step_count)
                    unbatched = False
            elif self._is_flat_pack_clock:
                if observation.grid.ndim == 2:
                    grid_b = (observation.grid[None, :, :, None].astype(jnp.float32)
                              / jnp.float32(self._fp_num_blocks))
                    tl = observation.time_left[None].astype(jnp.float32)
                    tf = jnp.clip(tl / self.time_limit, 0.0, 1.0)
                    time_b = jnp.stack([1.0 - tf, tf], axis=-1)
                    unbatched = True
                else:
                    grid_b = (observation.grid[:, :, :, None].astype(jnp.float32)
                              / jnp.float32(self._fp_num_blocks))
                    tl = observation.time_left.astype(jnp.float32)
                    tf = jnp.clip(tl / self.time_limit, 0.0, 1.0)
                    time_b = jnp.stack([1.0 - tf, tf], axis=-1)
                    unbatched = False
            else:
                grid = getattr(observation, "grid", observation)
                step_count = getattr(observation, "step_count")
                if grid.ndim == 3:
                    grid_b = grid[None, ...]
                    time_b = self._time_features(step_count[None])
                    unbatched = True
                else:
                    grid_b = grid
                    time_b = self._time_features(step_count)
                    unbatched = False

            (logits, _), _ = self.forward.apply(
                policy_params,
                net_state,
                grid_b,
                time_b,
                is_eval=True,
            )

            # Remove batch dim if we added it
            if unbatched:
                logits = logits[0]

            if stochastic:
                action = jax.random.categorical(key, logits, axis=-1)
            else:
                action = jnp.argmax(logits, axis=-1)

            log_prob = jnp.zeros_like(action, dtype=jnp.float32)
            return action, (log_prob, logits)

        return net_policy


    # ---- Loss ----

    def _loss(
        self,
        params: hk.Params,
        net_state: hk.State,
        acting_state: ActingState,
    ) -> Tuple[jnp.ndarray, Tuple[ActingState, Dict, hk.State]]:
        acting_state, rollout = self._rollout(params, net_state, acting_state)
        batch = self._targets(rollout)

        T, B = batch.obs_grid.shape[0], batch.obs_grid.shape[1]
        flat_obs = batch.obs_grid.reshape((T * B,) + batch.obs_grid.shape[2:])
        flat_time = batch.time_vec.reshape((T * B, 2))

        (flat_logits, flat_value), new_net_state = self.forward.apply(
            params,
            net_state,
            flat_obs,
            flat_time,
            is_eval=False,
        )
        logits = flat_logits.reshape((T, B, self.num_actions))
        value = flat_value.reshape((T, B))

        logp = jax.nn.log_softmax(logits, axis=-1)
        policy_loss = -jnp.sum(batch.policy_tgt * logp, axis=-1)  # (T,B)
        policy_loss = jnp.mean(policy_loss)

        value_err = (value - batch.value_tgt) ** 2
        denom = jnp.sum(batch.value_mask.astype(jnp.float32)) + 1e-8
        value_loss = jnp.sum(value_err * batch.value_mask.astype(jnp.float32)) / denom

        total = self.policy_loss_weight * policy_loss + self.value_loss_weight * value_loss

        metrics = {
            "total_loss": total,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "terminated_rate": jnp.mean(rollout.terminated.astype(jnp.float32)),
            "avg_reward": jnp.mean(rollout.reward),
            "avg_value_pred": jnp.mean(value),
        }
        return total, (acting_state, metrics, new_net_state)

    # ---- Rollout ----

    def _rollout(
        self,
        params: hk.Params,
        net_state: hk.State,
        acting_state: ActingState,
    ) -> Tuple[ActingState, AZRollout]:
        B = self.batch_size_per_device

        def one_step(carry: ActingState, _):
            key = carry.key
            key, key_mcts, key_next = jax.random.split(key, 3)

            ts = carry.timestep
            obs = ts.observation

            obs_grid, time_vec = self._get_grid_and_time(carry.state, obs)
            invalid = self._get_invalid_actions(self.raw_env_train, carry.state, obs)

            mcts_out = self._mcts_policy(
                raw_env=self.raw_env_train,
                params=params,
                net_state=net_state,
                rng_key=key_mcts,
                root_state=carry.state,
                root_obs_grid=obs_grid,
                root_time_vec=time_vec,
                invalid_actions=invalid,
            )
            mcts_out = jax.tree_util.tree_map(jax.lax.stop_gradient, mcts_out)

            action = mcts_out.action  # (B,) flat integer
            action = jax.lax.stop_gradient(action)
            policy_tgt = jax.lax.stop_gradient(mcts_out.action_weights)  # (B,A)

            action_env = self._prepare_action(action)   # (B,2) for Tetris, (B,) otherwise
            # For KT variants (TetrisRTKT, PacManKT), pass the current observation
            # so _k_step can call the policy network during the K-1 delay steps.
            _is_kt = self._is_tetris_rt_kt or self._is_pacman_kt or self._is_snake_kt
            next_state, next_ts, step_reward, disc, step_terminated = self._k_step(
                self.env.step,
                carry.state,
                action_env,
                params=params if _is_kt else None,
                net_state=net_state if _is_kt else None,
                initial_obs=carry.timestep.observation if _is_kt else None,
            )

            new_carry = ActingState(
                state=next_state,
                timestep=next_ts,
                key=key_next,
                episode_count=carry.episode_count + jax.lax.psum(step_terminated.sum(), "devices"),
                env_step_count=carry.env_step_count + jax.lax.psum(B, "devices"),
            )

            out = (
                obs_grid,
                time_vec.astype(jnp.float32),
                step_reward,
                disc,
                step_terminated,
                policy_tgt.astype(jnp.float32),
            )
            return new_carry, out

        acting_state, outs = jax.lax.scan(one_step, acting_state, xs=None, length=self.n_steps)
        obs_grid, time_vec, reward, discount, terminated, policy_tgt = outs

        return acting_state, AZRollout(
            obs_grid=obs_grid,
            time_vec=time_vec,
            reward=reward,
            discount=discount,
            terminated=terminated,
            policy_tgt=policy_tgt,
        )

    def _targets(self, rollout: AZRollout) -> AZBatch:
        T, B = rollout.reward.shape

        # True when a terminal was observed at or after time t in the rollout window.
        value_mask = (jnp.cumsum(rollout.terminated[::-1, :], axis=0)[::-1, :] >= 1)

        def body(carry, t_rev):
            t = T - 1 - t_rev
            g = rollout.reward[t] + rollout.discount[t] * carry
            return g, g

        _, g_rev = jax.lax.scan(body, jnp.zeros((B,), jnp.float32), jnp.arange(T))
        value_tgt = g_rev[::-1, :]

        return AZBatch(
            obs_grid=rollout.obs_grid,
            time_vec=rollout.time_vec,
            policy_tgt=rollout.policy_tgt,
            value_tgt=value_tgt,
            value_mask=value_mask,
        )

    # ---- MCTS ----

    def _mcts_policy(
        self,
        raw_env: Environment,
        params: hk.Params,
        net_state: hk.State,
        rng_key: chex.PRNGKey,
        root_state: Any,
        root_obs_grid: jnp.ndarray,
        root_time_vec: jnp.ndarray,
        invalid_actions: jnp.ndarray,
    ) -> mctx.PolicyOutput:
        (root_logits, root_value), _ = self.forward.apply(
            params,
            net_state,
            root_obs_grid,
            root_time_vec,
            is_eval=True,
        )

        minf = jnp.finfo(root_logits.dtype).min
        root_logits = jnp.where(~invalid_actions, root_logits, minf)

        root = mctx.RootFnOutput(
            prior_logits=root_logits,
            value=root_value,
            embedding=root_state,
        )

        def recurrent_fn(_ignored_params, rng_key, action, state):
            return self._recurrent_fn(raw_env, params, net_state, rng_key, action, state)

        return mctx.gumbel_muzero_policy(
            params=None,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=self.num_simulations,
            invalid_actions=invalid_actions,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=self.gumbel_scale,
        )

    def _recurrent_fn(
        self,
        raw_env: Environment,
        params: hk.Params,
        net_state: hk.State,
        rng_key: chex.PRNGKey,
        action: jnp.ndarray,
        state: Any,
    ) -> Tuple[mctx.RecurrentFnOutput, Any]:
        del rng_key

        action_env = self._prepare_action(action)   # (B,2) for Tetris, (B,) otherwise
        # For KT variants, reconstruct the observation at the current MCTS node so
        # that _k_step can call the policy network during the K-1 delay steps.
        is_kt = self._is_tetris_rt_kt or self._is_pacman_kt or self._is_snake_kt
        if self._is_tetris_rt_kt:
            kt_initial_obs = jax.vmap(raw_env._make_observation)(state)
        elif self._is_pacman_kt:
            kt_initial_obs = jax.vmap(lambda s: raw_env._observation_from_state(s))(state)
        elif self._is_snake_kt:
            kt_initial_obs = jax.vmap(raw_env._state_to_observation)(state)
        else:
            kt_initial_obs = None
        next_state, next_ts, reward, discount, _ = self._k_step(
            lambda s, a: jax.vmap(raw_env.step)(s, a),
            state,
            action_env,
            params=params if is_kt else None,
            net_state=net_state if is_kt else None,
            initial_obs=kt_initial_obs,
        )

        next_obs = next_ts.observation
        next_grid, next_time_vec = self._get_grid_and_time(next_state, next_obs)

        (logits, value), _ = self.forward.apply(
            params,
            net_state,
            next_grid,
            next_time_vec,
            is_eval=True,
        )

        invalid = self._get_invalid_actions(raw_env, next_state, next_obs)
        minf = jnp.finfo(logits.dtype).min
        logits = jnp.where(~invalid, logits, minf)

        value = jnp.where(next_ts.last(), 0.0, value)

        out = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        )
        return out, next_state

    # ---- Action delay (PacMan and TetrisRTKStep) ----

    _PACMAN_NOOP     = 4  # PacMan: stay-in-place
    _TETRIS_RT_NOOP  = 5  # TetrisRT: no horizontal/rotation change, gravity only

    def _k_step(
        self,
        env_step_fn: Callable,
        state: Any,
        action: jnp.ndarray,
        params=None,
        net_state=None,
        initial_obs=None,
        discount_within_option: float = None,
    ) -> Tuple[Any, Any, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Execute K-1 delay steps then 1 real action for K-step action delay.

        For PacMan and TetrisRTKStep: delay steps use noop (action 4 or 5).
        For TetrisRTKT: delay steps use argmax(policy_logits) — requires
            params, net_state, and initial_obs to be provided.
        Falls through to a single step when K=1 or for all other envs.

        Args:
            env_step_fn: batched (state, action) → (next_state, next_ts)
            state: batched env state
            action: (B,) flat integer actions
            params: network params (TetrisRTKT only)
            net_state: network state/running stats (TetrisRTKT only)
            initial_obs: observation at `state` (TetrisRTKT only)
            discount_within_option: if set, overrides self.gamma for within-option
                reward accumulation only (e.g. 1.0 for raw undiscounted reward sum).
                total_discount (γ^K) always uses self.gamma regardless of this flag.

        Returns:
            final_state, final_ts,
            total_reward   (B,) — accumulated reward across K steps
            total_discount (B,) — γ^K (product of self.gamma * d_i over K steps)
            any_terminated (B,) — True if any of the K steps ended the episode
        """
        use_delay = (
            (self._is_pacman and self.pacman_action_delay > 1) or
            (self._is_pacman_kt and self.pacman_action_delay > 1) or
            (self._is_tetris_rt_kstep and self.pacman_action_delay > 1) or
            (self._is_tetris_rt_kt and self.pacman_action_delay > 1) or
            (self._is_snake_kt and self.pacman_action_delay > 1)
        )
        if not use_delay:
            ns, ts = env_step_fn(state, action)
            return (
                ns, ts,
                ts.reward.astype(jnp.float32),
                self.gamma * ts.discount.astype(jnp.float32),
                ts.last(),
            )

        K = self.pacman_action_delay

        # cum_d_rew: discount factor applied to rewards within the option.
        #   Uses discount_within_option (e.g. 1.0 for raw) or self.gamma if None.
        # cum_d_smdp: always uses self.gamma; accumulates γ^K for total_discount.
        within_gamma = self.gamma if discount_within_option is None else discount_within_option

        cum_r      = jnp.zeros_like(action, dtype=jnp.float32)
        cum_d_rew  = jnp.ones_like(action,  dtype=jnp.float32)
        cum_d_smdp = jnp.ones_like(action,  dtype=jnp.float32)
        any_done   = jnp.zeros_like(action, dtype=bool)

        if self._is_tetris_rt_kt or self._is_pacman_kt or self._is_snake_kt:
            # Policy-guided delay steps: argmax(logits) at each step.
            # carry = (state, obs, cum_r, cum_d_rew, cum_d_smdp, any_done)
            def accum_policy(carry, _):
                s, obs, cr, cd_rew, cd_smdp, done = carry
                grid, time_vec = self._get_grid_and_time(s, obs)
                (logits, _), _ = self.forward.apply(
                    params, net_state, grid, time_vec, is_eval=True
                )
                if self._is_pacman_kt or self._is_snake_kt:
                    # Apply action mask before argmax (PacMan: 5 actions, Snake: 4 actions)
                    invalid = ~obs.action_mask
                    logits = jnp.where(~invalid, logits, jnp.finfo(logits.dtype).min)
                # else: TetrisRT actions always valid, no masking needed
                policy_action = jnp.argmax(logits, axis=-1)  # (B,)
                ns, ts = env_step_fn(s, policy_action)
                r    = ts.reward.astype(jnp.float32)
                disc = ts.discount.astype(jnp.float32)
                return (
                    ns, ts.observation,
                    cr + cd_rew * r,
                    cd_rew  * within_gamma  * disc,
                    cd_smdp * self.gamma    * disc,
                    done | ts.last(),
                ), None

            (s1, _, cum_r, cum_d_rew, cum_d_smdp, any_done), _ = jax.lax.scan(
                accum_policy,
                (state, initial_obs, cum_r, cum_d_rew, cum_d_smdp, any_done),
                xs=None,
                length=K - 1,
            )
        else:
            # Noop-based delay steps (plain PacMan and TetrisRTKStep)
            noop_val = self._PACMAN_NOOP if self._is_pacman else self._TETRIS_RT_NOOP

            def accum(carry, noop_a):
                s, cr, cd_rew, cd_smdp, done = carry
                ns, ts = env_step_fn(s, noop_a)
                r    = ts.reward.astype(jnp.float32)
                disc = ts.discount.astype(jnp.float32)
                return (
                    ns,
                    cr + cd_rew * r,
                    cd_rew  * within_gamma * disc,
                    cd_smdp * self.gamma   * disc,
                    done | ts.last(),
                ), None

            # K-1 no-op steps via scan
            noop_actions = jnp.broadcast_to(
                jnp.full_like(action, noop_val)[None, :], (K - 1, action.shape[0])
            )
            (s1, cum_r, cum_d_rew, cum_d_smdp, any_done), _ = jax.lax.scan(
                accum, (state, cum_r, cum_d_rew, cum_d_smdp, any_done), noop_actions
            )

        # Final real-action step
        final_state, final_ts = env_step_fn(s1, action)
        r_k    = final_ts.reward.astype(jnp.float32)
        disc_k = final_ts.discount.astype(jnp.float32)

        total_reward   = cum_r + cum_d_rew  * r_k
        total_discount = cum_d_smdp * (self.gamma * disc_k)   # always γ^K
        any_terminated = any_done | final_ts.last()

        return final_state, final_ts, total_reward, total_discount, any_terminated

    # ---- Env-specific dispatch helpers ----

    def _get_grid_and_time(
        self, state: Any, obs: Any
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns (obs_grid, time_vec) dispatching on environment type.

        PacMan:  encodes scattered fields → (B,31,28,6); step_count from state.
        Tetris:  encodes grid+piece+mask → (B,H,W,3); step_count from state.
        Sokoban: uses obs.grid (B,H,W,C) and obs.step_count directly.
        """
        if self._is_pacman or self._is_pacman_kt:
            obs_grid = self._encode_obs_pacman(obs)   # (B,31,28,6)
            step_count = state.step_count              # (B,)
        elif self._is_tetris:
            obs_grid = self._encode_obs_tetris(obs)   # (B,H,W,3)
            step_count = state.step_count              # (B,) — obs.step_count may be 0
        elif self._is_tetris_rt or self._is_tetris_rt_kstep or self._is_tetris_rt_kt:
            obs_grid = self._encode_obs_tetris_rt(obs)  # (B,H,W,2)
            step_count = obs.step_count                  # (B,)
        elif self._is_flat_pack_clock:
            obs_grid = (obs.grid[:, :, :, None].astype(jnp.float32)
                        / jnp.float32(self._fp_num_blocks))       # (B,7,7,1)
            time_frac = jnp.clip(
                obs.time_left.astype(jnp.float32) / self.time_limit, 0.0, 1.0
            )                                                       # (B,)
            time_vec = jnp.stack([1.0 - time_frac, time_frac], axis=-1)  # (B,2)
            return obs_grid, time_vec
        else:
            obs_grid = getattr(obs, "grid", obs)       # (B,H,W,C)
            step_count = getattr(obs, "step_count")    # (B,)
        return obs_grid, self._time_features(step_count)

    def _get_invalid_actions(
        self, raw_env: Any, state: Any, obs: Any
    ) -> jnp.ndarray:
        """Returns (B, A) bool mask where True = invalid action.

        PacMan:  ~obs.action_mask (B, 5)
        Tetris:  ~flatten(obs.action_mask) (B, 4*num_cols)
        Sokoban: detect_noop_action per (state, action) pair
        """
        if self._is_pacman or self._is_pacman_kt:
            return ~obs.action_mask                                           # (B, 5)
        elif self._is_snake or self._is_snake_kt:
            return ~obs.action_mask                                           # (B, 4)
        elif self._is_tetris:
            return ~jnp.reshape(obs.action_mask, (obs.action_mask.shape[0], -1))  # (B, 4*num_cols)
        elif self._is_tetris_rt or self._is_tetris_rt_kstep or self._is_tetris_rt_kt:
            # All 6 actions always valid; invalid moves are silently ignored in env.step
            B = obs.action_mask.shape[0]
            return jnp.zeros((B, 6), dtype=bool)
        elif self._is_flat_pack_clock:
            # action_mask: (B, num_blocks, 4, inner, inner) → (B, num_actions)
            return ~jnp.reshape(obs.action_mask, (obs.action_mask.shape[0], -1))
        else:
            return self._invalid_actions_sokoban(raw_env, state)

    # ---- PacMan observation encoding ----

    def _encode_obs_pacman_unbatched(self, obs: Any) -> jnp.ndarray:
        """Single (unbatched) PacMan observation → (H, W, 6) float32 grid.

        Coordinate convention (from env.py):
          Position.x = row (indexes first dim of grid, 0..30)
          Position.y = col (indexes second dim of grid, 0..27)
          ghost/pellet_locations[:, 0] = col, [:, 1] = row
        """
        H, W = 31, 28

        # Ch0: maze (1=passable, 0=wall)
        maze = obs.grid.astype(jnp.float32)  # (31, 28)

        # Ch1: player location — grid[row, col] = grid[player.x, player.y]
        player = jnp.zeros((H, W)).at[obs.player_locations.x, obs.player_locations.y].set(1.0)

        # Ch2: all 4 ghost locations combined into one channel
        # ghost_locations shape (4,2): dim0=col, dim1=row → scatter as grid[row, col]
        ghost = jnp.zeros((H, W)).at[
            obs.ghost_locations[:, 1], obs.ghost_locations[:, 0]
        ].add(1.0)

        # Ch3: pellet locations (316 positions, may have duplicates at (0,0) for eaten pellets)
        # pellet_locations shape (316,2): dim0=col, dim1=row
        pellets = jnp.zeros((H, W)).at[
            obs.pellet_locations[:, 1], obs.pellet_locations[:, 0]
        ].set(1.0)

        # Ch4: power-up locations (4 positions)
        # power_up_locations shape (4,2): dim0=col, dim1=row
        powerups = jnp.zeros((H, W)).at[
            obs.power_up_locations[:, 1], obs.power_up_locations[:, 0]
        ].set(1.0)

        # Ch5: frightened state time normalized to [0, 1]
        frightened = jnp.full(
            (H, W), obs.frightened_state_time.astype(jnp.float32) / self.time_limit
        )

        return jnp.stack([maze, player, ghost, pellets, powerups, frightened], axis=-1)

    def _encode_obs_pacman(self, obs: Any) -> jnp.ndarray:
        """Batched PacMan observation → (B, H, W, 6)."""
        return jax.vmap(self._encode_obs_pacman_unbatched)(obs)

    # ---- Tetris observation encoding ----

    def _encode_obs_tetris_unbatched(self, obs: Any) -> jnp.ndarray:
        """Single (unbatched) Tetris observation → (H, W, 3) float32 grid.

        Channel layout:
          Ch0: board occupancy (0=empty, 1=filled)
          Ch1: current tetromino (4×4) embedded in top-left corner, zeros elsewhere
          Ch2: action validity heatmap — action_mask (4, num_cols) in first 4 rows
        """
        H = self.raw_env_train.num_rows
        W = self.raw_env_train.num_cols

        board = obs.grid.astype(jnp.float32)  # (H, W)

        piece = jnp.zeros((H, W)).at[:4, :4].set(obs.tetromino.astype(jnp.float32))

        # action_mask shape (4, W) — embed as first 4 rows of (H, W) channel
        action_map = jnp.zeros((H, W)).at[:4, :].set(obs.action_mask.astype(jnp.float32))

        return jnp.stack([board, piece, action_map], axis=-1)  # (H, W, 3)

    def _encode_obs_tetris(self, obs: Any) -> jnp.ndarray:
        """Batched Tetris observation → (B, H, W, 3)."""
        return jax.vmap(self._encode_obs_tetris_unbatched)(obs)

    # ---- TetrisRT observation encoding ----

    def _encode_obs_tetris_rt_unbatched(self, obs: Any) -> jnp.ndarray:
        """Single (unbatched) TetrisRT observation → (H, W, 2) float32 grid.

        Channel layout:
          Ch0: locked board occupancy (0=empty, 1=filled)
          Ch1: current falling piece drawn at its live (x, y) position on an empty grid
        """
        H = self.raw_env_train.num_rows
        W = self.raw_env_train.num_cols

        board_ch = obs.board.astype(jnp.float32)  # (H, W)

        # Draw the falling piece on an empty (H+3, W+3) workspace then crop.
        workspace = jnp.zeros((H + 3, W + 3), dtype=jnp.float32)
        workspace = jax.lax.dynamic_update_slice(
            workspace,
            obs.tetromino.astype(jnp.float32),
            (obs.y_position, obs.x_position),
        )
        piece_ch = workspace[:H, :W]  # (H, W)

        return jnp.stack([board_ch, piece_ch], axis=-1)  # (H, W, 2)

    def _encode_obs_tetris_rt(self, obs: Any) -> jnp.ndarray:
        """Batched TetrisRT observation → (B, H, W, 2)."""
        return jax.vmap(self._encode_obs_tetris_rt_unbatched)(obs)

    # ---- Action conversion ----

    def _prepare_action(self, action: jnp.ndarray) -> jnp.ndarray:
        """Convert MCTS flat action index to env-compatible format.

        Tetris: flat int in [0, 4*num_cols) → [rotation, x_pos] shape (..., 2).
        All other envs: pass through unchanged.
        """
        if self._is_tetris:
            rotation = action // self.num_cols
            x_pos = action % self.num_cols
            return jnp.stack([rotation, x_pos], axis=-1)
        if self._is_flat_pack_clock:
            inner = self._fp_inner          # 5
            per_rot = inner * inner          # 25
            per_block = 4 * per_rot          # 100
            block_idx = action // per_block
            rem       = action  % per_block
            rotation  = rem // per_rot
            rem2      = rem  % per_rot
            row_idx   = rem2 // inner
            col_idx   = rem2  % inner
            return jnp.stack([block_idx, rotation, row_idx, col_idx], axis=-1)
        return action

    # ---- Sokoban invalid action mask ----

    def _invalid_actions_sokoban(self, raw_env: Any, state: Any) -> jnp.ndarray:
        """
        invalid[b, a] = True if detect_noop_action(...) would map action a -> NOOP (-1).
        state is expected to be batched: fields have leading dimension B.
        """
        from jumanji.environments.routing.sokoban.constants import NOOP

        actions = jnp.arange(self.num_actions, dtype=jnp.int32)  # (A,)

        def per_env(s):
            def check(a):
                mapped = raw_env.detect_noop_action(
                    s.variable_grid,
                    s.fixed_grid,
                    a,
                    s.agent_location,
                )
                return mapped == NOOP

            return jax.vmap(check)(actions)  # (A,)

        invalid = jax.vmap(per_env)(state)  # (B,A)
        return invalid

    def _time_features(self, step_count: jnp.ndarray) -> jnp.ndarray:
        """
        Converts step_count into a 2D vector:
          [time_seen_frac, time_left_frac]

        Args:
            step_count: shape () or (B,)

        Returns:
            shape (B,2) if input is batched, else (1,2) after caller batching
        """
        step_count = step_count.astype(jnp.float32)
        time_seen = jnp.clip(step_count / self.time_limit, 0.0, 1.0)
        time_left = 1.0 - time_seen
        return jnp.stack([time_seen, time_left], axis=-1)
