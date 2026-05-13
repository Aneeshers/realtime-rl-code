# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from typing import Any, Dict, Optional, Tuple

import chex
import haiku as hk
import jax
from jax import numpy as jnp

from jumanji.env import Environment
from jumanji.training.agents.a2c import A2CAgent
from jumanji.training.agents.base import Agent
from jumanji.training.agents.gumbel_alphazero import GumbelAlphaZeroAgent
from jumanji.training.agents.random import RandomAgent
from jumanji.training.training_types import (
    ActingState,
    AnyParamsState,
    ParamsState,
    AlphaZeroParamsState,
)

class Evaluator:
    """Class to run evaluations."""

    def __init__(
        self,
        eval_env: Environment,
        agent: Agent,
        total_batch_size: int,
        stochastic: bool,
    ):
        self.eval_env = eval_env
        self.agent = agent
        self.num_local_devices = jax.local_device_count()
        self.num_global_devices = jax.device_count()
        self.num_workers = self.num_global_devices // self.num_local_devices
        if total_batch_size % self.num_global_devices != 0:
            raise ValueError(
                "Expected eval total_batch_size to be a multiple of num_devices, "
                f"got {total_batch_size} and {self.num_global_devices}."
            )
        self.total_batch_size = total_batch_size
        self.batch_size_per_device = total_batch_size // self.num_global_devices
        self.generate_evaluations = jax.pmap(
            functools.partial(
                self._generate_evaluations, eval_batch_size=self.batch_size_per_device
            ),
            axis_name="devices",
        )
        self.stochastic = stochastic

    def _eval_one_episode(
        self,
        params_state: Optional[AnyParamsState],
        key: chex.PRNGKey,
    ) -> Dict:
        # --- Build policy callable depending on agent type ---
        if isinstance(self.agent, GumbelAlphaZeroAgent):
            assert isinstance(params_state, AlphaZeroParamsState)
            policy = self.agent.make_policy(
                params_state=params_state,
                stochastic=self.stochastic,
                eval_env=self.eval_env,
            )
    
            def acting_policy(state: Any, observation: Any, key: chex.PRNGKey) -> chex.Array:
                # Do NOT add batch dim unless your AZ policy expects it.
                # Most MCTS code will vmap internally over batch, but evaluation here is single-episode.
                return policy(state, observation, key)
    
        else:
            # Old path (A2C / Random): policy(observation, key) -> (action, aux) or action
            policy = self.agent.make_policy(policy_params=policy_params, stochastic=self.stochastic)
    
            if isinstance(self.agent, A2CAgent):
                def acting_policy(state: Any, observation: Any, key: chex.PRNGKey) -> chex.Array:
                    # A2C policy expects a batch dimension
                    obs_batched = jax.tree_util.tree_map(lambda x: x[None], observation)
                    action, _ = policy(obs_batched, key)
                    return jnp.squeeze(action, axis=0)
            else:
                def acting_policy(state: Any, observation: Any, key: chex.PRNGKey) -> chex.Array:
                    # RandomAgent / other simple agents: may already accept batched obs;
                    # use same batching pattern as before for consistency.
                    obs_batched = jax.tree_util.tree_map(lambda x: x[None], observation)
                    action = policy(obs_batched, key)
                    return jnp.squeeze(action, axis=0)
    
        # --- Episode loop (now policy can use state) ---
        def cond_fun(carry: Tuple[ActingState, float]) -> jnp.bool_:
            acting_state, _ = carry
            return ~acting_state.timestep.last()
    
        def body_fun(
            carry: Tuple[ActingState, float],
        ) -> Tuple[ActingState, float]:
            acting_state, return_ = carry
            key, action_key = jax.random.split(acting_state.key)
    
            action = acting_policy(
                acting_state.state,
                acting_state.timestep.observation,
                action_key,
            )
    
            state, timestep = self.eval_env.step(acting_state.state, action)
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
        state, timestep = self.eval_env.reset(reset_key)
        acting_state = ActingState(
            state=state,
            timestep=timestep,
            key=init_key,
            episode_count=jnp.array(0, jnp.int32),
            env_step_count=jnp.array(0, jnp.int32),
        )
        return_ = jnp.array(0, float)
    
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


    def _generate_evaluations(
        self,
        params_state: AnyParamsState,
        key: chex.PRNGKey,
        eval_batch_size: int,
    ) -> Dict:
        keys = jax.random.split(key, eval_batch_size)
    
        # Pass params_state, not policy_params
        eval_metrics = jax.vmap(self._eval_one_episode, in_axes=(None, 0))(
            params_state,
            keys,
        )
    
        eval_metrics: Dict = jax.lax.pmean(
            jax.tree_util.tree_map(jnp.mean, eval_metrics),
            axis_name="devices",
        )
        return eval_metrics


    def run_evaluation(self, params_state: Optional[AnyParamsState], eval_key: chex.PRNGKey) -> Dict:
        """Run one batch of evaluations."""
        eval_keys = jax.random.split(eval_key, self.num_global_devices).reshape(
            self.num_workers, self.num_local_devices, -1
        )
        eval_keys_per_worker = eval_keys[jax.process_index()]
        eval_metrics: Dict = self.generate_evaluations(
            params_state,
            eval_keys_per_worker,
        )
        return eval_metrics
