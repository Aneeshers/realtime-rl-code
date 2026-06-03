from functools import cached_property
from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji import specs
from jumanji.environments.packing.flat_pack.env import FlatPack
from jumanji.environments.packing.flat_pack.generator import RandomFlatPackGenerator
from jumanji.environments.packing.flat_pack.reward import CellDenseReward
from jumanji.environments.packing.flat_pack_clock.types import Observation, State
from jumanji.types import TimeStep, restart, termination, transition


class FlatPackClock(FlatPack):
    """FlatPack augmented with a per-step clock deduction.

    Each call to step() subtracts `sim_cost` ticks from `state.time_left`.
    The clock is clamped at zero; the episode still runs for exactly `num_blocks`
    steps. The remaining clock is exposed in both State and Observation so the
    network learns clock-aware value estimates.

    Default board: 3x3 blocks -> 9 blocks, 7x7 grid, 900 flat actions.

    Training K=1-4 uses sim_cost = Kx32:
      K=1 (32): 9x32=288 ticks used vs time_limit=576 -> lots of slack
      K=2 (64): 9x64=576 ticks used vs 576 -> exactly constrained
      K=3 (96): 576 ticks available, runs out at step 6
      K=4 (128): 576 ticks available, runs out at step ~4.5
    """

    def __init__(
        self,
        sim_cost: int = 32,
        time_limit: int = 576,
        num_row_blocks: int = 3,
        num_col_blocks: int = 3,
    ):
        # Set before super().__init__() because observation_spec (a cached_property)
        # is evaluated during the parent __init__ and reads these attributes.
        self.sim_cost = int(sim_cost)
        self.time_limit = int(time_limit)
        generator = RandomFlatPackGenerator(
            num_row_blocks=num_row_blocks,
            num_col_blocks=num_col_blocks,
        )
        super().__init__(generator=generator, reward_fn=CellDenseReward())

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        grid_state, base_ts = super().reset(key)

        time_left = jnp.array(self.time_limit, jnp.int32)

        state = State(
            grid=grid_state.grid,
            num_blocks=grid_state.num_blocks,
            blocks=grid_state.blocks,
            action_mask=grid_state.action_mask,
            placed_blocks=grid_state.placed_blocks,
            step_count=grid_state.step_count,
            time_left=time_left,
            key=grid_state.key,
        )

        obs = Observation(
            grid=base_ts.observation.grid,
            blocks=base_ts.observation.blocks,
            action_mask=base_ts.observation.action_mask,
            time_left=time_left,
            step_count=grid_state.step_count,
        )

        timestep = restart(observation=obs)
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        # Run base FlatPack step (operates on base State fields, ignores time_left)
        base_state_in = self._to_base_state(state)
        base_next, _ = super().step(base_state_in, action)

        # Deduct clock and clamp
        new_time_left = jnp.maximum(
            state.time_left - jnp.int32(self.sim_cost), jnp.int32(0)
        )

        next_state = State(
            grid=base_next.grid,
            num_blocks=base_next.num_blocks,
            blocks=base_next.blocks,
            action_mask=base_next.action_mask,
            placed_blocks=base_next.placed_blocks,
            step_count=base_next.step_count,
            time_left=new_time_left,
            key=base_next.key,
        )

        done = next_state.step_count >= next_state.num_blocks

        obs = Observation(
            grid=next_state.grid,
            blocks=next_state.blocks,
            action_mask=next_state.action_mask,
            time_left=new_time_left,
            step_count=next_state.step_count,
        )

        # Recompute reward using base logic
        from jumanji.environments.packing.flat_pack.utils import rotate_block
        block_idx, rotation, row_idx, col_idx = action
        chosen_block = state.blocks[block_idx]
        chosen_block = rotate_block(chosen_block, rotation)
        grid_block = self._expand_block_to_grid(chosen_block, row_idx, col_idx)
        action_is_legal = state.action_mask[block_idx, rotation, row_idx, col_idx]
        reward = self.reward_fn(
            self._to_base_state(state), grid_block,
            self._to_base_state(next_state), action_is_legal, done
        )

        timestep = jax.lax.cond(
            done,
            termination,
            transition,
            reward,
            obs,
        )

        return next_state, timestep

    # ------------------------------------------------------------------
    # Observation / action specs (extend parent with time_left + step_count)
    # ------------------------------------------------------------------

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        base_spec = super().observation_spec

        time_left_spec = specs.BoundedArray(
            shape=(),
            minimum=0,
            maximum=self.time_limit,
            dtype=jnp.int32,
            name="time_left",
        )
        step_count_spec = specs.DiscreteArray(
            num_values=self.num_blocks + 1,
            dtype=jnp.int32,
            name="step_count",
        )

        return specs.Spec(
            Observation,
            "ObservationSpec",
            grid=base_spec.grid,
            blocks=base_spec.blocks,
            action_mask=base_spec.action_mask,
            time_left=time_left_spec,
            step_count=step_count_spec,
        )

    # ------------------------------------------------------------------
    # Helper: convert FlatPackClock State -> base FlatPack State
    # ------------------------------------------------------------------

    def _to_base_state(self, state: State):
        from jumanji.environments.packing.flat_pack.types import State as BaseState
        return BaseState(
            grid=state.grid,
            num_blocks=state.num_blocks,
            blocks=state.blocks,
            action_mask=state.action_mask,
            placed_blocks=state.placed_blocks,
            step_count=state.step_count,
            key=state.key,
        )
