"""Real-time Tetris environment for Jumanji.

Unlike the placement-based Tetris-v0, this environment models each step as a single
gravity tick. The agent controls the falling piece via 6 actions (left, right, rotate CW,
rotate CCW, hard drop, noop). This makes a meaningful no-op possible, enabling K-step
action delay experiments.

Action index semantics:
  0: move left     - x -= 1 (silently ignored if it would cause a collision)
  1: move right    - x += 1 (silently ignored if it would cause a collision)
  2: rotate CW     - rotation = (rotation + 1) % 4 (silently ignored if invalid)
  3: rotate CCW    - rotation = (rotation - 1) % 4 (silently ignored if invalid)
  4: hard drop     - piece falls to its resting position immediately, then locks
  5: noop          - no horizontal/rotation change; gravity acts as normal

After every action, gravity tries to advance the piece one row. If gravity is
blocked (or action=hard_drop), the piece locks: it merges into the board, full
lines are cleared, a reward is given, and a new piece spawns at the top.

Episode ends when a newly spawned piece cannot be placed at the spawn position
(board is too full), or when step_count reaches time_limit.
"""

from functools import cached_property
from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.packing.tetris.constants import REWARD_LIST, TETROMINOES_LIST
from jumanji.environments.packing.tetris.utils import (
    check_valid_tetromino_placement,
    clean_lines,
    sample_tetromino_list,
)
from jumanji.environments.packing.tetris_rt.types import Observation, State
from jumanji.types import TimeStep, restart, termination, transition

# Action indices
_LEFT = 0
_RIGHT = 1
_ROTATE_CW = 2
_ROTATE_CCW = 3
_HARD_DROP = 4
_NOOP = 5


class TetrisRT(Environment[State, specs.DiscreteArray, Observation]):
    """Real-time Tetris where each step is one gravity tick.

    - observation: `Observation`
        - board: jax array (int32) of shape (num_rows, num_cols) - locked pieces only.
        - tetromino: jax array (int32) of shape (4, 4) - current falling piece shape.
        - x_position: int32 () - column of piece's 4x4 bounding-box top-left corner.
        - y_position: int32 () - row of piece's 4x4 bounding-box top-left corner.
        - action_mask: bool (6,) - always all-True.
        - step_count: int32 () - gravity ticks elapsed.

    - action: DiscreteArray(6)
        0=left, 1=right, 2=rotate_CW, 3=rotate_CCW, 4=hard_drop, 5=noop.

    - reward: convex in number of lines cleared (0, 40, 100, 300, 1200).

    - episode termination: new piece cannot be placed at spawn, or time_limit reached.

    ```python
    from jumanji.environments.packing.tetris_rt import TetrisRT
    env = TetrisRT()
    key = jax.random.PRNGKey(0)
    state, timestep = jax.jit(env.reset)(key)
    action = env.action_spec.generate_value()
    state, timestep = jax.jit(env.step)(state, action)
    ```
    """

    def __init__(
        self,
        num_rows: int = 20,
        num_cols: int = 10,
        time_limit: int = 2000,
        viewer: Optional[object] = None,
    ) -> None:
        if num_rows < 4:
            raise ValueError(f"num_rows must be >= 4, got {num_rows}")
        if num_cols < 4:
            raise ValueError(f"num_cols must be >= 4, got {num_cols}")

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.time_limit = time_limit

        self.TETROMINOES_LIST = jnp.array(TETROMINOES_LIST, jnp.int32)
        self.reward_list = jnp.array(REWARD_LIST, jnp.float32)

        # Spawn position: centre the 4x4 bounding box horizontally
        self._spawn_x = int(num_cols // 2 - 2)
        self._spawn_y = 0

        super().__init__()

    def __repr__(self) -> str:
        return (
            f"TetrisRT(num_rows={self.num_rows}, num_cols={self.num_cols}, "
            f"time_limit={self.time_limit})"
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        key, sample_key = jax.random.split(key)
        locked_grid = jnp.zeros((self.num_rows, self.num_cols), dtype=jnp.int32)
        _, tetromino_index = sample_tetromino_list(sample_key, self.TETROMINOES_LIST)

        state = State(
            locked_grid=locked_grid,
            tetromino_index=tetromino_index,
            rotation=jnp.int32(0),
            x_position=jnp.int32(self._spawn_x),
            y_position=jnp.int32(self._spawn_y),
            score=jnp.float32(0.0),
            reward=jnp.float32(0.0),
            key=key,
            step_count=jnp.int32(0),
        )
        observation = self._make_observation(state)
        return state, restart(observation=observation)

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        # Always split the key (needed in locked_branch for sampling).
        key, sample_key = jax.random.split(state.key)

        # ---- 1. Compute candidate (rotation, x) from action ----
        is_left = action == _LEFT
        is_right = action == _RIGHT
        is_cw = action == _ROTATE_CW
        is_ccw = action == _ROTATE_CCW
        is_hard_drop = action == _HARD_DROP

        new_rotation = jnp.where(
            is_cw, (state.rotation + 1) % 4,
            jnp.where(is_ccw, (state.rotation - 1 + 4) % 4, state.rotation)
        )
        new_x = jnp.where(
            is_left, state.x_position - 1,
            jnp.where(is_right, state.x_position + 1, state.x_position)
        )

        # Get the new piece shape
        new_piece = self.TETROMINOES_LIST[state.tetromino_index, new_rotation]

        # Validate lateral/rotation move at same y
        padded = self._make_padded(state.locked_grid)
        move_valid = self._is_valid_on_padded(padded, new_piece, new_x, state.y_position)

        # Apply move or revert
        actual_rotation = jnp.where(move_valid, new_rotation, state.rotation)
        actual_x = jnp.where(move_valid, new_x, state.x_position)
        actual_piece = self.TETROMINOES_LIST[state.tetromino_index, actual_rotation]

        # ---- 2. Gravity / hard-drop ----
        # Hard drop: find final resting y immediately.
        drop_y = self._find_drop_y(padded, actual_piece, actual_x, state.y_position)

        # Normal gravity: try y + 1
        gravity_y = state.y_position + 1
        gravity_valid = self._is_valid_on_padded(padded, actual_piece, actual_x, gravity_y)

        # Piece's y after this step
        new_y = jnp.where(
            is_hard_drop,
            drop_y,
            jnp.where(gravity_valid, gravity_y, state.y_position),
        )

        # Does the piece lock? hard_drop always locks; otherwise locks when gravity blocked.
        piece_locks = is_hard_drop | ~gravity_valid

        # ---- 3. Branch: locked vs. still falling ----
        def locked_branch(_):
            # Merge piece into board
            new_locked = self._lock_piece(state.locked_grid, actual_piece, actual_x, new_y)
            # Detect and clear full lines
            full_lines = jnp.all(new_locked != 0, axis=1)  # (num_rows,)
            new_locked = clean_lines(new_locked, full_lines)
            reward = self.reward_list[jnp.clip(full_lines.sum(), 0, 4)]
            # Sample next piece
            _, next_idx = sample_tetromino_list(sample_key, self.TETROMINOES_LIST)
            next_piece_shape = self.TETROMINOES_LIST[next_idx, 0]
            # Build padded grid with next piece for spawn check
            new_padded = self._make_padded(new_locked)
            spawn_x = jnp.int32(self._spawn_x)
            spawn_y = jnp.int32(self._spawn_y)
            game_over = ~self._is_valid_on_padded(new_padded, next_piece_shape, spawn_x, spawn_y)
            return (
                new_locked,
                next_idx,
                jnp.int32(0),    # reset rotation to 0 for new piece
                spawn_x,
                spawn_y,
                jnp.float32(reward),
                game_over,
                key,
            )

        def moving_branch(_):
            return (
                state.locked_grid,
                state.tetromino_index,
                actual_rotation,
                actual_x,
                new_y,
                jnp.float32(0.0),
                jnp.bool_(False),
                key,
            )

        (
            next_locked, next_tetromino_idx, next_rotation,
            next_x, next_y, reward, game_over, next_key,
        ) = jax.lax.cond(piece_locks, locked_branch, moving_branch, None)

        step_count = state.step_count + 1
        done = game_over | (step_count >= self.time_limit)

        next_state = State(
            locked_grid=next_locked,
            tetromino_index=next_tetromino_idx,
            rotation=next_rotation,
            x_position=next_x,
            y_position=next_y,
            score=state.score + reward,
            reward=reward,
            key=next_key,
            step_count=step_count,
        )

        next_observation = self._make_observation(next_state)

        next_timestep = jax.lax.cond(
            done,
            termination,
            transition,
            reward,
            next_observation,
        )
        return next_state, next_timestep

    # ------------------------------------------------------------------
    # Specs
    # ------------------------------------------------------------------

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        return specs.Spec(
            Observation,
            "ObservationSpec",
            board=specs.BoundedArray(
                shape=(self.num_rows, self.num_cols),
                dtype=jnp.int32,
                minimum=0,
                maximum=1,
                name="board",
            ),
            tetromino=specs.BoundedArray(
                shape=(4, 4),
                dtype=jnp.int32,
                minimum=0,
                maximum=1,
                name="tetromino",
            ),
            x_position=specs.DiscreteArray(
                num_values=self.num_cols,
                dtype=jnp.int32,
                name="x_position",
            ),
            y_position=specs.DiscreteArray(
                num_values=self.num_rows,
                dtype=jnp.int32,
                name="y_position",
            ),
            action_mask=specs.BoundedArray(
                shape=(6,),
                dtype=bool,
                minimum=False,
                maximum=True,
                name="action_mask",
            ),
            step_count=specs.DiscreteArray(
                num_values=self.time_limit,
                dtype=jnp.int32,
                name="step_count",
            ),
        )

    @cached_property
    def action_spec(self) -> specs.DiscreteArray:
        return specs.DiscreteArray(num_values=6, name="action", dtype=jnp.int32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_padded(self, locked_grid: chex.Array) -> chex.Array:
        """Build (num_rows+3, num_cols+3) padded grid with sentinel walls.

        Rows [num_rows:] and cols [num_cols:] are set to 1 (wall), so that
        check_valid_tetromino_placement naturally detects bottom and right walls.
        """
        g = jnp.zeros((self.num_rows + 3, self.num_cols + 3), dtype=jnp.int32)
        g = g.at[: self.num_rows, : self.num_cols].set(locked_grid)
        g = g.at[self.num_rows :, :].set(1)   # bottom wall
        g = g.at[:, self.num_cols :].set(1)   # right wall
        return g

    def _is_valid_on_padded(
        self,
        padded: chex.Array,
        tetromino: chex.Array,
        x: chex.Numeric,
        y: chex.Numeric,
    ) -> chex.Array:
        """Check if tetromino can be placed at (x, y) without collision or left-wall OOB.

        Right wall and bottom wall are handled by the sentinel values in `padded`.
        Left wall requires an explicit x >= 0 check (dynamic_slice clamps negatives).
        """
        x_ok = x >= 0
        y_ok = (y >= 0) & (y <= self.num_rows - 1)
        # Clamp to avoid dynamic_slice OOB (JAX clamps, but we want correct semantics).
        safe_x = jnp.clip(x, 0, self.num_cols - 1)
        safe_y = jnp.clip(y, 0, self.num_rows - 1)
        no_collision = check_valid_tetromino_placement(padded, tetromino, safe_y, safe_x)
        return x_ok & y_ok & no_collision

    def _find_drop_y(
        self,
        padded: chex.Array,
        tetromino: chex.Array,
        x: chex.Numeric,
        start_y: chex.Numeric,
    ) -> chex.Numeric:
        """Scan downward from start_y to find the lowest valid y (hard-drop landing row)."""
        safe_x = jnp.clip(x, 0, self.num_cols - 1)
        x_ok = x >= 0

        def body(i, y):
            next_y = y + 1
            in_y_bounds = next_y <= self.num_rows - 1
            safe_next_y = jnp.clip(next_y, 0, self.num_rows - 1)
            no_collision = check_valid_tetromino_placement(padded, tetromino, safe_next_y, safe_x)
            valid = x_ok & in_y_bounds & no_collision
            return jnp.where(valid, next_y, y)

        return jax.lax.fori_loop(0, self.num_rows + 3, body, start_y)

    def _lock_piece(
        self,
        locked_grid: chex.Array,
        tetromino: chex.Array,
        x: chex.Numeric,
        y: chex.Numeric,
    ) -> chex.Array:
        """Merge tetromino at (x, y) into locked_grid.

        Uses a (num_rows+3, num_cols+3) workspace so that dynamic_update_slice never
        goes out of bounds. Only game-area cells [:num_rows, :num_cols] are returned.
        """
        workspace = jnp.zeros((self.num_rows + 3, self.num_cols + 3), dtype=jnp.int32)
        workspace = jax.lax.dynamic_update_slice(
            workspace, jnp.clip(tetromino, 0, 1), (y, x)
        )
        piece_in_area = workspace[: self.num_rows, : self.num_cols]
        return jnp.clip(locked_grid + piece_in_area, 0, 1)

    def _make_observation(self, state: State) -> Observation:
        current_piece = self.TETROMINOES_LIST[state.tetromino_index, state.rotation]
        return Observation(
            board=state.locked_grid,
            tetromino=current_piece,
            x_position=state.x_position,
            y_position=state.y_position,
            action_mask=jnp.ones(6, dtype=bool),
            step_count=state.step_count,
        )


class TetrisRTKStep(TetrisRT):
    """TetrisRT variant that enables K-step action-delay MCTS in GumbelAlphaZeroAgent.

    Identical to TetrisRT in every way - same observation, action space, reward,
    and episode dynamics.  The distinct class name lets the agent detect this
    variant via isinstance and apply K-step tree expansion in _k_step, using
    action 5 (noop / gravity-only) for the K-1 delay steps.

    Use this env when you want training and MCTS search to both operate under
    K-step action delay (i.e. the MCTS tree simulates K env steps per edge).
    Use plain TetrisRT when you only want to apply delay at the outer eval loop.
    """
    pass


class TetrisRTKT(TetrisRT):
    """TetrisRT with policy-guided delay steps (KT = K-step with policy Transitions).

    Identical env dynamics to TetrisRT in every way.  The distinct class name
    enables isinstance dispatch in GumbelAlphaZeroAgent.

    During the K-1 delay steps, the agent uses argmax(policy_logits) instead
    of noop(5).  This means the MCTS tree simulates what the agent will actually
    do during the delay window rather than assuming the piece just falls freely.

    Inherits _make_observation(state) from TetrisRT, which is called by
    _recurrent_fn to obtain the initial observation from the MCTS node state
    before running policy-guided delay steps.
    """
    pass
