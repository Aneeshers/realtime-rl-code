from typing import TYPE_CHECKING, NamedTuple

import chex

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


class Observation(NamedTuple):
    """
    grid: 2D array with the current state of grid.
    blocks: 3D array with the blocks to be placed on the board.
    action_mask: 4D array showing where blocks can be placed on the grid.
    time_left: remaining clock ticks (scalar int32).
    step_count: number of steps taken so far.
    """

    grid: chex.Array        # (num_rows, num_cols)
    blocks: chex.Array      # (num_blocks, 3, 3)
    action_mask: chex.Array  # (num_blocks, 4, num_rows-2, num_cols-2)
    time_left: chex.Array   # ()
    step_count: chex.Array  # ()


@dataclass
class State:
    """
    grid: 2D array with the current state of grid.
    num_blocks: number of blocks in the full grid.
    blocks: 3D array with the blocks to be placed on the board.
    action_mask: 4D array showing where blocks can be placed on the grid.
    placed_blocks: 1D boolean array showing which blocks have been placed.
    step_count: number of steps taken in the environment.
    time_left: remaining clock ticks.
    key: random key used for board generation.
    """

    grid: chex.Array          # (num_rows, num_cols)
    num_blocks: chex.Numeric  # ()
    blocks: chex.Array        # (num_blocks, 3, 3)
    action_mask: chex.Array   # (num_blocks, 4, num_rows-2, num_cols-2)
    placed_blocks: chex.Array  # (num_blocks,)
    step_count: chex.Numeric  # ()
    time_left: chex.Numeric   # ()
    key: chex.PRNGKey         # (2,)
