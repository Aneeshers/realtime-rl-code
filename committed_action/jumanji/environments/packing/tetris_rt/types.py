from typing import TYPE_CHECKING, NamedTuple

import chex

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class State:
    """
    locked_grid: binary (0=empty, 1=filled) game grid of shape (num_rows, num_cols).
        Contains only pieces that have already locked in place. The currently falling
        piece is NOT included here; it is tracked by (tetromino_index, rotation, x, y).
    tetromino_index: index (0-6) identifying which of the 7 tetromino types is falling.
    rotation: current rotation of the falling piece (0=0 deg, 1=90 deg, 2=180 deg, 3=270 deg).
    x_position: column of the top-left corner of the piece's 4x4 bounding box.
    y_position: row of the top-left corner of the piece's 4x4 bounding box.
    score: cumulative reward earned so far in the episode.
    reward: reward earned on the most recent step.
    key: PRNG key used for sampling new pieces.
    step_count: number of environment steps taken so far (counts gravity ticks).
    """

    locked_grid: chex.Array    # (num_rows, num_cols) int32
    tetromino_index: chex.Numeric  # ()
    rotation: chex.Numeric     # ()
    x_position: chex.Numeric   # ()
    y_position: chex.Numeric   # ()
    score: chex.Array          # ()
    reward: chex.Array         # ()
    key: chex.PRNGKey          # (2,)
    step_count: chex.Numeric   # ()


class Observation(NamedTuple):
    """
    board: binary locked grid (num_rows, num_cols) - placed pieces only, not the falling piece.
    tetromino: 4x4 shape of the currently falling piece at its current rotation.
    x_position: column of the top-left of the piece's 4x4 bounding box.
    y_position: row of the top-left of the piece's 4x4 bounding box.
    action_mask: always all-True (6,) - all 6 actions are always valid; invalid moves
        are silently ignored (piece stays put) rather than terminating the episode.
    step_count: number of steps taken so far.
    """

    board: chex.Array          # (num_rows, num_cols) int32
    tetromino: chex.Array      # (4, 4) int32
    x_position: chex.Numeric   # ()
    y_position: chex.Numeric   # ()
    action_mask: chex.Array    # (6,) bool
    step_count: chex.Numeric   # ()
