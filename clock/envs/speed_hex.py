"""
speed_hex.py

A "speed" (clocked) Hex environment compatible with the interface we use for
speed_gardner_chess:

- State includes per-RL-player clocks: State._time_left (private) + State.time_left (read-only)
- Env.step accepts either:
    - action
    - (action, time_spent)
- time_spent is subtracted from the CURRENT RL player before the move flips.
- Exposes pure board helpers for MCTS:
    - _step_board(state, action): apply Hex rules only (no clocks)
    - _observe(state, player_id): observation only (no clocks)

Intended usage in your train_gate code:
    from speed_hex import Hex, _step_board, _observe
    env_speed = Hex()
    ...
    state_next = env_speed.step(env_state, (action, time_spent))
"""

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.games.hex import Game, GameState
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

# ----------------------------------------------------------------------
# Time-control knobs
# ----------------------------------------------------------------------
MAX_TERMINATION_STEPS = 256
DEFAULT_TIME = 300  # "ticks" per RL player for speed hex

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

# NOTE: pgx Hex is effectively fixed-shape in the wrapper (11x11).
# If you need other sizes, you'd typically create a separate env module
# so the JAX shapes are static. This file mirrors that convention.
DEFAULT_SIZE = 11

INIT_LEGAL_ACTION_MASK = (
    jnp.ones(DEFAULT_SIZE * DEFAULT_SIZE + 1, dtype=jnp.bool_)
    .at[-1]
    .set(FALSE)
)


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------
@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    # pgx._src.games.hex.observe returns (size, size, 4) bool:
    # [my_board, opp_board, color, can_swap]
    observation: Array = jnp.zeros((DEFAULT_SIZE, DEFAULT_SIZE, 4), dtype=jnp.bool_)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = INIT_LEGAL_ACTION_MASK
    _step_count: Array = jnp.int32(0)

    # --- Hex specific ---
    _player_order: Array = jnp.int32([0, 1])  # [0,1] or [1,0]
    _x: GameState = GameState()

    # --- Speed-hex time control (per RL player: 0/1) ---
    _time_left: Array = jnp.int32([DEFAULT_TIME, DEFAULT_TIME])

    @property
    def time_left(self) -> Array:
        """Public read-only view of the per-player clocks."""
        return self._time_left

    @property
    def env_id(self) -> core.EnvId:
        # Keep "hex" for compatibility with pretrained AZ checkpoints trained on pgx Hex.
        return "hex"


# ----------------------------------------------------------------------
# Pure board helpers (for MCTS)
# ----------------------------------------------------------------------
def _step_board(state: State, action: Array) -> State:
    """Pure Hex board step (NO time control)."""
    # We purposely instantiate Game at module scope via DEFAULT_SIZE.
    # This is a pure function w.r.t. State; no clocks are changed.
    game = _GAME_11  # fixed-shape helper
    x = game.step(state._x, action)
    return state.replace(  # type: ignore
        current_player=state._player_order[x.color],
        legal_action_mask=game.legal_action_mask(x),
        terminated=game.is_terminal(x),
        rewards=game.rewards(x)[state._player_order],
        _x=x,
    )


def _observe(state: State, player_id: Array) -> Array:
    """Pure Hex observation (NO time control)."""
    game = _GAME_11
    # Same logic as pgx hex.py:
    # If observing player is current_player, use current color; else flip color.
    color = jax.lax.select(
        player_id == state.current_player, state._x.color, 1 - state._x.color
    )
    return game.observe(state._x, color)


# A fixed Game instance for helpers (keeps JAX shapes static).
_GAME_11 = Game(size=DEFAULT_SIZE)


# ----------------------------------------------------------------------
# Env
# ----------------------------------------------------------------------
class Hex(core.Env):
    """
    Speed Hex environment: identical rules/obs/action space to pgx Hex,
    but adds a per-RL-player clock and allows step((action, time_spent)).
    """

    def __init__(self, *, size: int = DEFAULT_SIZE, default_time: int = DEFAULT_TIME):
        super().__init__()
        assert isinstance(size, int)
        assert isinstance(default_time, int)
        # To keep shapes static and match pgx Hex wrappers, we strongly assume size==11 here.
        # If you truly need variable size, you'd usually create a separate module per size.
        if size != DEFAULT_SIZE:
            raise ValueError(
                f"speed_hex currently supports size={DEFAULT_SIZE} only (got size={size})."
            )
        self.size = size
        self.default_time = default_time
        self._game = Game(size=size)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _init(self, key: PRNGKey) -> State:
        # Randomly assign which RL player is black/white (and thus who moves first).
        _player_order = jnp.array([[0, 1], [1, 0]])[
            jax.random.bernoulli(key).astype(jnp.int32)
        ]
        x = self._game.init()

        s = State(  # type: ignore
            current_player=_player_order[x.color],
            _player_order=_player_order,
            _x=x,
            _time_left=jnp.int32([self.default_time, self.default_time]),
        )

        # Make init fully self-contained: set masks/obs explicitly.
        s = s.replace(  # type: ignore
            legal_action_mask=self._game.legal_action_mask(x),
            terminated=self._game.is_terminal(x),
            rewards=self._game.rewards(x)[_player_order],
            observation=self._game.observe(x, x.color),
        )
        return s

    # ------------------------------------------------------------------
    # Public step: accepts action or (action, time_spent)
    # ------------------------------------------------------------------
    def step(self, state: State, action_input, key=None) -> State:
        """
        Step function that accepts either:
          - action (int / Array)
          - (action, time_spent)
        """
        # Unpack (action, time_spent) or default time_spent = 0
        if isinstance(action_input, (tuple, list)):
            action, time_spent = action_input
        else:
            action = action_input
            time_spent = jnp.int32(0)

        # PGX legality check must be based on *pre-step* mask
        is_illegal = ~state.legal_action_mask[action]
        current_player = state.current_player

        # If already terminated / truncated, just zero rewards
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(rewards=jnp.zeros_like(state.rewards)),
            lambda: self._step(
                state.replace(_step_count=state._step_count + 1),
                (action, time_spent),
                key,
            ),
        )

        # Illegal action -> immediate penalty / terminal (core.Env helper)
        state = jax.lax.cond(
            is_illegal,
            lambda: self._step_with_illegal_action(state, current_player),
            lambda: state,
        )

        # At terminal state: mask all actions as legal (PGX convention)
        state = jax.lax.cond(
            state.terminated,
            lambda: state.replace(
                legal_action_mask=jnp.ones_like(state.legal_action_mask)
            ),
            lambda: state,
        )

        # Update observation (for current_player view)
        state = state.replace(observation=self.observe(state))
        return state

    # ------------------------------------------------------------------
    # Internal _step: apply move + time control
    # ------------------------------------------------------------------
    def _step(self, state: core.State, action_input, key) -> State:  # type: ignore[override]
        del key
        assert isinstance(state, State)

        if isinstance(action_input, (tuple, list)):
            action, time_spent = action_input
        else:
            action = action_input
            time_spent = jnp.int32(0)

        # Subtract time for current RL player (before move flips player)
        idx = state.current_player
        new_time = state._time_left.at[idx].add(-time_spent)

        # Clamp BOTH clocks at 0: "no timeout losses"
        new_time = jnp.maximum(new_time, 0)

        # Apply pure board step (no clocks)
        state = _step_board(state, action)

        # Base termination and rewards from Hex rules ONLY
        terminated = state.terminated
        rewards = state.rewards

        # Global move cap safety
        terminated = terminated | (state._step_count >= MAX_TERMINATION_STEPS)

        state = state.replace(  # type: ignore
            _time_left=new_time,
            terminated=terminated,
            rewards=rewards,
        )
        return state  # type: ignore

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state, player_id)

    # ------------------------------------------------------------------
    # Env metadata
    # ------------------------------------------------------------------
    @property
    def id(self) -> core.EnvId:
        # Keep "hex" for compatibility; semantics now include time control.
        return "hex"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2