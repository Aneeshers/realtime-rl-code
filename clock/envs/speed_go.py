"""
speed_go.py

A "speed" (clocked) 9x9 Go environment analogous to speed_hex.py.

- State includes per-RL-player clocks: State._time_left (private) + State.time_left (read-only)
- Env.step accepts either:
    - action
    - (action, time_spent)
- time_spent is subtracted from the CURRENT RL player before the move flips.
- Exposes pure board helpers for MCTS:
    - _step_board(state, action): apply Go rules only (no clocks)
    - _observe(state, player_id): observation only (no clocks)

This is the non-timeout variant: clocks are clamped at zero and do not cause
immediate losses, matching speed_hex.py rather than speed_hex_timeout.py.
"""

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.games.go import Game, GameState
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

# ----------------------------------------------------------------------
# Time-control knobs
# ----------------------------------------------------------------------
DEFAULT_SIZE = 9
DEFAULT_KOMI = 7.5
DEFAULT_HISTORY_LENGTH = 8
MAX_TERMINATION_STEPS = DEFAULT_SIZE * DEFAULT_SIZE * 2
DEFAULT_TIME = 300

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

INIT_LEGAL_ACTION_MASK = jnp.ones(DEFAULT_SIZE * DEFAULT_SIZE + 1, dtype=jnp.bool_)


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------
@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = INIT_LEGAL_ACTION_MASK
    observation: Array = jnp.zeros(
        (DEFAULT_SIZE, DEFAULT_SIZE, DEFAULT_HISTORY_LENGTH * 2 + 1), dtype=jnp.bool_
    )
    _step_count: Array = jnp.int32(0)

    # --- Go specific ---
    _player_order: Array = jnp.int32([0, 1])
    _x: GameState = GameState(
        board=jnp.zeros(DEFAULT_SIZE * DEFAULT_SIZE, dtype=jnp.int32),
        board_history=jnp.full(
            (DEFAULT_HISTORY_LENGTH, DEFAULT_SIZE * DEFAULT_SIZE), 2, dtype=jnp.int32
        ),
        hash_history=jnp.zeros((MAX_TERMINATION_STEPS, 2), dtype=jnp.uint32),
    )

    # --- Speed-go time control (per RL player: 0/1) ---
    _time_left: Array = jnp.int32([DEFAULT_TIME, DEFAULT_TIME])

    @property
    def time_left(self) -> Array:
        return self._time_left

    @property
    def env_id(self) -> core.EnvId:
        return "go_9x9"


# ----------------------------------------------------------------------
# Pure board helpers (for MCTS)
# ----------------------------------------------------------------------
def _step_board(state: State, action: Array) -> State:
    x = _GAME_9.step(state._x, action)
    return state.replace(  # type: ignore
        current_player=state._player_order[x.color],
        legal_action_mask=_GAME_9.legal_action_mask(x),
        rewards=_GAME_9.rewards(x)[state._player_order],
        terminated=_GAME_9.is_terminal(x),
        _x=x,
    )


def _observe(state: State, player_id: Array) -> Array:
    curr_color = state._x.color
    my_turn = jax.lax.select(player_id == state.current_player, curr_color, 1 - curr_color)
    return _GAME_9.observe(state._x, my_turn)


_GAME_9 = Game(
    size=DEFAULT_SIZE,
    komi=DEFAULT_KOMI,
    history_length=DEFAULT_HISTORY_LENGTH,
    max_termination_steps=MAX_TERMINATION_STEPS,
)


# ----------------------------------------------------------------------
# Env
# ----------------------------------------------------------------------
class Go(core.Env):
    """
    Speed Go environment: identical rules/obs/action space to PGX Go 9x9,
    but adds a per-RL-player clock and allows step((action, time_spent)).
    """

    def __init__(
        self,
        *,
        size: int = DEFAULT_SIZE,
        komi: float = DEFAULT_KOMI,
        history_length: int = DEFAULT_HISTORY_LENGTH,
        default_time: int = DEFAULT_TIME,
    ):
        super().__init__()
        if size is None:
            size = DEFAULT_SIZE
        if size != DEFAULT_SIZE:
            raise ValueError(
                f"speed_go currently supports size={DEFAULT_SIZE} only (got size={size})."
            )
        if history_length != DEFAULT_HISTORY_LENGTH:
            raise ValueError(
                f"speed_go currently supports history_length={DEFAULT_HISTORY_LENGTH} only "
                f"(got history_length={history_length})."
            )
        self.size = size
        self.komi = komi
        self.history_length = history_length
        self.default_time = int(default_time)
        self._game = Game(
            size=size,
            komi=komi,
            history_length=history_length,
            max_termination_steps=MAX_TERMINATION_STEPS,
        )

    def _init(self, key: PRNGKey) -> State:
        _player_order = jnp.array([[0, 1], [1, 0]])[
            jax.random.bernoulli(key).astype(jnp.int32)
        ]
        x = self._game.init()
        s = State(  # type: ignore
            current_player=_player_order[x.color],
            legal_action_mask=self._game.legal_action_mask(x),
            _player_order=_player_order,
            _x=x,
            _time_left=jnp.int32([self.default_time, self.default_time]),
        )
        s = s.replace(  # type: ignore
            rewards=self._game.rewards(x)[_player_order],
            terminated=self._game.is_terminal(x),
            observation=self._game.observe(x, x.color),
        )
        return s

    def step(self, state: State, action_input, key=None) -> State:
        if isinstance(action_input, (tuple, list)):
            action, time_spent = action_input
        else:
            action = action_input
            time_spent = jnp.int32(0)

        is_illegal = ~state.legal_action_mask[action]
        current_player = state.current_player

        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(rewards=jnp.zeros_like(state.rewards)),
            lambda: self._step(
                state.replace(_step_count=state._step_count + 1),
                (action, time_spent),
                key,
            ),
        )

        state = jax.lax.cond(
            is_illegal,
            lambda: self._step_with_illegal_action(state, current_player),
            lambda: state,
        )

        state = jax.lax.cond(
            state.terminated,
            lambda: state.replace(legal_action_mask=jnp.ones_like(state.legal_action_mask)),
            lambda: state,
        )

        state = state.replace(observation=self.observe(state))
        return state

    def _step(self, state: core.State, action_input, key) -> State:  # type: ignore[override]
        del key
        assert isinstance(state, State)

        if isinstance(action_input, (tuple, list)):
            action, time_spent = action_input
        else:
            action = action_input
            time_spent = jnp.int32(0)

        idx = state.current_player
        new_time = state._time_left.at[idx].add(-time_spent)
        new_time = jnp.maximum(new_time, 0)

        state = _step_board(state, action)

        terminated = state.terminated | (state._step_count >= MAX_TERMINATION_STEPS)
        state = state.replace(  # type: ignore
            _time_left=new_time,
            terminated=terminated,
            rewards=state.rewards,
        )
        return state  # type: ignore

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def id(self) -> core.EnvId:
        return "go_9x9"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2


Env = Go
ENV_CLS = Go


def make_env(**kwargs):
    return Go(**kwargs)
