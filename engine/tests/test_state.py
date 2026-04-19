from __future__ import annotations

import numpy as np
import pytest

from pong_engine.config import BALL_INITIAL_SPEED
from pong_engine.errors import InvalidGameStateError
from pong_engine.state import GameState, create_initial_state


def test_create_initial_state_returns_canonical_centered_state() -> None:
    state = create_initial_state()

    assert state.ball_speed == pytest.approx(BALL_INITIAL_SPEED)
    assert state.ball_x == pytest.approx(39.0)
    assert state.ball_y == pytest.approx(29.0)
    assert state.ball_vx == pytest.approx(BALL_INITIAL_SPEED)
    assert state.ball_vy == pytest.approx(0.0)
    assert state.paddle_left_y == 24
    assert state.paddle_right_y == 24
    assert state.score_left == 0
    assert state.score_right == 0
    assert state.tick == 0


def test_game_state_rejects_negative_tick() -> None:
    with pytest.raises(InvalidGameStateError, match="tick"):
        GameState(
            ball_x=10.0,
            ball_y=10.0,
            ball_vx=1.0,
            ball_vy=0.0,
            ball_speed=1.0,
            paddle_left_y=2,
            paddle_right_y=2,
            score_left=0,
            score_right=0,
            tick=-1,
        )


def test_create_initial_state_with_rng_uses_random_launch_vector() -> None:
    rng = np.random.default_rng(42)

    state = create_initial_state(rng)

    assert state.ball_speed == pytest.approx(BALL_INITIAL_SPEED)
    assert abs(state.ball_vx) < BALL_INITIAL_SPEED
    assert state.ball_vy != pytest.approx(0.0)
    assert state.ball_vx**2 + state.ball_vy**2 == pytest.approx(BALL_INITIAL_SPEED**2)
