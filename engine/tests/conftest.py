from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from pong_engine.config import (
    ARENA_HEIGHT,
    ARENA_WIDTH,
    BALL_INITIAL_SPEED,
    BALL_SIZE,
    PADDLE_HEIGHT,
)
from pong_engine.state import GameState, create_initial_state


class StateBuilder:
    def __init__(self) -> None:
        self._state = create_initial_state()

    def with_updates(self, **updates: Any) -> "StateBuilder":
        self._state = replace(self._state, **updates)
        return self

    def build(self) -> GameState:
        return self._state


@pytest.fixture
def state_builder() -> StateBuilder:
    return StateBuilder()


@pytest.fixture
def canonical_state() -> GameState:
    return create_initial_state()


@pytest.fixture
def centered_ball_state() -> GameState:
    return GameState(
        ball_x=(ARENA_WIDTH - BALL_SIZE) / 2,
        ball_y=(ARENA_HEIGHT - BALL_SIZE) / 2,
        ball_vx=BALL_INITIAL_SPEED,
        ball_vy=0.0,
        ball_speed=BALL_INITIAL_SPEED,
        paddle_left_y=(ARENA_HEIGHT - PADDLE_HEIGHT) // 2,
        paddle_right_y=(ARENA_HEIGHT - PADDLE_HEIGHT) // 2,
        score_left=0,
        score_right=0,
        tick=0,
    )
