from __future__ import annotations

import numpy as np
import pytest

from pong_engine.config import (
    ARENA_HEIGHT,
    ARENA_WIDTH,
    BALL_ACCELERATION_FACTOR,
    BALL_INITIAL_SPEED,
    BALL_MAX_SPEED,
    BALL_SIZE,
    PADDLE_HEIGHT,
    PADDLE_SPEED,
    PADDLE_WIDTH,
)
from pong_engine.errors import MissingRngError
from pong_engine.physics import step


def test_step_moves_ball_and_paddles_without_collisions(centered_ball_state) -> None:
    rng = np.random.default_rng(42)

    next_state = step(centered_ball_state, "up", "down", rng)

    assert next_state.ball_x == pytest.approx(centered_ball_state.ball_x + BALL_INITIAL_SPEED)
    assert next_state.ball_y == pytest.approx(centered_ball_state.ball_y)
    assert next_state.paddle_left_y == centered_ball_state.paddle_left_y - PADDLE_SPEED
    assert next_state.paddle_right_y == centered_ball_state.paddle_right_y + PADDLE_SPEED
    assert next_state.tick == centered_ball_state.tick + 1


def test_step_keeps_diagonal_motion_active_when_no_collision_occurs(state_builder) -> None:
    rng = np.random.default_rng(42)
    state = state_builder.with_updates(
        ball_x=20.0,
        ball_y=20.0,
        ball_vx=1.1,
        ball_vy=0.7,
        ball_speed=(1.1**2 + 0.7**2) ** 0.5,
    ).build()

    next_state = step(state, "none", "none", rng)

    assert next_state.ball_x == pytest.approx(21.1)
    assert next_state.ball_y == pytest.approx(20.7)
    assert next_state.ball_vx == pytest.approx(1.1)
    assert next_state.ball_vy == pytest.approx(0.7)


def test_step_reflects_ball_on_top_wall_before_paddle_logic(state_builder) -> None:
    rng = np.random.default_rng(42)
    state = state_builder.with_updates(
        ball_x=PADDLE_WIDTH + 4.0,
        ball_y=0.5,
        ball_vx=-BALL_INITIAL_SPEED,
        ball_vy=-1.0,
        ball_speed=(BALL_INITIAL_SPEED**2 + 1.0) ** 0.5,
        paddle_left_y=0,
    ).build()

    next_state = step(state, "none", "none", rng)

    assert next_state.ball_vy == pytest.approx(1.0)
    assert next_state.ball_y == pytest.approx(0.5)
    assert next_state.ball_vx == pytest.approx(-BALL_INITIAL_SPEED)


def test_step_reflects_on_right_paddle_and_accelerates_ball(state_builder) -> None:
    rng = np.random.default_rng(42)
    state = state_builder.with_updates(
        ball_x=ARENA_WIDTH - PADDLE_WIDTH - BALL_SIZE - 0.25,
        ball_y=28.0,
        ball_vx=1.5,
        ball_vy=0.0,
        ball_speed=1.5,
        paddle_right_y=24,
    ).build()

    next_state = step(state, "none", "none", rng)

    assert next_state.ball_vx < 0
    assert next_state.ball_speed == pytest.approx(
        min(1.5 * BALL_ACCELERATION_FACTOR, BALL_MAX_SPEED)
    )
    assert abs(next_state.ball_vx) == pytest.approx(next_state.ball_speed)


def test_step_preserves_diagonal_motion_on_paddle_collision(state_builder) -> None:
    rng = np.random.default_rng(42)
    state = state_builder.with_updates(
        ball_x=ARENA_WIDTH - PADDLE_WIDTH - BALL_SIZE - 0.25,
        ball_y=28.0,
        ball_vx=1.2,
        ball_vy=0.6,
        ball_speed=(1.2**2 + 0.6**2) ** 0.5,
        paddle_right_y=24,
    ).build()

    next_state = step(state, "none", "none", rng)

    assert next_state.ball_vx < 0
    assert next_state.ball_vy > 0
    assert next_state.ball_speed == pytest.approx(
        min(state.ball_speed * BALL_ACCELERATION_FACTOR, BALL_MAX_SPEED)
    )
    assert next_state.ball_vx**2 + next_state.ball_vy**2 == pytest.approx(
        next_state.ball_speed**2
    )


def test_step_clamps_ball_speed_on_paddle_collision(state_builder) -> None:
    rng = np.random.default_rng(42)
    state = state_builder.with_updates(
        ball_x=PADDLE_WIDTH + 0.25,
        ball_y=30.0,
        ball_vx=-BALL_MAX_SPEED,
        ball_vy=0.0,
        ball_speed=BALL_MAX_SPEED,
        paddle_left_y=24,
    ).build()

    next_state = step(state, "none", "none", rng)

    assert next_state.ball_speed == pytest.approx(BALL_MAX_SPEED)
    assert next_state.ball_vx == pytest.approx(BALL_MAX_SPEED)


def test_step_awards_right_score_and_resets_ball_when_left_goal_occurs(state_builder) -> None:
    rng = np.random.default_rng(7)
    state = state_builder.with_updates(
        ball_x=0.25,
        ball_y=18.0,
        ball_vx=-1.5,
        ball_vy=0.0,
        ball_speed=1.5,
        paddle_left_y=ARENA_HEIGHT - PADDLE_HEIGHT,
    ).build()

    next_state = step(state, "none", "none", rng)

    assert next_state.score_right == state.score_right + 1
    assert next_state.score_left == state.score_left
    assert next_state.ball_x == pytest.approx((ARENA_WIDTH - BALL_SIZE) / 2)
    assert next_state.ball_y == pytest.approx((ARENA_HEIGHT - BALL_SIZE) / 2)
    assert next_state.ball_speed == pytest.approx(BALL_INITIAL_SPEED)
    assert abs(next_state.ball_vx) < BALL_INITIAL_SPEED
    assert next_state.ball_vy != pytest.approx(0.0)
    assert next_state.ball_vx**2 + next_state.ball_vy**2 == pytest.approx(BALL_INITIAL_SPEED**2)


def test_step_resets_accelerated_ball_to_min_speed_when_right_goal_occurs(state_builder) -> None:
    rng = np.random.default_rng(11)
    state = state_builder.with_updates(
        ball_x=ARENA_WIDTH - BALL_SIZE - 0.25,
        ball_y=22.0,
        ball_vx=BALL_MAX_SPEED,
        ball_vy=0.4,
        ball_speed=BALL_MAX_SPEED,
        paddle_right_y=0,
    ).build()

    next_state = step(state, "none", "none", rng)

    assert next_state.score_left == state.score_left + 1
    assert next_state.score_right == state.score_right
    assert next_state.ball_x == pytest.approx((ARENA_WIDTH - BALL_SIZE) / 2)
    assert next_state.ball_y == pytest.approx((ARENA_HEIGHT - BALL_SIZE) / 2)
    assert next_state.ball_speed == pytest.approx(BALL_INITIAL_SPEED)
    assert next_state.ball_vx**2 + next_state.ball_vy**2 == pytest.approx(BALL_INITIAL_SPEED**2)


def test_step_fails_with_clear_error_when_rng_is_missing(state_builder) -> None:
    state = state_builder.build()

    with pytest.raises(MissingRngError, match="rng"):
        step(state, "none", "none", None)
