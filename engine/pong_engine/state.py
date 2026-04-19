from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin
from typing import Literal

from pong_engine.config import (
    ARENA_HEIGHT,
    ARENA_WIDTH,
    BALL_INITIAL_SPEED,
    BALL_SIZE,
    PADDLE_HEIGHT,
)
from pong_engine.errors import InvalidGameStateError


Action = Literal["up", "down", "none"]
LAUNCH_MIN_ANGLE_DEGREES = 12.0
LAUNCH_MAX_ANGLE_DEGREES = 45.0


@dataclass(frozen=True)
class GameState:
    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    ball_speed: float
    paddle_left_y: int
    paddle_right_y: int
    score_left: int
    score_right: int
    tick: int

    def __post_init__(self) -> None:
        if self.ball_speed <= 0:
            raise InvalidGameStateError("ball_speed must be greater than zero.")
        if self.paddle_left_y < 0:
            raise InvalidGameStateError("paddle_left_y must be non-negative.")
        if self.paddle_right_y < 0:
            raise InvalidGameStateError("paddle_right_y must be non-negative.")
        if self.score_left < 0 or self.score_right < 0:
            raise InvalidGameStateError("scores must be non-negative.")
        if self.tick < 0:
            raise InvalidGameStateError("tick must be non-negative.")


def create_initial_state(rng: object | None = None) -> GameState:
    ball_vx, ball_vy = sample_launch_velocity(BALL_INITIAL_SPEED, rng)
    return GameState(
        ball_x=(ARENA_WIDTH - BALL_SIZE) / 2,
        ball_y=(ARENA_HEIGHT - BALL_SIZE) / 2,
        ball_vx=ball_vx,
        ball_vy=ball_vy,
        ball_speed=BALL_INITIAL_SPEED,
        paddle_left_y=(ARENA_HEIGHT - PADDLE_HEIGHT) // 2,
        paddle_right_y=(ARENA_HEIGHT - PADDLE_HEIGHT) // 2,
        score_left=0,
        score_right=0,
        tick=0,
    )


def sample_launch_velocity(speed: float, rng: object | None) -> tuple[float, float]:
    if rng is None:
        return speed, 0.0

    horizontal_direction = 1.0 if draw_random_value(rng) >= 0.5 else -1.0
    vertical_direction = 1.0 if draw_random_value(rng) >= 0.5 else -1.0
    angle_degrees = LAUNCH_MIN_ANGLE_DEGREES + draw_random_value(rng) * (
        LAUNCH_MAX_ANGLE_DEGREES - LAUNCH_MIN_ANGLE_DEGREES
    )
    angle_radians = radians(angle_degrees)
    return (
        speed * cos(angle_radians) * horizontal_direction,
        speed * sin(angle_radians) * vertical_direction,
    )


def draw_random_value(rng: object) -> float:
    if hasattr(rng, "random"):
        return float(getattr(rng, "random")())
    if hasattr(rng, "next"):
        return float(getattr(rng, "next")())
    raise InvalidGameStateError("rng must expose random() or next().")
