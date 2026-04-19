from __future__ import annotations

from math import hypot

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
from pong_engine.errors import InvalidActionError
from pong_engine.heuristics import ensure_rng
from pong_engine.state import Action, GameState, sample_launch_velocity


VALID_ACTIONS = {"up", "down", "none"}


def step(
    state: GameState,
    a_left: Action,
    a_right: Action,
    rng: object | None,
) -> GameState:
    ensure_rng(rng)
    validate_action(a_left)
    validate_action(a_right)

    paddle_left_y = move_paddle(state.paddle_left_y, a_left)
    paddle_right_y = move_paddle(state.paddle_right_y, a_right)

    next_x = state.ball_x + state.ball_vx
    next_y = state.ball_y + state.ball_vy
    next_vx = state.ball_vx
    next_vy = state.ball_vy
    next_speed = state.ball_speed

    if next_y < 0:
        next_y = -next_y
        next_vy = abs(next_vy)
    elif next_y + BALL_SIZE > ARENA_HEIGHT:
        overflow = next_y + BALL_SIZE - ARENA_HEIGHT
        next_y = ARENA_HEIGHT - BALL_SIZE - overflow
        next_vy = -abs(next_vy)

    if next_vx < 0 and collides_with_left_paddle(next_x, next_y, paddle_left_y):
        next_x = PADDLE_WIDTH
        next_speed = min(state.ball_speed * BALL_ACCELERATION_FACTOR, BALL_MAX_SPEED)
        next_vx, next_vy = rescale_velocity(abs(next_vx), next_vy, next_speed)
    elif next_vx > 0 and collides_with_right_paddle(next_x, next_y, paddle_right_y):
        next_x = ARENA_WIDTH - PADDLE_WIDTH - BALL_SIZE
        next_speed = min(state.ball_speed * BALL_ACCELERATION_FACTOR, BALL_MAX_SPEED)
        reflected_vx, next_vy = rescale_velocity(abs(next_vx), next_vy, next_speed)
        next_vx = -reflected_vx

    if next_x < 0:
        return create_reset_state(state, paddle_left_y, paddle_right_y, "right", rng)
    if next_x + BALL_SIZE > ARENA_WIDTH:
        return create_reset_state(state, paddle_left_y, paddle_right_y, "left", rng)

    return GameState(
        ball_x=next_x,
        ball_y=next_y,
        ball_vx=next_vx,
        ball_vy=next_vy,
        ball_speed=next_speed,
        paddle_left_y=paddle_left_y,
        paddle_right_y=paddle_right_y,
        score_left=state.score_left,
        score_right=state.score_right,
        tick=state.tick + 1,
    )


def validate_action(action: Action) -> None:
    if action not in VALID_ACTIONS:
        raise InvalidActionError(f"Invalid action: {action}")


def move_paddle(current_y: int, action: Action) -> int:
    delta = 0
    if action == "up":
        delta = -PADDLE_SPEED
    elif action == "down":
        delta = PADDLE_SPEED
    return clamp(current_y + delta, 0, ARENA_HEIGHT - PADDLE_HEIGHT)


def collides_with_left_paddle(ball_x: float, ball_y: float, paddle_y: int) -> bool:
    return ball_x <= PADDLE_WIDTH and overlaps_paddle(ball_y, paddle_y)


def collides_with_right_paddle(ball_x: float, ball_y: float, paddle_y: int) -> bool:
    paddle_x = ARENA_WIDTH - PADDLE_WIDTH
    return ball_x + BALL_SIZE >= paddle_x and overlaps_paddle(ball_y, paddle_y)


def overlaps_paddle(ball_y: float, paddle_y: int) -> bool:
    return ball_y + BALL_SIZE > paddle_y and ball_y < paddle_y + PADDLE_HEIGHT


def rescale_velocity(vx: float, vy: float, speed: float) -> tuple[float, float]:
    magnitude = hypot(vx, vy)
    if magnitude == 0:
        return speed, 0.0
    scale = speed / magnitude
    return vx * scale, vy * scale


def create_reset_state(
    state: GameState,
    paddle_left_y: int,
    paddle_right_y: int,
    scorer: str,
    rng: object | None,
) -> GameState:
    ball_vx, ball_vy = sample_launch_velocity(BALL_INITIAL_SPEED, rng)
    score_left = state.score_left + (1 if scorer == "left" else 0)
    score_right = state.score_right + (1 if scorer == "right" else 0)
    return GameState(
        ball_x=(ARENA_WIDTH - BALL_SIZE) / 2,
        ball_y=(ARENA_HEIGHT - BALL_SIZE) / 2,
        ball_vx=ball_vx,
        ball_vy=ball_vy,
        ball_speed=BALL_INITIAL_SPEED,
        paddle_left_y=paddle_left_y,
        paddle_right_y=paddle_right_y,
        score_left=score_left,
        score_right=score_right,
        tick=state.tick + 1,
    )


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))
