from __future__ import annotations

from pong_engine.config import BALL_SIZE, PADDLE_HEIGHT
from pong_engine.errors import MissingRngError
from pong_engine.state import Action, GameState


def partially_tracking(state: GameState, rng: object | None) -> Action:
    if state.ball_vx < 0:
        paddle_center = state.paddle_left_y + (PADDLE_HEIGHT / 2)
        ball_center = state.ball_y + (BALL_SIZE / 2)
        if ball_center < paddle_center - 1:
            return "up"
        if ball_center > paddle_center + 1:
            return "down"
        return "none"

    value = draw_random(rng)
    if value < 1 / 3:
        return "up"
    if value < 2 / 3:
        return "down"
    return "none"


def draw_random(rng: object | None) -> float:
    ensure_rng(rng)
    if hasattr(rng, "random"):
        value = getattr(rng, "random")()
        return float(value)
    value = getattr(rng, "next")()
    return float(value)


def ensure_rng(rng: object | None) -> None:
    if rng is None:
        raise MissingRngError("rng is required for deterministic Pong engine behavior.")
    if hasattr(rng, "random") or hasattr(rng, "next"):
        return
    raise MissingRngError("rng must expose random() or next().")
