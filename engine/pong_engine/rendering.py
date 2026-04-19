from __future__ import annotations

import numpy as np

from pong_engine.config import (
    ARENA_HEIGHT,
    ARENA_WIDTH,
    BALL_SIZE,
    PADDLE_HEIGHT,
    PADDLE_WIDTH,
)
from pong_engine.state import GameState


def bitmap_from_state(state: GameState) -> np.ndarray:
    bitmap = np.zeros((ARENA_HEIGHT, ARENA_WIDTH), dtype=np.uint8)
    bitmap[
        state.paddle_left_y : state.paddle_left_y + PADDLE_HEIGHT,
        0:PADDLE_WIDTH,
    ] = 1
    bitmap[
        state.paddle_right_y : state.paddle_right_y + PADDLE_HEIGHT,
        ARENA_WIDTH - PADDLE_WIDTH : ARENA_WIDTH,
    ] = 1
    ball_x = int(state.ball_x)
    ball_y = int(state.ball_y)
    bitmap[ball_y : ball_y + BALL_SIZE, ball_x : ball_x + BALL_SIZE] = 1
    return bitmap
