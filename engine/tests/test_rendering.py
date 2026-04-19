from __future__ import annotations

import numpy as np

from pong_engine.config import BALL_SIZE, PADDLE_HEIGHT, PADDLE_WIDTH
from pong_engine.rendering import bitmap_from_state


def test_bitmap_from_state_returns_binary_bitmap_with_ball_and_paddles(centered_ball_state) -> None:
    bitmap = bitmap_from_state(centered_ball_state)

    assert bitmap.shape == (60, 80)
    assert bitmap.dtype == np.uint8
    assert set(np.unique(bitmap)).issubset({0, 1})
    assert bitmap[centered_ball_state.paddle_left_y, 0] == 1
    assert bitmap[centered_ball_state.paddle_left_y + PADDLE_HEIGHT - 1, PADDLE_WIDTH - 1] == 1
    assert bitmap[int(centered_ball_state.ball_y), int(centered_ball_state.ball_x)] == 1
    assert (
        bitmap[int(centered_ball_state.ball_y) + BALL_SIZE - 1, int(centered_ball_state.ball_x)]
        == 1
    )
