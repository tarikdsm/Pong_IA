from __future__ import annotations

import numpy as np

from pong_engine.heuristics import partially_tracking


def test_partially_tracking_follows_ball_when_ball_moves_toward_left(state_builder) -> None:
    rng = np.random.default_rng(42)
    state = state_builder.with_updates(
        ball_y=5.0,
        ball_vx=-1.5,
        paddle_left_y=24,
    ).build()

    action = partially_tracking(state, rng)

    assert action == "up"


def test_partially_tracking_uses_rng_when_ball_moves_away_from_left(state_builder) -> None:
    rng = np.random.default_rng(42)
    state = state_builder.with_updates(
        ball_y=40.0,
        ball_vx=1.5,
        paddle_left_y=24,
    ).build()

    action = partially_tracking(state, rng)

    assert action == "none"
