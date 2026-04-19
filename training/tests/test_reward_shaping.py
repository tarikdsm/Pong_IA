from __future__ import annotations

import numpy as np
import pytest

from pong_engine.state import GameState
from reward_shaping import (
    centering_reward,
    compute_batched_rebound_rewards,
    rebound_reward,
    right_paddle_hit,
    right_paddle_miss,
)


def _state(
    *,
    ball_x: float = 40.0,
    ball_vx: float,
    paddle_right_y: int = 20,
    score_left: int = 0,
    score_right: int = 0,
    tick: int = 0,
) -> GameState:
    return GameState(
        ball_x=ball_x,
        ball_y=30.0,
        ball_vx=ball_vx,
        ball_vy=0.5,
        ball_speed=1.0,
        paddle_left_y=20,
        paddle_right_y=paddle_right_y,
        score_left=score_left,
        score_right=score_right,
        tick=tick,
    )


def test_right_paddle_hit_detects_velocity_flip_without_goal() -> None:
    before = _state(ball_vx=1.5)
    after = _state(ball_vx=-1.5, tick=1)

    assert right_paddle_hit(before, after) is True


def test_right_paddle_hit_returns_false_when_left_scores() -> None:
    before = _state(ball_vx=1.5, score_left=3)
    after = _state(ball_vx=-1.5, score_left=4, tick=1)

    assert right_paddle_hit(before, after) is False


def test_right_paddle_miss_detects_left_goal() -> None:
    before = _state(ball_vx=1.5, score_left=2)
    after = _state(ball_vx=1.5, score_left=3, tick=1)

    assert right_paddle_miss(before, after) is True


def test_right_paddle_miss_returns_false_on_right_goal() -> None:
    before = _state(ball_vx=-1.5, score_right=2)
    after = _state(ball_vx=-1.5, score_right=3, tick=1)

    assert right_paddle_miss(before, after) is False


def test_rebound_reward_returns_positive_reward_on_hit() -> None:
    before = _state(ball_vx=1.5)
    after = _state(ball_vx=-1.5, tick=1)

    shaped = rebound_reward(before, after, hit_reward=2.0, miss_penalty=-3.0)

    assert shaped == pytest.approx(2.0)


def test_rebound_reward_returns_negative_reward_on_miss() -> None:
    before = _state(ball_vx=1.5, score_left=1)
    after = _state(ball_vx=1.5, score_left=2, tick=1)

    shaped = rebound_reward(before, after, hit_reward=2.0, miss_penalty=-3.0)

    assert shaped == pytest.approx(-3.0)


def test_rebound_reward_returns_zero_when_center_bias_is_inactive() -> None:
    before = _state(ball_x=70.0, ball_vx=1.5)
    after = _state(ball_x=71.5, ball_vx=1.5, tick=1)

    shaped = rebound_reward(before, after)

    assert shaped == pytest.approx(0.0)


def test_centering_reward_is_positive_when_idle_paddle_moves_toward_center() -> None:
    before = _state(ball_x=30.0, ball_vx=1.5, paddle_right_y=32)
    after = _state(ball_x=31.5, ball_vx=1.5, paddle_right_y=30, tick=1)

    shaped = centering_reward(before, after)

    assert shaped > 0.0


def test_centering_reward_is_negative_when_idle_paddle_moves_away_from_center() -> None:
    before = _state(ball_x=25.0, ball_vx=-1.5, paddle_right_y=24)
    after = _state(ball_x=23.5, ball_vx=-1.5, paddle_right_y=26, tick=1)

    shaped = centering_reward(before, after)

    assert shaped < 0.0


def test_centering_reward_rewards_holding_the_center_while_waiting() -> None:
    before = _state(ball_x=15.0, ball_vx=-1.5, paddle_right_y=24)
    after = _state(ball_x=13.5, ball_vx=-1.5, paddle_right_y=24, tick=1)

    shaped = centering_reward(before, after)

    assert shaped > 0.0


def test_centering_reward_is_disabled_when_ball_is_close_and_approaching_right_paddle() -> None:
    before = _state(ball_x=70.0, ball_vx=1.5, paddle_right_y=24)
    after = _state(ball_x=71.5, ball_vx=1.5, paddle_right_y=22, tick=1)

    shaped = centering_reward(before, after)

    assert shaped == pytest.approx(0.0)


def test_batched_reward_matches_scalar_reward_for_idle_centering_case() -> None:
    before = _state(ball_x=30.0, ball_vx=1.5, paddle_right_y=32)
    after = _state(ball_x=31.5, ball_vx=1.5, paddle_right_y=30, tick=1)

    scalar_reward = rebound_reward(before, after)
    batched_reward = compute_batched_rebound_rewards(
        previous_ball_x=np.array([before.ball_x], dtype=np.float32),
        previous_ball_vx=np.array([before.ball_vx], dtype=np.float32),
        previous_paddle_right_y=np.array([before.paddle_right_y], dtype=np.int32),
        previous_score_left=np.array([before.score_left], dtype=np.int32),
        previous_score_right=np.array([before.score_right], dtype=np.int32),
        paddle_right_y=np.array([after.paddle_right_y], dtype=np.int32),
        score_left=np.array([after.score_left], dtype=np.int32),
        score_right=np.array([after.score_right], dtype=np.int32),
        ball_vx=np.array([after.ball_vx], dtype=np.float32),
    )[0]

    assert batched_reward == pytest.approx(scalar_reward)
