from __future__ import annotations

from dataclasses import astuple

import numpy as np
import pytest

from pong_engine.state import GameState
from rollout import InvalidRolloutError, run_episode


def test_run_episode_returns_requested_number_of_steps_with_flat_observations() -> None:
    observed_shapes: list[tuple[int, ...]] = []
    observed_dtypes: list[np.dtype[np.uint8]] = []

    def policy(observation: np.ndarray, rng: np.random.Generator) -> str:
        observed_shapes.append(observation.shape)
        observed_dtypes.append(observation.dtype)
        assert isinstance(rng, np.random.Generator)
        return "none"

    episode = run_episode(policy, seed=7, max_steps=3)

    assert len(episode.steps) == 3
    assert observed_shapes == [(24000,), (24000,), (24000,)]
    assert observed_dtypes == [np.dtype(np.uint8)] * 3
    assert episode.total_reward == pytest.approx(sum(step.reward for step in episode.steps))


def test_run_episode_is_deterministic_for_same_seed_and_policy() -> None:
    def policy(observation: np.ndarray, rng: np.random.Generator) -> str:
        del observation
        return "up" if rng.random() < 0.5 else "down"

    first = run_episode(policy, seed=42, max_steps=8)
    second = run_episode(policy, seed=42, max_steps=8)

    first_trace = [
        (step.action_right, step.reward, astuple(step.state_before), astuple(step.state_after))
        for step in first.steps
    ]
    second_trace = [
        (step.action_right, step.reward, astuple(step.state_before), astuple(step.state_after))
        for step in second.steps
    ]

    assert first_trace == second_trace
    assert astuple(first.final_state) == astuple(second.final_state)
    assert first.total_reward == second.total_reward


def test_run_episode_assigns_positive_reward_when_right_scores() -> None:
    initial_state = GameState(
        ball_x=0.1,
        ball_y=30.0,
        ball_vx=-1.0,
        ball_vy=0.0,
        ball_speed=1.0,
        paddle_left_y=0,
        paddle_right_y=24,
        score_left=0,
        score_right=20,
        tick=0,
    )

    episode = run_episode(lambda observation, rng: "none", seed=5, max_steps=5, initial_state=initial_state)

    assert len(episode.steps) == 1
    assert episode.steps[0].reward == pytest.approx(1.0)
    assert episode.final_state.score_right == 21


def test_run_episode_rejects_non_positive_max_steps() -> None:
    with pytest.raises(InvalidRolloutError, match="max_steps"):
        run_episode(lambda observation, rng: "none", seed=1, max_steps=0)
