from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import optim

from model import PongPolicyNetwork
from pong_engine.state import GameState
from reinforce import (
    InvalidDiscountError,
    InvalidReinforceLossError,
    ReinforceUpdateResult,
    compute_reinforce_loss,
    discount_episode_rewards,
    discount_rewards,
    normalize_returns,
    run_reinforce_update,
)
from rollout import EpisodeStep, RolloutEpisode


def test_discount_rewards_accumulates_future_rewards_with_gamma() -> None:
    discounted = discount_rewards([1.0, 0.0, -1.0], gamma=0.5)

    assert discounted.tolist() == pytest.approx([0.75, -0.5, -1.0])


def test_discount_episode_rewards_matches_episode_length() -> None:
    episode = RolloutEpisode(
        steps=(
            make_step(reward=1.0, tick=0),
            make_step(reward=0.0, tick=1),
            make_step(reward=-1.0, tick=2),
        ),
        final_state=make_state(tick=3),
        total_reward=0.0,
    )

    discounted = discount_episode_rewards(episode, gamma=0.5)

    assert discounted.shape == (3,)
    assert discounted.tolist() == pytest.approx([0.75, -0.5, -1.0])


def test_normalize_returns_centers_and_scales_values() -> None:
    normalized = normalize_returns(np.array([1.0, 2.0, 3.0], dtype=np.float64))

    assert float(normalized.mean()) == pytest.approx(0.0, abs=1e-7)
    assert float(normalized.std()) == pytest.approx(1.0, abs=1e-7)


def test_normalize_returns_returns_zeros_for_constant_vector() -> None:
    normalized = normalize_returns(np.array([2.0, 2.0, 2.0], dtype=np.float64))

    assert normalized.tolist() == [0.0, 0.0, 0.0]


def test_discount_rewards_rejects_gamma_outside_closed_interval() -> None:
    with pytest.raises(InvalidDiscountError, match="gamma"):
        discount_rewards([1.0], gamma=-0.1)

    with pytest.raises(InvalidDiscountError, match="gamma"):
        discount_rewards([1.0], gamma=1.1)


def test_compute_reinforce_loss_matches_expected_formula() -> None:
    log_probabilities = (
        torch.tensor(-0.2, dtype=torch.float32, requires_grad=True),
        torch.tensor(-0.4, dtype=torch.float32, requires_grad=True),
    )
    returns = torch.tensor([1.0, 2.0], dtype=torch.float32)

    loss = compute_reinforce_loss(log_probabilities, returns)

    expected = -((-0.2 * 1.0) + (-0.4 * 2.0)) / 2
    assert float(loss.item()) == pytest.approx(expected)


def test_compute_reinforce_loss_rejects_mismatched_lengths() -> None:
    log_probabilities = (torch.tensor(-0.2, dtype=torch.float32, requires_grad=True),)
    returns = torch.tensor([1.0, 2.0], dtype=torch.float32)

    with pytest.raises(InvalidReinforceLossError, match="same length"):
        compute_reinforce_loss(log_probabilities, returns)


def test_run_reinforce_update_returns_finite_metrics_and_updates_parameters() -> None:
    network = build_uniform_network()
    optimizer = optim.SGD(network.parameters(), lr=0.01)
    initial_state = GameState(
        ball_x=1.1,
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
    parameters_before = [parameter.detach().clone() for parameter in network.parameters()]

    result = run_reinforce_update(
        network,
        optimizer,
        seed=5,
        max_steps=5,
        gamma=0.9,
        initial_state=initial_state,
    )

    assert isinstance(result, ReinforceUpdateResult)
    assert np.isfinite(result.loss)
    assert np.isfinite(result.total_reward)
    assert result.episode_length == 2
    assert result.returns.shape == (2,)
    assert any(
        not torch.allclose(before, after.detach())
        for before, after in zip(parameters_before, network.parameters(), strict=True)
    )


def make_step(*, reward: float, tick: int) -> EpisodeStep:
    state_before = make_state(tick=tick)
    state_after = make_state(tick=tick + 1)
    return EpisodeStep(
        observation=np.zeros(24000, dtype=np.uint8),
        action_right="none",
        reward=reward,
        state_before=state_before,
        state_after=state_after,
    )


def make_state(*, tick: int) -> GameState:
    return GameState(
        ball_x=10.0,
        ball_y=10.0,
        ball_vx=1.0,
        ball_vy=0.0,
        ball_speed=1.0,
        paddle_left_y=20,
        paddle_right_y=20,
        score_left=0,
        score_right=0,
        tick=tick,
    )


def build_uniform_network() -> PongPolicyNetwork:
    network = PongPolicyNetwork()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
    return network
