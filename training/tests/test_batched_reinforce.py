from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import optim

from batched_reinforce import run_batched_reinforce_updates
from model import PongPolicyNetwork
from reinforce import ReinforceUpdateResult


def test_run_batched_reinforce_updates_returns_one_result_per_seed() -> None:
    network = build_uniform_network()
    optimizer = optim.SGD(network.parameters(), lr=0.01)

    results = run_batched_reinforce_updates(
        network,
        optimizer,
        seeds=[5, 6, 7],
        max_steps=5,
        gamma=0.9,
    )

    assert len(results) == 3
    assert all(isinstance(result, ReinforceUpdateResult) for result in results)
    assert all(np.isfinite(result.loss) for result in results)
    assert all(result.episode_length > 0 for result in results)


def test_run_batched_reinforce_updates_is_deterministic_for_same_seeds() -> None:
    first_network = build_uniform_network()
    second_network = build_uniform_network()
    first_optimizer = optim.SGD(first_network.parameters(), lr=0.01)
    second_optimizer = optim.SGD(second_network.parameters(), lr=0.01)

    first = run_batched_reinforce_updates(
        first_network,
        first_optimizer,
        seeds=[10, 11],
        max_steps=6,
        gamma=0.95,
    )
    second = run_batched_reinforce_updates(
        second_network,
        second_optimizer,
        seeds=[10, 11],
        max_steps=6,
        gamma=0.95,
    )

    for left, right in zip(first, second, strict=True):
        assert left.loss == pytest.approx(right.loss)
        assert left.total_reward == pytest.approx(right.total_reward)
        assert left.episode_length == right.episode_length
        assert left.returns.tolist() == pytest.approx(right.returns.tolist())


def build_uniform_network() -> PongPolicyNetwork:
    network = PongPolicyNetwork()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
    return network
