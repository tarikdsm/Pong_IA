from __future__ import annotations

import numpy as np
import pytest
import torch

from model import ACTIONS, INPUT_DIM, PongPolicyNetwork
from policy import InvalidPolicyObservationError, decide_action, make_policy_callback
from rollout import run_episode


@pytest.mark.parametrize("preferred_index, expected_action", enumerate(ACTIONS))
def test_decide_action_maps_logits_index_to_expected_action(
    preferred_index: int,
    expected_action: str,
) -> None:
    network = build_biased_network(preferred_index)
    observation = np.zeros(INPUT_DIM, dtype=np.uint8)
    rng = np.random.default_rng(7)

    decision = decide_action(network, observation, rng)

    assert decision.action == expected_action
    assert decision.action_index == preferred_index
    assert decision.probabilities.shape == (len(ACTIONS),)
    assert float(decision.probabilities.sum()) == pytest.approx(1.0)


def test_decide_action_is_deterministic_for_same_seed_and_model() -> None:
    network = build_uniform_network()
    observation = np.zeros(INPUT_DIM, dtype=np.uint8)

    first_rng = np.random.default_rng(42)
    second_rng = np.random.default_rng(42)

    first_sequence = [decide_action(network, observation, first_rng).action for _ in range(8)]
    second_sequence = [decide_action(network, observation, second_rng).action for _ in range(8)]

    assert first_sequence == second_sequence


def test_decide_action_rejects_invalid_observation_shape() -> None:
    network = build_uniform_network()
    invalid_observation = np.zeros((2, INPUT_DIM), dtype=np.uint8)

    with pytest.raises(InvalidPolicyObservationError, match=str(INPUT_DIM)):
        decide_action(network, invalid_observation, np.random.default_rng(1))


def test_make_policy_callback_plugs_directly_into_run_episode() -> None:
    network = build_uniform_network()
    episode = run_episode(make_policy_callback(network), seed=5, max_steps=3)

    assert len(episode.steps) == 3
    assert all(step.action_right in ACTIONS for step in episode.steps)


def build_uniform_network() -> PongPolicyNetwork:
    network = PongPolicyNetwork()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
    return network


def build_biased_network(preferred_index: int) -> PongPolicyNetwork:
    network = build_uniform_network()
    with torch.no_grad():
        network.output_layer.bias.fill_(-50.0)
        network.output_layer.bias[preferred_index] = 50.0
    return network
