from __future__ import annotations

import pytest
import torch

from model import (
    ACTIONS,
    HIDDEN_DIMS,
    INPUT_DIM,
    InvalidModelInputError,
    PongPolicyNetwork,
)


def test_policy_network_forward_returns_logits_and_hidden_activations() -> None:
    network = PongPolicyNetwork()
    inputs = torch.zeros((2, INPUT_DIM), dtype=torch.float32)

    outputs = network(inputs)

    assert outputs.logits.shape == (2, len(ACTIONS))
    assert outputs.logits.dtype == torch.float32
    assert tuple(activation.shape for activation in outputs.hidden_activations) == (
        (2, HIDDEN_DIMS[0]),
        (2, HIDDEN_DIMS[1]),
        (2, HIDDEN_DIMS[2]),
    )


def test_policy_network_uses_architecture_dimensions_from_project_contract() -> None:
    network = PongPolicyNetwork()

    assert ACTIONS == ("up", "down", "none")
    assert network.input_layer.weight.shape == (HIDDEN_DIMS[0], INPUT_DIM)
    assert network.output_layer.weight.shape == (len(ACTIONS), HIDDEN_DIMS[-1])


def test_policy_network_rejects_invalid_feature_dimension() -> None:
    network = PongPolicyNetwork()
    invalid_inputs = torch.zeros((2, INPUT_DIM - 1), dtype=torch.float32)

    with pytest.raises(InvalidModelInputError, match=str(INPUT_DIM)):
        network(invalid_inputs)


def test_policy_network_rejects_non_batched_input() -> None:
    network = PongPolicyNetwork()
    invalid_inputs = torch.zeros((INPUT_DIM,), dtype=torch.float32)

    with pytest.raises(InvalidModelInputError, match="2D"):
        network(invalid_inputs)
