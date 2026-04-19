from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pong_engine.config import BITMAP_HEIGHT, BITMAP_WIDTH, FRAME_STACK
from pong_engine.state import Action


ACTIONS: tuple[Action, Action, Action] = ("up", "down", "none")
HIDDEN_DIMS = (200, 200, 100)
INPUT_DIM = FRAME_STACK * BITMAP_HEIGHT * BITMAP_WIDTH


class InvalidModelInputError(ValueError):
    """Raised when policy-network inputs violate the expected shape."""


@dataclass(frozen=True)
class PolicyForward:
    logits: torch.Tensor
    hidden_activations: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class PongPolicyNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layer = nn.Linear(INPUT_DIM, HIDDEN_DIMS[0])
        self.hidden_layer_one = nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1])
        self.hidden_layer_two = nn.Linear(HIDDEN_DIMS[1], HIDDEN_DIMS[2])
        self.output_layer = nn.Linear(HIDDEN_DIMS[2], len(ACTIONS))
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> PolicyForward:
        validate_inputs(inputs)

        x = inputs.to(dtype=self.input_layer.weight.dtype)
        hidden_one = self.activation(self.input_layer(x))
        hidden_two = self.activation(self.hidden_layer_one(hidden_one))
        hidden_three = self.activation(self.hidden_layer_two(hidden_two))
        logits = self.output_layer(hidden_three)
        return PolicyForward(
            logits=logits,
            hidden_activations=(hidden_one, hidden_two, hidden_three),
        )


def validate_inputs(inputs: torch.Tensor) -> None:
    if inputs.ndim != 2:
        raise InvalidModelInputError(
            f"policy network inputs must be a 2D tensor with shape (batch, {INPUT_DIM})."
        )
    if inputs.shape[1] != INPUT_DIM:
        raise InvalidModelInputError(
            f"policy network inputs must have feature dimension {INPUT_DIM}, got {inputs.shape[1]}."
        )
