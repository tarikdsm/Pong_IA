from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from model import ACTIONS, INPUT_DIM, PongPolicyNetwork
from pong_engine.state import Action
from rollout import Policy


class InvalidPolicyObservationError(ValueError):
    """Raised when a policy observation does not match the expected flat input."""


@dataclass(frozen=True)
class PolicyDecision:
    action: Action
    action_index: int
    probabilities: np.ndarray
    logits: np.ndarray
    log_probability: float


def decide_action(
    network: PongPolicyNetwork,
    observation: np.ndarray,
    rng: np.random.Generator,
) -> PolicyDecision:
    observation_array = validate_observation(observation)
    inputs = torch.from_numpy(observation_array.astype(np.float32, copy=False)).unsqueeze(0)

    with torch.no_grad():
        outputs = network(inputs)
        logits_tensor = outputs.logits.squeeze(0)
        probabilities_tensor = torch.softmax(logits_tensor, dim=0)

    logits = logits_tensor.cpu().numpy()
    probabilities = probabilities_tensor.cpu().numpy()
    action_index = sample_action_index(probabilities, rng)
    action = ACTIONS[action_index]
    log_probability = float(np.log(probabilities[action_index]))
    return PolicyDecision(
        action=action,
        action_index=action_index,
        probabilities=probabilities,
        logits=logits,
        log_probability=log_probability,
    )


def make_policy_callback(network: PongPolicyNetwork) -> Policy:
    def policy(observation: np.ndarray, rng: np.random.Generator) -> Action:
        return decide_action(network, observation, rng).action

    return policy


def sample_action_index(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(ACTIONS), p=probabilities))


def validate_observation(observation: np.ndarray) -> np.ndarray:
    observation_array = np.asarray(observation, dtype=np.uint8)
    if observation_array.ndim != 1:
        raise InvalidPolicyObservationError(
            f"policy observation must be a flat vector with shape ({INPUT_DIM},)."
        )
    if observation_array.shape[0] != INPUT_DIM:
        raise InvalidPolicyObservationError(
            f"policy observation must have length {INPUT_DIM}, got {observation_array.shape[0]}."
        )
    return observation_array
