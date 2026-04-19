from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from frame_stack import FrameStack
from model import ACTIONS, PongPolicyNetwork
from policy import sample_action_index
from pong_engine import create_initial_state, partially_tracking, step
from pong_engine.state import GameState
from rollout import RolloutEpisode
from rollout import compute_reward, reached_score_limit


RewardFn = Callable[[GameState, GameState], float]


class InvalidDiscountError(ValueError):
    """Raised when discount configuration is invalid."""


class InvalidReinforceLossError(ValueError):
    """Raised when REINFORCE loss inputs are inconsistent."""


@dataclass(frozen=True)
class ReinforceUpdateResult:
    loss: float
    total_reward: float
    episode_length: int
    returns: np.ndarray


def discount_rewards(rewards: Sequence[float], gamma: float) -> np.ndarray:
    validate_gamma(gamma)
    rewards_array = np.asarray(rewards, dtype=np.float64)
    discounted = np.zeros_like(rewards_array, dtype=np.float64)
    running_total = 0.0

    for index in range(len(rewards_array) - 1, -1, -1):
        running_total = float(rewards_array[index]) + gamma * running_total
        discounted[index] = running_total

    return discounted


def discount_episode_rewards(episode: RolloutEpisode, gamma: float) -> np.ndarray:
    return discount_rewards([step.reward for step in episode.steps], gamma)


def normalize_returns(returns: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    returns_array = np.asarray(returns, dtype=np.float64)
    if returns_array.size == 0:
        return returns_array.copy()

    std = float(returns_array.std())
    if std <= epsilon:
        return np.zeros_like(returns_array, dtype=np.float64)

    mean = float(returns_array.mean())
    return (returns_array - mean) / std


def compute_reinforce_loss(
    log_probabilities: Sequence[torch.Tensor],
    returns: torch.Tensor,
) -> torch.Tensor:
    if len(log_probabilities) == 0:
        raise InvalidReinforceLossError("log_probabilities must contain at least one item.")
    if len(log_probabilities) != len(returns):
        raise InvalidReinforceLossError("log_probabilities and returns must have the same length.")

    log_probabilities_tensor = torch.stack(tuple(log_probabilities))
    aligned_returns = returns.to(
        device=log_probabilities_tensor.device,
        dtype=log_probabilities_tensor.dtype,
    )
    return -(log_probabilities_tensor * aligned_returns).mean()


def run_reinforce_update(
    network: PongPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    *,
    seed: int,
    max_steps: int,
    gamma: float,
    initial_state: GameState | None = None,
    normalize_returns_for_loss: bool = False,
    reward_fn: RewardFn | None = None,
) -> ReinforceUpdateResult:
    validate_gamma(gamma)
    if max_steps <= 0:
        raise InvalidReinforceLossError("max_steps must be greater than zero.")

    effective_reward_fn: RewardFn = reward_fn if reward_fn is not None else compute_reward

    rng = np.random.default_rng(seed)
    state = initial_state or create_initial_state(rng)
    frame_stack = FrameStack(debug_capacity=0)
    rewards: list[float] = []
    log_probabilities: list[torch.Tensor] = []
    model_device = next(network.parameters()).device

    network.train()

    while len(rewards) < max_steps and not reached_score_limit(state):
        frame_stack.push_state(state)
        observation = frame_stack.as_float32_flat(copy=False)
        inputs = torch.from_numpy(observation).to(device=model_device).unsqueeze(0)
        outputs = network(inputs)
        logits = outputs.logits.squeeze(0)
        probabilities = torch.softmax(logits, dim=0)
        action_index = sample_action_index(probabilities.detach().cpu().numpy(), rng)
        log_probability = torch.log_softmax(logits, dim=0)[action_index]
        action_right = ACTIONS[action_index]
        action_left = partially_tracking(state, rng)
        next_state = step(state, action_left, action_right, rng)
        rewards.append(effective_reward_fn(state, next_state))
        log_probabilities.append(log_probability)
        state = next_state

    returns = discount_rewards(rewards, gamma)
    if normalize_returns_for_loss:
        returns = normalize_returns(returns)
    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=model_device)
    loss = compute_reinforce_loss(log_probabilities, returns_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return ReinforceUpdateResult(
        loss=float(loss.item()),
        total_reward=float(sum(rewards)),
        episode_length=len(rewards),
        returns=returns.copy(),
    )


def validate_gamma(gamma: float) -> None:
    if gamma < 0.0 or gamma > 1.0:
        raise InvalidDiscountError("gamma must be within the closed interval [0, 1].")
