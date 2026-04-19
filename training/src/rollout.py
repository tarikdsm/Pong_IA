from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from frame_stack import FrameStack
from pong_engine import create_initial_state, partially_tracking, step
from pong_engine.config import SCORE_TO_WIN
from pong_engine.state import Action, GameState


Policy = Callable[[np.ndarray, np.random.Generator], Action]


class InvalidRolloutError(ValueError):
    """Raised when rollout configuration is invalid."""


@dataclass(frozen=True)
class EpisodeStep:
    observation: np.ndarray
    action_right: Action
    reward: float
    state_before: GameState
    state_after: GameState


@dataclass(frozen=True)
class RolloutEpisode:
    steps: tuple[EpisodeStep, ...]
    final_state: GameState
    total_reward: float


def run_episode(
    policy_right: Policy,
    *,
    seed: int,
    max_steps: int,
    initial_state: GameState | None = None,
) -> RolloutEpisode:
    if max_steps <= 0:
        raise InvalidRolloutError("max_steps must be greater than zero.")

    rng = np.random.default_rng(seed)
    state = initial_state or create_initial_state(rng)
    frame_stack = FrameStack()
    steps: list[EpisodeStep] = []

    while len(steps) < max_steps and not reached_score_limit(state):
        frame_stack.push_state(state)
        observation = frame_stack.as_flat(copy=False)
        action_right = policy_right(observation.copy(), rng)
        action_left = partially_tracking(state, rng)
        next_state = step(state, action_left, action_right, rng)
        reward = compute_reward(state, next_state)
        steps.append(
            EpisodeStep(
                observation=observation.copy(),
                action_right=action_right,
                reward=reward,
                state_before=state,
                state_after=next_state,
            )
        )
        state = next_state

    total_reward = float(sum(step_entry.reward for step_entry in steps))
    return RolloutEpisode(steps=tuple(steps), final_state=state, total_reward=total_reward)


def reached_score_limit(state: GameState) -> bool:
    return state.score_left >= SCORE_TO_WIN or state.score_right >= SCORE_TO_WIN


def compute_reward(previous_state: GameState, next_state: GameState) -> float:
    if next_state.score_right > previous_state.score_right:
        return 1.0
    if next_state.score_left > previous_state.score_left:
        return -1.0
    return 0.0
