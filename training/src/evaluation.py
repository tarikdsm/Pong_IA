from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import torch

from frame_stack import FrameStack
from model import ACTIONS, PongPolicyNetwork
from pong_engine import create_initial_state, partially_tracking, step
from pong_engine.config import INFERENCE_INTERVAL_TICKS, SCORE_TO_WIN
from reward_shaping import right_paddle_hit, right_paddle_miss


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: int
    hit_count: int
    miss_count: int
    attempt_count: int
    hit_rate: float
    hit_rate_lower_bound: float
    avg_hits_per_episode: float
    avg_misses_per_episode: float
    avg_episode_length: float


def wilson_lower_bound(successes: int, attempts: int, z: float = 1.96) -> float:
    if attempts <= 0:
        return 0.0

    proportion = successes / attempts
    z_squared = z * z
    denominator = 1.0 + z_squared / attempts
    center = proportion + z_squared / (2.0 * attempts)
    variance = (proportion * (1.0 - proportion) + z_squared / (4.0 * attempts)) / attempts
    margin = z * sqrt(variance)
    return max(0.0, (center - margin) / denominator)


def evaluate_policy(
    network: PongPolicyNetwork,
    *,
    seed: int,
    episodes: int,
    max_steps: int,
) -> EvaluationSummary:
    if episodes <= 0:
        raise ValueError("episodes must be a positive integer.")
    if max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")

    total_hits = 0
    total_misses = 0
    total_episode_length = 0
    model_device = next(network.parameters()).device
    was_training = network.training

    network.eval()
    with torch.no_grad():
        for episode_index in range(episodes):
            rng = np.random.default_rng(seed + episode_index)
            state = create_initial_state(rng)
            frame_stack = FrameStack(debug_capacity=0)
            last_action = "none"
            step_count = 0

            while step_count < max_steps and not _reached_score_limit(state):
                frame_stack.push_state(state)
                if frame_stack.is_ready() and state.tick % INFERENCE_INTERVAL_TICKS == 0:
                    observation = frame_stack.as_float32_flat(copy=False)
                    inputs = torch.from_numpy(observation).to(device=model_device).unsqueeze(0)
                    logits = network(inputs).logits.squeeze(0)
                    last_action = ACTIONS[int(torch.argmax(logits).item())]

                left_action = partially_tracking(state, rng)
                next_state = step(state, left_action, last_action, rng)
                if right_paddle_hit(state, next_state):
                    total_hits += 1
                elif right_paddle_miss(state, next_state):
                    total_misses += 1

                state = next_state
                step_count += 1

            total_episode_length += step_count

    if was_training:
        network.train()

    attempts = total_hits + total_misses
    hit_rate = (total_hits / attempts) if attempts > 0 else 0.0
    return EvaluationSummary(
        episodes=episodes,
        hit_count=total_hits,
        miss_count=total_misses,
        attempt_count=attempts,
        hit_rate=hit_rate,
        hit_rate_lower_bound=wilson_lower_bound(total_hits, attempts),
        avg_hits_per_episode=total_hits / episodes,
        avg_misses_per_episode=total_misses / episodes,
        avg_episode_length=total_episode_length / episodes,
    )


def _reached_score_limit(state) -> bool:
    return state.score_left >= SCORE_TO_WIN or state.score_right >= SCORE_TO_WIN
