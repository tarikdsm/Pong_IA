from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin
from typing import Sequence

import numpy as np
import torch

from frame_stack import FrameStack
from model import ACTIONS, PongPolicyNetwork
from pong_engine.config import (
    ARENA_HEIGHT,
    ARENA_WIDTH,
    BALL_ACCELERATION_FACTOR,
    BALL_INITIAL_SPEED,
    BALL_MAX_SPEED,
    BALL_SIZE,
    PADDLE_HEIGHT,
    PADDLE_SPEED,
    PADDLE_WIDTH,
    SCORE_TO_WIN,
)
from pong_engine.physics import collides_with_left_paddle, collides_with_right_paddle, rescale_velocity
from pong_engine.state import (
    LAUNCH_MAX_ANGLE_DEGREES,
    LAUNCH_MIN_ANGLE_DEGREES,
    Action,
    create_initial_state,
)
from policy import sample_action_index
from reinforce import (
    InvalidReinforceLossError,
    ReinforceUpdateResult,
    compute_reinforce_loss,
    discount_rewards,
    normalize_returns,
    validate_gamma,
)
from reward_shaping import (
    DEFAULT_CENTERING_REWARD_SCALE,
    DEFAULT_CENTERING_WINDOW_RATIO,
    DEFAULT_CENTER_HOLD_BONUS,
    DEFAULT_IDLE_MOVEMENT_PENALTY,
    DEFAULT_MISS_PENALTY,
    DEFAULT_REBOUND_REWARD,
    compute_batched_rebound_rewards,
)


@dataclass(slots=True)
class BatchedTrainingState:
    ball_x: np.ndarray
    ball_y: np.ndarray
    ball_vx: np.ndarray
    ball_vy: np.ndarray
    ball_speed: np.ndarray
    paddle_left_y: np.ndarray
    paddle_right_y: np.ndarray
    score_left: np.ndarray
    score_right: np.ndarray
    tick: np.ndarray


def run_batched_reinforce_updates(
    network: PongPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    *,
    seeds: Sequence[int],
    max_steps: int,
    gamma: float,
    normalize_returns_for_loss: bool = False,
    hit_reward: float = DEFAULT_REBOUND_REWARD,
    miss_penalty: float = DEFAULT_MISS_PENALTY,
    centering_reward_scale: float = DEFAULT_CENTERING_REWARD_SCALE,
    center_hold_bonus: float = DEFAULT_CENTER_HOLD_BONUS,
    idle_movement_penalty: float = DEFAULT_IDLE_MOVEMENT_PENALTY,
    centering_window_ratio: float = DEFAULT_CENTERING_WINDOW_RATIO,
) -> tuple[ReinforceUpdateResult, ...]:
    validate_gamma(gamma)
    if max_steps <= 0:
        raise InvalidReinforceLossError("max_steps must be greater than zero.")
    if len(seeds) == 0:
        raise InvalidReinforceLossError("seeds must contain at least one item.")

    rngs = [np.random.default_rng(seed) for seed in seeds]
    state = _create_initial_batch_state(rngs)
    frame_stacks = [FrameStack(debug_capacity=0) for _ in seeds]
    observations = np.zeros((len(seeds), network.input_layer.in_features), dtype=np.float32)
    rewards: list[list[float]] = [[] for _ in seeds]
    log_probabilities: list[list[torch.Tensor]] = [[] for _ in seeds]
    step_counts = np.zeros(len(seeds), dtype=np.int32)
    active = np.ones(len(seeds), dtype=bool)
    model_device = next(network.parameters()).device

    network.train()

    while np.any(active):
        active_indices = np.flatnonzero(active)
        for index in active_indices:
            frame_stacks[index].push_components(
                ball_x=state.ball_x[index],
                ball_y=state.ball_y[index],
                paddle_left_y=int(state.paddle_left_y[index]),
                paddle_right_y=int(state.paddle_right_y[index]),
            )
            observations[index] = frame_stacks[index].as_float32_flat(copy=False)

        inputs = torch.from_numpy(observations).to(device=model_device)
        logits = network(inputs).logits
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
        log_softmax = torch.log_softmax(logits, dim=1)

        action_indices = np.full(len(seeds), 2, dtype=np.int64)
        left_actions = np.full(len(seeds), "none", dtype=object)
        previous_ball_x = state.ball_x.copy()
        previous_ball_vx = state.ball_vx.copy()
        previous_paddle_right_y = state.paddle_right_y.copy()
        previous_score_left = state.score_left.copy()
        previous_score_right = state.score_right.copy()

        for index in active_indices:
            action_indices[index] = sample_action_index(probabilities[index], rngs[index])
            left_actions[index] = _left_heuristic_action(
                ball_x=state.ball_x[index],
                ball_y=state.ball_y[index],
                ball_vx=state.ball_vx[index],
                paddle_left_y=int(state.paddle_left_y[index]),
                rng=rngs[index],
            )
            _step_single_env(state, index, left_actions[index], ACTIONS[action_indices[index]], rngs[index])

        action_index_tensor = torch.tensor(action_indices, device=log_softmax.device, dtype=torch.int64)
        selected_log_probs = log_softmax[torch.arange(len(seeds), device=log_softmax.device), action_index_tensor]
        reward_values = _compute_batched_rewards(
            previous_ball_x=previous_ball_x,
            previous_score_left=previous_score_left,
            previous_score_right=previous_score_right,
            previous_ball_vx=previous_ball_vx,
            previous_paddle_right_y=previous_paddle_right_y,
            state=state,
            hit_reward=hit_reward,
            miss_penalty=miss_penalty,
            centering_reward_scale=centering_reward_scale,
            center_hold_bonus=center_hold_bonus,
            idle_movement_penalty=idle_movement_penalty,
            centering_window_ratio=centering_window_ratio,
        )

        for index in active_indices:
            rewards[index].append(float(reward_values[index]))
            log_probabilities[index].append(selected_log_probs[index])
            step_counts[index] += 1
            if step_counts[index] >= max_steps or _reached_score_limit_at_index(state, index):
                active[index] = False

    all_log_probabilities: list[torch.Tensor] = []
    all_returns: list[np.ndarray] = []
    results: list[ReinforceUpdateResult] = []

    for index in range(len(seeds)):
        episode_returns = discount_rewards(rewards[index], gamma)
        if normalize_returns_for_loss:
            episode_returns = normalize_returns(episode_returns)
        results.append(
            ReinforceUpdateResult(
                loss=0.0,
                total_reward=float(sum(rewards[index])),
                episode_length=len(rewards[index]),
                returns=episode_returns.copy(),
            )
        )
        all_log_probabilities.extend(log_probabilities[index])
        all_returns.append(episode_returns)

    returns_tensor = torch.tensor(
        np.concatenate(all_returns),
        dtype=torch.float32,
        device=model_device,
    )
    loss = compute_reinforce_loss(all_log_probabilities, returns_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return tuple(
        ReinforceUpdateResult(
            loss=float(loss.item()),
            total_reward=result.total_reward,
            episode_length=result.episode_length,
            returns=result.returns,
        )
        for result in results
    )


def _create_initial_batch_state(rngs: Sequence[np.random.Generator]) -> BatchedTrainingState:
    initial_states = [create_initial_state(rng) for rng in rngs]
    return BatchedTrainingState(
        ball_x=np.array([state.ball_x for state in initial_states], dtype=np.float32),
        ball_y=np.array([state.ball_y for state in initial_states], dtype=np.float32),
        ball_vx=np.array([state.ball_vx for state in initial_states], dtype=np.float32),
        ball_vy=np.array([state.ball_vy for state in initial_states], dtype=np.float32),
        ball_speed=np.array([state.ball_speed for state in initial_states], dtype=np.float32),
        paddle_left_y=np.array([state.paddle_left_y for state in initial_states], dtype=np.int32),
        paddle_right_y=np.array([state.paddle_right_y for state in initial_states], dtype=np.int32),
        score_left=np.array([state.score_left for state in initial_states], dtype=np.int32),
        score_right=np.array([state.score_right for state in initial_states], dtype=np.int32),
        tick=np.array([state.tick for state in initial_states], dtype=np.int32),
    )


def _left_heuristic_action(
    *,
    ball_x: float,
    ball_y: float,
    ball_vx: float,
    paddle_left_y: int,
    rng: np.random.Generator,
) -> Action:
    del ball_x
    if ball_vx < 0:
        paddle_center = paddle_left_y + (PADDLE_HEIGHT / 2)
        ball_center = ball_y + (BALL_SIZE / 2)
        if ball_center < paddle_center - 1:
            return "up"
        if ball_center > paddle_center + 1:
            return "down"
        return "none"

    value = float(rng.random())
    if value < 1 / 3:
        return "up"
    if value < 2 / 3:
        return "down"
    return "none"


def _step_single_env(
    state: BatchedTrainingState,
    index: int,
    a_left: Action,
    a_right: Action,
    rng: np.random.Generator,
) -> None:
    paddle_left_y = _move_paddle(int(state.paddle_left_y[index]), a_left)
    paddle_right_y = _move_paddle(int(state.paddle_right_y[index]), a_right)

    next_x = float(state.ball_x[index] + state.ball_vx[index])
    next_y = float(state.ball_y[index] + state.ball_vy[index])
    next_vx = float(state.ball_vx[index])
    next_vy = float(state.ball_vy[index])
    next_speed = float(state.ball_speed[index])

    if next_y < 0:
        next_y = -next_y
        next_vy = abs(next_vy)
    elif next_y + BALL_SIZE > ARENA_HEIGHT:
        overflow = next_y + BALL_SIZE - ARENA_HEIGHT
        next_y = ARENA_HEIGHT - BALL_SIZE - overflow
        next_vy = -abs(next_vy)

    if next_vx < 0 and collides_with_left_paddle(next_x, next_y, paddle_left_y):
        next_x = PADDLE_WIDTH
        next_speed = min(float(state.ball_speed[index]) * BALL_ACCELERATION_FACTOR, BALL_MAX_SPEED)
        next_vx, next_vy = rescale_velocity(abs(next_vx), next_vy, next_speed)
    elif next_vx > 0 and collides_with_right_paddle(next_x, next_y, paddle_right_y):
        next_x = ARENA_WIDTH - PADDLE_WIDTH - BALL_SIZE
        next_speed = min(float(state.ball_speed[index]) * BALL_ACCELERATION_FACTOR, BALL_MAX_SPEED)
        reflected_vx, next_vy = rescale_velocity(abs(next_vx), next_vy, next_speed)
        next_vx = -reflected_vx

    if next_x < 0:
        ball_vx, ball_vy = _sample_launch_velocity(BALL_INITIAL_SPEED, rng)
        state.ball_x[index] = (ARENA_WIDTH - BALL_SIZE) / 2
        state.ball_y[index] = (ARENA_HEIGHT - BALL_SIZE) / 2
        state.ball_vx[index] = ball_vx
        state.ball_vy[index] = ball_vy
        state.ball_speed[index] = BALL_INITIAL_SPEED
        state.paddle_left_y[index] = paddle_left_y
        state.paddle_right_y[index] = paddle_right_y
        state.score_right[index] += 1
        state.tick[index] += 1
        return

    if next_x + BALL_SIZE > ARENA_WIDTH:
        ball_vx, ball_vy = _sample_launch_velocity(BALL_INITIAL_SPEED, rng)
        state.ball_x[index] = (ARENA_WIDTH - BALL_SIZE) / 2
        state.ball_y[index] = (ARENA_HEIGHT - BALL_SIZE) / 2
        state.ball_vx[index] = ball_vx
        state.ball_vy[index] = ball_vy
        state.ball_speed[index] = BALL_INITIAL_SPEED
        state.paddle_left_y[index] = paddle_left_y
        state.paddle_right_y[index] = paddle_right_y
        state.score_left[index] += 1
        state.tick[index] += 1
        return

    state.ball_x[index] = next_x
    state.ball_y[index] = next_y
    state.ball_vx[index] = next_vx
    state.ball_vy[index] = next_vy
    state.ball_speed[index] = next_speed
    state.paddle_left_y[index] = paddle_left_y
    state.paddle_right_y[index] = paddle_right_y
    state.tick[index] += 1


def _compute_batched_rewards(
    *,
    previous_ball_x: np.ndarray,
    previous_score_left: np.ndarray,
    previous_score_right: np.ndarray,
    previous_ball_vx: np.ndarray,
    previous_paddle_right_y: np.ndarray,
    state: BatchedTrainingState,
    hit_reward: float,
    miss_penalty: float,
    centering_reward_scale: float,
    center_hold_bonus: float,
    idle_movement_penalty: float,
    centering_window_ratio: float,
) -> np.ndarray:
    return compute_batched_rebound_rewards(
        previous_ball_x=previous_ball_x,
        previous_ball_vx=previous_ball_vx,
        previous_paddle_right_y=previous_paddle_right_y,
        previous_score_left=previous_score_left,
        previous_score_right=previous_score_right,
        paddle_right_y=state.paddle_right_y,
        score_left=state.score_left,
        score_right=state.score_right,
        ball_vx=state.ball_vx,
        hit_reward=hit_reward,
        miss_penalty=miss_penalty,
        centering_reward_scale=centering_reward_scale,
        center_hold_bonus=center_hold_bonus,
        idle_movement_penalty=idle_movement_penalty,
        centering_window_ratio=centering_window_ratio,
    )


def _reached_score_limit_at_index(state: BatchedTrainingState, index: int) -> bool:
    return bool(state.score_left[index] >= SCORE_TO_WIN or state.score_right[index] >= SCORE_TO_WIN)


def _move_paddle(current_y: int, action: Action) -> int:
    delta = 0
    if action == "up":
        delta = -PADDLE_SPEED
    elif action == "down":
        delta = PADDLE_SPEED
    return max(0, min(current_y + delta, ARENA_HEIGHT - PADDLE_HEIGHT))


def _sample_launch_velocity(speed: float, rng: np.random.Generator) -> tuple[float, float]:
    horizontal_direction = 1.0 if float(rng.random()) >= 0.5 else -1.0
    vertical_direction = 1.0 if float(rng.random()) >= 0.5 else -1.0
    angle_degrees = LAUNCH_MIN_ANGLE_DEGREES + float(rng.random()) * (
        LAUNCH_MAX_ANGLE_DEGREES - LAUNCH_MIN_ANGLE_DEGREES
    )
    angle_radians = radians(angle_degrees)
    return (
        speed * cos(angle_radians) * horizontal_direction,
        speed * sin(angle_radians) * vertical_direction,
    )
