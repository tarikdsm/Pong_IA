from __future__ import annotations

import numpy as np

from pong_engine.config import ARENA_HEIGHT, ARENA_WIDTH, PADDLE_HEIGHT
from pong_engine.state import GameState


DEFAULT_REBOUND_REWARD = 1.0
DEFAULT_MISS_PENALTY = -1.0
DEFAULT_CENTERING_REWARD_SCALE = 0.02
DEFAULT_CENTER_HOLD_BONUS = 0.0005
DEFAULT_IDLE_MOVEMENT_PENALTY = 0.0005
DEFAULT_CENTERING_WINDOW_RATIO = 0.65

_ARENA_VERTICAL_CENTER = ARENA_HEIGHT / 2.0
_PADDLE_MAX_CENTER_DISTANCE = (ARENA_HEIGHT - PADDLE_HEIGHT) / 2.0


def should_apply_centering_bias(
    ball_x: float,
    ball_vx: float,
    *,
    centering_window_ratio: float = DEFAULT_CENTERING_WINDOW_RATIO,
) -> bool:
    return ball_vx <= 0.0 or ball_x < ARENA_WIDTH * centering_window_ratio


def normalized_right_paddle_center_distance(paddle_right_y: float | np.ndarray) -> float | np.ndarray:
    paddle_center = np.asarray(paddle_right_y, dtype=np.float32) + (PADDLE_HEIGHT / 2.0)
    normalized_distance = np.abs(paddle_center - _ARENA_VERTICAL_CENTER) / _PADDLE_MAX_CENTER_DISTANCE
    clipped = np.clip(normalized_distance, 0.0, 1.0)
    if np.ndim(clipped) == 0:
        return float(clipped)
    return clipped


def centering_reward(
    state_before: GameState,
    state_after: GameState,
    *,
    centering_reward_scale: float = DEFAULT_CENTERING_REWARD_SCALE,
    center_hold_bonus: float = DEFAULT_CENTER_HOLD_BONUS,
    idle_movement_penalty: float = DEFAULT_IDLE_MOVEMENT_PENALTY,
    centering_window_ratio: float = DEFAULT_CENTERING_WINDOW_RATIO,
) -> float:
    if not should_apply_centering_bias(
        state_before.ball_x,
        state_before.ball_vx,
        centering_window_ratio=centering_window_ratio,
    ):
        return 0.0

    before_distance = float(normalized_right_paddle_center_distance(state_before.paddle_right_y))
    after_distance = float(normalized_right_paddle_center_distance(state_after.paddle_right_y))
    reward = centering_reward_scale * (before_distance - after_distance)
    reward += center_hold_bonus * (1.0 - after_distance)
    if state_after.paddle_right_y != state_before.paddle_right_y:
        reward -= idle_movement_penalty
    return reward


def compute_batched_centering_rewards(
    *,
    previous_ball_x: np.ndarray,
    previous_ball_vx: np.ndarray,
    previous_paddle_right_y: np.ndarray,
    paddle_right_y: np.ndarray,
    centering_reward_scale: float = DEFAULT_CENTERING_REWARD_SCALE,
    center_hold_bonus: float = DEFAULT_CENTER_HOLD_BONUS,
    idle_movement_penalty: float = DEFAULT_IDLE_MOVEMENT_PENALTY,
    centering_window_ratio: float = DEFAULT_CENTERING_WINDOW_RATIO,
) -> np.ndarray:
    rewards = np.zeros_like(previous_ball_vx, dtype=np.float32)
    waiting_mask = (previous_ball_vx <= 0.0) | (previous_ball_x < ARENA_WIDTH * centering_window_ratio)
    before_distance = normalized_right_paddle_center_distance(previous_paddle_right_y)
    after_distance = normalized_right_paddle_center_distance(paddle_right_y)
    rewards[waiting_mask] += centering_reward_scale * (
        before_distance[waiting_mask] - after_distance[waiting_mask]
    )
    rewards[waiting_mask] += center_hold_bonus * (1.0 - after_distance[waiting_mask])
    moved_mask = waiting_mask & (paddle_right_y != previous_paddle_right_y)
    rewards[moved_mask] -= idle_movement_penalty
    return rewards


def compute_batched_rebound_rewards(
    *,
    previous_ball_x: np.ndarray,
    previous_ball_vx: np.ndarray,
    previous_paddle_right_y: np.ndarray,
    previous_score_left: np.ndarray,
    previous_score_right: np.ndarray,
    paddle_right_y: np.ndarray,
    score_left: np.ndarray,
    score_right: np.ndarray,
    ball_vx: np.ndarray,
    hit_reward: float = DEFAULT_REBOUND_REWARD,
    miss_penalty: float = DEFAULT_MISS_PENALTY,
    centering_reward_scale: float = DEFAULT_CENTERING_REWARD_SCALE,
    center_hold_bonus: float = DEFAULT_CENTER_HOLD_BONUS,
    idle_movement_penalty: float = DEFAULT_IDLE_MOVEMENT_PENALTY,
    centering_window_ratio: float = DEFAULT_CENTERING_WINDOW_RATIO,
) -> np.ndarray:
    rewards = compute_batched_centering_rewards(
        previous_ball_x=previous_ball_x,
        previous_ball_vx=previous_ball_vx,
        previous_paddle_right_y=previous_paddle_right_y,
        paddle_right_y=paddle_right_y,
        centering_reward_scale=centering_reward_scale,
        center_hold_bonus=center_hold_bonus,
        idle_movement_penalty=idle_movement_penalty,
        centering_window_ratio=centering_window_ratio,
    )

    miss_mask = score_left > previous_score_left
    rewards[miss_mask] = miss_penalty

    right_paddle_hit_mask = (
        (score_left == previous_score_left)
        & (score_right == previous_score_right)
        & (previous_ball_vx > 0.0)
        & (ball_vx < 0.0)
    )
    rewards[right_paddle_hit_mask] = hit_reward
    return rewards


def right_paddle_hit(state_before: GameState, state_after: GameState) -> bool:
    no_goal = (
        state_after.score_left == state_before.score_left
        and state_after.score_right == state_before.score_right
    )
    return no_goal and state_before.ball_vx > 0.0 and state_after.ball_vx < 0.0


def right_paddle_miss(state_before: GameState, state_after: GameState) -> bool:
    return state_after.score_left > state_before.score_left


def rebound_reward(
    state_before: GameState,
    state_after: GameState,
    *,
    hit_reward: float = DEFAULT_REBOUND_REWARD,
    miss_penalty: float = DEFAULT_MISS_PENALTY,
    centering_reward_scale: float = DEFAULT_CENTERING_REWARD_SCALE,
    center_hold_bonus: float = DEFAULT_CENTER_HOLD_BONUS,
    idle_movement_penalty: float = DEFAULT_IDLE_MOVEMENT_PENALTY,
    centering_window_ratio: float = DEFAULT_CENTERING_WINDOW_RATIO,
) -> float:
    if right_paddle_hit(state_before, state_after):
        return hit_reward
    if right_paddle_miss(state_before, state_after):
        return miss_penalty
    return centering_reward(
        state_before,
        state_after,
        centering_reward_scale=centering_reward_scale,
        center_hold_bonus=center_hold_bonus,
        idle_movement_penalty=idle_movement_penalty,
        centering_window_ratio=centering_window_ratio,
    )
