from __future__ import annotations

from collections import deque
from dataclasses import dataclass


class InvalidMetricsWindowError(ValueError):
    """Raised when the metrics window is not a positive integer."""


@dataclass(frozen=True)
class TrainingMetrics:
    episode: int
    loss: float
    total_reward: float
    episode_length: int
    avg_loss_window: float
    avg_reward_window: float


class MetricsTracker:
    def __init__(self, window: int) -> None:
        if window <= 0:
            raise InvalidMetricsWindowError("window must be a positive integer.")
        self._window = window
        self._losses: deque[float] = deque(maxlen=window)
        self._rewards: deque[float] = deque(maxlen=window)
        self._history: list[TrainingMetrics] = []

    def record(
        self,
        *,
        loss: float,
        total_reward: float,
        episode_length: int,
    ) -> TrainingMetrics:
        loss_value = float(loss)
        reward_value = float(total_reward)
        self._losses.append(loss_value)
        self._rewards.append(reward_value)
        metrics = TrainingMetrics(
            episode=len(self._history),
            loss=loss_value,
            total_reward=reward_value,
            episode_length=int(episode_length),
            avg_loss_window=sum(self._losses) / len(self._losses),
            avg_reward_window=sum(self._rewards) / len(self._rewards),
        )
        self._history.append(metrics)
        return metrics

    @property
    def history(self) -> tuple[TrainingMetrics, ...]:
        return tuple(self._history)

    @property
    def window(self) -> int:
        return self._window
