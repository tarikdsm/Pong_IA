from __future__ import annotations

import dataclasses

import pytest

from metrics import (
    InvalidMetricsWindowError,
    MetricsTracker,
    TrainingMetrics,
)


def test_training_metrics_is_a_frozen_dataclass_with_expected_fields() -> None:
    metrics = TrainingMetrics(
        episode=0,
        loss=1.0,
        total_reward=0.5,
        episode_length=10,
        avg_loss_window=1.0,
        avg_reward_window=0.5,
    )

    assert metrics.episode == 0
    assert metrics.loss == pytest.approx(1.0)
    assert metrics.total_reward == pytest.approx(0.5)
    assert metrics.episode_length == 10
    assert metrics.avg_loss_window == pytest.approx(1.0)
    assert metrics.avg_reward_window == pytest.approx(0.5)

    with pytest.raises(dataclasses.FrozenInstanceError):
        metrics.loss = 2.0  # type: ignore[misc]


def test_metrics_tracker_first_record_uses_itself_as_window_average() -> None:
    tracker = MetricsTracker(window=50)

    metrics = tracker.record(loss=1.5, total_reward=2.0, episode_length=5)

    assert metrics.episode == 0
    assert metrics.loss == pytest.approx(1.5)
    assert metrics.total_reward == pytest.approx(2.0)
    assert metrics.episode_length == 5
    assert metrics.avg_loss_window == pytest.approx(1.5)
    assert metrics.avg_reward_window == pytest.approx(2.0)


def test_metrics_tracker_increments_episode_index_on_each_record() -> None:
    tracker = MetricsTracker(window=50)

    first = tracker.record(loss=1.0, total_reward=0.0, episode_length=1)
    second = tracker.record(loss=2.0, total_reward=1.0, episode_length=2)
    third = tracker.record(loss=3.0, total_reward=2.0, episode_length=3)

    assert (first.episode, second.episode, third.episode) == (0, 1, 2)


def test_metrics_tracker_window_average_only_considers_last_n_records() -> None:
    tracker = MetricsTracker(window=2)

    tracker.record(loss=10.0, total_reward=100.0, episode_length=1)
    tracker.record(loss=20.0, total_reward=200.0, episode_length=2)
    third = tracker.record(loss=30.0, total_reward=300.0, episode_length=3)

    assert third.avg_loss_window == pytest.approx(25.0)
    assert third.avg_reward_window == pytest.approx(250.0)


def test_metrics_tracker_window_average_uses_available_records_when_below_window() -> None:
    tracker = MetricsTracker(window=10)

    tracker.record(loss=1.0, total_reward=0.0, episode_length=1)
    second = tracker.record(loss=3.0, total_reward=2.0, episode_length=1)

    assert second.avg_loss_window == pytest.approx(2.0)
    assert second.avg_reward_window == pytest.approx(1.0)


def test_metrics_tracker_history_returns_recorded_metrics_in_order() -> None:
    tracker = MetricsTracker(window=50)

    tracker.record(loss=1.0, total_reward=0.0, episode_length=1)
    tracker.record(loss=2.0, total_reward=1.0, episode_length=2)

    history = tracker.history

    assert len(history) == 2
    assert history[0].episode == 0
    assert history[0].loss == pytest.approx(1.0)
    assert history[1].episode == 1
    assert history[1].loss == pytest.approx(2.0)


def test_metrics_tracker_history_is_immutable_view() -> None:
    tracker = MetricsTracker(window=50)
    tracker.record(loss=1.0, total_reward=0.0, episode_length=1)

    history = tracker.history

    assert isinstance(history, tuple)
    with pytest.raises((TypeError, AttributeError)):
        history.append(  # type: ignore[attr-defined]
            TrainingMetrics(
                episode=99,
                loss=0.0,
                total_reward=0.0,
                episode_length=0,
                avg_loss_window=0.0,
                avg_reward_window=0.0,
            )
        )


def test_metrics_tracker_rejects_non_positive_window() -> None:
    with pytest.raises(InvalidMetricsWindowError, match="window"):
        MetricsTracker(window=0)

    with pytest.raises(InvalidMetricsWindowError, match="window"):
        MetricsTracker(window=-5)
