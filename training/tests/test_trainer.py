from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from checkpoint import load_checkpoint
from evaluation import EvaluationSummary, wilson_lower_bound
from reinforce import ReinforceUpdateResult
from trainer import (
    InvalidTrainerConfigError,
    TrainerConfig,
    TrainingRun,
    run_training,
)


def _evaluation_summary(hit_count: int, miss_count: int) -> EvaluationSummary:
    attempts = hit_count + miss_count
    hit_rate = hit_count / attempts if attempts > 0 else 0.0
    return EvaluationSummary(
        episodes=2,
        hit_count=hit_count,
        miss_count=miss_count,
        attempt_count=attempts,
        hit_rate=hit_rate,
        hit_rate_lower_bound=wilson_lower_bound(hit_count, attempts),
        avg_hits_per_episode=hit_count / 2.0,
        avg_misses_per_episode=miss_count / 2.0,
        avg_episode_length=10.0,
    )


def _config(tmp_path: Path, **overrides: Any) -> TrainerConfig:
    base: dict[str, Any] = dict(
        episodes=2,
        gamma=0.99,
        learning_rate=0.001,
        seed=42,
        max_steps=8,
        checkpoint_dir=tmp_path / "ckpts",
        checkpoint_every=1,
    )
    base.update(overrides)
    return TrainerConfig(**base)


def test_trainer_config_rejects_zero_episodes(tmp_path: Path) -> None:
    with pytest.raises(InvalidTrainerConfigError, match="episodes"):
        run_training(_config(tmp_path, episodes=0))


def test_trainer_config_rejects_non_positive_learning_rate(tmp_path: Path) -> None:
    with pytest.raises(InvalidTrainerConfigError, match="learning_rate"):
        run_training(_config(tmp_path, learning_rate=0.0))


def test_trainer_config_rejects_gamma_above_one(tmp_path: Path) -> None:
    with pytest.raises(InvalidTrainerConfigError, match="gamma"):
        run_training(_config(tmp_path, gamma=1.5))


def test_trainer_config_rejects_negative_gamma(tmp_path: Path) -> None:
    with pytest.raises(InvalidTrainerConfigError, match="gamma"):
        run_training(_config(tmp_path, gamma=-0.1))


def test_trainer_config_rejects_zero_checkpoint_every(tmp_path: Path) -> None:
    with pytest.raises(InvalidTrainerConfigError, match="checkpoint_every"):
        run_training(_config(tmp_path, checkpoint_every=0))


def test_trainer_config_rejects_zero_max_steps(tmp_path: Path) -> None:
    with pytest.raises(InvalidTrainerConfigError, match="max_steps"):
        run_training(_config(tmp_path, max_steps=0))


def test_trainer_config_rejects_invalid_device_name(tmp_path: Path) -> None:
    with pytest.raises(InvalidTrainerConfigError, match="device"):
        run_training(_config(tmp_path, device="tpu"))


def test_trainer_config_rejects_cuda_when_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("trainer.torch.cuda.is_available", lambda: False)

    with pytest.raises(InvalidTrainerConfigError, match="CUDA is unavailable"):
        run_training(_config(tmp_path, device="cuda"))


def test_trainer_config_rejects_non_positive_batch_envs(tmp_path: Path) -> None:
    with pytest.raises(InvalidTrainerConfigError, match="batch_envs"):
        run_training(_config(tmp_path, batch_envs=0))


def test_trainer_config_rejects_non_positive_hit_reward(tmp_path: Path) -> None:
    with pytest.raises(InvalidTrainerConfigError, match="hit_reward"):
        run_training(_config(tmp_path, hit_reward=0.0))


def test_trainer_config_rejects_non_negative_miss_penalty(tmp_path: Path) -> None:
    with pytest.raises(InvalidTrainerConfigError, match="miss_penalty"):
        run_training(_config(tmp_path, miss_penalty=0.0))


def test_trainer_config_rejects_negative_centering_reward_scale(tmp_path: Path) -> None:
    with pytest.raises(InvalidTrainerConfigError, match="centering_reward_scale"):
        run_training(_config(tmp_path, centering_reward_scale=-0.1))


def test_trainer_config_rejects_invalid_centering_window_ratio(tmp_path: Path) -> None:
    with pytest.raises(InvalidTrainerConfigError, match="centering_window_ratio"):
        run_training(_config(tmp_path, centering_window_ratio=0.0))


def test_trainer_config_rejects_negative_debug_snapshot_count(tmp_path: Path) -> None:
    with pytest.raises(InvalidTrainerConfigError, match="debug_snapshot_count"):
        run_training(_config(tmp_path, debug_snapshot_count=-1))


def test_trainer_config_rejects_negative_progress_interval(tmp_path: Path) -> None:
    with pytest.raises(InvalidTrainerConfigError, match="progress_interval"):
        run_training(_config(tmp_path, progress_interval=-1))


def test_trainer_config_rejects_partial_evaluation_configuration(tmp_path: Path) -> None:
    with pytest.raises(InvalidTrainerConfigError, match="evaluation_interval and evaluation_episodes"):
        run_training(_config(tmp_path, evaluation_interval=10, evaluation_episodes=0))


def test_trainer_config_rejects_early_stopping_without_evaluation(tmp_path: Path) -> None:
    with pytest.raises(InvalidTrainerConfigError, match="early_stopping_patience"):
        run_training(_config(tmp_path, early_stopping_patience=2))


def test_run_training_returns_training_run_with_n_metrics(tmp_path: Path) -> None:
    run = run_training(_config(tmp_path, episodes=3))

    assert isinstance(run, TrainingRun)
    assert len(run.metrics) == 3
    assert [m.episode for m in run.metrics] == [0, 1, 2]


def test_run_training_writes_latest_checkpoint_with_last_episode_index(
    tmp_path: Path,
) -> None:
    run = run_training(_config(tmp_path, episodes=2, checkpoint_every=2))

    assert run.final_checkpoint_path.exists()
    loaded = load_checkpoint(run.final_checkpoint_path)
    assert loaded.episode == 1
    assert len(loaded.metrics_history) == 2


def test_run_training_writes_best_checkpoint(tmp_path: Path) -> None:
    run = run_training(_config(tmp_path, episodes=2))

    assert run.best_checkpoint_path.exists()


def test_run_training_resumes_from_checkpoint_continues_episode_index(
    tmp_path: Path,
) -> None:
    first = run_training(_config(tmp_path, episodes=2))

    second = run_training(
        _config(
            tmp_path,
            episodes=2,
            resume_from=first.final_checkpoint_path,
        )
    )

    assert len(second.metrics) == 4
    assert [m.episode for m in second.metrics] == [0, 1, 2, 3]


def test_run_training_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    run_a = run_training(_config(tmp_path / "a", episodes=2, seed=7))
    run_b = run_training(_config(tmp_path / "b", episodes=2, seed=7))

    losses_a = [m.loss for m in run_a.metrics]
    losses_b = [m.loss for m in run_b.metrics]
    rewards_a = [m.total_reward for m in run_a.metrics]
    rewards_b = [m.total_reward for m in run_b.metrics]

    assert losses_a == pytest.approx(losses_b)
    assert rewards_a == pytest.approx(rewards_b)


def test_run_training_with_rebound_reward_completes(tmp_path: Path) -> None:
    run = run_training(_config(tmp_path, episodes=1))

    assert len(run.metrics) == 1


def test_run_training_with_batched_envs_returns_one_metric_per_episode(tmp_path: Path) -> None:
    run = run_training(_config(tmp_path, episodes=5, batch_envs=2))

    assert len(run.metrics) == 5
    assert [m.episode for m in run.metrics] == [0, 1, 2, 3, 4]


def test_run_training_records_hparams_in_checkpoint(tmp_path: Path) -> None:
    run = run_training(
        _config(
            tmp_path,
            episodes=1,
            seed=11,
            learning_rate=0.0005,
            gamma=0.95,
            device="cpu",
            batch_envs=3,
        )
    )

    loaded = load_checkpoint(run.final_checkpoint_path)
    assert loaded.hparams["seed"] == 11
    assert loaded.hparams["learning_rate"] == pytest.approx(0.0005)
    assert loaded.hparams["gamma"] == pytest.approx(0.95)
    assert loaded.hparams["reward_mode"] == "right_paddle_rebound_only"
    assert loaded.hparams["hit_reward"] == pytest.approx(1.0)
    assert loaded.hparams["miss_penalty"] == pytest.approx(-1.0)
    assert loaded.hparams["centering_reward_scale"] == pytest.approx(0.02)
    assert loaded.hparams["center_hold_bonus"] == pytest.approx(0.0005)
    assert loaded.hparams["idle_movement_penalty"] == pytest.approx(0.0005)
    assert loaded.hparams["centering_window_ratio"] == pytest.approx(0.65)
    assert loaded.hparams["device_requested"] == "cpu"
    assert loaded.hparams["device_used"] == "cpu"
    assert loaded.hparams["batch_envs"] == 3


def test_run_training_writes_only_five_debug_snapshots_per_run(tmp_path: Path) -> None:
    debug_output_dir = tmp_path / "debug"

    run_training(
        _config(
            tmp_path,
            episodes=8,
            max_steps=64,
            debug_snapshot_count=5,
            debug_output_dir=debug_output_dir,
        )
    )

    assert len(list(debug_output_dir.glob("observation-*.pgm"))) == 5


def test_run_training_saves_best_checkpoint_immediately_when_training_metric_improves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rewards = iter([5.0, -1.0, -2.0])

    def fake_update(*args, **kwargs) -> ReinforceUpdateResult:
        reward = next(rewards)
        return ReinforceUpdateResult(
            loss=0.0,
            total_reward=reward,
            episode_length=4,
            returns=np.array([reward], dtype=np.float64),
        )

    monkeypatch.setattr("trainer.run_reinforce_update", fake_update)

    run = run_training(
        _config(
            tmp_path,
            episodes=3,
            checkpoint_every=3,
            metrics_window=1,
        )
    )

    loaded_best = load_checkpoint(run.best_checkpoint_path)
    loaded_latest = load_checkpoint(run.final_checkpoint_path)

    assert loaded_best.episode == 0
    assert loaded_latest.episode == 2


def test_run_training_records_evaluations_and_stops_early_when_eval_stagnates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_summaries = iter(
        [
            _evaluation_summary(6, 4),
            _evaluation_summary(5, 5),
            _evaluation_summary(4, 6),
        ]
    )

    def fake_update(*args, **kwargs) -> ReinforceUpdateResult:
        return ReinforceUpdateResult(
            loss=0.0,
            total_reward=0.0,
            episode_length=4,
            returns=np.array([0.0], dtype=np.float64),
        )

    def fake_evaluate_policy(*args, **kwargs) -> EvaluationSummary:
        return next(eval_summaries)

    monkeypatch.setattr("trainer.run_reinforce_update", fake_update)
    monkeypatch.setattr("trainer.evaluate_policy", fake_evaluate_policy)

    run = run_training(
        _config(
            tmp_path,
            episodes=10,
            checkpoint_every=10,
            evaluation_interval=1,
            evaluation_episodes=2,
            early_stopping_patience=2,
        )
    )

    loaded_best = load_checkpoint(run.best_checkpoint_path)
    loaded_latest = load_checkpoint(run.final_checkpoint_path)

    assert run.stopped_early is True
    assert run.stop_reason is not None
    assert len(run.metrics) == 3
    assert len(run.evaluation_history) == 3
    assert loaded_best.episode == 0
    assert len(loaded_latest.evaluation_history) == 3
    assert loaded_latest.evaluation_history[0]["attempt_count"] == 10
    assert "hit_rate_lower_bound" in loaded_latest.evaluation_history[0]
