from __future__ import annotations

from pathlib import Path

import numpy as np

from trainer import TrainerConfig, run_training


def test_short_training_run_produces_finite_metrics_and_artifacts(tmp_path: Path) -> None:
    run = run_training(
        TrainerConfig(
            episodes=10,
            gamma=0.99,
            learning_rate=0.001,
            seed=42,
            max_steps=64,
            checkpoint_dir=tmp_path / "checkpoints",
            checkpoint_every=10,
            device="cpu",
            batch_envs=2,
            debug_snapshot_count=5,
            debug_output_dir=tmp_path / "debug",
        )
    )

    losses = np.array([metric.loss for metric in run.metrics], dtype=np.float64)
    rewards = np.array([metric.total_reward for metric in run.metrics], dtype=np.float64)

    assert np.isfinite(losses).all()
    assert np.isfinite(rewards).all()
    assert run.final_checkpoint_path.exists()
    assert run.best_checkpoint_path.exists()
    assert len(list((tmp_path / "debug").glob("observation-*.pgm"))) == 5
