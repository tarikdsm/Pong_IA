from __future__ import annotations

import numpy as np

from evaluation import evaluate_policy, wilson_lower_bound
from model import PongPolicyNetwork


def test_wilson_lower_bound_is_zero_without_attempts() -> None:
    assert wilson_lower_bound(0, 0) == 0.0


def test_wilson_lower_bound_is_conservative_for_partial_success_rate() -> None:
    lower_bound = wilson_lower_bound(50, 100)

    assert lower_bound > 0.0
    assert lower_bound < 0.5


def test_evaluate_policy_returns_finite_summary_on_small_run() -> None:
    network = PongPolicyNetwork()

    summary = evaluate_policy(
        network,
        seed=42,
        episodes=2,
        max_steps=64,
    )

    assert summary.episodes == 2
    assert summary.hit_count >= 0
    assert summary.miss_count >= 0
    assert summary.attempt_count == summary.hit_count + summary.miss_count
    assert np.isfinite(summary.hit_rate)
    assert np.isfinite(summary.hit_rate_lower_bound)
    assert 0.0 <= summary.hit_rate <= 1.0
    assert 0.0 <= summary.hit_rate_lower_bound <= summary.hit_rate
    assert np.isfinite(summary.avg_hits_per_episode)
    assert np.isfinite(summary.avg_misses_per_episode)
    assert np.isfinite(summary.avg_episode_length)
