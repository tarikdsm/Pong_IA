from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim

from batched_reinforce import run_batched_reinforce_updates
from checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from evaluation import EvaluationSummary, evaluate_policy, wilson_lower_bound
from frame_stack import FrameStack
from metrics import MetricsTracker, TrainingMetrics
from model import ACTIONS, PongPolicyNetwork
from policy import sample_action_index
from pong_engine import create_initial_state, partially_tracking, step
from pong_engine.config import INFERENCE_INTERVAL_TICKS
from reinforce import RewardFn, run_reinforce_update
from reward_shaping import DEFAULT_MISS_PENALTY, DEFAULT_REBOUND_REWARD, rebound_reward
from reward_shaping import (
    DEFAULT_CENTERING_REWARD_SCALE,
    DEFAULT_CENTERING_WINDOW_RATIO,
    DEFAULT_CENTER_HOLD_BONUS,
    DEFAULT_IDLE_MOVEMENT_PENALTY,
)


LATEST_CHECKPOINT_NAME = "latest.pt"
BEST_CHECKPOINT_NAME = "best.pt"


class InvalidTrainerConfigError(ValueError):
    """Raised when a trainer configuration value is invalid."""


@dataclass(frozen=True)
class TrainerConfig:
    episodes: int
    gamma: float
    learning_rate: float
    seed: int
    max_steps: int
    checkpoint_dir: Path
    checkpoint_every: int
    hit_reward: float = DEFAULT_REBOUND_REWARD
    miss_penalty: float = DEFAULT_MISS_PENALTY
    centering_reward_scale: float = DEFAULT_CENTERING_REWARD_SCALE
    center_hold_bonus: float = DEFAULT_CENTER_HOLD_BONUS
    idle_movement_penalty: float = DEFAULT_IDLE_MOVEMENT_PENALTY
    centering_window_ratio: float = DEFAULT_CENTERING_WINDOW_RATIO
    metrics_window: int = 50
    resume_from: Path | None = None
    device: str = "auto"
    batch_envs: int = 1
    debug_snapshot_count: int = 5
    debug_output_dir: Path | None = None
    progress_interval: int = 0
    evaluation_interval: int = 0
    evaluation_episodes: int = 0
    early_stopping_patience: int | None = None
    early_stopping_min_improvement: float = 0.0


@dataclass(frozen=True)
class EvaluationMetrics:
    episode: int
    hit_count: int
    miss_count: int
    attempt_count: int
    hit_rate: float
    hit_rate_lower_bound: float
    avg_hits_per_episode: float
    avg_misses_per_episode: float
    avg_episode_length: float


@dataclass(frozen=True)
class TrainingRun:
    metrics: tuple[TrainingMetrics, ...]
    final_checkpoint_path: Path
    best_checkpoint_path: Path
    evaluation_history: tuple[EvaluationMetrics, ...] = field(default_factory=tuple)
    stopped_early: bool = False
    stop_reason: str | None = None
    hparams: dict[str, Any] = field(default_factory=dict)


def run_training(config: TrainerConfig) -> TrainingRun:
    _validate(config)
    device = resolve_training_device(config.device)

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)

    network = PongPolicyNetwork().to(device)
    optimizer = optim.Adam(network.parameters(), lr=config.learning_rate)
    tracker = MetricsTracker(window=config.metrics_window)

    start_episode = 0
    best_avg_reward = float("-inf")
    best_eval_score = float("-inf")
    evaluation_history: list[EvaluationMetrics] = []
    if config.resume_from is not None:
        ckpt = load_checkpoint(config.resume_from, map_location=device)
        network.load_state_dict(ckpt.state_dict)
        optimizer.load_state_dict(ckpt.optimizer_state)
        _move_optimizer_state_to_device(optimizer, device)
        for entry in ckpt.metrics_history:
            recorded = tracker.record(
                loss=float(entry["loss"]),
                total_reward=float(entry["total_reward"]),
                episode_length=int(entry["episode_length"]),
            )
            best_avg_reward = max(best_avg_reward, recorded.avg_reward_window)
        evaluation_history = _restore_evaluation_history(ckpt.evaluation_history)
        for evaluation in evaluation_history:
            best_eval_score = max(best_eval_score, evaluation.hit_rate_lower_bound)
        start_episode = int(ckpt.episode) + 1

    reward_fn = _build_reward_fn(config)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = config.checkpoint_dir / LATEST_CHECKPOINT_NAME
    best_path = config.checkpoint_dir / BEST_CHECKPOINT_NAME
    hparams = _hparams(config)

    debug_targets = _build_debug_targets(start_episode, config.episodes, config.debug_snapshot_count)
    debug_output_dir = config.debug_output_dir
    if debug_output_dir is not None:
        debug_output_dir.mkdir(parents=True, exist_ok=True)
        for existing in debug_output_dir.glob("observation-*.pgm"):
            existing.unlink()
        for existing in debug_output_dir.glob("observation-*.png"):
            existing.unlink()

    completed = 0
    captured_debug_count = 0
    stopped_early = False
    stop_reason: str | None = None
    stale_evaluations = 0
    while completed < config.episodes:
        remaining = config.episodes - completed
        batch_size = min(config.batch_envs, remaining)
        episode = start_episode + completed
        if batch_size == 1:
            updates = (
                run_reinforce_update(
                    network,
                    optimizer,
                    seed=config.seed + episode,
                    max_steps=config.max_steps,
                    gamma=config.gamma,
                    reward_fn=reward_fn,
                ),
            )
        else:
            updates = run_batched_reinforce_updates(
                network,
                optimizer,
                seeds=[config.seed + episode + offset for offset in range(batch_size)],
                max_steps=config.max_steps,
                gamma=config.gamma,
                hit_reward=config.hit_reward,
                miss_penalty=config.miss_penalty,
                centering_reward_scale=config.centering_reward_scale,
                center_hold_bonus=config.center_hold_bonus,
                idle_movement_penalty=config.idle_movement_penalty,
                centering_window_ratio=config.centering_window_ratio,
            )

        for batch_offset, update in enumerate(updates):
            current_episode = episode + batch_offset
            recorded = tracker.record(
                loss=update.loss,
                total_reward=update.total_reward,
                episode_length=update.episode_length,
            )
            completed += 1
            current_checkpoint: Checkpoint | None = None

            is_periodic = completed % config.checkpoint_every == 0
            is_final = completed == config.episodes

            if not _evaluation_enabled(config) and recorded.avg_reward_window > best_avg_reward:
                best_avg_reward = recorded.avg_reward_window
                current_checkpoint = _build_checkpoint(
                    network,
                    optimizer,
                    hparams,
                    current_episode,
                    tracker,
                    evaluation_history,
                )
                save_checkpoint(best_path, current_checkpoint)

            if _should_run_evaluation(config, completed, is_final):
                evaluation_summary = evaluate_policy(
                    network,
                    seed=config.seed + 1_000_000 + current_episode * 10_000,
                    episodes=config.evaluation_episodes,
                    max_steps=config.max_steps,
                )
                evaluation = _record_evaluation(current_episode, evaluation_summary)
                evaluation_history.append(evaluation)
                current_checkpoint = _build_checkpoint(
                    network,
                    optimizer,
                    hparams,
                    current_episode,
                    tracker,
                    evaluation_history,
                )
                if (
                    evaluation.hit_rate_lower_bound
                    > best_eval_score + config.early_stopping_min_improvement
                ):
                    best_eval_score = evaluation.hit_rate_lower_bound
                    stale_evaluations = 0
                    save_checkpoint(best_path, current_checkpoint)
                else:
                    stale_evaluations += 1

                if (
                    config.early_stopping_patience is not None
                    and stale_evaluations >= config.early_stopping_patience
                ):
                    stopped_early = True
                    stop_reason = (
                        "Early stopping triggered after "
                        f"{stale_evaluations} evaluation rounds without evaluation-score improvement."
                    )
                    save_checkpoint(latest_path, current_checkpoint)
                    break

            if is_periodic or is_final:
                if current_checkpoint is None:
                    current_checkpoint = _build_checkpoint(
                        network,
                        optimizer,
                        hparams,
                        current_episode,
                        tracker,
                        evaluation_history,
                    )
                save_checkpoint(latest_path, current_checkpoint)
                if _evaluation_enabled(config) and not best_path.exists():
                    save_checkpoint(best_path, current_checkpoint)

            if debug_output_dir is not None and current_episode in debug_targets:
                bitmap = capture_model_debug_bitmap(
                    network,
                    seed=config.seed + 10_000 + current_episode,
                    max_steps=config.max_steps,
                )
                path = debug_output_dir / f"observation-{captured_debug_count:02d}.pgm"
                _write_pgm(path, bitmap)
                captured_debug_count += 1

            if _should_report_progress(config, completed, is_final, stopped_early):
                print(
                    f"progress {completed} de {config.episodes} "
                    f"(faltam {config.episodes - completed}) "
                    f"avg_reward_window={recorded.avg_reward_window:.2f}"
                )

        if stopped_early:
            break

    return TrainingRun(
        metrics=tracker.history,
        final_checkpoint_path=latest_path,
        best_checkpoint_path=best_path,
        evaluation_history=tuple(evaluation_history),
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        hparams=hparams,
    )


def _validate(config: TrainerConfig) -> None:
    if config.episodes <= 0:
        raise InvalidTrainerConfigError("episodes must be a positive integer.")
    if config.learning_rate <= 0.0:
        raise InvalidTrainerConfigError("learning_rate must be positive.")
    if config.gamma < 0.0 or config.gamma > 1.0:
        raise InvalidTrainerConfigError("gamma must be within the closed interval [0, 1].")
    if config.checkpoint_every <= 0:
        raise InvalidTrainerConfigError("checkpoint_every must be a positive integer.")
    if config.max_steps <= 0:
        raise InvalidTrainerConfigError("max_steps must be a positive integer.")
    if config.metrics_window <= 0:
        raise InvalidTrainerConfigError("metrics_window must be a positive integer.")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise InvalidTrainerConfigError("device must be one of: auto, cpu, cuda.")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise InvalidTrainerConfigError("device 'cuda' was requested but CUDA is unavailable.")
    if config.batch_envs <= 0:
        raise InvalidTrainerConfigError("batch_envs must be a positive integer.")
    if config.debug_snapshot_count < 0:
        raise InvalidTrainerConfigError("debug_snapshot_count must be non-negative.")
    if config.progress_interval < 0:
        raise InvalidTrainerConfigError("progress_interval must be non-negative.")
    if config.hit_reward <= 0.0:
        raise InvalidTrainerConfigError("hit_reward must be positive.")
    if config.miss_penalty >= 0.0:
        raise InvalidTrainerConfigError("miss_penalty must be negative.")
    if config.centering_reward_scale < 0.0:
        raise InvalidTrainerConfigError("centering_reward_scale must be non-negative.")
    if config.center_hold_bonus < 0.0:
        raise InvalidTrainerConfigError("center_hold_bonus must be non-negative.")
    if config.idle_movement_penalty < 0.0:
        raise InvalidTrainerConfigError("idle_movement_penalty must be non-negative.")
    if config.centering_window_ratio <= 0.0 or config.centering_window_ratio > 1.0:
        raise InvalidTrainerConfigError("centering_window_ratio must be within (0, 1].")
    if config.evaluation_interval < 0:
        raise InvalidTrainerConfigError("evaluation_interval must be non-negative.")
    if config.evaluation_episodes < 0:
        raise InvalidTrainerConfigError("evaluation_episodes must be non-negative.")
    if (config.evaluation_interval == 0) != (config.evaluation_episodes == 0):
        raise InvalidTrainerConfigError(
            "evaluation_interval and evaluation_episodes must both be zero or both be positive."
        )
    if config.early_stopping_patience is not None and config.early_stopping_patience <= 0:
        raise InvalidTrainerConfigError("early_stopping_patience must be a positive integer.")
    if config.early_stopping_patience is not None and not _evaluation_enabled(config):
        raise InvalidTrainerConfigError(
            "early_stopping_patience requires evaluation_interval and evaluation_episodes."
        )
    if config.early_stopping_min_improvement < 0.0:
        raise InvalidTrainerConfigError("early_stopping_min_improvement must be non-negative.")


def resolve_training_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_reward_fn(config: TrainerConfig) -> RewardFn:
    def reward(before, after):
        return rebound_reward(
            before,
            after,
            hit_reward=config.hit_reward,
            miss_penalty=config.miss_penalty,
            centering_reward_scale=config.centering_reward_scale,
            center_hold_bonus=config.center_hold_bonus,
            idle_movement_penalty=config.idle_movement_penalty,
            centering_window_ratio=config.centering_window_ratio,
        )

    return reward


def _hparams(config: TrainerConfig) -> dict[str, Any]:
    resolved_device = resolve_training_device(config.device)
    return {
        "seed": config.seed,
        "learning_rate": config.learning_rate,
        "gamma": config.gamma,
        "max_steps": config.max_steps,
        "reward_mode": "right_paddle_rebound_only",
        "hit_reward": config.hit_reward,
        "miss_penalty": config.miss_penalty,
        "centering_reward_scale": config.centering_reward_scale,
        "center_hold_bonus": config.center_hold_bonus,
        "idle_movement_penalty": config.idle_movement_penalty,
        "centering_window_ratio": config.centering_window_ratio,
        "metrics_window": config.metrics_window,
        "device_requested": config.device,
        "device_used": resolved_device.type,
        "batch_envs": config.batch_envs,
        "debug_snapshot_count": config.debug_snapshot_count,
        "progress_interval": config.progress_interval,
        "evaluation_interval": config.evaluation_interval,
        "evaluation_episodes": config.evaluation_episodes,
        "evaluation_selection_metric": "hit_rate_wilson_lower_bound_95",
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_improvement": config.early_stopping_min_improvement,
    }


def _build_checkpoint(
    network: PongPolicyNetwork,
    optimizer: optim.Optimizer,
    hparams: dict[str, Any],
    episode: int,
    tracker: MetricsTracker,
    evaluation_history: list[EvaluationMetrics],
) -> Checkpoint:
    return Checkpoint(
        state_dict={key: value.detach().clone() for key, value in network.state_dict().items()},
        optimizer_state=optimizer.state_dict(),
        hparams=hparams,
        episode=episode,
        metrics_history=[dataclasses.asdict(metrics) for metrics in tracker.history],
        evaluation_history=[dataclasses.asdict(evaluation) for evaluation in evaluation_history],
    )


def _move_optimizer_state_to_device(
    optimizer: optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _build_debug_targets(
    start_episode: int,
    episode_count: int,
    debug_snapshot_count: int,
) -> set[int]:
    if debug_snapshot_count == 0:
        return set()

    final_episode = start_episode + episode_count - 1
    if debug_snapshot_count == 1:
        return {final_episode}

    interval = max((episode_count - 1) / (debug_snapshot_count - 1), 1)
    return {
        min(final_episode, start_episode + round(index * interval))
        for index in range(debug_snapshot_count)
    }


def _record_evaluation(episode: int, summary: EvaluationSummary) -> EvaluationMetrics:
    return EvaluationMetrics(
        episode=episode,
        hit_count=summary.hit_count,
        miss_count=summary.miss_count,
        attempt_count=summary.attempt_count,
        hit_rate=summary.hit_rate,
        hit_rate_lower_bound=summary.hit_rate_lower_bound,
        avg_hits_per_episode=summary.avg_hits_per_episode,
        avg_misses_per_episode=summary.avg_misses_per_episode,
        avg_episode_length=summary.avg_episode_length,
    )


def _restore_evaluation_history(history: list[dict[str, Any]]) -> list[EvaluationMetrics]:
    restored: list[EvaluationMetrics] = []
    for entry in history:
        hit_count = int(entry["hit_count"])
        miss_count = int(entry["miss_count"])
        attempt_count = int(entry.get("attempt_count", hit_count + miss_count))
        restored.append(
            EvaluationMetrics(
                episode=int(entry["episode"]),
                hit_count=hit_count,
                miss_count=miss_count,
                attempt_count=attempt_count,
                hit_rate=float(entry["hit_rate"]),
                hit_rate_lower_bound=float(
                    entry.get(
                        "hit_rate_lower_bound",
                        wilson_lower_bound(hit_count, attempt_count),
                    )
                ),
                avg_hits_per_episode=float(entry["avg_hits_per_episode"]),
                avg_misses_per_episode=float(entry["avg_misses_per_episode"]),
                avg_episode_length=float(entry["avg_episode_length"]),
            )
        )
    return restored


def _evaluation_enabled(config: TrainerConfig) -> bool:
    return config.evaluation_interval > 0 and config.evaluation_episodes > 0


def _should_run_evaluation(config: TrainerConfig, completed: int, is_final: bool) -> bool:
    return _evaluation_enabled(config) and (completed % config.evaluation_interval == 0 or is_final)


def _should_report_progress(
    config: TrainerConfig,
    completed: int,
    is_final: bool,
    stopped_early: bool,
) -> bool:
    return (
        config.progress_interval > 0
        and (completed % config.progress_interval == 0 or is_final or stopped_early)
    )


def capture_model_debug_bitmap(
    network: PongPolicyNetwork,
    *,
    seed: int,
    max_steps: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = create_initial_state(rng)
    frame_stack = FrameStack(debug_capacity=1)
    model_device = next(network.parameters()).device
    last_action = "none"

    network.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            frame_stack.push_state(state)
            if frame_stack.is_ready() and state.tick % INFERENCE_INTERVAL_TICKS == 0:
                observation = frame_stack.as_float32_flat(copy=False)
                inputs = torch.from_numpy(observation).to(device=model_device).unsqueeze(0)
                logits = network(inputs).logits.squeeze(0)
                probabilities = torch.softmax(logits, dim=0).detach().cpu().numpy()
                last_action = ACTIONS[sample_action_index(probabilities, rng)]
                return frame_stack.debug_bitmaps()[-1]

            left_action = partially_tracking(state, rng)
            state = step(state, left_action, last_action, rng)

    if frame_stack.debug_bitmaps():
        return frame_stack.debug_bitmaps()[-1]
    raise InvalidTrainerConfigError("Unable to capture a model debug bitmap within max_steps.")


def _write_pgm(path: Path, bitmap: np.ndarray) -> None:
    header = f"P5\n{bitmap.shape[1]} {bitmap.shape[0]}\n255\n".encode("ascii")
    with path.open("wb") as output_file:
        output_file.write(header)
        output_file.write(bitmap.tobytes(order="C"))
