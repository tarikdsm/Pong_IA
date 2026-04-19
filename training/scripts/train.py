from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = REPO_ROOT / "engine"
TRAINING_SRC_PATH = REPO_ROOT / "training" / "src"

if str(ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(ENGINE_PATH))
if str(TRAINING_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC_PATH))


from trainer import (  # noqa: E402
    InvalidTrainerConfigError,
    TrainerConfig,
    run_training,
)


DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "training" / "checkpoints"
DEFAULT_DEBUG_DIR = REPO_ROOT / "training" / "debug" / "bitmaps"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Pong policy via REINFORCE."
    )
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
    )
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--hit-reward", type=float, default=1.0)
    parser.add_argument("--miss-penalty", type=float, default=-1.0)
    parser.add_argument("--centering-reward-scale", type=float, default=0.02)
    parser.add_argument("--center-hold-bonus", type=float, default=0.0005)
    parser.add_argument("--idle-movement-penalty", type=float, default=0.0005)
    parser.add_argument("--centering-window-ratio", type=float, default=0.65)
    parser.add_argument("--metrics-window", type=int, default=50)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-envs", type=int, default=1)
    parser.add_argument("--debug-snapshot-count", type=int, default=5)
    parser.add_argument("--debug-output-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--evaluation-interval", type=int, default=0)
    parser.add_argument("--evaluation-episodes", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-improvement", type=float, default=0.0)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> TrainerConfig:
    return TrainerConfig(
        episodes=args.episodes,
        gamma=args.gamma,
        learning_rate=args.lr,
        seed=args.seed,
        max_steps=args.max_steps,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        hit_reward=args.hit_reward,
        miss_penalty=args.miss_penalty,
        centering_reward_scale=args.centering_reward_scale,
        center_hold_bonus=args.center_hold_bonus,
        idle_movement_penalty=args.idle_movement_penalty,
        centering_window_ratio=args.centering_window_ratio,
        metrics_window=args.metrics_window,
        resume_from=args.resume_from,
        device=args.device,
        batch_envs=args.batch_envs,
        debug_snapshot_count=args.debug_snapshot_count,
        debug_output_dir=args.debug_output_dir,
        progress_interval=args.progress_interval,
        evaluation_interval=args.evaluation_interval,
        evaluation_episodes=args.evaluation_episodes,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_improvement=args.early_stopping_min_improvement,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)

    try:
        run = run_training(config)
    except InvalidTrainerConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    last = run.metrics[-1]
    evaluation_suffix = ""
    if run.evaluation_history:
        best_eval = max(run.evaluation_history, key=lambda item: item.hit_rate_lower_bound)
        evaluation_suffix = (
            f" eval_hit_rate={best_eval.hit_rate:.3f}"
            f" eval_hit_rate_lb={best_eval.hit_rate_lower_bound:.3f}"
            f" eval_episode={best_eval.episode}"
        )
    early_stop_suffix = ""
    if run.stopped_early and run.stop_reason:
        early_stop_suffix = f" stopped_early=true reason={run.stop_reason!r}"
    print(
        f"episodes={len(run.metrics)} "
        f"final_loss={last.loss:.4f} "
        f"final_reward={last.total_reward:.2f} "
        f"avg_reward_window={last.avg_reward_window:.2f} "
        f"device={run.hparams['device_used']} "
        f"batch_envs={run.hparams['batch_envs']} "
        f"latest={run.final_checkpoint_path} "
        f"best={run.best_checkpoint_path}"
        f"{evaluation_suffix}"
        f"{early_stop_suffix}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
