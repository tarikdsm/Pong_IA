from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = REPO_ROOT / "engine"
TRAINING_SRC_PATH = REPO_ROOT / "training" / "src"

if str(ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(ENGINE_PATH))
if str(TRAINING_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC_PATH))


from trainer import TrainerConfig, run_training  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a short Pong training run.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--target-episodes", type=int, default=5000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-envs", type=int, default=1)
    parser.add_argument("--debug-snapshot-count", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = Path(tempfile.mkdtemp(prefix="pong-ia-benchmark-"))

    try:
        config = TrainerConfig(
            episodes=args.episodes,
            gamma=args.gamma,
            learning_rate=args.lr,
            seed=args.seed,
            max_steps=args.max_steps,
            checkpoint_dir=workspace / "checkpoints",
            checkpoint_every=max(args.episodes, 1),
            device=args.device,
            batch_envs=args.batch_envs,
            debug_snapshot_count=args.debug_snapshot_count,
            debug_output_dir=workspace / "debug",
        )

        started = time.perf_counter()
        run = run_training(config)
        elapsed_seconds = time.perf_counter() - started
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    eps_per_second = args.episodes / elapsed_seconds if elapsed_seconds > 0 else 0.0
    estimated_total_seconds = args.target_episodes / eps_per_second if eps_per_second > 0 else float("inf")
    last = run.metrics[-1]
    print(
        f"episodes={args.episodes} "
        f"elapsed_seconds={elapsed_seconds:.3f} "
        f"episodes_per_second={eps_per_second:.3f} "
        f"estimated_total_seconds={estimated_total_seconds:.1f} "
        f"estimated_total_hours={estimated_total_seconds / 3600:.2f} "
        f"final_reward={last.total_reward:.2f} "
        f"avg_reward_window={last.avg_reward_window:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
