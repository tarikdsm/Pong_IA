from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = REPO_ROOT / "engine"
TRAINING_SRC_PATH = REPO_ROOT / "training" / "src"

if str(ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(ENGINE_PATH))
if str(TRAINING_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC_PATH))


from frame_stack import FrameStack  # noqa: E402
from pong_engine import create_initial_state, partially_tracking, step  # noqa: E402
from pong_engine.state import Action  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "training" / "debug" / "bitmaps"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the latest frame-stack debug bitmaps to disk."
    )
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--right-action",
        choices=["none", "up", "down"],
        default="none",
        help="Fixed action applied to the right paddle while generating samples.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.steps <= 0:
        print("error: --steps must be greater than zero.", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    state = create_initial_state(rng)
    frame_stack = FrameStack(debug_capacity=5)
    right_action: Action = args.right_action

    for _ in range(args.steps):
        frame_stack.push_state(state)
        left_action = partially_tracking(state, rng)
        state = step(state, left_action, right_action, rng)

    written = frame_stack.write_debug_bitmaps(args.output_dir)
    print(
        f"debug_bitmaps={len(written)} "
        f"output_dir={args.output_dir.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
