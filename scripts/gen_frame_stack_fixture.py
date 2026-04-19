from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_PATH = REPO_ROOT / "engine"
TRAINING_SRC_PATH = REPO_ROOT / "training" / "src"

if str(ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(ENGINE_PATH))

if str(TRAINING_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC_PATH))

OUTPUT_PATH = REPO_ROOT / "shared" / "fixtures" / "frame_stack_golden.json"


def main() -> None:
    from frame_stack import FrameStack
    from pong_engine.config import BITMAP_HEIGHT, BITMAP_WIDTH, FRAME_STACK, FRAME_STEP_TICKS

    history_size = 1 + (FRAME_STACK - 1) * FRAME_STEP_TICKS
    stack = FrameStack()
    frames: list[list[int]] = []

    for index in range(history_size):
        frame = np.zeros((BITMAP_HEIGHT, BITMAP_WIDTH), dtype=np.uint8)
        frame[index % BITMAP_HEIGHT, index % BITMAP_WIDTH] = 1
        stack.push(frame)
        frames.append(frame.reshape(-1).astype(int).tolist())

    payload = {
        "stack_size": FRAME_STACK,
        "frame_step_ticks": FRAME_STEP_TICKS,
        "frame_length": BITMAP_HEIGHT * BITMAP_WIDTH,
        "frames": frames,
        "expected_flat": stack.as_flat().astype(int).tolist(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Generated {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
