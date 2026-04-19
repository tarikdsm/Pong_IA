from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from frame_stack import FrameStack, InvalidFrameError
from pong_engine.config import FRAME_STEP_TICKS
from pong_engine.state import GameState


HEIGHT = 60
WIDTH = 80
STACK_SIZE = 5
HISTORY_SIZE = 1 + (STACK_SIZE - 1) * FRAME_STEP_TICKS


def make_frame(fill: int = 0, *, x: int = 0, y: int = 0) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH), fill, dtype=np.uint8)
    frame[y, x] = 1
    return frame


def test_frame_stack_returns_tensor_with_logical_shape_after_first_push() -> None:
    stack = FrameStack()

    stack.push(make_frame(x=3, y=4))

    tensor = stack.as_tensor()

    assert tensor.shape == (5, HEIGHT, WIDTH)
    assert tensor.dtype == np.uint8
    assert tensor[-1, 4, 3] == 1


def test_frame_stack_zero_fills_missing_initial_frames_deterministically() -> None:
    stack = FrameStack()

    stack.push(make_frame(x=2, y=1))

    tensor = stack.as_tensor()

    assert np.count_nonzero(tensor[0]) == 0
    assert np.count_nonzero(tensor[1]) == 0
    assert np.count_nonzero(tensor[2]) == 0
    assert np.count_nonzero(tensor[3]) == 0
    assert tensor[4, 1, 2] == 1


def test_frame_stack_returns_spaced_flat_vector_of_length_24000() -> None:
    stack = FrameStack()
    for offset in range(HISTORY_SIZE):
        stack.push(make_frame(x=offset, y=offset))

    flat = stack.as_flat()

    assert flat.shape == (STACK_SIZE * HEIGHT * WIDTH,)
    assert flat.dtype == np.uint8
    assert int(flat.sum()) == 5
    tensor = flat.reshape(STACK_SIZE, HEIGHT, WIDTH)
    assert tensor[0, 0, 0] == 1
    assert tensor[1, FRAME_STEP_TICKS, FRAME_STEP_TICKS] == 1
    assert tensor[2, FRAME_STEP_TICKS * 2, FRAME_STEP_TICKS * 2] == 1
    assert tensor[3, FRAME_STEP_TICKS * 3, FRAME_STEP_TICKS * 3] == 1
    assert tensor[4, FRAME_STEP_TICKS * 4, FRAME_STEP_TICKS * 4] == 1


def test_frame_stack_keeps_only_10_most_recent_debug_snapshots() -> None:
    stack = FrameStack(debug_capacity=10)

    for index in range(11):
        stack.push(make_frame(x=index % WIDTH, y=0))

    snapshots = stack.debug_snapshots()

    assert len(snapshots) == 10
    assert snapshots[0][0, 0] == 0
    assert snapshots[0][0, 1] == 1
    assert snapshots[-1][0, 10] == 1


def test_frame_stack_keeps_only_10_most_recent_debug_bitmaps() -> None:
    stack = FrameStack(debug_capacity=10)

    for index in range(11):
        stack.push(make_frame(x=index % WIDTH, y=0))

    bitmaps = stack.debug_bitmaps()

    assert len(bitmaps) == 10
    assert bitmaps[0].shape == (HEIGHT, WIDTH * 5)
    assert bitmaps[0].dtype == np.uint8
    assert int(bitmaps[0].max()) == 255
    assert bitmaps[-1][0, (4 * WIDTH) + 10] == 255


def test_frame_stack_can_write_debug_bitmaps_to_disk(tmp_path: Path) -> None:
    stack = FrameStack(debug_capacity=10)

    for index in range(HISTORY_SIZE + 1):
        stack.push(make_frame(x=index % WIDTH, y=0))

    written = stack.write_debug_bitmaps(tmp_path)

    assert len(written) == 10
    assert [path.name for path in written] == [f"observation-{index:02d}.pgm" for index in range(10)]

    first_bitmap = written[0].read_bytes()
    assert first_bitmap.startswith(b"P5\n400 60\n255\n")
    assert len(list(tmp_path.glob("observation-*.pgm"))) == 10


def test_frame_stack_rejects_invalid_shape() -> None:
    stack = FrameStack()
    invalid = np.zeros((HEIGHT - 1, WIDTH), dtype=np.uint8)

    with pytest.raises(InvalidFrameError, match="shape"):
        stack.push(invalid)


def test_frame_stack_rejects_values_outside_binary_range() -> None:
    stack = FrameStack()
    invalid = np.full((HEIGHT, WIDTH), 2, dtype=np.uint8)

    with pytest.raises(InvalidFrameError, match="binary"):
        stack.push(invalid)


def test_frame_stack_keeps_logical_order_after_wraparound() -> None:
    stack = FrameStack()

    for offset in range(HISTORY_SIZE + 5):
        stack.push(make_frame(x=offset, y=0))

    tensor = stack.as_tensor()

    assert tensor[0, 0, 5] == 1
    assert tensor[1, 0, 10] == 1
    assert tensor[2, 0, 15] == 1
    assert tensor[3, 0, 20] == 1
    assert tensor[4, 0, 25] == 1


def test_frame_stack_can_push_bitmap_generated_directly_from_state() -> None:
    stack = FrameStack()
    state = GameState(
        ball_x=10.0,
        ball_y=11.0,
        ball_vx=1.0,
        ball_vy=0.0,
        ball_speed=1.0,
        paddle_left_y=20,
        paddle_right_y=21,
        score_left=0,
        score_right=0,
        tick=0,
    )

    stack.push_state(state)
    tensor = stack.as_tensor()

    assert tensor[-1, 20, 0] == 1
    assert tensor[-1, 21, WIDTH - 1] == 1
    assert tensor[-1, 11, 10] == 1


def test_frame_stack_can_return_reusable_float32_flat_buffer() -> None:
    stack = FrameStack(debug_capacity=0)
    for offset in range(HISTORY_SIZE):
        stack.push(make_frame(x=offset, y=offset))

    flat_u8 = stack.as_flat()
    flat_f32 = stack.as_float32_flat(copy=False)

    assert flat_f32.shape == flat_u8.shape
    assert flat_f32.dtype == np.float32
    assert flat_f32.tolist() == pytest.approx(flat_u8.astype(np.float32).tolist())


def test_frame_stack_can_disable_debug_snapshots() -> None:
    stack = FrameStack(debug_capacity=0)

    for offset in range(3):
        stack.push(make_frame(x=offset, y=0))

    assert stack.debug_snapshots() == []


def test_frame_stack_is_ready_only_after_full_temporal_window() -> None:
    stack = FrameStack()

    for offset in range(HISTORY_SIZE - 1):
        stack.push(make_frame(x=offset, y=0))
        assert stack.is_ready() is False

    stack.push(make_frame(x=HISTORY_SIZE - 1, y=0))

    assert stack.is_ready() is True
