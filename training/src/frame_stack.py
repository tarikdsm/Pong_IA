from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from pong_engine.config import BALL_SIZE, FRAME_STEP_TICKS, PADDLE_HEIGHT, PADDLE_WIDTH
from pong_engine.state import GameState


class InvalidFrameError(ValueError):
    """Raised when a frame cannot be accepted into the stack."""


class FrameStack:
    def __init__(
        self,
        stack_size: int = 5,
        height: int = 60,
        width: int = 80,
        frame_step: int = FRAME_STEP_TICKS,
        debug_capacity: int = 0,
    ) -> None:
        if stack_size <= 0:
            raise InvalidFrameError("stack_size must be greater than zero.")
        if frame_step <= 0:
            raise InvalidFrameError("frame_step must be greater than zero.")

        self._stack_size = stack_size
        self._height = height
        self._width = width
        self._frame_step = frame_step
        self._frame_size = height * width
        self._history_size = 1 + (stack_size - 1) * frame_step
        self._history = np.zeros((self._history_size, height, width), dtype=np.uint8)
        self._tensor_buffer = np.zeros((stack_size, height, width), dtype=np.uint8)
        self._flat_buffer = np.zeros(stack_size * self._frame_size, dtype=np.uint8)
        self._flat_float32_buffer = np.zeros(stack_size * self._frame_size, dtype=np.float32)
        self._bitmap_buffer = np.zeros((height, width), dtype=np.uint8)
        self._history_count = 0
        self._history_next_index = 0
        self._debug_snapshots: deque[np.ndarray] = deque(maxlen=debug_capacity)
        self._debug_bitmaps: deque[np.ndarray] = deque(maxlen=debug_capacity)

    def push(self, frame: np.ndarray) -> None:
        validated = self._validate(frame)
        self._store_frame(validated)

    def push_state(self, state: GameState) -> None:
        self.push_components(
            ball_x=state.ball_x,
            ball_y=state.ball_y,
            paddle_left_y=state.paddle_left_y,
            paddle_right_y=state.paddle_right_y,
        )

    def push_components(
        self,
        *,
        ball_x: float,
        ball_y: float,
        paddle_left_y: int,
        paddle_right_y: int,
    ) -> None:
        self._bitmap_buffer.fill(0)
        self._bitmap_buffer[
            paddle_left_y : paddle_left_y + PADDLE_HEIGHT,
            0:PADDLE_WIDTH,
        ] = 1
        self._bitmap_buffer[
            paddle_right_y : paddle_right_y + PADDLE_HEIGHT,
            self._width - PADDLE_WIDTH : self._width,
        ] = 1
        ball_x_int = int(ball_x)
        ball_y_int = int(ball_y)
        self._bitmap_buffer[ball_y_int : ball_y_int + BALL_SIZE, ball_x_int : ball_x_int + BALL_SIZE] = 1
        self._store_frame(self._bitmap_buffer)

    def as_tensor(self, *, copy: bool = True) -> np.ndarray:
        self._tensor_buffer.fill(0)

        for target_index in range(self._stack_size):
            frames_back = (self._stack_size - 1 - target_index) * self._frame_step
            if self._history_count <= frames_back:
                continue
            history_index = self._history_count - 1 - frames_back
            self._tensor_buffer[target_index] = self._history_at(history_index)

        if copy:
            return self._tensor_buffer.copy()
        return self._tensor_buffer

    def as_flat(self, *, copy: bool = True) -> np.ndarray:
        tensor = self.as_tensor(copy=False)
        self._flat_buffer[:] = tensor.reshape(-1)
        if copy:
            return self._flat_buffer.copy()
        return self._flat_buffer

    def as_float32_flat(self, *, copy: bool = True) -> np.ndarray:
        tensor = self.as_tensor(copy=False)
        self._flat_float32_buffer[:] = tensor.reshape(-1)
        if copy:
            return self._flat_float32_buffer.copy()
        return self._flat_float32_buffer

    def is_ready(self) -> bool:
        return self._history_count >= self._history_size

    def debug_snapshots(self) -> list[np.ndarray]:
        return [frame.copy() for frame in self._debug_snapshots]

    def debug_bitmaps(self) -> list[np.ndarray]:
        return [bitmap.copy() for bitmap in self._debug_bitmaps]

    def write_debug_bitmaps(self, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        for existing in output_dir.glob("observation-*.pgm"):
            existing.unlink()

        written_paths: list[Path] = []
        for index, bitmap in enumerate(self._debug_bitmaps):
            path = output_dir / f"observation-{index:02d}.pgm"
            self._write_pgm(path, bitmap)
            written_paths.append(path)
        return written_paths

    def _store_frame(self, frame: np.ndarray) -> None:
        self._history[self._history_next_index] = frame
        self._history_next_index = (self._history_next_index + 1) % self._history_size
        self._history_count = min(self._history_count + 1, self._history_size)

        if self._debug_snapshots.maxlen:
            self._debug_snapshots.append(frame.copy())
            self._debug_bitmaps.append(self._build_debug_bitmap())

    def _history_at(self, logical_index: int) -> np.ndarray:
        oldest_index = (self._history_next_index - self._history_count) % self._history_size
        physical_index = (oldest_index + logical_index) % self._history_size
        return self._history[physical_index]

    def _validate(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape != (self._height, self._width):
            raise InvalidFrameError(
                f"Frame shape must be {(self._height, self._width)}, got {frame.shape}."
            )
        if not np.issubdtype(frame.dtype, np.integer):
            raise InvalidFrameError("Frame dtype must be integer-compatible.")
        unique_values = np.unique(frame)
        if not np.all(np.isin(unique_values, [0, 1])):
            raise InvalidFrameError("Frame values must be binary in {0, 1}.")
        return frame.astype(np.uint8, copy=False)

    def _build_debug_bitmap(self) -> np.ndarray:
        tensor = self.as_tensor(copy=False)
        frames = [tensor[index] * np.uint8(255) for index in range(self._stack_size)]
        return np.concatenate(frames, axis=1).copy()

    def _write_pgm(self, path: Path, bitmap: np.ndarray) -> None:
        header = f"P5\n{bitmap.shape[1]} {bitmap.shape[0]}\n255\n".encode("ascii")
        with path.open("wb") as output_file:
            output_file.write(header)
            output_file.write(bitmap.tobytes(order="C"))
