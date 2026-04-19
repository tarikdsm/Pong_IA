from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from debug_bitmap_codec import (
    InvalidPortableGraymapError,
    convert_directory_of_pgms,
    convert_pgm_to_png,
)


def test_convert_pgm_to_png_writes_valid_grayscale_png(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.pgm"
    source_path.write_bytes(b"P5\n4 2\n255\n" + bytes([0, 32, 64, 255, 255, 64, 32, 0]))

    output_path = convert_pgm_to_png(source_path, tmp_path / "sample.png")

    payload = output_path.read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert payload[12:16] == b"IHDR"
    width, height = struct.unpack(">II", payload[16:24])
    assert (width, height) == (4, 2)


def test_convert_directory_of_pgms_converts_all_files(tmp_path: Path) -> None:
    for index in range(2):
        path = tmp_path / f"observation-{index:02d}.pgm"
        pixels = np.full((2, 3), index * 127, dtype=np.uint8)
        path.write_bytes(b"P5\n3 2\n255\n" + pixels.tobytes(order="C"))

    converted = convert_directory_of_pgms(tmp_path)

    assert [path.name for path in converted] == ["observation-00.png", "observation-01.png"]
    assert all(path.exists() for path in converted)


def test_convert_pgm_to_png_rejects_non_p5_input(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.pgm"
    source_path.write_text("P2\n1 1\n255\n0\n", encoding="ascii")

    with pytest.raises(InvalidPortableGraymapError, match="P5"):
        convert_pgm_to_png(source_path, tmp_path / "sample.png")
