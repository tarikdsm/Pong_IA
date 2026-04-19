from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


class InvalidPortableGraymapError(ValueError):
    """Raised when a .pgm file is malformed or unsupported."""


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def convert_pgm_to_png(source_path: Path, output_path: Path) -> Path:
    bitmap = read_pgm(source_path)
    write_grayscale_png(output_path, bitmap)
    return output_path


def convert_directory_of_pgms(input_dir: Path, output_dir: Path | None = None) -> list[Path]:
    resolved_output_dir = output_dir if output_dir is not None else input_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    converted_paths: list[Path] = []
    for source_path in sorted(input_dir.glob("*.pgm")):
        output_path = resolved_output_dir / f"{source_path.stem}.png"
        convert_pgm_to_png(source_path, output_path)
        converted_paths.append(output_path)
    return converted_paths


def read_pgm(path: Path) -> np.ndarray:
    with path.open("rb") as input_file:
        magic = _read_non_comment_line(input_file)
        if magic != b"P5":
            raise InvalidPortableGraymapError("Only binary PGM (P5) files are supported.")

        dimensions = _read_non_comment_line(input_file).split()
        if len(dimensions) != 2:
            raise InvalidPortableGraymapError("PGM dimensions must contain width and height.")
        width, height = (int(value) for value in dimensions)

        max_value = int(_read_non_comment_line(input_file))
        if max_value != 255:
            raise InvalidPortableGraymapError("Only 8-bit PGM files with max value 255 are supported.")

        payload = input_file.read()

    expected_size = width * height
    if len(payload) != expected_size:
        raise InvalidPortableGraymapError(
            f"PGM payload size mismatch: expected {expected_size} bytes, got {len(payload)}."
        )
    return np.frombuffer(payload, dtype=np.uint8).reshape((height, width))


def write_grayscale_png(path: Path, bitmap: np.ndarray) -> None:
    if bitmap.ndim != 2:
        raise ValueError("PNG writer expects a 2D grayscale bitmap.")
    if bitmap.dtype != np.uint8:
        bitmap = bitmap.astype(np.uint8, copy=False)

    height, width = bitmap.shape
    raw_rows = b"".join(b"\x00" + bitmap[row_index].tobytes(order="C") for row_index in range(height))
    compressed = zlib.compress(raw_rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output_file:
        output_file.write(PNG_SIGNATURE)
        output_file.write(_png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)))
        output_file.write(_png_chunk(b"IDAT", compressed))
        output_file.write(_png_chunk(b"IEND", b""))


def _read_non_comment_line(input_file) -> bytes:
    while True:
        line = input_file.readline()
        if not line:
            raise InvalidPortableGraymapError("Unexpected end of file while reading PGM header.")
        stripped = line.strip()
        if stripped and not stripped.startswith(b"#"):
            return stripped


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc)
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc & 0xFFFFFFFF)
