from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_SRC_PATH = REPO_ROOT / "training" / "src"

if str(TRAINING_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC_PATH))


from debug_bitmap_codec import convert_directory_of_pgms  # noqa: E402


DEFAULT_INPUT_DIR = REPO_ROOT / "training" / "debug" / "bitmaps"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all debug PGM bitmaps to PNG."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input_dir.exists():
        print(f"error: input directory not found: {args.input_dir}", file=sys.stderr)
        return 2

    converted = convert_directory_of_pgms(args.input_dir, args.output_dir)
    output_dir = args.output_dir if args.output_dir is not None else args.input_dir
    print(
        f"converted={len(converted)} "
        f"output_dir={output_dir.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
