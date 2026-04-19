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


import numpy as np  # noqa: E402

from checkpoint import (  # noqa: E402
    CheckpointNotFoundError,
    InvalidCheckpointError,
    load_checkpoint,
)
from export_onnx import (  # noqa: E402
    DEFAULT_OPSET_VERSION,
    export_to_onnx,
    verify_onnx_parity,
)
from export_viz import (  # noqa: E402
    DEFAULT_METADATA_FILENAME,
    DEFAULT_WEIGHTS_FILENAME,
    export_model_visualization,
)
from model import INPUT_DIM, PongPolicyNetwork  # noqa: E402


DEFAULT_OUTPUT_PATH = REPO_ROOT / "web" / "public" / "model.onnx"
DEFAULT_PARITY_SAMPLES = 50
DEFAULT_PARITY_TOL = 1e-4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained Pong policy checkpoint to ONNX."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET_VERSION)
    parser.add_argument("--parity-samples", type=int, default=DEFAULT_PARITY_SAMPLES)
    parser.add_argument("--parity-tol", type=float, default=DEFAULT_PARITY_TOL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-parity", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    except CheckpointNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except InvalidCheckpointError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    network = PongPolicyNetwork()
    network.load_state_dict(checkpoint.state_dict)
    network.eval()

    export_to_onnx(network, args.output, opset_version=args.opset)
    print(f"exported checkpoint={args.checkpoint} -> onnx={args.output} opset={args.opset}")

    metadata_filename, weights_filename = build_visualization_filenames(args.output)
    viz_result = export_model_visualization(
        network,
        args.output.parent,
        metadata_filename=metadata_filename,
        weights_filename=weights_filename,
    )
    print(
        "exported visualization "
        f"metadata={viz_result.metadata_path} "
        f"weights={viz_result.weights_path} "
        f"neurons={viz_result.neuron_count}"
    )

    if args.skip_parity:
        return 0

    rng = np.random.default_rng(args.seed)
    samples = rng.uniform(0.0, 1.0, size=(args.parity_samples, INPUT_DIM)).astype(np.float32)
    result = verify_onnx_parity(network, args.output, samples, tol=args.parity_tol)

    print(
        f"parity samples={result.sample_count} "
        f"max_abs_diff={result.max_abs_diff:.3e} "
        f"tol={args.parity_tol:.3e} "
        f"passed={result.passed}"
    )
    return 0 if result.passed else 1


def build_visualization_filenames(onnx_path: Path) -> tuple[str, str]:
    if onnx_path.stem == "model":
        return DEFAULT_METADATA_FILENAME, DEFAULT_WEIGHTS_FILENAME
    return (
        f"{onnx_path.stem}-viz.json",
        f"{onnx_path.stem}-first-layer.uint8.bin",
    )


if __name__ == "__main__":
    raise SystemExit(main())
