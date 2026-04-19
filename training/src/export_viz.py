from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from model import HIDDEN_DIMS, PongPolicyNetwork
from pong_engine.config import BITMAP_HEIGHT, BITMAP_WIDTH, FRAME_STACK


DEFAULT_METADATA_FILENAME = "model-viz.json"
DEFAULT_WEIGHTS_FILENAME = "model-first-layer.uint8.bin"
UINT8_ZERO_LEVEL = 128


@dataclass(frozen=True)
class VisualizationExportResult:
    metadata_path: Path
    weights_path: Path
    neuron_count: int


def export_model_visualization(
    network: PongPolicyNetwork,
    output_dir: Path,
    *,
    metadata_filename: str = DEFAULT_METADATA_FILENAME,
    weights_filename: str = DEFAULT_WEIGHTS_FILENAME,
) -> VisualizationExportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / metadata_filename
    weights_path = output_dir / weights_filename

    weights = (
        network.input_layer.weight.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    neuron_count = int(weights.shape[0])
    reshaped = weights.reshape(neuron_count, FRAME_STACK, BITMAP_HEIGHT, BITMAP_WIDTH)
    quantized = quantize_first_layer_weights(reshaped)

    weights_path.write_bytes(quantized.tobytes())
    metadata_path.write_text(
        json.dumps(
            {
                "version": 1,
                "frameStack": FRAME_STACK,
                "bitmapWidth": BITMAP_WIDTH,
                "bitmapHeight": BITMAP_HEIGHT,
                "hiddenDims": list(HIDDEN_DIMS),
                "firstLayerNeurons": neuron_count,
                "weightsUrl": weights_filename,
                "weightsEncoding": "uint8-centered-per-neuron-absmax",
                "zeroLevel": UINT8_ZERO_LEVEL,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return VisualizationExportResult(
        metadata_path=metadata_path,
        weights_path=weights_path,
        neuron_count=neuron_count,
    )


def quantize_first_layer_weights(weights: np.ndarray) -> np.ndarray:
    if weights.ndim != 4:
        raise ValueError("first-layer weights must have shape (neurons, frames, height, width).")

    abs_max = np.max(np.abs(weights), axis=(1, 2, 3), keepdims=True)
    safe_abs_max = np.where(abs_max > 0.0, abs_max, 1.0)
    normalized = np.clip(weights / safe_abs_max, -1.0, 1.0)
    scaled = np.rint((normalized + 1.0) * 127.5)
    return scaled.astype(np.uint8, copy=False)
