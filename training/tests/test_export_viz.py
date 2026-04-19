from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from export_viz import (
    DEFAULT_METADATA_FILENAME,
    DEFAULT_WEIGHTS_FILENAME,
    UINT8_ZERO_LEVEL,
    export_model_visualization,
    quantize_first_layer_weights,
)
from model import HIDDEN_DIMS, PongPolicyNetwork
from pong_engine.config import BITMAP_HEIGHT, BITMAP_WIDTH, FRAME_STACK


@pytest.fixture(scope="module")
def trained_model() -> PongPolicyNetwork:
    torch.manual_seed(321)
    return PongPolicyNetwork()


def test_quantize_first_layer_weights_centers_zero_weights() -> None:
    weights = np.zeros((2, FRAME_STACK, BITMAP_HEIGHT, BITMAP_WIDTH), dtype=np.float32)

    quantized = quantize_first_layer_weights(weights)

    assert quantized.dtype == np.uint8
    assert np.all(quantized == UINT8_ZERO_LEVEL)


def test_export_model_visualization_writes_metadata_and_binary(
    trained_model: PongPolicyNetwork, tmp_path: Path
) -> None:
    result = export_model_visualization(trained_model, tmp_path)

    assert result.metadata_path == tmp_path / DEFAULT_METADATA_FILENAME
    assert result.weights_path == tmp_path / DEFAULT_WEIGHTS_FILENAME
    assert result.neuron_count == HIDDEN_DIMS[0]
    assert result.metadata_path.exists()
    assert result.weights_path.exists()

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    expected_bytes = HIDDEN_DIMS[0] * FRAME_STACK * BITMAP_HEIGHT * BITMAP_WIDTH

    assert metadata["frameStack"] == FRAME_STACK
    assert metadata["bitmapWidth"] == BITMAP_WIDTH
    assert metadata["bitmapHeight"] == BITMAP_HEIGHT
    assert metadata["hiddenDims"] == list(HIDDEN_DIMS)
    assert metadata["firstLayerNeurons"] == HIDDEN_DIMS[0]
    assert metadata["weightsUrl"] == DEFAULT_WEIGHTS_FILENAME
    assert result.weights_path.stat().st_size == expected_bytes
