from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from export_onnx import (
    DEFAULT_OPSET_VERSION,
    InvalidOnnxParityError,
    OnnxParityResult,
    OUTPUT_NAMES,
    export_to_onnx,
    verify_onnx_parity,
)
from model import INPUT_DIM, PongPolicyNetwork


@pytest.fixture(scope="module")
def trained_model() -> PongPolicyNetwork:
    torch.manual_seed(123)
    return PongPolicyNetwork()


def test_export_to_onnx_creates_file_at_path(
    trained_model: PongPolicyNetwork, tmp_path: Path
) -> None:
    path = tmp_path / "model.onnx"

    export_to_onnx(trained_model, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_export_to_onnx_creates_missing_parent_directories(
    trained_model: PongPolicyNetwork, tmp_path: Path
) -> None:
    path = tmp_path / "nested" / "deep" / "model.onnx"

    export_to_onnx(trained_model, path)

    assert path.exists()


def test_export_to_onnx_produces_valid_onnx_model(
    trained_model: PongPolicyNetwork, tmp_path: Path
) -> None:
    path = tmp_path / "model.onnx"
    export_to_onnx(trained_model, path)

    model_proto = onnx.load(str(path))
    onnx.checker.check_model(model_proto)


def test_export_to_onnx_uses_expected_input_and_output_names(
    trained_model: PongPolicyNetwork, tmp_path: Path
) -> None:
    path = tmp_path / "model.onnx"
    export_to_onnx(trained_model, path)

    model_proto = onnx.load(str(path))
    input_names = [item.name for item in model_proto.graph.input]
    output_names = [item.name for item in model_proto.graph.output]

    assert input_names == ["observation"]
    assert output_names == list(OUTPUT_NAMES)


def test_export_to_onnx_supports_dynamic_batch_dimension(
    trained_model: PongPolicyNetwork, tmp_path: Path
) -> None:
    path = tmp_path / "model.onnx"
    export_to_onnx(trained_model, path)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    single = np.zeros((1, INPUT_DIM), dtype=np.float32)
    batch = np.zeros((4, INPUT_DIM), dtype=np.float32)

    out_single = session.run(list(OUTPUT_NAMES), {"observation": single})
    out_batch = session.run(list(OUTPUT_NAMES), {"observation": batch})

    assert out_single[0].shape == (1, 3)
    assert out_single[1].shape == (1, 200)
    assert out_single[2].shape == (1, 200)
    assert out_single[3].shape == (1, 100)
    assert out_batch[0].shape == (4, 3)
    assert out_batch[1].shape == (4, 200)
    assert out_batch[2].shape == (4, 200)
    assert out_batch[3].shape == (4, 100)


def test_export_to_onnx_uses_default_opset_version(
    trained_model: PongPolicyNetwork, tmp_path: Path
) -> None:
    path = tmp_path / "model.onnx"
    export_to_onnx(trained_model, path)

    model_proto = onnx.load(str(path))
    opsets = [opset.version for opset in model_proto.opset_import if opset.domain in ("", "ai.onnx")]

    assert DEFAULT_OPSET_VERSION in opsets


def test_verify_onnx_parity_returns_passing_result_for_same_export(
    trained_model: PongPolicyNetwork, tmp_path: Path
) -> None:
    path = tmp_path / "model.onnx"
    export_to_onnx(trained_model, path)

    rng = np.random.default_rng(0)
    samples = rng.uniform(0.0, 1.0, size=(50, INPUT_DIM)).astype(np.float32)

    result = verify_onnx_parity(trained_model, path, samples, tol=1e-4)

    assert isinstance(result, OnnxParityResult)
    assert result.passed is True
    assert result.max_abs_diff < 1e-4
    assert result.sample_count == 50


def test_verify_onnx_parity_detects_mismatch_when_models_differ(
    trained_model: PongPolicyNetwork, tmp_path: Path
) -> None:
    path = tmp_path / "model.onnx"
    export_to_onnx(trained_model, path)

    different_model = PongPolicyNetwork()
    with torch.no_grad():
        for parameter in different_model.parameters():
            parameter.add_(1.0)

    samples = np.zeros((5, INPUT_DIM), dtype=np.float32)
    samples[:, 0] = 1.0

    result = verify_onnx_parity(different_model, path, samples, tol=1e-4)

    assert result.passed is False
    assert result.max_abs_diff > 1e-4


def test_verify_onnx_parity_rejects_wrong_sample_shape(
    trained_model: PongPolicyNetwork, tmp_path: Path
) -> None:
    path = tmp_path / "model.onnx"
    export_to_onnx(trained_model, path)

    bad_samples = np.zeros((5, 100), dtype=np.float32)

    with pytest.raises(InvalidOnnxParityError, match="shape"):
        verify_onnx_parity(trained_model, path, bad_samples)


def test_verify_onnx_parity_rejects_empty_samples(
    trained_model: PongPolicyNetwork, tmp_path: Path
) -> None:
    path = tmp_path / "model.onnx"
    export_to_onnx(trained_model, path)

    empty_samples = np.zeros((0, INPUT_DIM), dtype=np.float32)

    with pytest.raises(InvalidOnnxParityError, match="empty"):
        verify_onnx_parity(trained_model, path, empty_samples)
