from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from model import INPUT_DIM, PongPolicyNetwork


DEFAULT_OPSET_VERSION = 17
OUTPUT_NAMES = ("logits", "hidden_one", "hidden_two", "hidden_three")


class InvalidOnnxParityError(ValueError):
    """Raised when ONNX parity inputs violate the expected shape or are empty."""


@dataclass(frozen=True)
class OnnxParityResult:
    passed: bool
    max_abs_diff: float
    sample_count: int


class _PolicyOutputs(nn.Module):
    def __init__(self, network: PongPolicyNetwork) -> None:
        super().__init__()
        self.network = network

    def forward(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        forward = self.network(observation)
        hidden_one, hidden_two, hidden_three = forward.hidden_activations
        return forward.logits, hidden_one, hidden_two, hidden_three


def export_to_onnx(
    network: PongPolicyNetwork,
    path: Path,
    *,
    opset_version: int = DEFAULT_OPSET_VERSION,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _PolicyOutputs(network)
    wrapper.eval()
    dummy_input = torch.zeros((1, INPUT_DIM), dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(path),
        input_names=["observation"],
        output_names=list(OUTPUT_NAMES),
        dynamic_axes={
            "observation": {0: "batch"},
            **{name: {0: "batch"} for name in OUTPUT_NAMES},
        },
        opset_version=opset_version,
        dynamo=False,
    )


def verify_onnx_parity(
    network: PongPolicyNetwork,
    onnx_path: Path,
    samples: np.ndarray,
    *,
    tol: float = 1e-4,
) -> OnnxParityResult:
    if samples.ndim != 2 or samples.shape[1] != INPUT_DIM:
        raise InvalidOnnxParityError(
            f"samples must have shape (N, {INPUT_DIM}), got {samples.shape}."
        )
    if samples.shape[0] == 0:
        raise InvalidOnnxParityError("samples must not be empty.")

    samples_f32 = samples.astype(np.float32, copy=False)

    network.eval()
    with torch.no_grad():
        forward = network(torch.from_numpy(samples_f32))
        torch_outputs = {
            "logits": forward.logits.numpy(),
            "hidden_one": forward.hidden_activations[0].numpy(),
            "hidden_two": forward.hidden_activations[1].numpy(),
            "hidden_three": forward.hidden_activations[2].numpy(),
        }

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_values = session.run(list(OUTPUT_NAMES), {"observation": samples_f32})
    onnx_outputs = dict(zip(OUTPUT_NAMES, onnx_values, strict=True))

    max_abs_diff = max(
        float(np.max(np.abs(torch_outputs[name] - onnx_outputs[name])))
        for name in OUTPUT_NAMES
    )
    return OnnxParityResult(
        passed=max_abs_diff < tol,
        max_abs_diff=max_abs_diff,
        sample_count=int(samples.shape[0]),
    )
