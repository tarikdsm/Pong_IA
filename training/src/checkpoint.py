from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when a checkpoint file is expected but does not exist."""


class InvalidCheckpointError(ValueError):
    """Raised when a checkpoint file is corrupted or has an unexpected layout."""


@dataclass(frozen=True)
class Checkpoint:
    state_dict: dict[str, Any]
    optimizer_state: dict[str, Any]
    hparams: dict[str, Any]
    episode: int
    metrics_history: list[dict[str, Any]]
    evaluation_history: list[dict[str, Any]] = field(default_factory=list)


_REQUIRED_KEYS = frozenset(
    {"state_dict", "optimizer_state", "hparams", "episode", "metrics_history"}
)


def save_checkpoint(path: Path, ckpt: Checkpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": _clone_to_cpu(ckpt.state_dict),
        "optimizer_state": _clone_to_cpu(ckpt.optimizer_state),
        "hparams": ckpt.hparams,
        "episode": ckpt.episode,
        "metrics_history": ckpt.metrics_history,
        "evaluation_history": ckpt.evaluation_history,
    }
    torch.save(payload, str(path))


def load_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device | None = None,
) -> Checkpoint:
    if not path.exists():
        raise CheckpointNotFoundError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(str(path), weights_only=False, map_location=map_location)
    except Exception as exc:
        raise InvalidCheckpointError(
            f"Checkpoint file is corrupted or unreadable: {path}"
        ) from exc

    if not isinstance(payload, dict) or not _REQUIRED_KEYS.issubset(payload.keys()):
        raise InvalidCheckpointError(
            f"Checkpoint file is corrupted or has unexpected layout: {path}"
        )

    return Checkpoint(
        state_dict=payload["state_dict"],
        optimizer_state=payload["optimizer_state"],
        hparams=payload["hparams"],
        episode=int(payload["episode"]),
        metrics_history=list(payload["metrics_history"]),
        evaluation_history=list(payload.get("evaluation_history", [])),
    )


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return value
