from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch
from torch import nn, optim

from checkpoint import (
    Checkpoint,
    CheckpointNotFoundError,
    InvalidCheckpointError,
    load_checkpoint,
    save_checkpoint,
)


def _make_state() -> tuple[Checkpoint, nn.Module, optim.Optimizer]:
    network = nn.Linear(4, 2)
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    loss = network(torch.zeros(1, 4)).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ckpt = Checkpoint(
        state_dict={key: value.detach().clone() for key, value in network.state_dict().items()},
        optimizer_state=optimizer.state_dict(),
        hparams={"lr": 0.001, "gamma": 0.99, "seed": 42, "reward_shaping": False},
        episode=10,
        metrics_history=[
            {"episode": 0, "loss": 1.0, "total_reward": 0.0, "episode_length": 5},
            {"episode": 1, "loss": 0.9, "total_reward": 1.0, "episode_length": 6},
        ],
        evaluation_history=[
            {"episode": 1, "hit_count": 4, "miss_count": 2, "hit_rate": 2 / 3},
        ],
    )
    return ckpt, network, optimizer


def test_checkpoint_is_frozen_dataclass_with_expected_fields() -> None:
    ckpt = Checkpoint(
        state_dict={},
        optimizer_state={},
        hparams={"lr": 0.001},
        episode=0,
        metrics_history=[],
        evaluation_history=[],
    )

    assert ckpt.state_dict == {}
    assert ckpt.optimizer_state == {}
    assert ckpt.hparams == {"lr": 0.001}
    assert ckpt.episode == 0
    assert ckpt.metrics_history == []
    assert ckpt.evaluation_history == []

    with pytest.raises(dataclasses.FrozenInstanceError):
        ckpt.episode = 1  # type: ignore[misc]


def test_save_and_load_checkpoint_preserves_state_dict(tmp_path: Path) -> None:
    ckpt, _, _ = _make_state()
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, ckpt)
    loaded = load_checkpoint(path)

    assert set(loaded.state_dict.keys()) == set(ckpt.state_dict.keys())
    for key, tensor in ckpt.state_dict.items():
        assert torch.equal(loaded.state_dict[key], tensor)


def test_save_and_load_checkpoint_preserves_optimizer_state(tmp_path: Path) -> None:
    ckpt, network, _ = _make_state()
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, ckpt)
    loaded = load_checkpoint(path)

    fresh_optimizer = optim.Adam(network.parameters(), lr=0.001)
    fresh_optimizer.load_state_dict(loaded.optimizer_state)

    assert (
        fresh_optimizer.state_dict()["param_groups"][0]["lr"]
        == ckpt.optimizer_state["param_groups"][0]["lr"]
    )


def test_save_and_load_checkpoint_preserves_hparams_and_episode(tmp_path: Path) -> None:
    ckpt, _, _ = _make_state()
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, ckpt)
    loaded = load_checkpoint(path)

    assert loaded.hparams == ckpt.hparams
    assert loaded.episode == ckpt.episode


def test_save_and_load_checkpoint_preserves_metrics_history(tmp_path: Path) -> None:
    ckpt, _, _ = _make_state()
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, ckpt)
    loaded = load_checkpoint(path)

    assert loaded.metrics_history == ckpt.metrics_history


def test_save_and_load_checkpoint_preserves_evaluation_history(tmp_path: Path) -> None:
    ckpt, _, _ = _make_state()
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, ckpt)
    loaded = load_checkpoint(path)

    assert loaded.evaluation_history == ckpt.evaluation_history


def test_save_checkpoint_creates_missing_parent_directories(tmp_path: Path) -> None:
    ckpt, _, _ = _make_state()
    path = tmp_path / "nested" / "deep" / "ckpt.pt"

    save_checkpoint(path, ckpt)

    assert path.exists()


def test_save_checkpoint_overwrites_existing_file(tmp_path: Path) -> None:
    first, _, _ = _make_state()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, first)

    second = Checkpoint(
        state_dict={},
        optimizer_state={},
        hparams={"lr": 0.5},
        episode=999,
        metrics_history=[],
        evaluation_history=[],
    )
    save_checkpoint(path, second)
    loaded = load_checkpoint(path)

    assert loaded.episode == 999
    assert loaded.hparams == {"lr": 0.5}


def test_load_checkpoint_raises_clear_error_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.pt"

    with pytest.raises(CheckpointNotFoundError, match="does_not_exist"):
        load_checkpoint(missing)


def test_load_checkpoint_raises_clear_error_when_file_corrupted(tmp_path: Path) -> None:
    corrupted = tmp_path / "corrupted.pt"
    corrupted.write_bytes(b"this is not a torch file")

    with pytest.raises(InvalidCheckpointError, match="corrupted"):
        load_checkpoint(corrupted)


def test_load_checkpoint_accepts_map_location_argument(tmp_path: Path) -> None:
    ckpt, _, _ = _make_state()
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, ckpt)
    loaded = load_checkpoint(path, map_location="cpu")

    assert loaded.episode == ckpt.episode
    for tensor in loaded.state_dict.values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"
