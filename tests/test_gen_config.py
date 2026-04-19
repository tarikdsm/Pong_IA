from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_gen_config_module():
    module_path = REPO_ROOT / "scripts" / "gen_config.py"
    spec = importlib.util.spec_from_file_location("gen_config", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_config_artifacts_writes_python_and_ts_files(tmp_path: Path) -> None:
    module = load_gen_config_module()
    config_path = tmp_path / "shared" / "config.json"
    python_output_path = tmp_path / "engine" / "pong_engine" / "config.py"
    ts_output_path = tmp_path / "web" / "src" / "engine" / "config.ts"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
        {
          "arena": { "width": 80, "height": 60 },
          "paddle": { "width": 2, "height": 12, "speed": 2 },
          "ball": {
            "size": 2,
            "initial_speed": 1.5,
            "max_speed": 6.0,
            "acceleration_factor": 1.03
          },
          "match": { "score_to_win": 21, "fps": 60 },
          "bitmap": { "width": 80, "height": 60, "frame_stack": 5, "frame_step_ticks": 5 },
          "ai": { "inference_interval_ticks": 30, "heuristic_interval_ticks": 60 },
          "random": { "default_seed": 42 }
        }
        """.strip(),
        encoding="utf-8",
    )

    module.generate_config_artifacts(config_path, python_output_path, ts_output_path)

    python_config = python_output_path.read_text(encoding="utf-8")
    ts_config = ts_output_path.read_text(encoding="utf-8")

    assert "# AUTO-GENERATED - do not edit manually." in python_config
    assert "ARENA_WIDTH = 80" in python_config
    assert "FRAME_STEP_TICKS = 5" in python_config
    assert "BALL_ACCELERATION_FACTOR = 1.03" in python_config
    assert "CONFIG = {" in python_config

    assert "// AUTO-GENERATED - do not edit manually." in ts_config
    assert "export const ARENA_WIDTH = 80 as const;" in ts_config
    assert "export const FRAME_STEP_TICKS = 5 as const;" in ts_config
    assert "export const BALL_ACCELERATION_FACTOR = 1.03 as const;" in ts_config
    assert "export const CONFIG = {" in ts_config


def test_generate_config_artifacts_rejects_invalid_config_without_partial_writes(
    tmp_path: Path,
) -> None:
    module = load_gen_config_module()
    config_path = tmp_path / "shared" / "config.json"
    python_output_path = tmp_path / "engine" / "pong_engine" / "config.py"
    ts_output_path = tmp_path / "web" / "src" / "engine" / "config.ts"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
        {
          "arena": { "width": 80, "height": 60 },
          "paddle": { "width": 2, "height": 12, "speed": 2 },
          "ball": {
            "size": 2,
            "initial_speed": -1.5,
            "max_speed": 6.0,
            "acceleration_factor": 1.03
          },
          "match": { "score_to_win": 21, "fps": 60 },
          "bitmap": { "width": 80, "height": 60, "frame_stack": 5, "frame_step_ticks": 5 },
          "ai": { "inference_interval_ticks": 30, "heuristic_interval_ticks": 60 },
          "random": { "default_seed": 42 }
        }
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(module.ConfigValidationError, match="ball.initial_speed"):
        module.generate_config_artifacts(config_path, python_output_path, ts_output_path)

    assert not python_output_path.exists()
    assert not ts_output_path.exists()
