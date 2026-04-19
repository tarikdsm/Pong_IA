from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "shared" / "config.json"
DEFAULT_PYTHON_OUTPUT_PATH = REPO_ROOT / "engine" / "pong_engine" / "config.py"
DEFAULT_TS_OUTPUT_PATH = REPO_ROOT / "web" / "src" / "engine" / "config.ts"

FIELD_SPECS = (
    ("arena.width", "ARENA_WIDTH", "arenaWidth", int),
    ("arena.height", "ARENA_HEIGHT", "arenaHeight", int),
    ("paddle.width", "PADDLE_WIDTH", "paddleWidth", int),
    ("paddle.height", "PADDLE_HEIGHT", "paddleHeight", int),
    ("paddle.speed", "PADDLE_SPEED", "paddleSpeed", int),
    ("ball.size", "BALL_SIZE", "ballSize", int),
    ("ball.initial_speed", "BALL_INITIAL_SPEED", "ballInitialSpeed", float),
    ("ball.max_speed", "BALL_MAX_SPEED", "ballMaxSpeed", float),
    (
        "ball.acceleration_factor",
        "BALL_ACCELERATION_FACTOR",
        "ballAccelerationFactor",
        float,
    ),
    ("match.score_to_win", "SCORE_TO_WIN", "scoreToWin", int),
    ("match.fps", "FPS", "fps", int),
    ("bitmap.width", "BITMAP_WIDTH", "bitmapWidth", int),
    ("bitmap.height", "BITMAP_HEIGHT", "bitmapHeight", int),
    ("bitmap.frame_stack", "FRAME_STACK", "frameStack", int),
    (
        "bitmap.frame_step_ticks",
        "FRAME_STEP_TICKS",
        "frameStepTicks",
        int,
    ),
    (
        "ai.inference_interval_ticks",
        "INFERENCE_INTERVAL_TICKS",
        "inferenceIntervalTicks",
        int,
    ),
    (
        "ai.heuristic_interval_ticks",
        "HEURISTIC_INTERVAL_TICKS",
        "heuristicIntervalTicks",
        int,
    ),
    ("random.default_seed", "DEFAULT_SEED", "defaultSeed", int),
)

MINIMUM_VALUES = {
    "arena.width": 1,
    "arena.height": 1,
    "paddle.width": 1,
    "paddle.height": 1,
    "paddle.speed": 1,
    "ball.size": 1,
    "ball.initial_speed": 0.000001,
    "ball.max_speed": 0.000001,
    "ball.acceleration_factor": 1.0,
    "match.score_to_win": 1,
    "match.fps": 1,
    "bitmap.width": 1,
    "bitmap.height": 1,
    "bitmap.frame_stack": 1,
    "bitmap.frame_step_ticks": 1,
    "ai.inference_interval_ticks": 1,
    "ai.heuristic_interval_ticks": 1,
    "random.default_seed": 0,
}


class ConfigValidationError(ValueError):
    """Raised when shared/config.json contains invalid values."""


def load_config(config_path: Path) -> dict[str, Any]:
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigValidationError(f"Config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(f"Invalid JSON in {config_path}: {exc.msg}") from exc

    if not isinstance(raw_config, dict):
        raise ConfigValidationError("Config root must be an object.")

    return raw_config


def flatten_config(raw_config: dict[str, Any]) -> dict[str, int | float]:
    values = {
        constant_name: read_number(raw_config, path, expected_type)
        for path, constant_name, _, expected_type in FIELD_SPECS
    }
    validate_cross_field_rules(values)
    return values


def read_number(
    raw_config: dict[str, Any],
    path: str,
    expected_type: type[int] | type[float],
) -> int | float:
    minimum = MINIMUM_VALUES[path]
    value = read_path(raw_config, path)
    if expected_type is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigValidationError(f"{path} must be an integer.")
        if value < int(minimum):
            raise ConfigValidationError(f"{path} must be greater than or equal to {int(minimum)}.")
        return value

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigValidationError(f"{path} must be a number.")
    numeric_value = float(value)
    if numeric_value < float(minimum):
        raise ConfigValidationError(
            f"{path} must be greater than or equal to {format_python_value(minimum)}."
        )
    return numeric_value


def read_path(raw_config: dict[str, Any], path: str) -> Any:
    current: Any = raw_config
    traversed: list[str] = []
    for segment in path.split("."):
        traversed.append(segment)
        if not isinstance(current, dict) or segment not in current:
            joined_path = ".".join(traversed)
            raise ConfigValidationError(f"Missing required config key: {joined_path}")
        current = current[segment]
    return current


def validate_cross_field_rules(values: dict[str, int | float]) -> None:
    ensure_at_least(
        values,
        "BALL_MAX_SPEED",
        float(values["BALL_INITIAL_SPEED"]),
        "ball.max_speed must be greater than or equal to ball.initial_speed.",
    )
    ensure_at_most(
        values,
        "PADDLE_HEIGHT",
        int(values["ARENA_HEIGHT"]),
        "paddle.height must be less than or equal to arena.height.",
    )
    ensure_at_most(
        values,
        "PADDLE_WIDTH",
        int(values["ARENA_WIDTH"]),
        "paddle.width must be less than or equal to arena.width.",
    )
    ensure_at_most(
        values,
        "BALL_SIZE",
        min(int(values["ARENA_WIDTH"]), int(values["ARENA_HEIGHT"])),
        "ball.size must fit inside the arena dimensions.",
    )

    if values["BITMAP_WIDTH"] != values["ARENA_WIDTH"]:
        raise ConfigValidationError("bitmap.width must match arena.width.")
    if values["BITMAP_HEIGHT"] != values["ARENA_HEIGHT"]:
        raise ConfigValidationError("bitmap.height must match arena.height.")


def ensure_at_least(
    values: dict[str, int | float],
    key: str,
    minimum: float,
    message: str | None = None,
) -> None:
    if float(values[key]) < minimum:
        raise ConfigValidationError(message or f"{key} must be at least {minimum}.")


def ensure_at_most(
    values: dict[str, int | float],
    key: str,
    maximum: int,
    message: str,
) -> None:
    if int(values[key]) > maximum:
        raise ConfigValidationError(message)


def generate_config_artifacts(
    config_path: Path = DEFAULT_CONFIG_PATH,
    python_output_path: Path = DEFAULT_PYTHON_OUTPUT_PATH,
    ts_output_path: Path = DEFAULT_TS_OUTPUT_PATH,
) -> None:
    values = flatten_config(load_config(config_path))
    python_source = render_python_config(values)
    ts_source = render_ts_config(values)
    write_text_file(python_output_path, python_source)
    write_text_file(ts_output_path, ts_source)


def render_python_config(values: dict[str, int | float]) -> str:
    assignments = "\n".join(
        f"{constant_name} = {format_python_value(values[constant_name])}"
        for _, constant_name, _, _ in FIELD_SPECS
    )
    config_lines = "\n".join(
        f'    "{python_key}": {constant_name},'
        for _, constant_name, camel_key, _ in FIELD_SPECS
        for python_key in [camel_to_snake(camel_key)]
    )
    return "\n".join(
        (
            "# AUTO-GENERATED - do not edit manually.",
            "# Source: shared/config.json",
            "",
            assignments,
            "",
            "CONFIG = {",
            config_lines,
            "}",
            "",
        )
    )


def render_ts_config(values: dict[str, int | float]) -> str:
    assignments = "\n".join(
        f"export const {constant_name} = {format_ts_value(values[constant_name])} as const;"
        for _, constant_name, _, _ in FIELD_SPECS
    )
    config_lines = "\n".join(
        f"  {camel_key}: {constant_name},"
        for _, constant_name, camel_key, _ in FIELD_SPECS
    )
    return "\n".join(
        (
            "// AUTO-GENERATED - do not edit manually.",
            "// Source: shared/config.json",
            "",
            assignments,
            "",
            "export const CONFIG = {",
            config_lines,
            "} as const;",
            "",
            "export type Config = typeof CONFIG;",
            "",
        )
    )


def format_python_value(value: int | float) -> str:
    return str(int(value)) if isinstance(value, int) else repr(value)


def format_ts_value(value: int | float) -> str:
    return str(int(value)) if isinstance(value, int) else repr(value)


def camel_to_snake(value: str) -> str:
    characters: list[str] = []
    for character in value:
        if character.isupper():
            characters.append("_")
            characters.append(character.lower())
            continue
        characters.append(character)
    return "".join(characters)


def write_text_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(contents, encoding="utf-8", newline="\n")
    temp_path.replace(path)


def main() -> None:
    generate_config_artifacts()


if __name__ == "__main__":
    main()
