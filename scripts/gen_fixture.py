from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = REPO_ROOT / "engine"

if str(ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(ENGINE_PATH))

if TYPE_CHECKING:
    from pong_engine.state import Action, GameState

FIXTURES_DIR = REPO_ROOT / "shared" / "fixtures"


class RecordingRng:
    def __init__(self, seed: int) -> None:
        self._generator = np.random.default_rng(seed)
        self.current_step_values: list[float] = []

    def next(self) -> float:
        value = float(self._generator.random())
        self.current_step_values.append(value)
        return value

    def begin_step(self) -> None:
        self.current_step_values = []


def serialize_state(state: GameState) -> dict[str, float | int]:
    return asdict(state)


def right_paddle_tracking(state: "GameState") -> "Action":
    from pong_engine.config import BALL_SIZE, PADDLE_HEIGHT

    paddle_center = state.paddle_right_y + (PADDLE_HEIGHT / 2)
    ball_center = state.ball_y + (BALL_SIZE / 2)
    if ball_center < paddle_center - 1:
        return "up"
    if ball_center > paddle_center + 1:
        return "down"
    return "none"


def build_fixture(
    name: str,
    seed: int,
    steps_count: int,
    initial_state: "GameState",
    right_action_selector: Callable[["GameState"], "Action"],
) -> dict[str, object]:
    from pong_engine.heuristics import partially_tracking
    from pong_engine.physics import step

    rng = RecordingRng(seed)
    state = initial_state
    steps: list[dict[str, object]] = []

    for tick in range(steps_count):
        rng.begin_step()
        left_action = partially_tracking(state, rng)
        right_action = right_action_selector(state)
        state = step(state, left_action, right_action, rng)
        steps.append(
            {
                "tick": tick,
                "leftAction": left_action,
                "rightAction": right_action,
                "randomValues": list(rng.current_step_values),
                "expectedState": serialize_state(state),
            }
        )

    return {
        "name": name,
        "seed": seed,
        "initialState": serialize_state(initial_state),
        "steps": steps,
    }


def build_default_fixtures() -> list[dict[str, object]]:
    from pong_engine.config import ARENA_HEIGHT, ARENA_WIDTH, BALL_MAX_SPEED, BALL_SIZE, PADDLE_HEIGHT
    from pong_engine.state import GameState, create_initial_state

    short_game = build_fixture(
        name="short_game_low_speed",
        seed=42,
        steps_count=300,
        initial_state=create_initial_state(np.random.default_rng(42)),
        right_action_selector=right_paddle_tracking,
    )
    long_game = build_fixture(
        name="long_game_accelerated",
        seed=99,
        steps_count=2000,
        initial_state=create_initial_state(np.random.default_rng(99)),
        right_action_selector=right_paddle_tracking,
    )
    max_speed_start = GameState(
        ball_x=ARENA_WIDTH - BALL_SIZE - 4.0,
        ball_y=(ARENA_HEIGHT / 2) - 4.0,
        ball_vx=BALL_MAX_SPEED * 0.98,
        ball_vy=0.0,
        ball_speed=BALL_MAX_SPEED * 0.98,
        paddle_left_y=(ARENA_HEIGHT - PADDLE_HEIGHT) // 2,
        paddle_right_y=(ARENA_HEIGHT - PADDLE_HEIGHT) // 2,
        score_left=0,
        score_right=0,
        tick=0,
    )
    max_speed = build_fixture(
        name="max_speed_clamped",
        seed=7,
        steps_count=500,
        initial_state=max_speed_start,
        right_action_selector=right_paddle_tracking,
    )
    return [short_game, long_game, max_speed]


def write_fixture(path: Path, fixture: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=2), encoding="utf-8", newline="\n")


def generate_fixtures() -> list[Path]:
    output_paths: list[Path] = []
    for fixture in build_default_fixtures():
        path = FIXTURES_DIR / f"{fixture['name']}.json"
        write_fixture(path, fixture)
        output_paths.append(path)
    return output_paths


def main() -> None:
    generate_fixtures()


if __name__ == "__main__":
    main()
