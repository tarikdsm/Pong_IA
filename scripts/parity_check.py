from __future__ import annotations

import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = REPO_ROOT / "engine"

if str(ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(ENGINE_PATH))

if TYPE_CHECKING:
    from pong_engine.state import GameState

FIXTURES_DIR = REPO_ROOT / "shared" / "fixtures"
FLOAT_FIELDS = {"ball_x", "ball_y", "ball_vx", "ball_vy", "ball_speed"}


class ReplayRng:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)
        self._index = 0

    def next(self) -> float:
        if self._index >= len(self._values):
            raise AssertionError("Fixture RNG values were exhausted during replay.")
        value = self._values[self._index]
        self._index += 1
        return value

    def assert_consumed(self) -> None:
        if self._index != len(self._values):
            raise AssertionError("Replay consumed fewer RNG values than the fixture recorded.")


def load_state(payload: dict[str, object]) -> GameState:
    from pong_engine.state import GameState

    return GameState(**payload)


def compare_states(actual: GameState, expected: GameState) -> None:
    from pong_engine.state import GameState

    for field in fields(GameState):
        actual_value = getattr(actual, field.name)
        expected_value = getattr(expected, field.name)
        if field.name in FLOAT_FIELDS:
            if not math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=1e-6):
                raise AssertionError(
                    f"State mismatch for {field.name}: expected {expected_value}, got {actual_value}"
                )
            continue
        if actual_value != expected_value:
            raise AssertionError(
                f"State mismatch for {field.name}: expected {expected_value}, got {actual_value}"
            )


def replay_fixture(path: Path) -> None:
    from pong_engine.heuristics import partially_tracking
    from pong_engine.physics import step

    fixture = json.loads(path.read_text(encoding="utf-8"))
    state = load_state(fixture["initialState"])

    for step_payload in fixture["steps"]:
        rng = ReplayRng(step_payload["randomValues"])
        left_action = partially_tracking(state, rng)
        if left_action != step_payload["leftAction"]:
            raise AssertionError(
                f"Left action mismatch at tick {step_payload['tick']}: "
                f"expected {step_payload['leftAction']}, got {left_action}"
            )
        actual_state = step(state, left_action, step_payload["rightAction"], rng)
        rng.assert_consumed()
        expected_state = load_state(step_payload["expectedState"])
        compare_states(actual_state, expected_state)
        state = actual_state


def main() -> None:
    fixture_paths = sorted(
        path
        for path in FIXTURES_DIR.glob("*.json")
        if path.name != "frame_stack_golden.json"
    )
    if not fixture_paths:
        raise SystemExit("No fixtures found. Run python scripts/gen_fixture.py first.")

    for path in fixture_paths:
        replay_fixture(path)
        print(f"OK {path.name}")


if __name__ == "__main__":
    main()
