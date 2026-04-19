import fs from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { partiallyTracking } from "@/engine/heuristics";
import { step } from "@/engine/physics";
import type { GameState } from "@/engine/state";

const FIXTURES_DIR = path.resolve(import.meta.dirname, "../../shared/fixtures");
const FLOAT_KEYS = new Set(["ballX", "ballY", "ballVx", "ballVy", "ballSpeed"]);

interface FixtureStep {
  readonly tick: number;
  readonly leftAction: "up" | "down" | "none";
  readonly rightAction: "up" | "down" | "none";
  readonly randomValues: number[];
  readonly expectedState: Record<string, number>;
}

interface FixtureFile {
  readonly name: string;
  readonly seed: number;
  readonly initialState: Record<string, number>;
  readonly steps: FixtureStep[];
}

class ReplayRng {
  private index = 0;

  constructor(private readonly values: readonly number[]) {}

  next(): number {
    const value = this.values[this.index];
    if (value === undefined) {
      throw new Error("Fixture RNG values were exhausted during replay.");
    }
    this.index += 1;
    return value;
  }

  assertConsumed(): void {
    expect(this.index).toBe(this.values.length);
  }
}

function normalizeState(state: Record<string, number>): GameState {
  return {
    ballX: state.ball_x,
    ballY: state.ball_y,
    ballVx: state.ball_vx,
    ballVy: state.ball_vy,
    ballSpeed: state.ball_speed,
    paddleLeftY: state.paddle_left_y,
    paddleRightY: state.paddle_right_y,
    scoreLeft: state.score_left,
    scoreRight: state.score_right,
    tick: state.tick,
  };
}

describe("python-ts parity", () => {
  const fixtureFiles = fs
    .readdirSync(FIXTURES_DIR)
    .filter((filename) => filename !== "frame_stack_golden.json")
    .filter((filename) => filename.endsWith(".json"))
    .sort();

  it("loads the expected fixture set", () => {
    expect(fixtureFiles).toEqual([
      "long_game_accelerated.json",
      "max_speed_clamped.json",
      "short_game_low_speed.json",
    ]);
  });

  for (const fixtureFile of fixtureFiles) {
    it(`replays ${fixtureFile} with matching actions and states`, () => {
      const fixture = JSON.parse(
        fs.readFileSync(path.join(FIXTURES_DIR, fixtureFile), "utf-8"),
      ) as FixtureFile;
      let state = normalizeState(fixture.initialState);

      for (const fixtureStep of fixture.steps) {
        const rng = new ReplayRng(fixtureStep.randomValues);
        const leftAction = partiallyTracking(state, rng);
        expect(leftAction).toBe(fixtureStep.leftAction);

        state = step(state, leftAction, fixtureStep.rightAction, rng);
        rng.assertConsumed();

        const expectedState = normalizeState(fixtureStep.expectedState);
        for (const [key, value] of Object.entries(expectedState)) {
          const actualValue = state[key as keyof GameState];
          if (FLOAT_KEYS.has(key)) {
            expect(actualValue).toBeCloseTo(value, 6);
          } else {
            expect(actualValue).toBe(value);
          }
        }
      }
    });
  }
});
