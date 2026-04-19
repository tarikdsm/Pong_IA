import { describe, expect, it } from "vitest";

import { createInitialState } from "@/engine/state";
import { didScore, POINT_SERVE_PAUSE_FRAMES, tickPauseFrames } from "@/ui/round-flow";

describe("round flow", () => {
  it("detects when a point changes the score", () => {
    const previousState = createInitialState();
    const nextState = { ...previousState, scoreRight: previousState.scoreRight + 1 };

    expect(didScore(previousState, nextState)).toBe(true);
  });

  it("does not report score when only movement changes", () => {
    const previousState = createInitialState();
    const nextState = { ...previousState, ballX: previousState.ballX + 1.5 };

    expect(didScore(previousState, nextState)).toBe(false);
  });

  it("counts pause frames down to zero", () => {
    expect(POINT_SERVE_PAUSE_FRAMES).toBeGreaterThan(0);
    expect(tickPauseFrames(2)).toBe(1);
    expect(tickPauseFrames(1)).toBe(0);
    expect(tickPauseFrames(0)).toBe(0);
  });
});
