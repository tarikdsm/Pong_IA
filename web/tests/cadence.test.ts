import { describe, expect, it } from "vitest";

import { FPS, HEURISTIC_INTERVAL_TICKS, INFERENCE_INTERVAL_TICKS } from "@/engine/config";

describe("runtime cadence", () => {
  it("keeps the model responding twice as fast as the heuristic", () => {
    expect(INFERENCE_INTERVAL_TICKS).toBe(30);
    expect(HEURISTIC_INTERVAL_TICKS).toBe(60);
    expect(HEURISTIC_INTERVAL_TICKS).toBe(INFERENCE_INTERVAL_TICKS * 2);
  });

  it("maps cadence values to the expected wall-clock timing", () => {
    expect((INFERENCE_INTERVAL_TICKS / FPS) * 1000).toBeCloseTo(500, 5);
    expect((HEURISTIC_INTERVAL_TICKS / FPS) * 1000).toBeCloseTo(1000, 5);
  });
});
