import fs from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { BrowserFrameStack } from "@/ai/frame_stack";

const FIXTURE_PATH = path.resolve(import.meta.dirname, "../../shared/fixtures/frame_stack_golden.json");

interface FrameStackFixture {
  readonly stack_size: number;
  readonly frame_length: number;
  readonly frames: number[][];
  readonly expected_flat: number[];
}

describe("python-ts frame stack parity", () => {
  it("matches the golden flattened vector exactly", () => {
    const fixture = JSON.parse(fs.readFileSync(FIXTURE_PATH, "utf-8")) as FrameStackFixture;
    const stack = new BrowserFrameStack();

    for (const frame of fixture.frames) {
      stack.push(Uint8Array.from(frame));
    }

    expect(Array.from(stack.flatten())).toEqual(fixture.expected_flat);
  });
});
