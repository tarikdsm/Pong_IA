import { describe, expect, it } from "vitest";

import { BITMAP_HEIGHT, BITMAP_WIDTH, FRAME_STACK, FRAME_STEP_TICKS } from "@/engine/config";
import { BrowserFrameStack, InvalidFrameStackInputError } from "@/ai/frame_stack";

const FRAME_LENGTH = BITMAP_WIDTH * BITMAP_HEIGHT;
const HISTORY_SIZE = 1 + (FRAME_STACK - 1) * FRAME_STEP_TICKS;

function makeFrame(index: number): Uint8Array {
  const frame = new Uint8Array(FRAME_LENGTH);
  frame[index] = 1;
  return frame;
}

describe("browser frame stack", () => {
  it("zero-fills missing initial frames and flattens to 24000", () => {
    const stack = new BrowserFrameStack();

    stack.push(makeFrame(3));

    const flat = stack.flatten();

    expect(flat).toBeInstanceOf(Float32Array);
    expect(flat).toHaveLength(FRAME_STACK * FRAME_LENGTH);
    expect(flat[3]).toBe(0);
    expect(flat[(FRAME_STACK - 1) * FRAME_LENGTH + 3]).toBe(1);
  });

  it("reports readiness only after receiving the full temporal window", () => {
    const stack = new BrowserFrameStack();

    for (let index = 0; index < HISTORY_SIZE - 1; index += 1) {
      stack.push(makeFrame(index));
      expect(stack.isReady()).toBe(false);
    }

    stack.push(makeFrame(HISTORY_SIZE - 1));

    expect(stack.isReady()).toBe(true);
  });

  it("samples frames with a temporal stride of five ticks", () => {
    const stack = new BrowserFrameStack();

    for (let index = 0; index < HISTORY_SIZE; index += 1) {
      stack.push(makeFrame(index));
    }

    const flat = stack.flatten();

    expect(flat[0]).toBe(1);
    expect(flat[FRAME_LENGTH + FRAME_STEP_TICKS]).toBe(1);
    expect(flat[2 * FRAME_LENGTH + FRAME_STEP_TICKS * 2]).toBe(1);
    expect(flat[3 * FRAME_LENGTH + FRAME_STEP_TICKS * 3]).toBe(1);
    expect(flat[4 * FRAME_LENGTH + FRAME_STEP_TICKS * 4]).toBe(1);
  });

  it("resets to an empty deterministic state", () => {
    const stack = new BrowserFrameStack();
    stack.push(makeFrame(1));
    stack.push(makeFrame(2));

    stack.reset();

    expect(stack.isReady()).toBe(false);
    expect(Array.from(stack.flatten())).toEqual(new Array(FRAME_STACK * FRAME_LENGTH).fill(0));
  });

  it("fails clearly when frame length is invalid", () => {
    const stack = new BrowserFrameStack();

    expect(() => stack.push(new Uint8Array(FRAME_LENGTH - 1))).toThrow(InvalidFrameStackInputError);
  });
});
