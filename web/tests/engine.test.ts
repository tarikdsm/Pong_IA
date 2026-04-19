import { describe, expect, it } from "vitest";

import {
  ARENA_HEIGHT,
  ARENA_WIDTH,
  BALL_ACCELERATION_FACTOR,
  BALL_INITIAL_SPEED,
  BALL_MAX_SPEED,
  BALL_SIZE,
  PADDLE_HEIGHT,
  PADDLE_SPEED,
  PADDLE_WIDTH,
} from "@/engine/config";
import { partiallyTracking } from "@/engine/heuristics";
import { bitmapFromState, mirrorBitmapHorizontally } from "@/engine/bitmap";
import { step } from "@/engine/physics";
import { createInitialState, type GameState } from "@/engine/state";
import { MissingRngError } from "@/engine/errors";

class StubRng {
  private readonly values: number[];

  private index = 0;

  constructor(...values: number[]) {
    this.values = values;
  }

  next(): number {
    const value = this.values[this.index] ?? this.values[this.values.length - 1] ?? 0;
    this.index += 1;
    return value;
  }
}

function withUpdates(state: GameState, updates: Partial<GameState>): GameState {
  return { ...state, ...updates };
}

describe("engine state", () => {
  it("creates the canonical initial state", () => {
    expect(createInitialState()).toEqual({
      ballX: 39,
      ballY: 29,
      ballVx: BALL_INITIAL_SPEED,
      ballVy: 0,
      ballSpeed: BALL_INITIAL_SPEED,
      paddleLeftY: 24,
      paddleRightY: 24,
      scoreLeft: 0,
      scoreRight: 0,
      tick: 0,
    });
  });

  it("creates a random launch vector when rng is provided", () => {
    const state = createInitialState(new StubRng(0.7739560485559633, 0.4388784397520523, 0.8585979199113825));

    expect(state.ballX).toBe(39);
    expect(state.ballY).toBe(29);
    expect(state.ballSpeed).toBeCloseTo(BALL_INITIAL_SPEED);
    expect(Math.abs(state.ballVx)).toBeLessThan(BALL_INITIAL_SPEED);
    expect(state.ballVy).not.toBeCloseTo(0);
    expect(state.ballVx ** 2 + state.ballVy ** 2).toBeCloseTo(BALL_INITIAL_SPEED ** 2);
  });
});

describe("physics", () => {
  it("moves the ball and paddles without collisions", () => {
    const nextState = step(createInitialState(), "up", "down", new StubRng(0.9));

    expect(nextState.ballX).toBeCloseTo(39 + BALL_INITIAL_SPEED);
    expect(nextState.ballY).toBeCloseTo(29);
    expect(nextState.paddleLeftY).toBe(24 - PADDLE_SPEED);
    expect(nextState.paddleRightY).toBe(24 + PADDLE_SPEED);
    expect(nextState.tick).toBe(1);
  });

  it("keeps diagonal motion active when no collision occurs", () => {
    const state = withUpdates(createInitialState(), {
      ballX: 20,
      ballY: 20,
      ballVx: 1.1,
      ballVy: 0.7,
      ballSpeed: Math.hypot(1.1, 0.7),
    });

    const nextState = step(state, "none", "none", new StubRng(0.9));

    expect(nextState.ballX).toBeCloseTo(21.1);
    expect(nextState.ballY).toBeCloseTo(20.7);
    expect(nextState.ballVx).toBeCloseTo(1.1);
    expect(nextState.ballVy).toBeCloseTo(0.7);
  });

  it("reflects on the top wall before paddle logic", () => {
    const state = withUpdates(createInitialState(), {
      ballX: PADDLE_WIDTH + 4,
      ballY: 0.5,
      ballVx: -BALL_INITIAL_SPEED,
      ballVy: -1,
      ballSpeed: Math.hypot(BALL_INITIAL_SPEED, 1),
      paddleLeftY: 0,
    });

    const nextState = step(state, "none", "none", new StubRng(0.9));

    expect(nextState.ballVy).toBeCloseTo(1);
    expect(nextState.ballY).toBeCloseTo(0.5);
    expect(nextState.ballVx).toBeCloseTo(-BALL_INITIAL_SPEED);
  });

  it("reflects on the right paddle and accelerates the ball", () => {
    const state = withUpdates(createInitialState(), {
      ballX: ARENA_WIDTH - PADDLE_WIDTH - BALL_SIZE - 0.25,
      ballY: 28,
      ballVx: 1.5,
      ballVy: 0,
      ballSpeed: 1.5,
      paddleRightY: 24,
    });

    const nextState = step(state, "none", "none", new StubRng(0.9));

    expect(nextState.ballVx).toBeLessThan(0);
    expect(nextState.ballSpeed).toBeCloseTo(
      Math.min(1.5 * BALL_ACCELERATION_FACTOR, BALL_MAX_SPEED),
    );
    expect(Math.abs(nextState.ballVx)).toBeCloseTo(nextState.ballSpeed);
  });

  it("preserves diagonal motion on paddle collision", () => {
    const state = withUpdates(createInitialState(), {
      ballX: ARENA_WIDTH - PADDLE_WIDTH - BALL_SIZE - 0.25,
      ballY: 28,
      ballVx: 1.2,
      ballVy: 0.6,
      ballSpeed: Math.hypot(1.2, 0.6),
      paddleRightY: 24,
    });

    const nextState = step(state, "none", "none", new StubRng(0.9));

    expect(nextState.ballVx).toBeLessThan(0);
    expect(nextState.ballVy).toBeGreaterThan(0);
    expect(nextState.ballSpeed).toBeCloseTo(
      Math.min(state.ballSpeed * BALL_ACCELERATION_FACTOR, BALL_MAX_SPEED),
    );
    expect(nextState.ballVx ** 2 + nextState.ballVy ** 2).toBeCloseTo(
      nextState.ballSpeed ** 2,
    );
  });

  it("clamps ball speed when max speed is reached", () => {
    const state = withUpdates(createInitialState(), {
      ballX: PADDLE_WIDTH + 0.25,
      ballY: 30,
      ballVx: -BALL_MAX_SPEED,
      ballVy: 0,
      ballSpeed: BALL_MAX_SPEED,
      paddleLeftY: 24,
    });

    const nextState = step(state, "none", "none", new StubRng(0.9));

    expect(nextState.ballSpeed).toBeCloseTo(BALL_MAX_SPEED);
    expect(nextState.ballVx).toBeCloseTo(BALL_MAX_SPEED);
  });

  it("awards the right score and resets the ball on left goal", () => {
    const state = withUpdates(createInitialState(), {
      ballX: 0.25,
      ballY: 18,
      ballVx: -1.5,
      ballVy: 0,
      ballSpeed: 1.5,
      paddleLeftY: ARENA_HEIGHT - PADDLE_HEIGHT,
    });

    const nextState = step(state, "none", "none", new StubRng(0.7));

    expect(nextState.scoreRight).toBe(1);
    expect(nextState.scoreLeft).toBe(0);
    expect(nextState.ballX).toBeCloseTo((ARENA_WIDTH - BALL_SIZE) / 2);
    expect(nextState.ballY).toBeCloseTo((ARENA_HEIGHT - BALL_SIZE) / 2);
    expect(nextState.ballSpeed).toBeCloseTo(BALL_INITIAL_SPEED);
    expect(Math.abs(nextState.ballVx)).toBeLessThan(BALL_INITIAL_SPEED);
    expect(nextState.ballVy).not.toBeCloseTo(0);
    expect(nextState.ballVx ** 2 + nextState.ballVy ** 2).toBeCloseTo(BALL_INITIAL_SPEED ** 2);
  });

  it("resets an accelerated ball to minimum speed when right goal occurs", () => {
    const state = withUpdates(createInitialState(), {
      ballX: ARENA_WIDTH - BALL_SIZE - 0.25,
      ballY: 22,
      ballVx: BALL_MAX_SPEED,
      ballVy: 0.4,
      ballSpeed: BALL_MAX_SPEED,
      paddleRightY: 0,
    });

    const nextState = step(state, "none", "none", new StubRng(0.7, 0.2, 0.5));

    expect(nextState.scoreLeft).toBe(1);
    expect(nextState.scoreRight).toBe(0);
    expect(nextState.ballX).toBeCloseTo((ARENA_WIDTH - BALL_SIZE) / 2);
    expect(nextState.ballY).toBeCloseTo((ARENA_HEIGHT - BALL_SIZE) / 2);
    expect(nextState.ballSpeed).toBeCloseTo(BALL_INITIAL_SPEED);
    expect(nextState.ballVx ** 2 + nextState.ballVy ** 2).toBeCloseTo(BALL_INITIAL_SPEED ** 2);
  });

  it("fails clearly when rng is missing", () => {
    expect(() => step(createInitialState(), "none", "none", null)).toThrow(MissingRngError);
  });
});

describe("heuristics", () => {
  it("tracks the ball when it moves toward the left paddle", () => {
    const action = partiallyTracking(
      withUpdates(createInitialState(), {
        ballY: 5,
        ballVx: -1.5,
        paddleLeftY: 24,
      }),
      new StubRng(0.5),
    );

    expect(action).toBe("up");
  });

  it("uses rng when the ball moves away from the left paddle", () => {
    const action = partiallyTracking(
      withUpdates(createInitialState(), {
        ballY: 40,
        ballVx: 1.5,
        paddleLeftY: 24,
      }),
      new StubRng(0.7739560485559633),
    );

    expect(action).toBe("none");
  });
});

describe("bitmap", () => {
  it("returns a binary flattened bitmap with paddles and ball", () => {
    const bitmap = bitmapFromState(createInitialState());

    expect(bitmap).toHaveLength(ARENA_WIDTH * ARENA_HEIGHT);
    expect(new Set(bitmap)).toEqual(new Set([0, 1]));
    expect(bitmap[24 * ARENA_WIDTH]).toBe(1);
    expect(bitmap[(24 + PADDLE_HEIGHT - 1) * ARENA_WIDTH + (PADDLE_WIDTH - 1)]).toBe(1);
    expect(bitmap[29 * ARENA_WIDTH + 39]).toBe(1);
  });

  it("mirrors the bitmap horizontally for left-side model inference", () => {
    const state = withUpdates(createInitialState(), {
      ballX: 12,
      ballY: 10,
      paddleLeftY: 8,
      paddleRightY: 32,
    });

    const mirrored = mirrorBitmapHorizontally(bitmapFromState(state));

    expect(mirrored[8 * ARENA_WIDTH + (ARENA_WIDTH - PADDLE_WIDTH)]).toBe(1);
    expect(mirrored[32 * ARENA_WIDTH]).toBe(1);
    expect(mirrored[10 * ARENA_WIDTH + (ARENA_WIDTH - 1 - 12)]).toBe(1);
  });
});
