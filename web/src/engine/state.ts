import { ARENA_HEIGHT, ARENA_WIDTH, BALL_INITIAL_SPEED, BALL_SIZE, PADDLE_HEIGHT } from "@/engine/config";
import { InvalidGameStateError } from "@/engine/errors";
import type { Rng } from "@/engine/rng";

export type Action = "up" | "down" | "none";

export interface GameState {
  readonly ballX: number;
  readonly ballY: number;
  readonly ballVx: number;
  readonly ballVy: number;
  readonly ballSpeed: number;
  readonly paddleLeftY: number;
  readonly paddleRightY: number;
  readonly scoreLeft: number;
  readonly scoreRight: number;
  readonly tick: number;
}

const LAUNCH_MIN_ANGLE_DEGREES = 12;
const LAUNCH_MAX_ANGLE_DEGREES = 45;

export function assertValidGameState(state: GameState): void {
  if (state.ballSpeed <= 0) {
    throw new InvalidGameStateError("ballSpeed must be greater than zero.");
  }
  if (state.paddleLeftY < 0) {
    throw new InvalidGameStateError("paddleLeftY must be non-negative.");
  }
  if (state.paddleRightY < 0) {
    throw new InvalidGameStateError("paddleRightY must be non-negative.");
  }
  if (state.scoreLeft < 0 || state.scoreRight < 0) {
    throw new InvalidGameStateError("scores must be non-negative.");
  }
  if (state.tick < 0) {
    throw new InvalidGameStateError("tick must be non-negative.");
  }
}

export function createInitialState(rng?: Rng | null): GameState {
  const [ballVx, ballVy] = sampleLaunchVelocity(BALL_INITIAL_SPEED, rng);
  const state: GameState = {
    ballX: (ARENA_WIDTH - BALL_SIZE) / 2,
    ballY: (ARENA_HEIGHT - BALL_SIZE) / 2,
    ballVx,
    ballVy,
    ballSpeed: BALL_INITIAL_SPEED,
    paddleLeftY: Math.floor((ARENA_HEIGHT - PADDLE_HEIGHT) / 2),
    paddleRightY: Math.floor((ARENA_HEIGHT - PADDLE_HEIGHT) / 2),
    scoreLeft: 0,
    scoreRight: 0,
    tick: 0,
  };
  assertValidGameState(state);
  return state;
}

export function sampleLaunchVelocity(speed: number, rng?: Rng | null): [number, number] {
  if (!rng) {
    return [speed, 0];
  }

  const horizontalDirection = rng.next() >= 0.5 ? 1 : -1;
  const verticalDirection = rng.next() >= 0.5 ? 1 : -1;
  const angleDegrees =
    LAUNCH_MIN_ANGLE_DEGREES + rng.next() * (LAUNCH_MAX_ANGLE_DEGREES - LAUNCH_MIN_ANGLE_DEGREES);
  const angleRadians = (angleDegrees * Math.PI) / 180;

  return [
    speed * Math.cos(angleRadians) * horizontalDirection,
    speed * Math.sin(angleRadians) * verticalDirection,
  ];
}
