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
import { InvalidActionError } from "@/engine/errors";
import { assertRng, type Rng } from "@/engine/rng";
import { sampleLaunchVelocity, type Action, type GameState } from "@/engine/state";

const VALID_ACTIONS = new Set<Action>(["up", "down", "none"]);

export function step(
  state: GameState,
  aLeft: Action,
  aRight: Action,
  rng: Rng | null | undefined,
): GameState {
  assertRng(rng);
  validateAction(aLeft);
  validateAction(aRight);

  const paddleLeftY = movePaddle(state.paddleLeftY, aLeft);
  const paddleRightY = movePaddle(state.paddleRightY, aRight);

  let nextX = state.ballX + state.ballVx;
  let nextY = state.ballY + state.ballVy;
  let nextVx = state.ballVx;
  let nextVy = state.ballVy;
  let nextSpeed = state.ballSpeed;

  if (nextY < 0) {
    nextY = -nextY;
    nextVy = Math.abs(nextVy);
  } else if (nextY + BALL_SIZE > ARENA_HEIGHT) {
    const overflow = nextY + BALL_SIZE - ARENA_HEIGHT;
    nextY = ARENA_HEIGHT - BALL_SIZE - overflow;
    nextVy = -Math.abs(nextVy);
  }

  if (nextVx < 0 && collidesWithLeftPaddle(nextX, nextY, paddleLeftY)) {
    nextX = PADDLE_WIDTH;
    nextSpeed = Math.min(state.ballSpeed * BALL_ACCELERATION_FACTOR, BALL_MAX_SPEED);
    [nextVx, nextVy] = rescaleVelocity(Math.abs(nextVx), nextVy, nextSpeed);
  } else if (nextVx > 0 && collidesWithRightPaddle(nextX, nextY, paddleRightY)) {
    nextX = ARENA_WIDTH - PADDLE_WIDTH - BALL_SIZE;
    nextSpeed = Math.min(state.ballSpeed * BALL_ACCELERATION_FACTOR, BALL_MAX_SPEED);
    const [reflectedVx, reflectedVy] = rescaleVelocity(Math.abs(nextVx), nextVy, nextSpeed);
    nextVx = -reflectedVx;
    nextVy = reflectedVy;
  }

  if (nextX < 0) {
    return createResetState(state, paddleLeftY, paddleRightY, "right", rng);
  }
  if (nextX + BALL_SIZE > ARENA_WIDTH) {
    return createResetState(state, paddleLeftY, paddleRightY, "left", rng);
  }

  return {
    ballX: nextX,
    ballY: nextY,
    ballVx: nextVx,
    ballVy: nextVy,
    ballSpeed: nextSpeed,
    paddleLeftY,
    paddleRightY,
    scoreLeft: state.scoreLeft,
    scoreRight: state.scoreRight,
    tick: state.tick + 1,
  };
}

function validateAction(action: Action): void {
  if (!VALID_ACTIONS.has(action)) {
    throw new InvalidActionError(`Invalid action: ${action}`);
  }
}

function movePaddle(currentY: number, action: Action): number {
  let delta = 0;
  if (action === "up") {
    delta = -PADDLE_SPEED;
  } else if (action === "down") {
    delta = PADDLE_SPEED;
  }
  return clamp(currentY + delta, 0, ARENA_HEIGHT - PADDLE_HEIGHT);
}

function collidesWithLeftPaddle(ballX: number, ballY: number, paddleY: number): boolean {
  return ballX <= PADDLE_WIDTH && overlapsPaddle(ballY, paddleY);
}

function collidesWithRightPaddle(ballX: number, ballY: number, paddleY: number): boolean {
  const paddleX = ARENA_WIDTH - PADDLE_WIDTH;
  return ballX + BALL_SIZE >= paddleX && overlapsPaddle(ballY, paddleY);
}

function overlapsPaddle(ballY: number, paddleY: number): boolean {
  return ballY + BALL_SIZE > paddleY && ballY < paddleY + PADDLE_HEIGHT;
}

function rescaleVelocity(vx: number, vy: number, speed: number): [number, number] {
  const magnitude = Math.hypot(vx, vy);
  if (magnitude === 0) {
    return [speed, 0];
  }
  const scale = speed / magnitude;
  return [vx * scale, vy * scale];
}

function createResetState(
  state: GameState,
  paddleLeftY: number,
  paddleRightY: number,
  scorer: "left" | "right",
  rng: Rng,
): GameState {
  const [ballVx, ballVy] = sampleLaunchVelocity(BALL_INITIAL_SPEED, rng);
  return {
    ballX: (ARENA_WIDTH - BALL_SIZE) / 2,
    ballY: (ARENA_HEIGHT - BALL_SIZE) / 2,
    ballVx,
    ballVy,
    ballSpeed: BALL_INITIAL_SPEED,
    paddleLeftY,
    paddleRightY,
    scoreLeft: state.scoreLeft + (scorer === "left" ? 1 : 0),
    scoreRight: state.scoreRight + (scorer === "right" ? 1 : 0),
    tick: state.tick + 1,
  };
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(value, maximum));
}
