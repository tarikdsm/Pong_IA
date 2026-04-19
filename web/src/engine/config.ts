// AUTO-GENERATED - do not edit manually.
// Source: shared/config.json

export const ARENA_WIDTH = 80 as const;
export const ARENA_HEIGHT = 60 as const;
export const PADDLE_WIDTH = 2 as const;
export const PADDLE_HEIGHT = 12 as const;
export const PADDLE_SPEED = 2 as const;
export const BALL_SIZE = 2 as const;
export const BALL_INITIAL_SPEED = 0.15 as const;
export const BALL_MAX_SPEED = 6.0 as const;
export const BALL_ACCELERATION_FACTOR = 1.05 as const;
export const SCORE_TO_WIN = 21 as const;
export const FPS = 60 as const;
export const BITMAP_WIDTH = 80 as const;
export const BITMAP_HEIGHT = 60 as const;
export const FRAME_STACK = 5 as const;
export const FRAME_STEP_TICKS = 5 as const;
export const INFERENCE_INTERVAL_TICKS = 30 as const;
export const HEURISTIC_INTERVAL_TICKS = 60 as const;
export const DEFAULT_SEED = 42 as const;

export const CONFIG = {
  arenaWidth: ARENA_WIDTH,
  arenaHeight: ARENA_HEIGHT,
  paddleWidth: PADDLE_WIDTH,
  paddleHeight: PADDLE_HEIGHT,
  paddleSpeed: PADDLE_SPEED,
  ballSize: BALL_SIZE,
  ballInitialSpeed: BALL_INITIAL_SPEED,
  ballMaxSpeed: BALL_MAX_SPEED,
  ballAccelerationFactor: BALL_ACCELERATION_FACTOR,
  scoreToWin: SCORE_TO_WIN,
  fps: FPS,
  bitmapWidth: BITMAP_WIDTH,
  bitmapHeight: BITMAP_HEIGHT,
  frameStack: FRAME_STACK,
  frameStepTicks: FRAME_STEP_TICKS,
  inferenceIntervalTicks: INFERENCE_INTERVAL_TICKS,
  heuristicIntervalTicks: HEURISTIC_INTERVAL_TICKS,
  defaultSeed: DEFAULT_SEED,
} as const;

export type Config = typeof CONFIG;
