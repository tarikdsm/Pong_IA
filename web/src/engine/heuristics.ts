import { BALL_SIZE, PADDLE_HEIGHT } from "@/engine/config";
import { drawRandom, type Rng } from "@/engine/rng";
import type { Action, GameState } from "@/engine/state";

export function partiallyTracking(state: GameState, rng: Rng | null | undefined): Action {
  if (state.ballVx < 0) {
    const paddleCenter = state.paddleLeftY + PADDLE_HEIGHT / 2;
    const ballCenter = state.ballY + BALL_SIZE / 2;
    if (ballCenter < paddleCenter - 1) {
      return "up";
    }
    if (ballCenter > paddleCenter + 1) {
      return "down";
    }
    return "none";
  }

  const value = drawRandom(rng);
  if (value < 1 / 3) {
    return "up";
  }
  if (value < 2 / 3) {
    return "down";
  }
  return "none";
}
