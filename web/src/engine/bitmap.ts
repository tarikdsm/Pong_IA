import { ARENA_HEIGHT, ARENA_WIDTH, BALL_SIZE, PADDLE_HEIGHT, PADDLE_WIDTH } from "@/engine/config";
import type { GameState } from "@/engine/state";

export function bitmapFromState(state: GameState): Uint8Array {
  const bitmap = new Uint8Array(ARENA_WIDTH * ARENA_HEIGHT);

  fillRect(bitmap, 0, state.paddleLeftY, PADDLE_WIDTH, PADDLE_HEIGHT);
  fillRect(bitmap, ARENA_WIDTH - PADDLE_WIDTH, state.paddleRightY, PADDLE_WIDTH, PADDLE_HEIGHT);
  fillRect(bitmap, Math.trunc(state.ballX), Math.trunc(state.ballY), BALL_SIZE, BALL_SIZE);

  return bitmap;
}

export function mirrorBitmapHorizontally(bitmap: Uint8Array): Uint8Array {
  const mirrored = new Uint8Array(bitmap.length);

  for (let row = 0; row < ARENA_HEIGHT; row += 1) {
    const rowOffset = row * ARENA_WIDTH;
    for (let column = 0; column < ARENA_WIDTH; column += 1) {
      mirrored[rowOffset + column] = bitmap[rowOffset + (ARENA_WIDTH - 1 - column)];
    }
  }

  return mirrored;
}

function fillRect(bitmap: Uint8Array, x: number, y: number, width: number, height: number): void {
  for (let row = y; row < y + height; row += 1) {
    for (let column = x; column < x + width; column += 1) {
      bitmap[row * ARENA_WIDTH + column] = 1;
    }
  }
}
