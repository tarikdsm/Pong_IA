import { ARENA_HEIGHT, ARENA_WIDTH, BALL_SIZE, PADDLE_HEIGHT, PADDLE_WIDTH } from "@/engine/config";
import type { GameState } from "@/engine/state";

export function drawGame(ctx: CanvasRenderingContext2D, state: GameState): void {
  ctx.fillStyle = "#101820";
  ctx.fillRect(0, 0, ARENA_WIDTH, ARENA_HEIGHT);

  ctx.strokeStyle = "#4c9f70";
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 2]);
  ctx.beginPath();
  ctx.moveTo(ARENA_WIDTH / 2, 0);
  ctx.lineTo(ARENA_WIDTH / 2, ARENA_HEIGHT);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#f2c14e";
  ctx.fillRect(0, state.paddleLeftY, PADDLE_WIDTH, PADDLE_HEIGHT);
  ctx.fillRect(ARENA_WIDTH - PADDLE_WIDTH, state.paddleRightY, PADDLE_WIDTH, PADDLE_HEIGHT);

  ctx.fillStyle = "#e8505b";
  ctx.fillRect(state.ballX, state.ballY, BALL_SIZE, BALL_SIZE);
}
