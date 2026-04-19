import { FPS } from "@/engine/config";
import type { GameState } from "@/engine/state";

export const POINT_SERVE_PAUSE_FRAMES = Math.round(FPS * 0.5);

export function didScore(previousState: GameState, nextState: GameState): boolean {
  return (
    previousState.scoreLeft !== nextState.scoreLeft ||
    previousState.scoreRight !== nextState.scoreRight
  );
}

export function tickPauseFrames(remainingFrames: number): number {
  return Math.max(remainingFrames - 1, 0);
}
