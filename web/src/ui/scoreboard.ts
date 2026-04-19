import { SCORE_TO_WIN } from "@/engine/config";
import type { GameState } from "@/engine/state";
import { getModeLabel, type ControlMode } from "@/ui/mode_toggle";

export interface ScoreboardMeta {
  readonly leftMode: ControlMode;
  readonly rightMode: ControlMode;
  readonly warning: string | null;
}

export function getScoreText(state: GameState): string {
  return `${state.scoreLeft} : ${state.scoreRight}`;
}

export function getStatusText(state: GameState, meta: ScoreboardMeta): string {
  const baseStatus = `Esquerda: ${getModeLabel(meta.leftMode)} | Direita: ${getModeLabel(meta.rightMode)}`;

  if (state.scoreLeft >= SCORE_TO_WIN || state.scoreRight >= SCORE_TO_WIN) {
    return `Partida reiniciada | ${baseStatus}`;
  }
  if (meta.warning) {
    return `${baseStatus} | ${meta.warning}`;
  }
  return baseStatus;
}

export function renderScoreboard(
  scoreElement: HTMLElement,
  statusElement: HTMLElement,
  state: GameState,
  meta: ScoreboardMeta,
): void {
  scoreElement.textContent = getScoreText(state);
  statusElement.textContent = getStatusText(state, meta);
}
