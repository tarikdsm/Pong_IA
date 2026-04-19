import { BITMAP_HEIGHT, BITMAP_WIDTH, FRAME_STACK, FRAME_STEP_TICKS } from "@/engine/config";

const FRAME_LENGTH = BITMAP_WIDTH * BITMAP_HEIGHT;
const HISTORY_SIZE = 1 + (FRAME_STACK - 1) * FRAME_STEP_TICKS;

export interface FrameStackTS {
  push(frame: Uint8Array): void;
  flatten(): Float32Array;
  isReady(): boolean;
  reset(): void;
}

export class InvalidFrameStackInputError extends Error {}

export class BrowserFrameStack implements FrameStackTS {
  private readonly history = Array.from({ length: HISTORY_SIZE }, () => new Uint8Array(FRAME_LENGTH));
  private readonly flat = new Float32Array(FRAME_STACK * FRAME_LENGTH);
  private historyCount = 0;
  private nextHistoryIndex = 0;

  push(frame: Uint8Array): void {
    if (frame.length !== FRAME_LENGTH) {
      throw new InvalidFrameStackInputError(
        `frame must have length ${FRAME_LENGTH}, got ${frame.length}.`,
      );
    }

    this.history[this.nextHistoryIndex].set(frame);
    this.nextHistoryIndex = (this.nextHistoryIndex + 1) % HISTORY_SIZE;
    this.historyCount = Math.min(this.historyCount + 1, HISTORY_SIZE);
  }

  flatten(): Float32Array {
    this.flat.fill(0);

    for (let targetIndex = 0; targetIndex < FRAME_STACK; targetIndex += 1) {
      const framesBack = (FRAME_STACK - 1 - targetIndex) * FRAME_STEP_TICKS;
      if (this.historyCount <= framesBack) {
        continue;
      }

      const historyIndex = this.historyCount - 1 - framesBack;
      const frame = this.historyAt(historyIndex);
      this.flat.set(frame, targetIndex * FRAME_LENGTH);
    }

    return this.flat.slice();
  }

  isReady(): boolean {
    return this.historyCount >= HISTORY_SIZE;
  }

  reset(): void {
    this.historyCount = 0;
    this.nextHistoryIndex = 0;
    for (const frame of this.history) {
      frame.fill(0);
    }
    this.flat.fill(0);
  }

  private historyAt(logicalIndex: number): Uint8Array {
    const oldestIndex = (this.nextHistoryIndex - this.historyCount + HISTORY_SIZE) % HISTORY_SIZE;
    const physicalIndex = (oldestIndex + logicalIndex) % HISTORY_SIZE;
    return this.history[physicalIndex];
  }
}
