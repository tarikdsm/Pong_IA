// @vitest-environment jsdom

import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import type { ModelInference } from "@/ai/inference";
import { mirrorBitmapHorizontally } from "@/engine/bitmap";
import type { Rng } from "@/engine/rng";
import type { FrameStackTS } from "@/ai/frame_stack";

interface RafHarness {
  readonly requestAnimationFrame: (callback: FrameRequestCallback) => number;
  readonly cancelAnimationFrame: (handle: number) => void;
  flush(frames: number): Promise<void>;
}

interface AppHandle {
  dispose(): void;
  getMode(): string;
  getModes(): { left: string; right: string };
  readonly elements: {
    readonly viewport: HTMLElement;
    readonly layout: HTMLElement;
    readonly status: HTMLElement;
  };
}

let startApp: (options?: object) => Promise<AppHandle>;

beforeAll(async () => {
  globalThis.__PONG_IA_DISABLE_AUTO_START__ = true;
  ({ startApp } = await import("@/main"));
});

beforeEach(() => {
  vi.restoreAllMocks();
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => createCanvasContext());
  document.body.innerHTML = "";
});

afterEach(() => {
  document.body.innerHTML = "";
});

describe("startApp", () => {
  it("runs 100 ticks in model mode without errors and uses inference output", async () => {
    const raf = createRafHarness();
    const inference = createInference();
    const rng = createRng(0.02);

    const app = await startApp({
      fetchImpl: createFetchImpl(),
      inference,
      requestAnimationFrameImpl: raf.requestAnimationFrame,
      cancelAnimationFrameImpl: raf.cancelAnimationFrame,
      rng,
    });

    await raf.flush(100);

    expect(app.getMode()).toBe("model");
    expect(app.getModes()).toEqual({ left: "heuristic", right: "model" });
    expect(String(inference.load.mock.calls[0]?.[0] ?? "")).toMatch(/model\.onnx$/);
    expect(inference.forward).toHaveBeenCalled();
    expect(inference.forward.mock.calls.length).toBeGreaterThanOrEqual(2);
    const firstObservation = inference.forward.mock.calls[0]?.[0] as Float32Array;
    expect(firstObservation).toBeInstanceOf(Float32Array);
    expect(firstObservation).toHaveLength(5 * 60 * 80);
    expect(app.elements.status.textContent).toContain("Esquerda: heuristica");
    expect(app.elements.status.textContent).toContain("Direita: modelo");
    expect(app.elements.viewport.style.overflow).toBe("hidden");
    expect(app.elements.layout.style.transform).toContain("scale(");
    expect(document.body.textContent ?? "").not.toContain("IA direita");
    expect(document.body.textContent ?? "").not.toContain("argmax=");
    expect(document.body.textContent ?? "").toContain("Camada 1");
    expect(document.body.textContent ?? "").toContain("Pesos");

    app.dispose();
  });

  it("falls back to keyboard when model.onnx is absent", async () => {
    const raf = createRafHarness();
    const inference = createInference();

    const app = await startApp({
      fetchImpl: createFetchImpl({ modelAvailable: false }),
      inference,
      requestAnimationFrameImpl: raf.requestAnimationFrame,
      cancelAnimationFrameImpl: raf.cancelAnimationFrame,
    });

    await raf.flush(40);

    expect(app.getMode()).toBe("keyboard");
    expect(app.getModes()).toEqual({ left: "heuristic", right: "keyboard" });
    expect(inference.load).not.toHaveBeenCalled();
    expect(app.elements.status.textContent).toContain("teclado ativado");

    app.dispose();
  });

  it("allows selecting model for the left paddle and mirrors its bitmap before inference", async () => {
    const raf = createRafHarness();
    const inference = createInference();
    const leftFrameStack = createInspectableFrameStack();
    const rightFrameStack = createInspectableFrameStack();

    const app = await startApp({
      fetchImpl: createFetchImpl(),
      inference,
      leftFrameStack,
      rightFrameStack,
      requestAnimationFrameImpl: raf.requestAnimationFrame,
      cancelAnimationFrameImpl: raf.cancelAnimationFrame,
    });

    const selects = Array.from(document.querySelectorAll("select"));
    const leftSelect = selects[0] as HTMLSelectElement | undefined;
    const rightSelect = selects[1] as HTMLSelectElement | undefined;
    expect(leftSelect).toBeDefined();
    expect(rightSelect).toBeDefined();
    expect(leftSelect?.parentElement?.parentElement).toBe(rightSelect?.parentElement?.parentElement);

    rightSelect!.value = "keyboard";
    rightSelect!.dispatchEvent(new Event("change"));
    await Promise.resolve();
    rightSelect!.value = "model";
    rightSelect!.dispatchEvent(new Event("change"));
    leftSelect!.value = "model";
    leftSelect!.dispatchEvent(new Event("change"));

    await Promise.resolve();
    await raf.flush(40);

    expect(app.getModes()).toEqual({ left: "model", right: "model" });
    expect(app.elements.status.textContent).toContain("Esquerda: modelo");
    expect(app.elements.status.textContent).toContain("Direita: modelo");
    expect(leftFrameStack.pushedFrames.length).toBeGreaterThan(0);
    expect(rightFrameStack.pushedFrames.length).toBeGreaterThan(0);
    expect(leftFrameStack.pushedFrames[0]).toEqual(
      mirrorBitmapHorizontally(rightFrameStack.pushedFrames[0]),
    );

    app.dispose();
  });
});

function createInference(): ModelInference & {
  load: ReturnType<typeof vi.fn>;
  forward: ReturnType<typeof vi.fn>;
} {
  return {
    load: vi.fn().mockResolvedValue(undefined),
    forward: vi.fn().mockResolvedValue({
      logits: new Float32Array([0.1, 2.5, -0.5]),
      probs: new Float32Array([0.08, 0.86, 0.06]),
      action: "down",
      hiddenActivations: {
        hiddenOne: new Float32Array([0.1, 0.9, 0.3]),
        hiddenTwo: new Float32Array([0.4, 0.2]),
        hiddenThree: new Float32Array([0.7]),
      },
    }),
    isLoaded: vi.fn().mockReturnValue(true),
  };
}

function createFetchImpl(options: {
  modelAvailable?: boolean;
  visualizationAvailable?: boolean;
} = {}): typeof fetch {
  const {
    modelAvailable = true,
    visualizationAvailable = true,
  } = options;
  const metadata = {
    frameStack: 5,
    bitmapWidth: 80,
    bitmapHeight: 60,
    firstLayerNeurons: 3,
    weightsUrl: "model-first-layer.uint8.bin",
    zeroLevel: 128,
  };
  const weights = new Uint8Array(3 * 5 * 60 * 80).fill(128);

  return vi.fn(async (input: string | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith("model.onnx")) {
      return { ok: modelAvailable } as Response;
    }
    if (url.endsWith("model-viz.json")) {
      return visualizationAvailable
        ? ({
            ok: true,
            json: async () => metadata,
          } as Response)
        : ({ ok: false } as Response);
    }
    if (url.endsWith("model-first-layer.uint8.bin")) {
      return visualizationAvailable
        ? ({
            ok: true,
            arrayBuffer: async () =>
              weights.buffer.slice(
                weights.byteOffset,
                weights.byteOffset + weights.byteLength,
              ),
          } as Response)
        : ({ ok: false } as Response);
    }
    if (init?.method === "HEAD") {
      return { ok: modelAvailable } as Response;
    }
    return { ok: false } as Response;
  }) as typeof fetch;
}

function createInspectableFrameStack(): FrameStackTS & {
  pushedFrames: Uint8Array[];
} {
  const pushedFrames: Uint8Array[] = [];

  return {
    pushedFrames,
    push(frame: Uint8Array): void {
      pushedFrames.push(frame.slice());
    },
    flatten(): Float32Array {
      return new Float32Array(5 * 60 * 80);
    },
    isReady(): boolean {
      return false;
    },
    reset(): void {
      pushedFrames.length = 0;
    },
  };
}

function createRng(value: number): Rng {
  return {
    next(): number {
      return value;
    },
  };
}

function createRafHarness(): RafHarness {
  let nextHandle = 1;
  let now = 0;
  const queue = new Map<number, FrameRequestCallback>();

  return {
    requestAnimationFrame(callback): number {
      const handle = nextHandle;
      nextHandle += 1;
      queue.set(handle, callback);
      return handle;
    },
    cancelAnimationFrame(handle): void {
      queue.delete(handle);
    },
    async flush(frames: number): Promise<void> {
      for (let frame = 0; frame < frames; frame += 1) {
        const callbacks = Array.from(queue.values());
        queue.clear();
        now += 1000 / 60;
        for (const callback of callbacks) {
          callback(now);
        }
        await Promise.resolve();
        await Promise.resolve();
      }
    },
  };
}

function createCanvasContext(): CanvasRenderingContext2D {
  return {
    beginPath: vi.fn(),
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    lineTo: vi.fn(),
    moveTo: vi.fn(),
    setLineDash: vi.fn(),
    stroke: vi.fn(),
    fillStyle: "",
    lineWidth: 1,
    strokeStyle: "",
  } as unknown as CanvasRenderingContext2D;
}
