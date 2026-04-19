// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from "vitest";

import { createModelViz, type ModelVizSnapshot } from "@/viz/model_viz";

beforeEach(() => {
  vi.restoreAllMocks();
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => createCanvasContext());
  document.body.innerHTML = "";
});

describe("model visualization", () => {
  it("loads artifacts and renders both the input frames and the selected neuron weights", async () => {
    const container = document.createElement("div");
    const controller = createModelViz(container, createFetchImpl());

    await controller.load("model-viz.json");
    controller.render({
      left: createSnapshot("left", 1),
      right: null,
    });

    expect(container.textContent).toContain("Esquerda");
    expect(container.textContent).toContain("Camada 1");
    expect(container.textContent).toContain("Neuronio 2");
    expect(container.textContent).toContain("Subir 10.0%");
    expect(container.querySelectorAll("canvas").length).toBeGreaterThanOrEqual(13);
  });

  it("degrades gracefully when the weight artifacts are unavailable", async () => {
    const container = document.createElement("div");
    const controller = createModelViz(container, createFetchImpl({ visualizationAvailable: false }));

    await controller.load("model-viz.json");
    controller.render({
      left: createSnapshot("left", 1),
      right: null,
    });

    expect(container.textContent).toContain("artefato de pesos ausente");
  });
});

function createSnapshot(side: "left" | "right", sequence: number): ModelVizSnapshot {
  const observation = new Float32Array(5 * 60 * 80);
  observation[0] = 1;
  observation[100] = 1;

  return {
    side,
    sequence,
    action: "down",
    probs: new Float32Array([0.1, 0.8, 0.1]),
    observation,
    hiddenActivations: {
      hiddenOne: new Float32Array([0.2, 0.9, 0.3]),
      hiddenTwo: new Float32Array([0.5, 0.4]),
      hiddenThree: new Float32Array([0.7]),
    },
  };
}

function createFetchImpl(options: { visualizationAvailable?: boolean } = {}): typeof fetch {
  const { visualizationAvailable = true } = options;
  const metadata = {
    frameStack: 5,
    bitmapWidth: 80,
    bitmapHeight: 60,
    firstLayerNeurons: 3,
    weightsUrl: "model-first-layer.uint8.bin",
    zeroLevel: 128,
  };
  const weights = new Uint8Array(3 * 5 * 60 * 80).fill(128);

  return vi.fn(async (input: string | URL) => {
    const url = String(input);
    if (url.endsWith("model-viz.json")) {
      return visualizationAvailable
        ? ({
            ok: true,
            json: async () => metadata,
          } as Response)
        : ({ ok: false } as Response);
    }
    if (url.endsWith("model-first-layer.uint8.bin")) {
      return {
        ok: true,
        arrayBuffer: async () =>
          weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength),
      } as Response;
    }
    return { ok: false } as Response;
  }) as typeof fetch;
}

function createCanvasContext(): CanvasRenderingContext2D {
  return {
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    fillStyle: "",
  } as unknown as CanvasRenderingContext2D;
}
