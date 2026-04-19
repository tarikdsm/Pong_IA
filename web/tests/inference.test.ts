import { beforeEach, describe, expect, it, vi } from "vitest";
import * as ort from "onnxruntime-web/wasm";

import { BITMAP_HEIGHT, BITMAP_WIDTH, FRAME_STACK } from "@/engine/config";
import {
  BrowserModelInference,
  InvalidInferenceInputError,
  ModelNotLoadedError,
} from "@/ai/inference";

const ACTIONS = ["up", "down", "none"] as const;
const INPUT_DIM = FRAME_STACK * BITMAP_WIDTH * BITMAP_HEIGHT;
const { createMock, runMock } = vi.hoisted(() => ({
  createMock: vi.fn(),
  runMock: vi.fn(),
}));

vi.mock("onnxruntime-web/ort-wasm-simd-threaded.mjs?url", () => ({
  default: "/mock/ort-wasm-simd-threaded.mjs",
}));

vi.mock("onnxruntime-web/ort-wasm-simd-threaded.wasm?url", () => ({
  default: "/mock/ort-wasm-simd-threaded.wasm",
}));

vi.mock("onnxruntime-web/wasm", () => {
  class Tensor {
    constructor(
      public readonly type: string,
      public readonly data: Float32Array,
      public readonly dims: number[],
    ) {}
  }

  return {
    env: {
      wasm: {},
    },
    InferenceSession: {
      create: createMock,
    },
    Tensor,
  };
});

describe("browser inference", () => {
  beforeEach(() => {
    createMock.mockReset();
    runMock.mockReset();
    createMock.mockResolvedValue({
      run: runMock,
    });
    runMock.mockResolvedValue({
      logits: {
        data: Float32Array.from([1, 3, 2]),
      },
      hidden_one: {
        data: Float32Array.from([0.2, 0.7, 0.1]),
      },
      hidden_two: {
        data: Float32Array.from([0.5, 0.4]),
      },
      hidden_three: {
        data: Float32Array.from([0.9]),
      },
    });
  });

  it("loads a model and runs forward to logits, probabilities and action", async () => {
    const inference = new BrowserModelInference();
    await inference.load("model.onnx");

    const result = await inference.forward(new Float32Array(INPUT_DIM));

    expect(ort.env.wasm.numThreads).toBe(1);
    expect(ort.env.wasm.proxy).toBe(false);
    expect(ort.env.wasm.wasmPaths).toEqual({
      mjs: "/mock/ort-wasm-simd-threaded.mjs",
      wasm: "/mock/ort-wasm-simd-threaded.wasm",
    });
    expect(createMock).toHaveBeenCalledWith(
      "model.onnx",
      expect.objectContaining({ executionProviders: ["wasm"] }),
    );
    expect(runMock).toHaveBeenCalledTimes(1);
    expect(result.logits).toEqual(Float32Array.from([1, 3, 2]));
    expect(result.probs).toHaveLength(ACTIONS.length);
    expect(result.action).toBe("down");
    expect(result.hiddenActivations.hiddenOne).toEqual(Float32Array.from([0.2, 0.7, 0.1]));
    expect(result.hiddenActivations.hiddenTwo).toEqual(Float32Array.from([0.5, 0.4]));
    expect(result.hiddenActivations.hiddenThree).toEqual(Float32Array.from([0.9]));
    expect(result.probs[0] + result.probs[1] + result.probs[2]).toBeCloseTo(1, 6);
  });

  it("fails clearly when forward is called before load", async () => {
    const inference = new BrowserModelInference();

    await expect(inference.forward(new Float32Array(INPUT_DIM))).rejects.toThrow(ModelNotLoadedError);
  });

  it("fails clearly when observation length is invalid", async () => {
    const inference = new BrowserModelInference();
    await inference.load("model.onnx");

    await expect(inference.forward(new Float32Array(INPUT_DIM - 1))).rejects.toThrow(
      InvalidInferenceInputError,
    );
  });
});
