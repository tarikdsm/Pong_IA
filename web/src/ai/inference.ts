import * as ort from "onnxruntime-web/wasm";
import ortWasmMjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";

import { BITMAP_HEIGHT, BITMAP_WIDTH, FRAME_STACK } from "@/engine/config";
import type { Action } from "@/engine/state";

const INPUT_DIM = FRAME_STACK * BITMAP_WIDTH * BITMAP_HEIGHT;
const ACTIONS: readonly Action[] = ["up", "down", "none"];

export interface HiddenActivations {
  readonly hiddenOne: Float32Array;
  readonly hiddenTwo: Float32Array;
  readonly hiddenThree: Float32Array;
}

export interface InferenceResult {
  readonly logits: Float32Array;
  readonly probs: Float32Array;
  readonly action: Action;
  readonly hiddenActivations: HiddenActivations;
}

export interface ModelInference {
  load(url: string): Promise<void>;
  forward(obs: Float32Array): Promise<InferenceResult>;
  isLoaded(): boolean;
}

export class ModelNotLoadedError extends Error {}

export class InvalidInferenceInputError extends Error {}

let ortConfigured = false;

export class BrowserModelInference implements ModelInference {
  private session: ort.InferenceSession | null = null;

  private loadedUrl: string | null = null;

  async load(url: string): Promise<void> {
    if (this.session && this.loadedUrl === url) {
      return;
    }

    configureOrtWasm();
    this.session = await ort.InferenceSession.create(url, {
      executionProviders: ["wasm"],
    });
    this.loadedUrl = url;
  }

  isLoaded(): boolean {
    return this.session !== null;
  }

  async forward(obs: Float32Array): Promise<InferenceResult> {
    if (!this.session) {
      throw new ModelNotLoadedError("model must be loaded before running inference.");
    }
    if (obs.length !== INPUT_DIM) {
      throw new InvalidInferenceInputError(
        `observation must have length ${INPUT_DIM}, got ${obs.length}.`,
      );
    }

    const input = new ort.Tensor("float32", obs, [1, INPUT_DIM]);
    const outputs = await this.session.run({ observation: input });
    const logits = toFloat32Array(outputs.logits?.data);
    const probs = softmax(logits);
    const action = ACTIONS[argmax(probs)];
    return {
      logits,
      probs,
      action,
      hiddenActivations: {
        hiddenOne: toOptionalFloat32Array(outputs.hidden_one?.data),
        hiddenTwo: toOptionalFloat32Array(outputs.hidden_two?.data),
        hiddenThree: toOptionalFloat32Array(outputs.hidden_three?.data),
      },
    };
  }
}

function configureOrtWasm(): void {
  if (ortConfigured) {
    return;
  }

  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = {
    mjs: ortWasmMjsUrl,
    wasm: ortWasmUrl,
  };
  ortConfigured = true;
}

function toFloat32Array(data: unknown): Float32Array {
  if (data instanceof Float32Array) {
    return data;
  }
  if (Array.isArray(data)) {
    return Float32Array.from(data);
  }
  if (ArrayBuffer.isView(data)) {
    return Float32Array.from(Array.from(data as unknown as ArrayLike<number>));
  }
  throw new InvalidInferenceInputError("inference output logits must be numeric.");
}

function toOptionalFloat32Array(data: unknown): Float32Array {
  if (data === undefined) {
    return new Float32Array(0);
  }
  return toFloat32Array(data);
}

function softmax(logits: Float32Array): Float32Array {
  let max = Number.NEGATIVE_INFINITY;
  for (const value of logits) {
    if (value > max) {
      max = value;
    }
  }

  const exps = new Float32Array(logits.length);
  let sum = 0;
  for (let index = 0; index < logits.length; index += 1) {
    const exp = Math.exp(logits[index] - max);
    exps[index] = exp;
    sum += exp;
  }

  for (let index = 0; index < exps.length; index += 1) {
    exps[index] /= sum;
  }

  return exps;
}

function argmax(values: Float32Array): number {
  let bestIndex = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[bestIndex]) {
      bestIndex = index;
    }
  }
  return bestIndex;
}
