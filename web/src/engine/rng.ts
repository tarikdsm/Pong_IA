import { MissingRngError } from "@/engine/errors";

export interface Rng {
  next(): number;
}

export function assertRng(rng: Rng | null | undefined): asserts rng is Rng {
  if (!rng || typeof rng.next !== "function") {
    throw new MissingRngError("rng is required for deterministic Pong engine behavior.");
  }
}

export function drawRandom(rng: Rng | null | undefined): number {
  assertRng(rng);
  return Number(rng.next());
}

export function createMulberry32(seed: number): Rng {
  let state = seed >>> 0;
  return {
    next(): number {
      state += 0x6d2b79f5;
      let value = Math.imul(state ^ (state >>> 15), 1 | state);
      value ^= value + Math.imul(value ^ (value >>> 7), 61 | value);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    },
  };
}
