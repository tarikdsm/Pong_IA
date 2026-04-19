import type { HiddenActivations } from "@/ai/inference";
import { BITMAP_HEIGHT, BITMAP_WIDTH, FRAME_STACK } from "@/engine/config";
import type { Action } from "@/engine/state";

type PaddleSide = "left" | "right";
type FetchLike = (
  input: string,
  init?: RequestInit,
) => Promise<Pick<Response, "ok" | "json" | "arrayBuffer">>;

interface VisualizationMetadata {
  readonly frameStack: number;
  readonly bitmapWidth: number;
  readonly bitmapHeight: number;
  readonly firstLayerNeurons: number;
  readonly weightsUrl: string;
  readonly zeroLevel: number;
}

interface VisualizationArtifacts {
  readonly metadata: VisualizationMetadata;
  readonly weights: Uint8Array;
}

interface SidePanel {
  readonly root: HTMLElement;
  readonly status: HTMLParagraphElement;
  readonly probabilities: HTMLParagraphElement;
  readonly frameCanvases: readonly HTMLCanvasElement[];
  readonly hiddenOneCanvas: HTMLCanvasElement;
  readonly hiddenTwoCanvas: HTMLCanvasElement;
  readonly hiddenThreeCanvas: HTMLCanvasElement;
  readonly neuronLabel: HTMLParagraphElement;
  readonly weightCanvases: readonly HTMLCanvasElement[];
}

export interface ModelVizSnapshot {
  readonly side: PaddleSide;
  readonly sequence: number;
  readonly action: Action;
  readonly probs: Float32Array;
  readonly observation: Float32Array;
  readonly hiddenActivations: HiddenActivations;
}

export interface ModelVizRenderState {
  readonly left: ModelVizSnapshot | null;
  readonly right: ModelVizSnapshot | null;
}

export interface ModelVizController {
  load(metadataUrl?: string): Promise<void>;
  render(state: ModelVizRenderState): void;
}

export function createModelViz(
  container: HTMLElement,
  fetchImpl: FetchLike = fetch,
): ModelVizController {
  const root = document.createElement("section");
  root.style.display = "grid";
  root.style.gridTemplateColumns = "repeat(2, minmax(0, 1fr))";
  root.style.gap = "10px";
  root.style.width = "100%";

  const panels = {
    left: createSidePanel("Esquerda"),
    right: createSidePanel("Direita"),
  } satisfies Record<PaddleSide, SidePanel>;
  root.append(panels.left.root, panels.right.root);
  container.replaceChildren(root);

  let artifacts: VisualizationArtifacts | null = null;
  let loadWarning: string | null = null;
  let lastSignature = "";
  let lastState: ModelVizRenderState = { left: null, right: null };

  return {
    async load(metadataUrl = "model-viz.json"): Promise<void> {
      try {
        const metadataResponse = await fetchImpl(metadataUrl, {
          cache: "no-store",
          method: "GET",
        });
        if (!metadataResponse.ok) {
          throw new Error("artefato de pesos ausente");
        }

        const metadata = (await metadataResponse.json()) as VisualizationMetadata;
        const weightsResponse = await fetchImpl(metadata.weightsUrl, {
          cache: "no-store",
          method: "GET",
        });
        if (!weightsResponse.ok) {
          throw new Error("bitmap de pesos ausente");
        }

        artifacts = {
          metadata,
          weights: new Uint8Array(await weightsResponse.arrayBuffer()),
        };
        loadWarning = null;
      } catch (error) {
        artifacts = null;
        loadWarning = formatLoadWarning(error);
      }

      lastSignature = "";
      this.render(lastState);
    },
    render(state: ModelVizRenderState): void {
      lastState = state;
      const signature = [
        state.left?.sequence ?? "left-off",
        state.right?.sequence ?? "right-off",
        artifacts?.metadata.firstLayerNeurons ?? "no-weights",
        loadWarning ?? "ok",
      ].join("|");
      if (signature === lastSignature) {
        return;
      }
      lastSignature = signature;

      renderSidePanel(panels.left, state.left, artifacts, loadWarning);
      renderSidePanel(panels.right, state.right, artifacts, loadWarning);
    },
  };
}

function createSidePanel(title: string): SidePanel {
  const root = document.createElement("section");
  root.style.display = "grid";
  root.style.gap = "4px";
  root.style.padding = "6px";
  root.style.border = "1px solid #4c9f70";
  root.style.borderRadius = "8px";
  root.style.background = "#101820";
  root.style.minWidth = "0";

  const heading = document.createElement("h2");
  heading.textContent = title;
  heading.style.margin = "0";
  heading.style.fontSize = "14px";
  heading.style.textAlign = "center";

  const status = document.createElement("p");
  status.style.margin = "0";
  status.style.fontSize = "11px";
  status.style.textAlign = "center";

  const probabilities = document.createElement("p");
  probabilities.style.margin = "0";
  probabilities.style.fontSize = "11px";
  probabilities.style.textAlign = "center";

  const frameCanvases = createCanvasStrip(FRAME_STACK);
  const hiddenOneCanvas = createCanvas(200, 24);
  const hiddenTwoCanvas = createCanvas(200, 24);
  const hiddenThreeCanvas = createCanvas(100, 24);
  const neuronLabel = document.createElement("p");
  neuronLabel.style.margin = "0";
  neuronLabel.style.fontSize = "11px";
  neuronLabel.style.textAlign = "center";
  const weightCanvases = createCanvasStrip(FRAME_STACK);

  root.append(
    heading,
    status,
    probabilities,
    createLabeledStrip("Entrada", frameCanvases),
    createLabeledCanvas("Camada 1", hiddenOneCanvas),
    createLabeledCanvas("Camada 2", hiddenTwoCanvas),
    createLabeledCanvas("Camada 3", hiddenThreeCanvas),
    neuronLabel,
    createLabeledStrip("Pesos", weightCanvases),
  );

  return {
    root,
    status,
    probabilities,
    frameCanvases,
    hiddenOneCanvas,
    hiddenTwoCanvas,
    hiddenThreeCanvas,
    neuronLabel,
    weightCanvases,
  };
}

function createLabeledCanvas(labelText: string, canvas: HTMLCanvasElement): HTMLElement {
  const wrapper = document.createElement("div");
  wrapper.style.display = "grid";
  wrapper.style.gap = "2px";

  const label = document.createElement("p");
  label.textContent = labelText;
  label.style.margin = "0";
  label.style.fontSize = "11px";
  label.style.textAlign = "center";

  canvas.style.width = "100%";
  canvas.style.height = "24px";

  wrapper.append(label, canvas);
  return wrapper;
}

function createLabeledStrip(
  labelText: string,
  canvases: readonly HTMLCanvasElement[],
): HTMLElement {
  const wrapper = document.createElement("div");
  wrapper.style.display = "grid";
  wrapper.style.gap = "2px";

  const label = document.createElement("p");
  label.textContent = labelText;
  label.style.margin = "0";
  label.style.fontSize = "11px";
  label.style.textAlign = "center";

  const strip = document.createElement("div");
  strip.style.display = "grid";
  strip.style.gridTemplateColumns = "repeat(5, minmax(0, 1fr))";
  strip.style.gap = "4px";
  strip.style.minWidth = "0";

  for (const canvas of canvases) {
    canvas.style.width = "100%";
    canvas.style.height = "32px";
    strip.append(canvas);
  }

  wrapper.append(label, strip);
  return wrapper;
}

function createCanvasStrip(count: number): HTMLCanvasElement[] {
  return Array.from({ length: count }, () => createCanvas(BITMAP_WIDTH, BITMAP_HEIGHT));
}

function createCanvas(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.style.aspectRatio = `${width} / ${height}`;
  canvas.style.imageRendering = "pixelated";
  canvas.style.border = "1px solid #35513f";
  canvas.style.borderRadius = "4px";
  canvas.style.background = "#0b1116";
  return canvas;
}

function renderSidePanel(
  panel: SidePanel,
  snapshot: ModelVizSnapshot | null,
  artifacts: VisualizationArtifacts | null,
  loadWarning: string | null,
): void {
  if (!snapshot) {
    panel.status.textContent = "Ative o modo modelo nesta raquete.";
    panel.probabilities.textContent = "";
    panel.neuronLabel.textContent = loadWarning ?? "";
    clearCanvasGroup(panel.frameCanvases);
    clearCanvasGroup(panel.weightCanvases);
    clearActivationCanvas(panel.hiddenOneCanvas);
    clearActivationCanvas(panel.hiddenTwoCanvas);
    clearActivationCanvas(panel.hiddenThreeCanvas);
    return;
  }

  panel.status.textContent = `Acao: ${translateAction(snapshot.action)}`;
  panel.probabilities.textContent = [
    `Subir ${formatPercent(snapshot.probs[0])}`,
    `Descer ${formatPercent(snapshot.probs[1])}`,
    `Parado ${formatPercent(snapshot.probs[2])}`,
  ].join(" | ");

  drawObservationFrames(panel.frameCanvases, snapshot.observation);
  drawActivationCanvas(panel.hiddenOneCanvas, snapshot.hiddenActivations.hiddenOne, "#72c28d");
  drawActivationCanvas(panel.hiddenTwoCanvas, snapshot.hiddenActivations.hiddenTwo, "#72c28d");
  drawActivationCanvas(panel.hiddenThreeCanvas, snapshot.hiddenActivations.hiddenThree, "#72c28d");

  const selectedNeuron = argmax(snapshot.hiddenActivations.hiddenOne);
  if (selectedNeuron === null) {
    panel.neuronLabel.textContent = "Ativacoes indisponiveis neste modelo.";
    clearCanvasGroup(panel.weightCanvases);
    return;
  }

  panel.neuronLabel.textContent = `Neuronio ${selectedNeuron + 1}`;
  if (!artifacts) {
    clearCanvasGroup(panel.weightCanvases);
    if (loadWarning) {
      panel.neuronLabel.textContent += ` | ${loadWarning}`;
    }
    return;
  }

  drawWeightFrames(panel.weightCanvases, artifacts, selectedNeuron);
}

function drawObservationFrames(
  canvases: readonly HTMLCanvasElement[],
  observation: Float32Array,
): void {
  const frameArea = BITMAP_WIDTH * BITMAP_HEIGHT;
  for (let index = 0; index < canvases.length; index += 1) {
    const start = index * frameArea;
    const end = start + frameArea;
    drawBitmapFrame(canvases[index], observation.subarray(start, end));
  }
}

function drawBitmapFrame(canvas: HTMLCanvasElement, values: ArrayLike<number>): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let y = 0; y < canvas.height; y += 1) {
    for (let x = 0; x < canvas.width; x += 1) {
      const value = values[y * canvas.width + x] ?? 0;
      ctx.fillStyle = value > 0.5 ? "#f7f7f2" : "#0b1116";
      ctx.fillRect(x, y, 1, 1);
    }
  }
}

function drawWeightFrames(
  canvases: readonly HTMLCanvasElement[],
  artifacts: VisualizationArtifacts,
  neuronIndex: number,
): void {
  const { bitmapWidth, bitmapHeight, frameStack, firstLayerNeurons, zeroLevel } = artifacts.metadata;
  if (neuronIndex < 0 || neuronIndex >= firstLayerNeurons) {
    clearCanvasGroup(canvases);
    return;
  }

  const frameArea = bitmapWidth * bitmapHeight;
  const neuronOffset = neuronIndex * frameStack * frameArea;
  for (let frameIndex = 0; frameIndex < canvases.length; frameIndex += 1) {
    const start = neuronOffset + frameIndex * frameArea;
    const end = start + frameArea;
    drawSignedWeightFrame(
      canvases[frameIndex],
      artifacts.weights.subarray(start, end),
      zeroLevel,
    );
  }
}

function drawSignedWeightFrame(
  canvas: HTMLCanvasElement,
  values: Uint8Array,
  zeroLevel: number,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let y = 0; y < canvas.height; y += 1) {
    for (let x = 0; x < canvas.width; x += 1) {
      const value = values[y * canvas.width + x] ?? zeroLevel;
      ctx.fillStyle = toSignedWeightColor(value, zeroLevel);
      ctx.fillRect(x, y, 1, 1);
    }
  }
}

function toSignedWeightColor(value: number, zeroLevel: number): string {
  if (value >= zeroLevel) {
    const intensity = (value - zeroLevel) / Math.max(255 - zeroLevel, 1);
    return `rgb(${Math.round(44 + intensity * 36)}, ${Math.round(120 + intensity * 115)}, ${Math.round(60 + intensity * 70)})`;
  }
  const intensity = (zeroLevel - value) / Math.max(zeroLevel, 1);
  return `rgb(${Math.round(120 + intensity * 115)}, ${Math.round(44 + intensity * 30)}, ${Math.round(44 + intensity * 30)})`;
}

function drawActivationCanvas(
  canvas: HTMLCanvasElement,
  activations: Float32Array,
  fillColor: string,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (activations.length === 0) {
    return;
  }

  const maxValue = Math.max(maxAbs(activations), 1e-6);
  const barWidth = canvas.width / activations.length;
  for (let index = 0; index < activations.length; index += 1) {
    const normalized = Math.max(0, activations[index] / maxValue);
    const barHeight = Math.max(1, Math.round(normalized * canvas.height));
    ctx.fillStyle = fillColor;
    ctx.fillRect(index * barWidth, canvas.height - barHeight, Math.max(1, barWidth), barHeight);
  }
}

function clearActivationCanvas(canvas: HTMLCanvasElement): void {
  const ctx = canvas.getContext("2d");
  ctx?.clearRect(0, 0, canvas.width, canvas.height);
}

function clearCanvasGroup(canvases: readonly HTMLCanvasElement[]): void {
  for (const canvas of canvases) {
    clearActivationCanvas(canvas);
  }
}

function argmax(values: Float32Array): number | null {
  if (values.length === 0) {
    return null;
  }
  let bestIndex = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[bestIndex]) {
      bestIndex = index;
    }
  }
  return bestIndex;
}

function maxAbs(values: Float32Array): number {
  let maxValue = 0;
  for (let index = 0; index < values.length; index += 1) {
    const value = Math.abs(values[index]);
    if (value > maxValue) {
      maxValue = value;
    }
  }
  return maxValue;
}

function translateAction(action: Action): string {
  if (action === "up") {
    return "subir";
  }
  if (action === "down") {
    return "descer";
  }
  return "parado";
}

function formatPercent(value: number | undefined): string {
  return `${((value ?? 0) * 100).toFixed(1)}%`;
}

function formatLoadWarning(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return "pesos indisponiveis";
}
