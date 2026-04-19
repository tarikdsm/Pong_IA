import { BrowserFrameStack, type FrameStackTS } from "@/ai/frame_stack";
import { BrowserModelInference, type ModelInference } from "@/ai/inference";
import {
  INFERENCE_INTERVAL_TICKS,
  HEURISTIC_INTERVAL_TICKS,
  ARENA_HEIGHT,
  ARENA_WIDTH,
  BALL_SIZE,
  DEFAULT_SEED,
  FPS,
  PADDLE_HEIGHT,
  SCORE_TO_WIN,
} from "@/engine/config";
import { bitmapFromState, mirrorBitmapHorizontally } from "@/engine/bitmap";
import { partiallyTracking } from "@/engine/heuristics";
import type { Rng } from "@/engine/rng";
import { createInitialState } from "@/engine/state";
import type { Action, GameState } from "@/engine/state";
import { step } from "@/engine/physics";
import { drawGame } from "@/engine/rendering";
import { createMulberry32 } from "@/engine/rng";
import { createControls } from "@/ui/controls";
import type { Controls } from "@/ui/controls";
import { createModeToggle, resolveInitialMode, type ControlMode } from "@/ui/mode_toggle";
import { didScore, POINT_SERVE_PAUSE_FRAMES, tickPauseFrames } from "@/ui/round-flow";
import { renderScoreboard } from "@/ui/scoreboard";
import { createModelViz, type ModelVizSnapshot } from "@/viz/model_viz";
import type { HiddenActivations } from "@/ai/inference";

const MODEL_URL = "model.onnx";
const MODEL_VIZ_URL = "model-viz.json";
const BASE_LAYOUT_WIDTH = 1180;
const BASE_LAYOUT_HEIGHT = 980;
const VIEWPORT_PADDING = 24;

declare global {
  var __PONG_IA_DISABLE_AUTO_START__: boolean | undefined;
}

interface AppElements {
  readonly app: HTMLElement;
  readonly viewport: HTMLDivElement;
  readonly layout: HTMLElement;
  readonly controlsRow: HTMLDivElement;
  readonly score: HTMLDivElement;
  readonly status: HTMLDivElement;
  readonly canvas: HTMLCanvasElement;
  readonly visualization: HTMLDivElement;
}

interface WindowTargetLike {
  addEventListener(type: "keydown" | "keyup", listener: (event: KeyboardEvent) => void): void;
  removeEventListener(type: "keydown" | "keyup", listener: (event: KeyboardEvent) => void): void;
}

interface StartAppOptions {
  readonly document?: Document;
  readonly windowTarget?: WindowTargetLike;
  readonly requestAnimationFrameImpl?: (callback: FrameRequestCallback) => number;
  readonly cancelAnimationFrameImpl?: (handle: number) => void;
  readonly fetchImpl?: typeof fetch;
  readonly controls?: Controls;
  readonly inference?: ModelInference;
  readonly frameStack?: FrameStackTS;
  readonly leftFrameStack?: FrameStackTS;
  readonly rightFrameStack?: FrameStackTS;
  readonly modelUrl?: string;
  readonly rng?: Rng;
}

export interface GameApp {
  dispose(): void;
  getMode(): ControlMode;
  getModes(): { left: ControlMode; right: ControlMode };
  getState(): GameState;
  readonly elements: AppElements;
}

type PaddleSide = "left" | "right";

interface SideController {
  mode: ControlMode;
  lastModelAction: Action;
  lastHeuristicAction: Action;
  inferencePending: boolean;
  inferenceToken: number;
  inferenceSequence: number;
  lastVizSnapshot: ModelVizSnapshot | null;
  readonly frameStack: FrameStackTS;
}

export async function startApp(options: StartAppOptions = {}): Promise<GameApp> {
  const doc = options.document ?? document;
  const win = options.windowTarget ?? window;
  const requestFrame = options.requestAnimationFrameImpl ?? requestAnimationFrame;
  const cancelFrame = options.cancelAnimationFrameImpl ?? cancelAnimationFrame;
  const fetchImpl = options.fetchImpl ?? fetch;
  const controls = options.controls ?? createControls(win);
  const inference = options.inference ?? new BrowserModelInference();
  const fallbackRightFrameStack = options.rightFrameStack ?? options.frameStack;
  const modelUrl = options.modelUrl ?? MODEL_URL;
  const rng = options.rng ?? createMulberry32(DEFAULT_SEED);
  const layoutWindow = doc.defaultView;

  const elements = createAppElements(doc);
  const modelViz = createModelViz(elements.visualization, fetchImpl);
  doc.body.style.margin = "0";
  doc.body.style.overflowY = "hidden";
  doc.documentElement.style.overflowY = "hidden";
  doc.body.append(elements.app);
  const fitViewport = (): void => fitLayoutToViewport(elements, layoutWindow);
  fitViewport();
  layoutWindow?.addEventListener("resize", fitViewport);
  void modelViz.load(MODEL_VIZ_URL).finally(fitViewport);

  const context = elements.canvas.getContext("2d");
  if (!context) {
    throw new Error("Canvas 2D context is unavailable.");
  }
  const ctx = context;

  let state = createInitialState(rng);
  let servePauseFrames = POINT_SERVE_PAUSE_FRAMES;
  let warning: string | null = null;
  let animationFrameHandle: number | null = null;
  let disposed = false;
  let frames = 0;
  let lastFpsSample = 0;

  const controllers: Record<PaddleSide, SideController> = {
    left: createSideController(options.leftFrameStack ?? new BrowserFrameStack(), "heuristic"),
    right: createSideController(fallbackRightFrameStack ?? new BrowserFrameStack(), "keyboard"),
  };

  const modeToggles = {
    left: createModeToggle({
      labelText: "Controle esquerda",
      initialMode: controllers.left.mode,
      onChange(mode): void {
        void setMode("left", mode);
      },
    }),
    right: createModeToggle({
      labelText: "Controle direita",
      initialMode: controllers.right.mode,
      onChange(mode): void {
        void setMode("right", mode);
      },
    }),
  };
  elements.controlsRow.append(modeToggles.left.element, modeToggles.right.element);

  function render(): void {
    drawGame(ctx, state);
    renderScoreboard(elements.score, elements.status, state, {
      leftMode: controllers.left.mode,
      rightMode: controllers.right.mode,
      warning,
    });
    modelViz.render({
      left: controllers.left.mode === "model" ? controllers.left.lastVizSnapshot : null,
      right: controllers.right.mode === "model" ? controllers.right.lastVizSnapshot : null,
    });
  }

  function resetModelController(side: PaddleSide): void {
    const controller = controllers[side];
    controller.inferenceToken += 1;
    controller.inferencePending = false;
    controller.lastModelAction = "none";
    controller.inferenceSequence = 0;
    controller.lastVizSnapshot = null;
    controller.frameStack.reset();
  }

  function resetHeuristicController(side: PaddleSide): void {
    controllers[side].lastHeuristicAction = "none";
  }

  function resetAllControllers(): void {
    resetModelController("left");
    resetModelController("right");
    resetHeuristicController("left");
    resetHeuristicController("right");
  }

  function applyFallbackMode(side: PaddleSide, message: string): void {
    const fallbackMode = getFallbackMode(side);
    controllers[side].mode = fallbackMode;
    warning = message;
    modeToggles[side].setMode(fallbackMode);
    resetModelController(side);
    resetHeuristicController(side);
  }

  async function setMode(side: PaddleSide, nextMode: ControlMode): Promise<void> {
    if (disposed) {
      return;
    }

    if (nextMode === "model") {
      try {
        await inference.load(modelUrl);
        if (disposed) {
          return;
        }
        controllers[side].mode = "model";
        warning = null;
        resetModelController(side);
        resetHeuristicController(side);
      } catch (error) {
        console.error("Falha ao carregar o modelo ONNX no browser.", error);
        applyFallbackMode(
          side,
          `Falha ao carregar o modelo; fallback ativado. ${formatRuntimeError(error)}`,
        );
      }
    } else {
      controllers[side].mode = nextMode;
      warning = null;
      resetModelController(side);
      resetHeuristicController(side);
    }

    modeToggles[side].setMode(controllers[side].mode);
    fitViewport();
    render();
  }

  function scheduleInferenceIfNeeded(side: PaddleSide, activeState: GameState): void {
    const controller = controllers[side];
    if (controller.mode !== "model" || controller.inferencePending) {
      return;
    }
    if (!controller.frameStack.isReady()) {
      return;
    }
    if (activeState.tick % INFERENCE_INTERVAL_TICKS !== 0) {
      return;
    }

    const observation = controller.frameStack.flatten();
    const token = controller.inferenceToken;
    controller.inferencePending = true;

    void inference
      .forward(observation)
      .then((result) => {
        if (!disposed && controllers[side].mode === "model" && token === controller.inferenceToken) {
          const sampledAction = sampleModelAction(result.probs, rng);
          controller.lastModelAction = sampledAction;
          controller.inferenceSequence += 1;
          controller.lastVizSnapshot = {
            side,
            sequence: controller.inferenceSequence,
            action: sampledAction,
            probs: result.probs.slice(),
            observation: observation.slice(),
            hiddenActivations: cloneHiddenActivations(result.hiddenActivations),
          };
        }
      })
      .catch((error) => {
        if (!disposed && token === controller.inferenceToken) {
          console.error("Falha durante a inferencia ONNX no browser.", error);
          applyFallbackMode(
            side,
            `Falha na inferencia; fallback ativado. ${formatRuntimeError(error)}`,
          );
          render();
        }
      })
      .finally(() => {
        if (token === controller.inferenceToken) {
          controller.inferencePending = false;
        }
      });
  }

  function getRightAction(activeState: GameState): Action {
    return getActionForSide("right", activeState);
  }

  function getLeftAction(activeState: GameState): Action {
    return getActionForSide("left", activeState);
  }

  function getActionForSide(side: PaddleSide, activeState: GameState): Action {
    const controller = controllers[side];

    if (controller.mode === "model") {
      const bitmap = bitmapFromState(activeState);
      controller.frameStack.push(
        side === "left" ? mirrorBitmapHorizontally(bitmap) : bitmap,
      );
      scheduleInferenceIfNeeded(side, activeState);
      return controller.lastModelAction;
    }
    if (controller.mode === "heuristic") {
      if (activeState.tick % HEURISTIC_INTERVAL_TICKS === 0) {
        controller.lastHeuristicAction =
          side === "left"
            ? partiallyTracking(activeState, rng)
            : getRightHeuristicAction(activeState, rng);
      }
      return controller.lastHeuristicAction;
    }
    return side === "left" ? controls.getLeftAction() : controls.getRightAction();
  }

  function gameLoop(now: number): void {
    if (disposed) {
      return;
    }

    frames += 1;
    if (now - lastFpsSample >= 1000) {
      frames = 0;
      lastFpsSample = now;
    }

    if (servePauseFrames === 0) {
      const previousState = state;
      const leftAction = getLeftAction(state);
      const rightAction = getRightAction(state);
      state = step(state, leftAction, rightAction, rng);

      if (didScore(previousState, state)) {
        servePauseFrames = POINT_SERVE_PAUSE_FRAMES;
      }

      if (state.scoreLeft >= SCORE_TO_WIN || state.scoreRight >= SCORE_TO_WIN) {
        state = createInitialState(rng);
        servePauseFrames = POINT_SERVE_PAUSE_FRAMES;
        resetAllControllers();
      }
    } else {
      servePauseFrames = tickPauseFrames(servePauseFrames);
    }

    render();
    animationFrameHandle = requestFrame(gameLoop);
  }

  const initialMode = await resolveInitialMode(fetchImpl, modelUrl);
  controllers.left.mode = "heuristic";
  controllers.right.mode = initialMode.mode;
  warning = initialMode.warning;
  modeToggles.left.setMode(controllers.left.mode);
  modeToggles.right.setMode(controllers.right.mode);

  if (controllers.right.mode === "model") {
    await setMode("right", "model");
  } else {
    render();
  }

  console.log(`Target FPS: ${FPS}`);
  animationFrameHandle = requestFrame(gameLoop);

  return {
    dispose(): void {
      disposed = true;
      if (animationFrameHandle !== null) {
        cancelFrame(animationFrameHandle);
      }
      controls.dispose();
      layoutWindow?.removeEventListener("resize", fitViewport);
      elements.app.remove();
    },
    getMode(): ControlMode {
      return controllers.right.mode;
    },
    getModes(): { left: ControlMode; right: ControlMode } {
      return {
        left: controllers.left.mode,
        right: controllers.right.mode,
      };
    },
    getState(): GameState {
      return state;
    },
    elements,
  };
}

function formatRuntimeError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function sampleModelAction(probs: Float32Array, rng: Rng): Action {
  let threshold = rng.next();
  for (let index = 0; index < probs.length; index += 1) {
    threshold -= probs[index];
    if (threshold <= 0) {
      return toAction(index);
    }
  }
  return toAction(probs.length - 1);
}

function toAction(index: number): Action {
  if (index === 0) {
    return "up";
  }
  if (index === 1) {
    return "down";
  }
  return "none";
}

function createSideController(frameStack: FrameStackTS, mode: ControlMode): SideController {
  return {
    mode,
    lastModelAction: "none",
    lastHeuristicAction: "none",
    inferencePending: false,
    inferenceToken: 0,
    inferenceSequence: 0,
    lastVizSnapshot: null,
    frameStack,
  };
}

function getFallbackMode(side: PaddleSide): ControlMode {
  return side === "left" ? "heuristic" : "keyboard";
}

function createAppElements(doc: Document): AppElements {
  const app = doc.createElement("main");
  app.style.minHeight = "100vh";
  app.style.height = "100vh";
  app.style.display = "grid";
  app.style.placeItems = "center";
  app.style.background = "#0b1116";
  app.style.color = "#f7f7f2";
  app.style.fontFamily = "Arial, sans-serif";
  app.style.padding = "12px";
  app.style.boxSizing = "border-box";
  app.style.width = "100%";

  const viewport = doc.createElement("div");
  viewport.style.width = "100%";
  viewport.style.height = "100%";
  viewport.style.display = "grid";
  viewport.style.placeItems = "center";
  viewport.style.overflow = "hidden";

  const layout = doc.createElement("section");
  layout.style.display = "grid";
  layout.style.gap = "10px";
  layout.style.justifyItems = "center";
  layout.style.alignContent = "start";
  layout.style.width = `${BASE_LAYOUT_WIDTH}px`;
  layout.style.maxWidth = `${BASE_LAYOUT_WIDTH}px`;
  layout.style.transformOrigin = "center center";
  layout.style.willChange = "transform";

  const controlsRow = doc.createElement("div");
  controlsRow.style.display = "grid";
  controlsRow.style.gridTemplateColumns = "repeat(2, minmax(0, 1fr))";
  controlsRow.style.alignItems = "center";
  controlsRow.style.width = "100%";
  controlsRow.style.gap = "12px";

  const score = doc.createElement("div");
  score.style.fontSize = "30px";
  score.style.textAlign = "center";

  const status = doc.createElement("div");
  status.style.fontSize = "13px";
  status.style.textAlign = "center";
  status.style.maxWidth = "100%";
  status.style.wordBreak = "break-word";

  const canvas = doc.createElement("canvas");
  canvas.width = ARENA_WIDTH;
  canvas.height = ARENA_HEIGHT;
  canvas.style.width = "820px";
  canvas.style.maxWidth = "820px";
  canvas.style.height = "auto";
  canvas.style.aspectRatio = `${ARENA_WIDTH} / ${ARENA_HEIGHT}`;
  canvas.style.border = "2px solid #4c9f70";
  canvas.style.borderRadius = "8px";
  canvas.style.background = "#101820";
  canvas.style.imageRendering = "pixelated";
  canvas.style.display = "block";

  const canvasStage = doc.createElement("div");
  canvasStage.style.width = "100%";
  canvasStage.style.display = "grid";
  canvasStage.style.placeItems = "center";
  canvasStage.append(canvas);

  const visualization = doc.createElement("div");
  visualization.style.width = "100%";
  visualization.style.minWidth = "0";

  layout.append(score, status, controlsRow, canvasStage, visualization);
  viewport.append(layout);
  app.append(viewport);

  return { app, viewport, layout, controlsRow, score, status, canvas, visualization };
}

function getRightHeuristicAction(state: GameState, rng: Rng): Action {
  if (state.ballVx > 0) {
    const paddleCenter = state.paddleRightY + PADDLE_HEIGHT / 2;
    const ballCenter = state.ballY + BALL_SIZE / 2;
    if (ballCenter < paddleCenter - 1) {
      return "up";
    }
    if (ballCenter > paddleCenter + 1) {
      return "down";
    }
    return "none";
  }

  const value = rng.next();
  if (value < 1 / 3) {
    return "up";
  }
  if (value < 2 / 3) {
    return "down";
  }
  return "none";
}

if (
  typeof window !== "undefined" &&
  typeof document !== "undefined" &&
  !globalThis.__PONG_IA_DISABLE_AUTO_START__
) {
  void startApp();
}

function cloneHiddenActivations(hiddenActivations: HiddenActivations): HiddenActivations {
  return {
    hiddenOne: hiddenActivations.hiddenOne.slice(),
    hiddenTwo: hiddenActivations.hiddenTwo.slice(),
    hiddenThree: hiddenActivations.hiddenThree.slice(),
  };
}

function fitLayoutToViewport(elements: AppElements, layoutWindow: Window | null): void {
  const viewportWidth = layoutWindow?.innerWidth ?? BASE_LAYOUT_WIDTH;
  const viewportHeight = layoutWindow?.innerHeight ?? BASE_LAYOUT_HEIGHT;
  const layoutWidth = elements.layout.scrollWidth || BASE_LAYOUT_WIDTH;
  const layoutHeight = elements.layout.scrollHeight || BASE_LAYOUT_HEIGHT;
  const availableWidth = Math.max(viewportWidth - VIEWPORT_PADDING, 1);
  const availableHeight = Math.max(viewportHeight - VIEWPORT_PADDING, 1);
  const scale = Math.min(availableWidth / layoutWidth, availableHeight / layoutHeight, 1);

  elements.layout.style.transform = `scale(${scale})`;
}
