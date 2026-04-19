import type { Action } from "@/engine/state";

export interface Controls {
  dispose(): void;
  getLeftAction(): Action;
  getRightAction(): Action;
  handleKeyDown(event: Pick<KeyboardEvent, "key" | "preventDefault">): void;
  handleKeyUp(event: Pick<KeyboardEvent, "key" | "preventDefault">): void;
}

export const RIGHT_PADDLE_REPEAT_INTERVAL_FRAMES = 3;
export const LEFT_PADDLE_REPEAT_INTERVAL_FRAMES = RIGHT_PADDLE_REPEAT_INTERVAL_FRAMES;

interface EventTargetLike {
  addEventListener(type: "keydown" | "keyup", listener: (event: KeyboardEvent) => void): void;
  removeEventListener(type: "keydown" | "keyup", listener: (event: KeyboardEvent) => void): void;
}

interface ControlChannelState {
  readonly pressedKeys: Set<string>;
  pendingAction: Action;
  holdFrames: number;
  readonly repeatIntervalFrames: number;
}

export function resolveActionFromPressedKeys(pressedKeys: ReadonlySet<string>): Action {
  const upPressed = pressedKeys.has("up");
  const downPressed = pressedKeys.has("down");
  if (upPressed === downPressed) {
    return "none";
  }
  return upPressed ? "up" : "down";
}

export function normalizeLeftControlKey(key: string): "up" | "down" | null {
  const normalized = key.toLowerCase();
  if (normalized === "w") {
    return "up";
  }
  if (normalized === "s") {
    return "down";
  }
  return null;
}

export function normalizeRightControlKey(key: string): "up" | "down" | null {
  const normalized = key.toLowerCase();
  if (normalized === "arrowup") {
    return "up";
  }
  if (normalized === "arrowdown") {
    return "down";
  }
  return null;
}

export function createControls(target: EventTargetLike): Controls {
  const leftChannel = createControlChannel(LEFT_PADDLE_REPEAT_INTERVAL_FRAMES);
  const rightChannel = createControlChannel(RIGHT_PADDLE_REPEAT_INTERVAL_FRAMES);

  const handleKeyDown = (event: Pick<KeyboardEvent, "key" | "preventDefault">): void => {
    const leftKey = normalizeLeftControlKey(event.key);
    const rightKey = normalizeRightControlKey(event.key);

    if (leftKey || rightKey) {
      event.preventDefault();
    }
    if (leftKey) {
      pushKeyDown(leftChannel, leftKey);
    }
    if (rightKey) {
      pushKeyDown(rightChannel, rightKey);
    }
  };

  const handleKeyUp = (event: Pick<KeyboardEvent, "key" | "preventDefault">): void => {
    const leftKey = normalizeLeftControlKey(event.key);
    const rightKey = normalizeRightControlKey(event.key);

    if (leftKey || rightKey) {
      event.preventDefault();
    }
    if (leftKey) {
      leftChannel.pressedKeys.delete(leftKey);
    }
    if (rightKey) {
      rightChannel.pressedKeys.delete(rightKey);
    }
  };

  const onKeyDown = (event: KeyboardEvent): void => handleKeyDown(event);
  const onKeyUp = (event: KeyboardEvent): void => handleKeyUp(event);

  target.addEventListener("keydown", onKeyDown);
  target.addEventListener("keyup", onKeyUp);

  return {
    dispose(): void {
      target.removeEventListener("keydown", onKeyDown);
      target.removeEventListener("keyup", onKeyUp);
    },
    getLeftAction(): Action {
      return consumeAction(leftChannel);
    },
    getRightAction(): Action {
      return consumeAction(rightChannel);
    },
    handleKeyDown,
    handleKeyUp,
  };
}

function createControlChannel(repeatIntervalFrames: number): ControlChannelState {
  return {
    pressedKeys: new Set<string>(),
    pendingAction: "none",
    holdFrames: 0,
    repeatIntervalFrames,
  };
}

function pushKeyDown(channel: ControlChannelState, key: "up" | "down"): void {
  if (!channel.pressedKeys.has(key)) {
    channel.pendingAction = key;
    channel.holdFrames = 0;
  }
  channel.pressedKeys.add(key);
}

function consumeAction(channel: ControlChannelState): Action {
  if (channel.pendingAction !== "none") {
    const action = channel.pendingAction;
    channel.pendingAction = "none";
    channel.holdFrames = 1;
    return action;
  }

  const heldAction = resolveActionFromPressedKeys(channel.pressedKeys);
  if (heldAction === "none") {
    channel.holdFrames = 0;
    return "none";
  }

  channel.holdFrames += 1;
  if (channel.holdFrames >= channel.repeatIntervalFrames) {
    channel.holdFrames = 0;
    return heldAction;
  }

  return "none";
}
