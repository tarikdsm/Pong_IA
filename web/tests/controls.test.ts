import { describe, expect, it } from "vitest";

import {
  LEFT_PADDLE_REPEAT_INTERVAL_FRAMES,
  RIGHT_PADDLE_REPEAT_INTERVAL_FRAMES,
  createControls,
  normalizeLeftControlKey,
  normalizeRightControlKey,
  resolveActionFromPressedKeys,
} from "@/ui/controls";

describe("controls", () => {
  it("normalizes W/S only for the left paddle", () => {
    expect(normalizeLeftControlKey("w")).toBe("up");
    expect(normalizeLeftControlKey("S")).toBe("down");
    expect(normalizeLeftControlKey("ArrowUp")).toBeNull();
  });

  it("normalizes arrow keys only for the right paddle", () => {
    expect(normalizeRightControlKey("ArrowUp")).toBe("up");
    expect(normalizeRightControlKey("ArrowDown")).toBe("down");
    expect(normalizeRightControlKey("w")).toBeNull();
  });

  it("resolves action from the pressed-key set", () => {
    expect(resolveActionFromPressedKeys(new Set())).toBe("none");
    expect(resolveActionFromPressedKeys(new Set(["up"]))).toBe("up");
    expect(resolveActionFromPressedKeys(new Set(["down"]))).toBe("down");
    expect(resolveActionFromPressedKeys(new Set(["up", "down"]))).toBe("none");
  });

  it("emits one immediate action pulse on keydown for the left paddle", () => {
    const controls = createControls({
      addEventListener() {
        return undefined;
      },
      removeEventListener() {
        return undefined;
      },
    });

    controls.handleKeyDown({ key: "w", preventDefault() {} });

    expect(controls.getLeftAction()).toBe("up");
    expect(controls.getLeftAction()).toBe("none");
  });

  it("repeats held left movement more slowly than every frame", () => {
    const controls = createControls({
      addEventListener() {
        return undefined;
      },
      removeEventListener() {
        return undefined;
      },
    });

    controls.handleKeyDown({ key: "s", preventDefault() {} });

    expect(controls.getLeftAction()).toBe("down");
    for (let frame = 1; frame < LEFT_PADDLE_REPEAT_INTERVAL_FRAMES - 1; frame += 1) {
      expect(controls.getLeftAction()).toBe("none");
    }
    expect(controls.getLeftAction()).toBe("down");
  });

  it("repeats held right movement more slowly than every frame", () => {
    const controls = createControls({
      addEventListener() {
        return undefined;
      },
      removeEventListener() {
        return undefined;
      },
    });

    controls.handleKeyDown({ key: "ArrowDown", preventDefault() {} });

    expect(controls.getRightAction()).toBe("down");
    for (let frame = 1; frame < RIGHT_PADDLE_REPEAT_INTERVAL_FRAMES - 1; frame += 1) {
      expect(controls.getRightAction()).toBe("none");
    }
    expect(controls.getRightAction()).toBe("down");
  });

  it("prevents the browser default scroll behavior for both control schemes", () => {
    const controls = createControls({
      addEventListener() {
        return undefined;
      },
      removeEventListener() {
        return undefined;
      },
    });
    let prevented = false;

    controls.handleKeyDown({
      key: "ArrowUp",
      preventDefault() {
        prevented = true;
      },
    });

    expect(prevented).toBe(true);

    prevented = false;
    controls.handleKeyDown({
      key: "w",
      preventDefault() {
        prevented = true;
      },
    });

    expect(prevented).toBe(true);
  });

  it("keeps left and right keyboard control independent", () => {
    const controls = createControls({
      addEventListener() {
        return undefined;
      },
      removeEventListener() {
        return undefined;
      },
    });

    controls.handleKeyDown({ key: "w", preventDefault() {} });
    controls.handleKeyDown({ key: "ArrowDown", preventDefault() {} });

    expect(controls.getLeftAction()).toBe("up");
    expect(controls.getRightAction()).toBe("down");
  });
});
