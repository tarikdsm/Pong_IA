export type ControlMode = "model" | "heuristic" | "keyboard";

export interface ModeResolution {
  readonly mode: ControlMode;
  readonly modelAvailable: boolean;
  readonly warning: string | null;
}

export interface ModeToggleOptions {
  readonly labelText: string;
  readonly initialMode: ControlMode;
  readonly onChange: (mode: ControlMode) => void;
}

export interface ModeToggle {
  readonly element: HTMLLabelElement;
  readonly select: HTMLSelectElement;
  getMode(): ControlMode;
  setMode(mode: ControlMode): void;
}

type FetchLike = (input: string, init?: RequestInit) => Promise<Pick<Response, "ok">>;

const MODE_LABELS: Record<ControlMode, string> = {
  model: "modelo",
  heuristic: "heuristica",
  keyboard: "teclado",
};

export function createModeToggle(options: ModeToggleOptions): ModeToggle {
  const label = document.createElement("label");
  label.style.display = "flex";
  label.style.alignItems = "center";
  label.style.justifyContent = "center";
  label.style.gap = "8px";
  label.style.fontSize = "14px";
  label.style.width = "100%";
  label.style.minWidth = "0";
  label.style.flexWrap = "nowrap";

  const text = document.createElement("span");
  text.textContent = options.labelText;
  text.style.whiteSpace = "nowrap";

  const select = document.createElement("select");
  select.style.borderRadius = "6px";
  select.style.border = "1px solid #4c9f70";
  select.style.background = "#101820";
  select.style.color = "#f7f7f2";
  select.style.padding = "6px 8px";
  select.style.minWidth = "0";

  for (const mode of ["model", "heuristic", "keyboard"] satisfies ControlMode[]) {
    const option = document.createElement("option");
    option.value = mode;
    option.textContent = getModeLabel(mode);
    select.append(option);
  }

  select.value = options.initialMode;
  select.addEventListener("change", () => options.onChange(select.value as ControlMode));

  label.append(text, select);

  return {
    element: label,
    select,
    getMode(): ControlMode {
      return select.value as ControlMode;
    },
    setMode(mode: ControlMode): void {
      select.value = mode;
    },
  };
}

export function getModeLabel(mode: ControlMode): string {
  return MODE_LABELS[mode];
}

export async function resolveInitialMode(
  fetchImpl: FetchLike,
  modelUrl: string,
): Promise<ModeResolution> {
  try {
    const response = await fetchImpl(modelUrl, {
      cache: "no-store",
      method: "HEAD",
    });

    if (response.ok) {
      return {
        mode: "model",
        modelAvailable: true,
        warning: null,
      };
    }
  } catch {
    return {
      mode: "keyboard",
      modelAvailable: false,
      warning: "Nao foi possivel localizar model.onnx; teclado ativado.",
    };
  }

  return {
    mode: "keyboard",
    modelAvailable: false,
    warning: "model.onnx ausente; teclado ativado.",
  };
}
