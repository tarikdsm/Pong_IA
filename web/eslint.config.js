import tseslint from "@typescript-eslint/eslint-plugin";
import parser from "@typescript-eslint/parser";

const browserGlobals = {
  console: "readonly",
  document: "readonly",
  window: "readonly",
  performance: "readonly",
  requestAnimationFrame: "readonly",
  cancelAnimationFrame: "readonly",
  KeyboardEvent: "readonly",
  HTMLElement: "readonly",
  HTMLDivElement: "readonly",
  HTMLCanvasElement: "readonly",
  CanvasRenderingContext2D: "readonly",
};

export default [
  {
    files: ["src/**/*.ts", "tests/**/*.ts", "vite.config.ts"],
    languageOptions: {
      parser,
      parserOptions: {
        ecmaVersion: "latest",
        sourceType: "module",
        project: "./tsconfig.json",
        tsconfigRootDir: import.meta.dirname,
      },
      globals: browserGlobals,
    },
    plugins: {
      "@typescript-eslint": tseslint,
    },
    rules: {
      "no-console": "off",
      "@typescript-eslint/consistent-type-imports": "error",
      "@typescript-eslint/no-explicit-any": "error",
    },
  },
];
