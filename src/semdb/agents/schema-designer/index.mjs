import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

export const config = {
  name: "Schema Designer",
  configKey: "schema_designer",
  promptPath: resolve(__dirname, "prompt.md"),
  userPromptPath: resolve(__dirname, "user-prompt.md"),
  allowedTools: ["Read", "Write", "Edit", "Glob", "Grep"],
  model: "opus",
};
