import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
const __dirname = dirname(fileURLToPath(import.meta.url));
export const config = {
  name: "VADAR API", configKey: "vadar_api",
  promptPath: resolve(__dirname, "prompt.md"),
  promptPathText: resolve(__dirname, "prompt-text.md"),
  userPromptPath: resolve(__dirname, "user-prompt.md"),
  allowedTools: ["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
  model: "sonnet",
};
