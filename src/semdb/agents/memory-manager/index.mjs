import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

export const config = {
  name: "Memory Manager",
  configKey: "memory_manager",
  promptPath: resolve(here, "prompt.md"),
  userPromptPath: resolve(here, "user-prompt.md"),
  // It authors skills; it does not consume them, so it gets no Skill tool. Edit is
  // needed to extend an existing skill rather than duplicating it.
  allowedTools: ["Read", "Write", "Edit", "Glob", "Grep"],
};
