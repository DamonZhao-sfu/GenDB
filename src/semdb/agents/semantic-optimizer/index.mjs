import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

export const config = {
  name: "Semantic Optimizer",
  configKey: "semantic_optimizer",
  promptPath: resolve(here, "prompt.md"),
  userPromptPath: resolve(here, "user-prompt.md"),
  skillPath: resolve(here, "..", "..", "skills", "optimize-semantic-program", "SKILL.md"),
  skillName: "optimize-semantic-program",
  useSkills: true,
  allowedTools: ["Read", "Write", "Glob", "Grep"],
};
