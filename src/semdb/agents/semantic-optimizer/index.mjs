import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const skillPath = resolve(here, "..", "..", "skills", "optimize-semantic-program", "SKILL.md");

export const config = {
  name: "Semantic Optimizer",
  configKey: "semantic_optimizer",
  promptPath: skillPath,
  userPromptPath: resolve(here, "user-prompt.md"),
  skillPath,
  skillName: "optimize-semantic-program",
  allowedTools: ["Read", "Write", "Glob", "Grep", "Skill"],
};
