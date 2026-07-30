import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

export const config = {
  name: "Semantic Code Generator",
  configKey: "semantic_code_generator",
  promptPath: resolve(here, "prompt.md"),
  userPromptPath: resolve(here, "user-prompt.md"),
  skillPath: resolve(here, "..", "..", "skills", "generate-semantic-program", "SKILL.md"),
  skillName: "generate-semantic-program",
  allowedTools: ["Read", "Write", "Edit", "Glob", "Grep", "Bash", "Skill"],
};
