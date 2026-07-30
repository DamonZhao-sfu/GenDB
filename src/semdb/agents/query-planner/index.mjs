import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

export const config = {
  name: "Semantic Query Planner",
  configKey: "query_planner",
  promptPath: resolve(here, "prompt.md"),
  userPromptPath: resolve(here, "user-prompt.md"),
  skillPath: resolve(here, "..", "..", "skills", "plan-semantic-query", "SKILL.md"),
  skillName: "plan-semantic-query",
  allowedTools: ["Read", "Write", "Glob", "Grep", "Skill"],
};
