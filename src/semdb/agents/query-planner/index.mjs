import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const skillPath = resolve(here, "..", "..", "skills", "plan-semantic-query", "SKILL.md");

export const config = {
  name: "Semantic Query Planner",
  configKey: "query_planner",
  promptPath: skillPath,
  userPromptPath: resolve(here, "user-prompt.md"),
  skillPath,
  skillName: "plan-semantic-query",
  allowedTools: ["Read", "Write", "Glob", "Grep", "Skill"],
};
