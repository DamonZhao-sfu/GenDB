import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import {
  loadAgentSkill,
  loadBoundAgentSkill,
} from "../agent-runtime/skill-loader.mjs";
import { config as planner } from "../agents/query-planner/index.mjs";
import { config as generator } from "../agents/semantic-code-generator/index.mjs";
import { config as optimizer } from "../agents/semantic-optimizer/index.mjs";
import { parseArgs } from "../orchestrator.mjs";

for (const config of [planner, generator, optimizer]) {
  const skill = await loadAgentSkill(config.skillPath, config.skillName);
  assert.equal(skill.name, config.skillName);
  assert.ok(skill.description);
  assert.ok(skill.body.startsWith("# "));
  assert.ok(skill.prompt.includes("statically bound"));
}

assert.equal(await loadBoundAgentSkill(planner, false), null);

const root = await mkdtemp(resolve(tmpdir(), "semdb-skills-"));
const bad = resolve(root, "SKILL.md");

await writeFile(bad, "---\nname: valid-name\ndescription: x\nextra: no\n---\n# X\n");
await assert.rejects(() => loadAgentSkill(bad), /unknown frontmatter field/);

await writeFile(bad, "---\nname: camelCase\ndescription: x\n---\n# X\n");
await assert.rejects(() => loadAgentSkill(bad), /lowercase hyphen-case/);

await writeFile(bad, "---\nname: valid-name\n---\n# X\n");
await assert.rejects(() => loadAgentSkill(bad), /name and description/);

await assert.rejects(
  () => loadAgentSkill(planner.skillPath, "wrong-name"),
  /skill mismatch/i,
);

const disabled = parseArgs([
  "node",
  "orchestrator.mjs",
  "--direct",
  "--agent-architecture",
  "pgo",
  "--max-replans",
  "0",
  "--no-agent-skills",
]);
assert.equal(disabled.agentArchitecture, "pgo");
assert.equal(disabled.maxReplans, 0);
assert.equal(disabled.enableAgentSkills, false);
assert.throws(
  () => parseArgs(["node", "orchestrator.mjs", "--agent-architecture", "invalid"]),
  /agent-architecture/,
);
assert.throws(
  () => parseArgs(["node", "orchestrator.mjs", "--max-replans", "-1"]),
  /non-negative integer/,
);

console.log("test_skill_loader OK");
