/**
 * Role skills after the removal of static binding.
 *
 * They used to be injected into every system prompt by skill-loader.mjs, which
 * also told the agent "Do not discover or load any other skill". Both are gone:
 * the three procedures are now published into the isolated skill root and
 * discovered like any other skill. This file replaces test_skill_loader.mjs and
 * keeps its content assertions, which check the skills' substance, not the
 * delivery mechanism.
 */

import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { config as planner } from "../agents/query-planner/index.mjs";
import { config as generator } from "../agents/semantic-code-generator/index.mjs";
import { config as optimizer } from "../agents/semantic-optimizer/index.mjs";
import {
  ROLE_SKILLS,
  initSkillRoot,
  lintSkill,
  listSkills,
  publishRoleSkills,
  resolveSkillRoot,
  skillsDirFor,
} from "../memory/skills.mjs";
import { parseArgs } from "../orchestrator.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const semdbDir = resolve(here, "..");

// --- binding is gone --------------------------------------------------------

assert.ok(
  !existsSync(resolve(semdbDir, "agent-runtime", "skill-loader.mjs")),
  "skill-loader.mjs must be gone: skills are discovered, not bound",
);

const sources = [];
for (const rel of ["orchestrator.mjs", "agent-runtime", "skills", "agents"]) {
  const target = resolve(semdbDir, rel);
  const stack = [target];
  while (stack.length) {
    const current = stack.pop();
    if (!existsSync(current)) continue;
    let entries = [];
    try {
      entries = await readdir(current, { withFileTypes: true });
    } catch {
      sources.push(await readFile(current, "utf8"));   // it was a file
      continue;
    }
    for (const e of entries) {
      const p = resolve(current, e.name);
      if (e.isDirectory()) stack.push(p);
      else if (/\.(mjs|md)$/.test(e.name)) sources.push(await readFile(p, "utf8"));
    }
  }
}
const allSource = sources.join("\n");
assert.ok(
  !/Do not discover or load any other skill/i.test(allSource),
  "the sentence forbidding discovery must not survive anywhere in semdb",
);
assert.ok(
  !/statically bound/i.test(allSource),
  "no prompt may still claim a skill is statically bound to a role",
);

// --- the Skill tool is actually granted -------------------------------------

for (const config of [planner, generator, optimizer]) {
  assert.ok(
    config.allowedTools.includes("Skill"),
    `${config.name} needs the Skill tool — the provider filters the capability out without it`,
  );
  assert.ok(config.skillName, `${config.name} still names its procedure skill`);
}

// --- role skills publish into an isolated root and pass lint ----------------

const out = await mkdtemp(resolve(tmpdir(), "semdb-out-"));
const root = resolveSkillRoot({ memoryDir: null, out });
assert.equal(root, resolve(out, "_skills"),
  "without --memory-dir the roles still need a root, or they lose their procedures");
await initSkillRoot(root);
const published = await publishRoleSkills(root);
assert.deepEqual(published.sort(), [...ROLE_SKILLS].sort());

const discovered = await listSkills(root);
assert.deepEqual(
  discovered.map((s) => s.name).sort(),
  [...ROLE_SKILLS].sort(),
  "all three procedures are discoverable",
);
for (const s of discovered) {
  assert.equal(s.role, true, `${s.name} is tagged as a role skill, so it stays out of the memory catalog`);
  assert.ok(s.description, `${s.name} needs a description — discovery matches on it`);
}

for (const name of ROLE_SKILLS) {
  const result = await lintSkill(resolve(skillsDirFor(root), name), {
    expectedName: name, requireEvidence: false, requireTriggerPhrasing: false,
  });
  assert.deepEqual(result.errors, [], `${name} must pass lint`);
}

const bodyOf = async (name) => readFile(resolve(skillsDirFor(root), name, "SKILL.md"), "utf8");

// --- content assertions carried over from test_skill_loader.mjs -------------

const plannerBody = await bodyOf("plan-semantic-query");
assert.ok(!plannerBody.includes("Signature agent"));
assert.ok(!plannerBody.includes("API agent"));
for (const primitive of [
  "classify_multi", "detect_open", "regions_propose", "pair_score", "topk_similar", "topk_text",
]) {
  assert.ok(
    plannerBody.includes(`\`${primitive}\``),
    `Planner skill preserves vis-operator guidance for ${primitive}`,
  );
}

const generatorBody = await bodyOf("generate-semantic-program");
assert.ok(generatorBody.includes("resolve_image_path"));
assert.ok(generatorBody.includes("vadar.predefined"));
assert.ok(!generatorBody.includes("predefined_text"));
assert.ok(!/\bimport semvision\b/.test(generatorBody));
assert.ok(generatorBody.includes("branch counters"));

const optimizerBody = await bodyOf("optimize-semantic-program");
for (const metric of [
  "adjusted_rand_index", "relative_error", "spearman_correlation", "macro_f1",
]) {
  assert.ok(
    optimizerBody.includes(`\`${metric}\``),
    `Optimizer skill has query-specific guidance for ${metric}`,
  );
}
assert.ok(optimizerBody.includes("must never replace"));

// --- CLI ---------------------------------------------------------------------

const disabled = parseArgs([
  "node", "orchestrator.mjs", "--direct", "--agent-architecture", "pgo",
  "--max-replans", "0", "--no-agent-skills",
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

console.log("test_role_skills: PASS");
