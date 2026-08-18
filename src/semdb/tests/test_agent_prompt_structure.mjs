import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { renderTemplate } from "../../gendb/shared.mjs";
import { config as plannerConfig } from "../agents/query-planner/index.mjs";
import { config as generatorConfig } from "../agents/semantic-code-generator/index.mjs";
import { config as optimizerConfig } from "../agents/semantic-optimizer/index.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const semdbDir = resolve(here, "..");

async function read(relativePath) {
  return readFile(resolve(semdbDir, relativePath), "utf8");
}

const plannerPrompt = await read("agents/query-planner/prompt.md");
const plannerUser = await read("agents/query-planner/user-prompt.md");
const generatorPrompt = await read("agents/semantic-code-generator/prompt.md");
const generatorUser = await read("agents/semantic-code-generator/user-prompt.md");
const optimizerPrompt = await read("agents/semantic-optimizer/prompt.md");
const optimizerUser = await read("agents/semantic-optimizer/user-prompt.md");
const orchestrator = await read("orchestrator.mjs");

for (const heading of [
  "## Identity",
  "## Thinking Discipline",
  "## Semantic Planning Framework",
  "## Plan JSON Structure",
  "## Key Rules",
  "## Output Contract",
]) {
  assert.ok(plannerPrompt.includes(heading), `Planner prompt is missing ${heading}`);
}
assert.ok(!plannerPrompt.includes("VADAR Signature"));
assert.ok(!plannerPrompt.includes("VADAR API"));
assert.ok(!plannerPrompt.includes("legacy VADAR"));
assert.ok(plannerPrompt.includes("### Step 8: Replan from evidence"));
assert.ok(plannerUser.includes("{{agent_context_path}}"));
assert.ok(plannerUser.includes("{{agent_context_sha256}}"));

for (const heading of [
  "## Identity",
  "## Thinking Discipline",
  "## Authority and Generation Modes",
  "## Implementation Framework",
  "## Candidate Manifest Draft Structure",
  "## Key Rules",
  "## Output Contract",
]) {
  assert.ok(generatorPrompt.includes(heading), `Generator prompt is missing ${heading}`);
}
assert.ok(generatorPrompt.includes("### `PATCH_CODE`"));
assert.ok(generatorPrompt.includes("### `REPLAN`"));
assert.ok(generatorPrompt.includes("resolve_image_path"));
assert.ok(generatorPrompt.includes("from vadar import ImagePatch, get_encoder, resolve_image_path"));
assert.ok(generatorUser.includes("{{generation_mode}}"));
assert.ok(generatorUser.includes("{{replan_action_path}}"));

for (const heading of [
  "## Identity",
  "## Thinking Discipline",
  "## Ownership Boundary",
  "## Diagnostic Framework",
  "## Optimizer Action JSON Structure",
  "## Key Rules",
  "## Output Contract",
]) {
  assert.ok(optimizerPrompt.includes(heading), `Optimizer prompt is missing ${heading}`);
}
for (let step = 0; step <= 8; step += 1) {
  assert.ok(
    optimizerPrompt.includes(`### Step ${step}:`),
    `Optimizer prompt is missing diagnostic step ${step}`,
  );
}
assert.ok(optimizerPrompt.includes("Never use `PATCH_CODE` to tune a planned phrase"));
assert.ok(optimizerUser.includes("{{agent_context_path}}"));
assert.ok(optimizerUser.includes("{{agent_context_sha256}}"));

for (const binding of [
  "plan_schema_path:",
  "generation_mode:",
  "optimizer_action_schema_path:",
  "local_primitive_files:",
]) {
  assert.ok(orchestrator.includes(binding), `Orchestrator is missing ${binding}`);
}

// Planner/Optimizer receive their canonical procedure directly as the system prompt and
// must not spend a tool round-trip loading it again. Generator keeps discovery because it
// remains the unconstrained coding role in both execution modes.
for (const [label, config, userPrompt, skill] of [
  ["Planner", plannerConfig, plannerUser, "plan-semantic-query"],
  ["Optimizer", optimizerConfig, optimizerUser, "optimize-semantic-program"],
]) {
  assert.ok(config.allowedTools.includes("Skill"), `${label} must be granted the Skill tool`);
  assert.ok(
    userPrompt.includes(`canonical \`${skill}\` procedure is already active`),
    `${label} user prompt must state that its canonical procedure is already active`,
  );
  assert.ok(userPrompt.includes("Do not load its `SKILL.md` again"));
}
assert.ok(generatorConfig.allowedTools.includes("Skill"));
assert.ok(generatorUser.includes("Load the `generate-semantic-program` skill"));
for (const guard of ["memory_pre_injection", "memory_catalog", "memory_inline_skills"]) {
  assert.ok(generatorUser.includes(`{{#if ${guard}}}`));
}

function assertFullyRendered(template, vars, label) {
  const rendered = renderTemplate(template, vars);
  assert.ok(!rendered.includes("{{"), `${label} left an unresolved template placeholder`);
  return rendered;
}

const renderedPlannerReplan = assertFullyRendered(plannerUser, {
  query_id: "q",
  agent_context_path: "/run/iter_1/_agent_context_query_planner.md",
  agent_context_sha256: "abc123",
}, "Planner user prompt");
assert.ok(renderedPlannerReplan.includes("_agent_context_query_planner.md"));
assert.ok(renderedPlannerReplan.includes("abc123"));

const patchPrompt = assertFullyRendered(generatorUser, {
  query_id: "q",
  generation_mode: "PATCH_CODE",
  plan_path: "/run/iter_1/plan.json",
  plan_schema_path: "/repo/contracts/semantic-plan.schema.json",
  parent_candidate_manifest_path: "/run/iter_0/candidate_manifest.json",
  optimizer_action_path: "/run/iter_1/optimizer_action.json",
  replan_action_path: "",
  tables_doc: "- images.csv",
  local_primitive_files: "- predefined.py",
  semdb_dir: "/repo/src/semdb",
  solve_path: "/run/iter_1/solve_q.py",
  helpers_path: "/run/iter_1/helpers_q.py",
  manifest_draft_path: "/run/iter_1/candidate_manifest.json",
  runtime_args: " --image-dir <dir>",
}, "Generator PATCH_CODE user prompt");
assert.ok(patchPrompt.includes("## PATCH_CODE Context"));
assert.ok(!patchPrompt.includes("## REPLAN Context"));

const replanPrompt = assertFullyRendered(generatorUser, {
  query_id: "q",
  generation_mode: "REPLAN",
  plan_path: "/run/iter_1/plan.json",
  plan_schema_path: "/repo/contracts/semantic-plan.schema.json",
  parent_candidate_manifest_path: "",
  optimizer_action_path: "/run/iter_1/optimizer_action.json",
  replan_action_path: "/run/iter_1/optimizer_action.json",
  tables_doc: "- images.csv",
  local_primitive_files: "- predefined.py",
  semdb_dir: "/repo/src/semdb",
  solve_path: "/run/iter_1/solve_q.py",
  helpers_path: "/run/iter_1/helpers_q.py",
  manifest_draft_path: "/run/iter_1/candidate_manifest.json",
  runtime_args: " --image-dir <dir>",
}, "Generator REPLAN user prompt");
assert.ok(replanPrompt.includes("## REPLAN Context"));
assert.ok(!replanPrompt.includes("## PATCH_CODE Context"));

assertFullyRendered(optimizerUser, {
  query_id: "q",
  agent_context_path: "/run/iter_1/_agent_context_semantic_optimizer.md",
  agent_context_sha256: "def456",
}, "Optimizer user prompt");

console.log("test_agent_prompt_structure OK");
