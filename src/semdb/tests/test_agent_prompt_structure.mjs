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
assert.ok(plannerUser.includes("{{plan_schema_path}}"));
assert.ok(plannerUser.includes("{{optimizer_action_path}}"));

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
assert.ok(optimizerUser.includes("{{optimizer_action_schema_path}}"));
assert.ok(optimizerUser.includes("{{local_primitive_files}}"));

for (const binding of [
  "plan_schema_path:",
  "generation_mode:",
  "optimizer_action_schema_path:",
  "local_primitive_files:",
]) {
  assert.ok(orchestrator.includes(binding), `Orchestrator is missing ${binding}`);
}

// Skills are discovered, not bound: each role must be granted the Skill tool, told to
// load its own procedure, and given the three memory blocks.
for (const [label, config, systemPrompt, userPrompt, skill] of [
  ["Planner", plannerConfig, plannerPrompt, plannerUser, "plan-semantic-query"],
  ["Generator", generatorConfig, generatorPrompt, generatorUser, "generate-semantic-program"],
  ["Optimizer", optimizerConfig, optimizerPrompt, optimizerUser, "optimize-semantic-program"],
]) {
  assert.ok(config.allowedTools.includes("Skill"), `${label} must be granted the Skill tool`);
  assert.ok(
    userPrompt.includes(`Load the \`${skill}\` skill`),
    `${label} user prompt must direct the agent to load its procedure skill`,
  );
  assert.ok(
    systemPrompt.includes("## Prior knowledge is advisory"),
    `${label} system prompt must mark prior knowledge advisory`,
  );
  assert.ok(
    systemPrompt.includes("never contains ground-truth answers"),
    `${label} system prompt must state the ground-truth boundary`,
  );
  for (const guard of ["memory_pre_injection", "memory_catalog", "memory_inline_skills"]) {
    assert.ok(
      userPrompt.includes(`{{#if ${guard}}}`),
      `${label} user prompt must guard ${guard} so it vanishes when memory is off`,
    );
  }
}
assert.ok(
  plannerUser.includes("{{#if memory_reference_plan_path}}"),
  "Planner needs the warm-start reference branch (a reference, not a replan)",
);

function assertFullyRendered(template, vars, label) {
  const rendered = renderTemplate(template, vars);
  assert.ok(!rendered.includes("{{"), `${label} left an unresolved template placeholder`);
  return rendered;
}

assertFullyRendered(plannerUser, {
  query_id: "q",
  query_sql: "SELECT * FROM images",
  query_nl: "find matching images",
  modality: "image",
  tables_doc: "- images.csv",
  local_primitive_files: "- predefined.py",
  trace_contract: "row key: id",
  previous_plan_path: "/run/iter_0/plan.json",
  optimizer_action_path: "/run/iter_1/optimizer_action.json",
  planner_lint: "",
  plan_schema_path: "/repo/contracts/semantic-plan.schema.json",
  plan_path: "/run/iter_1/plan.json",
}, "Planner user prompt");

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
  plan_path: "/run/iter_0/plan.json",
  candidate_manifest_path: "/run/iter_0/candidate_manifest.json",
  iteration_feedback_path: "/run/iter_0/iteration_feedback.json",
  history_manifest_paths: "- /run/iter_0/candidate_manifest.json",
  local_primitive_files: "- predefined.py",
  remaining_iteration_budget: 2,
  remaining_replan_budget: 1,
  optimizer_action_schema_path: "/repo/contracts/optimizer-action.schema.json",
  optimizer_action_path: "/run/iter_1/optimizer_action.json",
}, "Optimizer user prompt");

console.log("test_agent_prompt_structure OK");
