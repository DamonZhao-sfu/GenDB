import assert from "node:assert/strict";
import { mkdtemp, readFile, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import { buildStructuredResponsesRequest } from "../../gendb/providers/vllm.mjs";
import { config as plannerConfig } from "../agents/query-planner/index.mjs";
import { config as generatorConfig } from "../agents/semantic-code-generator/index.mjs";
import { config as optimizerConfig } from "../agents/semantic-optimizer/index.mjs";
import {
  buildPrimitiveCatalog,
  extractPrimitiveSignatures,
  prepareAgentRole,
  prepareStructuredRole,
  supportsStructuredRole,
  validateStructuredOptimizerEvidence,
} from "../agent-runtime/structured-role.mjs";
import { writeTextAtomic } from "../agent-runtime/contracts.mjs";
import { parseArgs, selectAgentSkills } from "../orchestrator.mjs";

const root = await mkdtemp(resolve(tmpdir(), "semdb-structured-"));
const primitivePath = resolve("src/semdb/vadar/predefined.py");
const primitiveSource = await readFile(primitivePath, "utf8");
const signatures = extractPrimitiveSignatures(primitiveSource, primitivePath);
const catalog = await buildPrimitiveCatalog([primitivePath]);
assert.ok(signatures.includes("def classify("));
assert.ok(signatures.includes("def text_classify_detail("));
assert.ok(!signatures.includes("return image.classify("), "catalog must omit implementation bodies");
assert.ok(catalog.includes("SHA256:"));
assert.ok(catalog.length < primitiveSource.length * 0.7, "catalog must be materially smaller than source");

const previousPlanPath = resolve(root, "previous-plan.json");
const actionPath = resolve(root, "action.json");
const manifestPath = resolve(root, "manifest.json");
const feedbackPath = resolve(root, "feedback.json");
const helpersPath = resolve(root, "helpers.py");
const solverPath = resolve(root, "solver.py");
const diffPath = resolve(root, "diff.json");
await writeFile(previousPlanPath, JSON.stringify({ query_id: "q2a", plan_version: 1 }));
await writeFile(actionPath, JSON.stringify({ action: "REPLAN", candidate_id: "q2a-iter-0" }));
await writeFile(manifestPath, JSON.stringify({ candidate_id: "q2a-iter-0" }));
await writeFile(feedbackPath, JSON.stringify({
  data_boundary: { source: "select_validation", cert_accessed: false },
  objective: { name: "f1", value: 0.1 },
  trace_summary: { true_count: 0 },
}));
await writeFile(helpersPath, "def matches(value):\n    return bool(value)\n");
await writeFile(solverPath, "def solve():\n    return []\n");
await writeFile(diffPath, JSON.stringify({ mistakes: [] }));

const planner = await prepareStructuredRole(plannerConfig, {
  query_id: "q2a",
  query_sql: "SELECT * FROM images WHERE AI.IF(track, image)",
  query_nl: "match track logos",
  modality: "image",
  tables_doc: "- images.csv",
  planner_table_profile: {
    profile_version: "1.0",
    tables: [{ table: "images", sample_rows: 2 }],
    same_name_join_candidates: [],
  },
  trace_contract: "pair key",
  local_primitive_files: `- ${primitivePath}`,
  plan_schema_path: resolve("src/semdb/contracts/semantic-plan.schema.json"),
  plan_path: resolve(root, "plan.json"),
  previous_plan_path: previousPlanPath,
  previous_plan_version: 1,
  required_plan_version: 2,
  required_parent_plan_version: 1,
  optimizer_action_path: actionPath,
  planner_lint: "",
}, {});
assert.equal(planner.maxOutputTokens, 24_000);
assert.equal(planner.retryMaxOutputTokens, 32_000);
assert.equal(planner.effortLevel, "medium");
assert.ok(planner.systemPrompt.includes("# Plan Semantic Query"));
assert.ok(planner.systemPrompt.includes("You have no tools"));
assert.ok(!planner.systemPrompt.includes("You are the SemDB Semantic Query Planner"),
  "structured mode must not duplicate the legacy system procedure");
assert.ok(planner.userPrompt.includes("Required plan_version: 2"));
assert.ok(planner.userPrompt.includes("runtime_contract"));
assert.ok(planner.userPrompt.includes('"action":"REPLAN"'));
assert.ok(planner.userPrompt.includes("def classify("));
assert.ok(!planner.userPrompt.includes("return image.classify("));
assert.ok(planner.userPrompt.includes("runtime_table_profile_json"));
assert.ok(planner.userPrompt.includes('"sample_rows": 2'));

const agentPlanner = await prepareAgentRole(plannerConfig, {
  query_id: "q2a",
  query_sql: "SELECT * FROM images WHERE AI.IF(track, image)",
  query_nl: "match track logos",
  modality: "image",
  tables_doc: "- images.csv",
  planner_table_profile: {
    profile_version: "2.0",
    scan_mode: "full_file_streaming",
    tables: [{ table: "images", rows_scanned: 200 }],
    same_name_join_candidates: [],
  },
  trace_contract: "pair key",
  local_primitive_files: `- ${primitivePath}`,
  plan_schema_path: resolve("src/semdb/contracts/semantic-plan.schema.json"),
  plan_path: resolve(root, "agent-plan.json"),
  previous_plan_path: previousPlanPath,
  previous_plan_version: 1,
  required_plan_version: 2,
  required_parent_plan_version: 1,
  optimizer_action_path: actionPath,
  planner_lint: "",
  memory_catalog: "UNRELATED SKILL CATALOG",
});
assert.ok(agentPlanner.systemPrompt.includes("# Plan Semantic Query"));
assert.ok(agentPlanner.systemPrompt.includes("procedure above is already loaded"));
assert.ok(!agentPlanner.systemPrompt.includes("You have no tools"));
assert.ok(!agentPlanner.systemPrompt.includes("You are the SemDB Semantic Query Planner"));
assert.ok(agentPlanner.contextText.includes("full_file_streaming"));
assert.ok(agentPlanner.contextText.includes("output_json_schema"));
assert.ok(agentPlanner.contextText.includes("agent-plan.json"));
assert.ok(!agentPlanner.contextText.includes("UNRELATED SKILL CATALOG"),
  "agent discovery advertises selected learned skills; the bundle must not duplicate its catalog");
assert.match(agentPlanner.contextSha256, /^[a-f0-9]{64}$/);
const bundlePath = resolve(root, "_agent_context_query_planner.md");
await writeTextAtomic(bundlePath, agentPlanner.contextText, { mode: 0o444 });
assert.equal((await stat(bundlePath)).mode & 0o777, 0o444);
assert.equal(await readFile(bundlePath, "utf8"), agentPlanner.contextText);

const optimizer = await prepareStructuredRole(optimizerConfig, {
  query_id: "q2a",
  plan_path: previousPlanPath,
  candidate_manifest_path: manifestPath,
  iteration_feedback_path: feedbackPath,
  candidate_helpers_path: helpersPath,
  candidate_solver_path: solverPath,
  candidate_diff_path: diffPath,
  history_manifest_paths: "(none)",
  optimizer_action_path: resolve(root, "optimizer-action.json"),
  optimizer_action_schema_path: resolve("src/semdb/contracts/optimizer-action.schema.json"),
  local_primitive_files: `- ${primitivePath}`,
  remaining_iteration_budget: 1,
  remaining_replan_budget: 1,
}, {});
assert.equal(optimizer.maxOutputTokens, 12_000);
assert.equal(optimizer.retryMaxOutputTokens, 16_000);
assert.equal(optimizer.effortLevel, "medium");
assert.ok(optimizer.systemPrompt.includes("# Optimize Semantic Program"));
assert.ok(optimizer.userPrompt.includes("q2a-iter-0"));
assert.ok(optimizer.userPrompt.includes("select_validation"));
assert.ok(optimizer.userPrompt.includes('"diagnosis":{"category"'));
assert.ok(optimizer.userPrompt.includes("helpers:L<number>"));
assert.ok(optimizer.userPrompt.includes("L1 | def matches(value):"));
assert.ok(optimizer.userPrompt.includes("L1 | def solve():"));
assert.ok(optimizer.userPrompt.includes("candidate_validation_diff_json"));

const agentOptimizer = await prepareAgentRole(optimizerConfig, {
  query_id: "q2a",
  plan_path: previousPlanPath,
  candidate_manifest_path: manifestPath,
  iteration_feedback_path: feedbackPath,
  candidate_helpers_path: helpersPath,
  candidate_solver_path: solverPath,
  candidate_diff_path: diffPath,
  history_manifest_paths: "(none)",
  optimizer_action_path: resolve(root, "agent-optimizer-action.json"),
  optimizer_action_schema_path: resolve("src/semdb/contracts/optimizer-action.schema.json"),
  local_primitive_files: `- ${primitivePath}`,
  remaining_iteration_budget: 1,
  remaining_replan_budget: 1,
});
assert.ok(agentOptimizer.systemPrompt.includes("# Optimize Semantic Program"));
assert.ok(agentOptimizer.contextText.includes("L1 | def matches(value):"));
assert.ok(agentOptimizer.contextText.includes("candidate_validation_diff_json"));
assert.ok(agentOptimizer.contextText.includes("agent-optimizer-action.json"));

const availableSkills = [
  { name: "plan-semantic-query", role: true },
  { name: "generate-semantic-program", role: true },
  { name: "optimize-semantic-program", role: true },
  { name: "semdb-relevant", role: false },
  { name: "semdb-unrelated", role: false },
];
assert.deepEqual(
  selectAgentSkills(plannerConfig, availableSkills, ["semdb-relevant"], {
    procedureInSystem: true,
  }).map((skill) => skill.name),
  ["semdb-relevant"],
  "a canonical Planner must not rediscover its role skill or unrelated learned skills",
);
assert.deepEqual(
  selectAgentSkills(generatorConfig, availableSkills, ["semdb-relevant"], {
    procedureInSystem: false,
  }).map((skill) => skill.name),
  ["generate-semantic-program", "semdb-relevant"],
  "Generator retains its role procedure plus retrieval-selected learned skills",
);

const citedPatch = {
  action: "PATCH_CODE",
  diagnosis: {
    summary: "The solver returns before writing results.",
    evidence: ["solver:L2 returns an empty list unconditionally"],
  },
  targets: [{ artifact: "solver", symbol: "solve", intent: "write planned results" }],
};
await validateStructuredOptimizerEvidence(citedPatch, {
  candidate_helpers_path: helpersPath,
  candidate_solver_path: solverPath,
  iteration_feedback_path: feedbackPath,
  remaining_replan_budget: 1,
});
await assert.rejects(
  validateStructuredOptimizerEvidence({
    ...citedPatch,
    diagnosis: {
      summary: "gate_pass means trace entries are true",
      evidence: ["solver:L2 writes 30 trace entries as true"],
    },
  }, {
    candidate_helpers_path: helpersPath,
    candidate_solver_path: solverPath,
    iteration_feedback_path: feedbackPath,
    remaining_replan_budget: 1,
  }),
  /true_count is 0/,
);
await assert.rejects(
  validateStructuredOptimizerEvidence({
    ...citedPatch,
    diagnosis: { summary: "guess", evidence: ["no source citation"] },
  }, {
    candidate_helpers_path: helpersPath,
    candidate_solver_path: solverPath,
    iteration_feedback_path: feedbackPath,
    remaining_replan_budget: 1,
  }),
  /must cite a valid solver:L<number>/,
);

const request = buildStructuredResponsesRequest({
  model: "qwen3.8",
  systemPrompt: optimizer.systemPrompt,
  userPrompt: optimizer.userPrompt,
  schema: optimizer.schema,
  schemaName: optimizer.schemaName,
  maxOutputTokens: optimizer.maxOutputTokens,
  reasoningEffort: optimizer.effortLevel,
});
assert.equal(request.max_output_tokens, 12_000);
assert.equal(request.reasoning.effort, "medium");
assert.equal(request.text.format.type, "json_schema");
assert.equal(request.text.format.strict, true);
assert.deepEqual(request.tools, undefined);

assert.ok(supportsStructuredRole("query_planner"));
assert.ok(supportsStructuredRole("semantic_optimizer"));
assert.ok(!supportsStructuredRole("semantic_code_generator"));
assert.equal(parseArgs(["node", "orchestrator"]).agentExecution, "agent");
assert.equal(parseArgs(["node", "orchestrator", "--agent-execution", "agent"]).agentExecution, "agent");
assert.equal(
  parseArgs(["node", "orchestrator", "--structured-reasoning-effort", "medium"])
    .structuredReasoningEffort,
  "medium",
);
assert.throws(
  () => parseArgs(["node", "orchestrator", "--agent-execution", "invalid"]),
  /--agent-execution/,
);
assert.throws(
  () => parseArgs(["node", "orchestrator", "--structured-reasoning-effort", "minimal"]),
  /low, medium, or xhigh/,
);

console.log("test_structured_roles OK");
