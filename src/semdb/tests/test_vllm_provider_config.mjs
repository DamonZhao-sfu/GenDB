import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import Ajv2020 from "ajv/dist/2020.js";

import { defaults as gendbDefaults } from "../../gendb/gendb.config.mjs";
import {
  getAvailableProviders,
  setAgentProvider,
} from "../../gendb/providers/index.mjs";
import {
  canonicalizeStructuredOutput,
  parseStructuredResponseText,
  selectServedModel,
} from "../../gendb/providers/vllm.mjs";
import {
  defaults as semdbDefaults,
  getAgentModel,
  getProviderConfig,
} from "../semdb.config.mjs";

const model = "Qwen/Qwen3.8-27B-FP8";

assert.ok(getAvailableProviders().includes("vllm"));
assert.doesNotThrow(() => setAgentProvider("vllm"));

assert.equal(gendbDefaults.providers.vllm.model, model);
assert.equal(gendbDefaults.providers.vllm.baseUrl, "http://localhost:8000/v1");
assert.ok(
  Object.values(gendbDefaults.providers.vllm.agentEffortLevels)
    .every((effort) => effort === "medium"),
  "every GenDB vLLM agent should default to medium effort",
);
assert.equal(gendbDefaults.providers.vllm.escalationEffortLevel, "medium");
assert.equal(gendbDefaults.providers.vllm.singleAgent.effortLevel, "medium");
assert.equal(semdbDefaults.agentProvider, "vllm");
assert.equal(semdbDefaults.agentExecution, "agent");
assert.equal(getProviderConfig("vllm").model, model);
assert.ok(
  Object.values(getProviderConfig("vllm").agentEffortLevels)
    .every((effort) => effort === "medium"),
  "every SemDB vLLM agent should default to medium effort",
);
for (const setting of [
  "smallImageModel",
  "strongImageModel",
  "smallTextModel",
  "escalationImageModel",
  "captionModel",
]) {
  assert.equal(semdbDefaults.extraction[setting], model, `${setting} should use local Qwen`);
}
assert.equal(selectServedModel(model, { data: [{ id: model }] }), model);
assert.equal(
  selectServedModel(model, { data: [{ id: "qwen3.8", root: model }] }),
  "qwen3.8",
);
assert.throws(
  () => selectServedModel(model, { data: [{ id: "another-model" }] }),
  /does not serve/,
);

assert.deepEqual(
  parseStructuredResponseText('```json\n{"action":"STOP"}\n```'),
  { action: "STOP" },
  "a single JSON presentation fence should not force a model retry",
);
assert.throws(
  () => parseStructuredResponseText('commentary\n```json\n{"action":"STOP"}\n```'),
  /Unexpected token|Unexpected non-whitespace character/,
  "surrounding prose is not a harmless presentation wrapper",
);

const patchAction = {
  schema_version: "1.0",
  query_id: "q2a",
  candidate_id: "q2a-iter-0",
  action: "PATCH_CODE",
  replan_reason: "",
  diagnosis: { category: "implementation", summary: "mismatch", evidence: [] },
  targets: [{ artifact: "solver", symbol: "solve", intent: "repair mismatch" }],
  preserve: ["plan semantics"],
  expected_effect: { primary_metric: "f1", direction: "increase", risk: "low" },
};
const canonicalPatch = canonicalizeStructuredOutput(patchAction, {
  schemaName: "semdb_optimizer_action",
});
assert.equal(Object.hasOwn(canonicalPatch.value, "replan_reason"), false);
assert.equal(Object.hasOwn(patchAction, "replan_reason"), true,
  "canonicalization must not mutate the provider response object");
assert.deepEqual(canonicalPatch.changes, ["removed_replan_reason_for_non_replan_action"]);

const optimizerSchema = JSON.parse(await readFile(
  new URL("../contracts/optimizer-action.schema.json", import.meta.url),
  "utf8",
));
const validateOptimizer = new Ajv2020({ allErrors: true, strict: true })
  .compile(optimizerSchema);
assert.equal(validateOptimizer(patchAction), false,
  "the raw harmless extra field reproduces the schema fallback");
assert.equal(validateOptimizer(canonicalPatch.value), true,
  "canonicalization should avoid retry/fallback without changing PATCH_CODE semantics");

const replanAction = { action: "REPLAN", replan_reason: "wrong sampling unit" };
assert.deepEqual(
  canonicalizeStructuredOutput(replanAction, { configName: "semantic_optimizer" }),
  { value: replanAction, changes: [] },
  "a semantic REPLAN reason must never be rewritten",
);

for (const role of [
  "schema_designer",
  "extractor",
  "code_generator",
  "query_planner",
  "semantic_code_generator",
  "semantic_optimizer",
  "memory_manager",
]) {
  assert.equal(getAgentModel(role, "vllm"), model, `${role} should use local Qwen`);
}

console.log("vLLM provider configuration tests passed");
