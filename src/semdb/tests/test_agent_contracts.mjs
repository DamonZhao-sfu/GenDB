import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import {
  assertPlanGeneratable,
  finalizeCandidateManifest,
  readAndValidateFeedback,
  readAndValidateManifest,
  readAndValidateOptimizerAction,
  readAndValidatePlan,
  writeJsonAtomic,
} from "../agent-runtime/contracts.mjs";
import { assertSelectValidationPayload } from "../agent-runtime/feedback.mjs";

const root = await mkdtemp(resolve(tmpdir(), "semdb-contracts-"));
const planPath = resolve(root, "plan.json");
const helpersPath = resolve(root, "_semantic_helpers_q.py");
const solverPath = resolve(root, "solve_q.py");
const manifestPath = resolve(root, "candidate_manifest.json");
const feedbackPath = resolve(root, "iteration_feedback.json");
const actionPath = resolve(root, "optimizer_action.json");

const plan = {
  schema_version: "1.0",
  query_id: "q",
  plan_version: 1,
  parent_plan_version: null,
  modality: "image",
  compilability: { class: "exact", obligations: [], unresolved: [] },
  semantic_sites: [{
    site_id: "site_0",
    operator: "AI_FILTER",
    predicate: "contains a cat",
    inputs: [{ table: "images", column: "filename", type: "image_ref" }],
    sampling_unit: "row",
    output_type: "boolean",
    value_space: null,
  }],
  helper_dag: [{
    helper_id: "h0",
    name: "matches_predicate",
    args: [{ name: "image_path", type: "path" }],
    return_type: "boolean",
    depends_on: [],
    primitive_steps: [{
      primitive: "classify",
      source: "src/semdb/vadar/predefined.py",
      inputs: ["image_path"],
      output_type: "label",
    }],
    confidence_signal: {
      source: "primitive_score",
      range: [0, 1],
      higher_is_more_confident: true,
    },
  }],
  relational_plan: [{ operator: "filter", site_id: "site_0" }],
  trace_contract: { sampling_unit: "row", key: "id" },
  runtime_contract: { offline: true },
  validation_contract: { source: "select_validation" },
  assumptions: [],
  invariants: ["offline-only runtime"],
};
await writeJsonAtomic(planPath, plan);
await writeFile(helpersPath, "def matches_predicate(path): return True\n");
await writeFile(solverPath, "print('ok')\n");

assert.deepEqual(await readAndValidatePlan(planPath, { queryId: "q" }), plan);

const manifest = await finalizeCandidateManifest(manifestPath, {
  candidate_id: "q-iter-0",
  query_id: "q",
  iteration: 0,
  plan_version: 1,
  parent_candidate_id: null,
  trigger_action: "INITIAL",
  artifacts: {
    plan: "plan.json",
    helpers: "_semantic_helpers_q.py",
    solver: "solve_q.py",
  },
}, { queryId: "q", candidateId: "q-iter-0" });
assert.equal(
  (await readAndValidateManifest(manifestPath)).candidate_id,
  manifest.candidate_id,
);

const feedback = {
  schema_version: "1.0",
  query_id: "q",
  candidate_id: "q-iter-0",
  iteration: 0,
  execution: {
    status: "ok",
    stage: "score",
    preflight: null,
    stderr_tail: "",
    runtime_ms: 10,
  },
  objective: {
    name: "f1",
    value: 0.5,
    precision: 0.5,
    recall: 0.5,
    weighted: true,
    direction: "maximize",
    scope: "query_metric",
    details: {
      metric_family: "QueryMetricRetrieval",
      f1_score: 0.5,
    },
  },
  operator_fidelity: {
    accuracy: 0.5,
    precision: 0.5,
    recall: 0.5,
    f1: 0.5,
    n: 10,
    correct: 5,
    weighted: true,
  },
  errors: {
    kind: "classification",
    false_positive_total: 1,
    false_negative_total: 1,
    mismatch_total: 2,
    mistakes: [],
  },
  runtime_branches: {},
  history: [],
  data_boundary: {
    source: "select_validation",
    cert_accessed: false,
    full_ground_truth_accessed: false,
  },
};
await writeJsonAtomic(feedbackPath, feedback);
assert.equal(
  (await readAndValidateFeedback(feedbackPath, {
    queryId: "q",
    candidateId: "q-iter-0",
  })).objective.value,
  0.5,
);

const action = {
  schema_version: "1.0",
  query_id: "q",
  candidate_id: "q-iter-0",
  action: "PATCH_CODE",
  diagnosis: {
    category: "threshold_or_mapping",
    summary: "Recall is low",
    evidence: ["feedback.errors.false_negative_total"],
  },
  targets: [{
    artifact: "helpers",
    symbol: "matches_predicate",
    intent: "broaden the positive condition",
  }],
  preserve: ["trace key format"],
  expected_effect: {
    primary_metric: "recall",
    direction: "increase",
    risk: "precision may decrease",
  },
};
await writeJsonAtomic(actionPath, action);
assert.equal(
  (await readAndValidateOptimizerAction(actionPath, {
    queryId: "q",
    candidateId: "q-iter-0",
  })).action,
  "PATCH_CODE",
);

await writeJsonAtomic(planPath, { ...plan, schema_version: "2.0" });
await assert.rejects(() => readAndValidatePlan(planPath), /schema_version/);
await writeJsonAtomic(planPath, { ...plan, modality: "audio" });
await assert.rejects(() => readAndValidatePlan(planPath), /modality/);
await writeJsonAtomic(planPath, plan);
await assert.rejects(
  () => readAndValidatePlan(planPath, { queryId: "other" }),
  /query id mismatch/,
);

await writeJsonAtomic(actionPath, { ...action, action: "REPLAN" });
await assert.rejects(
  () => readAndValidateOptimizerAction(actionPath),
  /replan_reason/,
);

await writeFile(helpersPath, "def changed(): return False\n");
await assert.rejects(
  () => readAndValidateManifest(manifestPath),
  /hash mismatch/,
);

assert.throws(
  () => assertPlanGeneratable({
    ...plan,
    compilability: {
      class: "not_compilable",
      obligations: [],
      unresolved: ["missing primitive"],
    },
  }),
  /cannot enter the Generator/,
);
assert.equal(
  assertSelectValidationPayload({ split: "select", labels: {} }).split,
  "select",
);
assert.throws(
  () => assertSelectValidationPayload({ split: "cert", labels: {} }),
  /requires a SELECT validation artifact/,
);
assert.throws(
  () => assertSelectValidationPayload({ labels: {} }),
  /split=<missing>/,
);

console.log("test_agent_contracts OK");
