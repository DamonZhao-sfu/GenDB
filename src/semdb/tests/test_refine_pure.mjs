import assert from "node:assert";
import {
  checkSemdbImprovement,
  directTimingBreakdown,
  sanitizeAgentQueryMetadata,
  semdbObjective,
  shouldContinueSemdb,
} from "../orchestrator.mjs";

// improvement rule
assert.equal(checkSemdbImprovement({ status: "ok", f1: 0.4 }, { status: "ok", f1: 0.6 }), true);
assert.equal(checkSemdbImprovement({ status: "ok", f1: 0.6 }, { status: "ok", f1: 0.6 }), false);
assert.equal(checkSemdbImprovement({ status: "ok", f1: 0.6 }, { status: "crash", f1: null }), false);
assert.equal(checkSemdbImprovement({ status: "crash", f1: null }, { status: "ok", f1: 0.1 }), true);
assert.equal(checkSemdbImprovement(
  { status: "ok", objective: { name: "relative_error", value: 0.4, direction: "minimize" } },
  { status: "ok", objective: { name: "relative_error", value: 0.2, direction: "minimize" } },
), true, "a lower minimization objective improves");
assert.equal(checkSemdbImprovement(
  { status: "ok", objective: { name: "ari", value: 0.4, direction: "maximize" } },
  { status: "ok", objective: { name: "ari", value: 0.2, direction: "maximize" } },
), false, "a lower maximization objective regresses");
assert.equal(checkSemdbImprovement(
  { status: "ok", objective: { name: "ari", value: 0.4, direction: "maximize" } },
  { status: "ok", objective: { name: "f1", value: 0.9, direction: "maximize" } },
), false, "objectives with different semantics are never compared");
assert.equal(checkSemdbImprovement(
  { status: "ok", f1: 0.8, objective: { name: "top1", value: 1, direction: "maximize" } },
  { status: "ok", f1: 1.0, objective: { name: "top1", value: 1, direction: "maximize" } },
), true, "operator fidelity breaks an exact query-objective tie");
assert.equal(checkSemdbImprovement(
  { status: "ok", f1: null, objective: { name: "ari", value: null, direction: "maximize" } },
  { status: "ok", f1: null, objective: { name: "ari", value: null, direction: "maximize" } },
), false, "null is not a numeric objective");
assert.deepEqual(
  semdbObjective({
    f1: 0.99,
    objective: {
      name: "query_metric_unavailable",
      value: null,
      direction: "maximize",
      details: { reason: "multi-site" },
    },
  }),
  {
    name: "query_metric_unavailable",
    value: null,
    direction: "maximize",
    details: { reason: "multi-site" },
  },
  "an explicit unavailable objective never falls back to legacy F1",
);
const safeIntent = sanitizeAgentQueryMetadata({
  nl_question: "Which images contain the logo?",
  modalities: ["table", "image"],
  ground_truth: [["secret-id", "secret.png"]],
});
assert.equal(safeIntent, "Which images contain the logo?");
assert.ok(!safeIntent.includes("secret"),
  "agent-facing natural-language metadata excludes colocated ground truth");
assert.equal(
  sanitizeAgentQueryMetadata({ ground_truth: ["secret"] }),
  "(natural-language intent unavailable; use the SQL)",
);

// stop logic
assert.equal(shouldContinueSemdb([{ iter: 0, f1: 1.0, status: "ok", improved: true }], 1, 5, 2).action, "continue",
  "perfect validation F1 must not shorten the requested run");
assert.equal(shouldContinueSemdb([{ iter: 0, f1: 0.5, status: "ok", improved: true }], 6, 5, 2).action, "stop");
assert.equal(shouldContinueSemdb([{ iter: 0, f1: null, status: "crash", improved: false }], 1, 5, 2).action, "continue");
const stalled = [
  { iter: 0, f1: 0.5, status: "ok", improved: true },
  { iter: 1, f1: 0.5, status: "ok", improved: false },
  { iter: 2, f1: 0.5, status: "ok", improved: false },
];
assert.equal(shouldContinueSemdb(stalled, 3, 5, 2).action, "continue",
  "stalled validation F1 must not shorten the requested run");
assert.equal(shouldContinueSemdb([{ iter: 0, f1: 0.5, status: "ok", improved: true }], 1, 5, 2).action, "continue");

// DIRECT timing must count only actual generated-program runs as code execution.
assert.deepEqual(
  directTimingBreakdown(1000, 700, 40, [
    { duration_ms: 60 },
    { duration_ms: 140 },
  ]),
  {
    agent_stage_ms: 700,
    validation_sampling_llm_ms: 40,
    code_execution_ms: 200,
    other_overhead_ms: 60,
  },
);
console.log("test_refine_pure OK");
