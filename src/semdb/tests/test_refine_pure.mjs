import assert from "node:assert";
import {
  checkSemdbImprovement,
  directTimingBreakdown,
  shouldContinueSemdb,
} from "../orchestrator.mjs";

// improvement rule
assert.equal(checkSemdbImprovement({ status: "ok", f1: 0.4 }, { status: "ok", f1: 0.6 }), true);
assert.equal(checkSemdbImprovement({ status: "ok", f1: 0.6 }, { status: "ok", f1: 0.6 }), false);
assert.equal(checkSemdbImprovement({ status: "ok", f1: 0.6 }, { status: "crash", f1: null }), false);
assert.equal(checkSemdbImprovement({ status: "crash", f1: null }, { status: "ok", f1: 0.1 }), true);

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
