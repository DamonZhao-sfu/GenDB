import assert from "node:assert/strict";

import {
  buildDeterministicRepairAction,
  routeOptimizerAction,
} from "../agent-runtime/pgo-loop.mjs";

assert.equal(routeOptimizerAction({
  action: { action: "PATCH_CODE", candidate_id: "q-iter-0" },
  currentCandidateId: "q-iter-0",
  replansUsed: 0,
  maxReplans: 1,
}).route, "generator", "compile failure patches code without invoking Planner");

assert.equal(routeOptimizerAction({
  action: { action: "REPLAN", candidate_id: "q-iter-0" },
  currentCandidateId: "q-iter-0",
  replansUsed: 0,
  maxReplans: 1,
}).route, "planner_then_generator");

assert.deepEqual(routeOptimizerAction({
  action: { action: "REPLAN", candidate_id: "q-iter-0" },
  currentCandidateId: "q-iter-0",
  replansUsed: 1,
  maxReplans: 1,
}), { route: "stop", reason: "replan_budget_exhausted" });

assert.equal(routeOptimizerAction({
  action: { action: "STOP", candidate_id: "q-iter-0" },
  currentCandidateId: "q-iter-0",
}).route, "stop");

assert.throws(() => routeOptimizerAction({
  action: { action: "PATCH_CODE", candidate_id: "wrong" },
  currentCandidateId: "q-iter-0",
}), /candidate id mismatch/);

const compileRepair = buildDeterministicRepairAction({
  candidate: { candidate_id: "q1-iter-0" },
  feedback: {
    query_id: "q1",
    execution: { status: "crash", stage: "compile", stderr_tail: "NameError: warnings" },
    trace_summary: { status: "missing" },
    objective: { name: "f1" },
  },
});
assert.equal(compileRepair.action, "PATCH_CODE");
assert.equal(compileRepair.candidate_id, "q1-iter-0");
assert.equal(compileRepair.diagnosis.category, "deterministic_preflight_failure");
assert.equal(buildDeterministicRepairAction({
  candidate: { candidate_id: "q1-iter-0" },
  feedback: {
    query_id: "q1",
    execution: { status: "ok", stage: null },
    trace_summary: { status: "ok" },
  },
}), null);

console.log("test_optimizer_routing OK");
