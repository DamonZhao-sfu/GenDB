import assert from "node:assert/strict";

import { routeOptimizerAction } from "../agent-runtime/pgo-loop.mjs";

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

console.log("test_optimizer_routing OK");
