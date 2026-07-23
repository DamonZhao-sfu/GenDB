import assert from "node:assert";
import { checkSemdbImprovement, shouldContinueSemdb } from "../orchestrator.mjs";

// improvement rule
assert.equal(checkSemdbImprovement({ status: "ok", f1: 0.4 }, { status: "ok", f1: 0.6 }), true);
assert.equal(checkSemdbImprovement({ status: "ok", f1: 0.6 }, { status: "ok", f1: 0.6 }), false);
assert.equal(checkSemdbImprovement({ status: "ok", f1: 0.6 }, { status: "crash", f1: null }), false);
assert.equal(checkSemdbImprovement({ status: "crash", f1: null }, { status: "ok", f1: 0.1 }), true);

// stop logic
assert.equal(shouldContinueSemdb([{ iter: 0, f1: 1.0, status: "ok", improved: true }], 1, 5, 2).action, "stop");
assert.equal(shouldContinueSemdb([{ iter: 0, f1: 0.5, status: "ok", improved: true }], 6, 5, 2).action, "stop");
assert.equal(shouldContinueSemdb([{ iter: 0, f1: null, status: "crash", improved: false }], 1, 5, 2).action, "continue");
const stalled = [
  { iter: 0, f1: 0.5, status: "ok", improved: true },
  { iter: 1, f1: 0.5, status: "ok", improved: false },
  { iter: 2, f1: 0.5, status: "ok", improved: false },
];
assert.equal(shouldContinueSemdb(stalled, 3, 5, 2).action, "stop");
assert.equal(shouldContinueSemdb([{ iter: 0, f1: 0.5, status: "ok", improved: true }], 1, 5, 2).action, "continue");
console.log("test_refine_pure OK");
