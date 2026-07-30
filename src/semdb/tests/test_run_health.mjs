/** Execution health must separate "ran fine, matched nothing" from "threw on every row".
 *
 *  mmqa q2a: the solver passed a bare ClipEncoder where ImagePatch wanted the ctx
 *  mapping, so all 200 images raised TypeError inside the mandated per-row try/except.
 *  The program still exited 0 and still wrote a well-formed trace, so the loop scored it
 *  "ok" with objective 0.0. The generator FIXED it in iter_1 (0 errors, 1 row out) but
 *  0.0 > 0.0 is false, so `checkSemdbImprovement` rolled back and promoted the crashing
 *  iter_0 — F1 0, pred_count 0, four iterations of budget spent.
 *
 *  Log fixtures below are verbatim from that run.
 */
import assert from "node:assert";
import {
  checkSemdbImprovement, parseRunHealth, runErrorRate, runIsBroken,
} from "../orchestrator.mjs";

// --- verbatim from runs/mmqa_mem_demo/mmqa-q2a/iter_{0,1}/run.log -------------------
const ITER0 = [
  "[solve] WARN inference_error id=0-1sz0kf8wcmj0q8n3pu6mg61gl158vvz1.png type=TypeError",
  "[solve] branch=error_path n=74",
  "[solve] branch=equality_fail n=56",
  "[solve] WARN-TOTAL inference_error n=74",
  "[solve] WARN-TOTAL predicate_miss n=56",
  "[solve] rows_in=2600 rows_out=0 elapsed=5.8s",
].join("\n");

const ITER1 = [
  "[solve] branch=equality_fail n=129",
  "[solve] branch=cache_miss n=74",
  "[solve] branch=classify_match n=9",
  "[solve] branch=equality_pass n=1",
  "[solve] WARN-TOTAL predicate_miss n=129",
  "[solve] rows_in=2600 rows_out=1 elapsed=6.0s",
].join("\n");

const h0 = parseRunHealth(ITER0);
const h1 = parseRunHealth(ITER1);

assert.deepEqual(h0, { rowsIn: 2600, rowsOut: 0, errorRows: 74 });
// No error marker BUT a summary line present ⇒ zero errors, not "unknown". Collapsing
// those two would make a clean run incomparable to a failing one.
assert.deepEqual(h1, { rowsIn: 2600, rowsOut: 1, errorRows: 0 });

// The contract's own `errors=` field wins over the inferred counters.
assert.equal(parseRunHealth("[solve] rows_in=100 rows_out=3 errors=7 elapsed=1s").errorRows, 7);
// A program that said nothing stays unknown — never punished for silence.
assert.deepEqual(parseRunHealth("some unrelated output"),
  { rowsIn: null, rowsOut: null, errorRows: null });
assert.equal(runErrorRate(parseRunHealth("nothing")), null);

// iter_0 produced nothing AND lost rows to exceptions. The plain rate test cannot catch
// it — a join counts rows_in as its PAIR domain (2600) while errors are per image (74),
// which rates under 3% — so the empty-with-errors rule is what fires.
assert.equal(runIsBroken(h0), true, "all-errors run must be broken");
assert.equal(runIsBroken(h1), false, "healthy run must not be broken");
// An empty result with NO errors is a legitimate finding, not a crash.
assert.equal(runIsBroken(parseRunHealth("[solve] rows_in=200 rows_out=0 errors=0")), false);
// Majority-failure still trips the rate rule even when some rows came out.
assert.equal(runIsBroken(parseRunHealth("[solve] rows_in=200 rows_out=5 errors=150")), true);

// --- the promotion decision --------------------------------------------------------
const cand = (status, health) => ({
  status, f1: 0, objective: { name: "f1", value: 0, direction: "maximize" }, health,
});

// Once iter_0 is classified a crash, the pre-existing "fixed a crash/empty" rule applies.
assert.equal(checkSemdbImprovement(cand("crash", h0), cand("ok", h1)), true,
  "the repaired iteration must win");
assert.equal(checkSemdbImprovement(cand("ok", h1), cand("crash", h0)), false,
  "a crash must never win back");

// Health also breaks a tie between two runs that both scored 0.0 and both count as ok.
const noisy = parseRunHealth("[solve] rows_in=200 rows_out=4 errors=40");
const clean = parseRunHealth("[solve] rows_in=200 rows_out=4 errors=0");
assert.equal(checkSemdbImprovement(cand("ok", noisy), cand("ok", clean)), true);
assert.equal(checkSemdbImprovement(cand("ok", clean), cand("ok", noisy)), false);

// Equal health and equal objective is genuinely no improvement — do not churn.
assert.equal(checkSemdbImprovement(cand("ok", clean), cand("ok", clean)), false);

// A real objective gain still wins regardless of health.
const better = { status: "ok", f1: 0.4, health: noisy,
                 objective: { name: "f1", value: 0.4, direction: "maximize" } };
assert.equal(checkSemdbImprovement(cand("ok", clean), better), true);

// Unknown health on either side must not flip a tie either way.
assert.equal(checkSemdbImprovement(cand("ok", clean), cand("ok", parseRunHealth(""))), false);

console.log("test_run_health OK");
