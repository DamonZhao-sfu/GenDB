import assert from "node:assert";
import { renderFeedback } from "../orchestrator.mjs";

// per-row branch: diff has `mistakes`
const out = renderFeedback({
  status: "ok", f1: 0.75,
  metrics: { accuracy: 0.75, n: 4, correct: 3 },
  diff: { n_mistakes: 1, mistakes: [
    { id: "m4", text: "a dark thriller", predicted: "comedy", expected: "drama" },
  ] },
  history: [{ iter: 0, f1: 0.75, status: "ok", improved: true }],
});
assert.ok(out.includes("PER-ROW INFERENCE"), "has per-row header");
assert.ok(out.includes("accuracy=0.75"), "shows accuracy");
assert.ok(out.includes("m4") && out.includes("predicted=comedy") && out.includes("expected=drama"),
  "shows the mislabeled row");
assert.ok(!out.includes("FALSE POSITIVES"), "does NOT use the F1 block");

// F1 branch still works (no `mistakes` key)
const f1out = renderFeedback({
  status: "ok", f1: 0.6, metrics: { precision: 0.5, recall: 0.7 },
  diff: { fp_total: 2, false_positives: ["x"], fn_total: 1, false_negatives: ["z"] },
  history: [],
});
assert.ok(f1out.includes("FALSE POSITIVES"), "F1 block preserved");

const ariOut = renderFeedback({
  status: "ok", f1: null,
  objective: {
    name: "adjusted_rand_index", value: 0.7, direction: "maximize",
    details: { metric_type: "adjusted-rand-index", accuracy: 0.7, n: 20 },
  },
  metrics: {
    metric: "adjusted-rand-index", metric_family: "SingleAccuracyScore",
    metric_type: "adjusted-rand-index", accuracy: 0.7, adjusted_rand_index: 0.7,
  },
  diff: { fp_total: 0, false_positives: [], fn_total: 0, false_negatives: [] },
  history: [],
});
assert.ok(ariOut.includes("QUERY METRIC adjusted_rand_index=0.7"),
  "shows the true non-F1 objective");
assert.ok(ariOut.includes('"metric_type": "adjusted-rand-index"'),
  "shows the SemBench metric type");
assert.ok(!ariOut.includes("FALSE POSITIVES"), "does not render meaningless id-set F1 feedback");
console.log("test_val_feedback OK");
