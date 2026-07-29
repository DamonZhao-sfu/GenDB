import assert from "node:assert";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { renderFeedback } from "../orchestrator.mjs";
import { buildIterationFeedback } from "../agent-runtime/feedback.mjs";
import {
  readAndValidateFeedback,
  writeJsonAtomic,
} from "../agent-runtime/contracts.mjs";

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

const structuredAri = buildIterationFeedback({
  query: { query_id: "q3" },
  candidate: { candidate_id: "q3-iter-0", iteration: 0 },
  runOutcome: { status: "ok", execMs: 12 },
  scoreOutcome: {
    status: "ok",
    f1: 0,
    objective: {
      name: "adjusted_rand_index",
      value: 0.8,
      direction: "maximize",
      details: {
        metric_type: "adjusted-rand-index",
        accuracy: 0.8,
        n: 20,
      },
    },
    metrics: {
      accuracy: 0.25,
      n: 20,
      correct: 5,
      quality: {
        precision: 0.4,
        recall: 0.5,
        f1: 0.44,
        accuracy: 0.25,
      },
    },
    diff: {
      n_mistakes: 15,
      mistakes: [{ id: "1", predicted: "a", expected: "nike" }],
    },
  },
  dataBoundary: {
    source: "select_validation",
    cert_accessed: false,
    full_ground_truth_accessed: false,
  },
});
assert.equal(structuredAri.objective.name, "adjusted_rand_index");
assert.equal(structuredAri.objective.value, 0.8);
assert.equal(structuredAri.objective.scope, "query_metric");
assert.equal(structuredAri.objective.details.metric_type, "adjusted-rand-index");
assert.equal(structuredAri.objective.precision, null,
  "binary operator precision must not be presented as ARI precision");
assert.equal(structuredAri.operator_fidelity.f1, 0.44,
  "operator fidelity remains available as a separate diagnostic");
assert.equal(structuredAri.errors.kind, "label_mismatch");
assert.equal(structuredAri.errors.false_positive_total, null,
  "ARI feedback must not invent binary false-positive totals");
assert.equal(structuredAri.errors.mismatch_total, 15);
const feedbackDir = await mkdtemp(resolve(tmpdir(), "semdb-metric-feedback-"));
const feedbackPath = resolve(feedbackDir, "iteration_feedback.json");
await writeJsonAtomic(feedbackPath, structuredAri);
assert.equal(
  (await readAndValidateFeedback(feedbackPath, {
    queryId: "q3",
    candidateId: "q3-iter-0",
  })).objective.details.metric_type,
  "adjusted-rand-index",
);
console.log("test_val_feedback OK");
