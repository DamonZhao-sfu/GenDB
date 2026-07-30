/** A filter site bound to an argmax primitive with no abstention threshold.
 *
 *  `classify`/`classify_detail` always return one of `options` — they can never return
 *  "none" — so mmqa q7's `if classify_detail(...)[0] != "none"` was a guard that is
 *  always true. All 200 images were assigned an airline (196 of them not logos at all),
 *  the confidence it computed was discarded, and the query scored tp=0 fp=16. The plan
 *  had declared a `confidence_signal`; it just never named a threshold.
 */
import assert from "node:assert";
import { lintImagePlan } from "../agent-runtime/contracts.mjs";
import {
  Q2A_MULTILABEL_FILTER, Q7_ARGMAX_NO_THRESHOLD, clonePlan as clone,
} from "./fixtures/plans.mjs";

const q7 = Q7_ARGMAX_NO_THRESHOLD;
const hasArgmaxFinding = (p) =>
  lintImagePlan(p).some((f) => /argmax primitive and no abstention threshold/.test(f));

// The real shipped plan must be flagged.
assert.ok(hasArgmaxFinding(q7), "q7's plan must be flagged");

// Naming the cutoff clears it — that is the whole ask.
const withThreshold = clone(q7);
for (const h of withThreshold.helper_dag) {
  if (h.confidence_signal) h.confidence_signal.threshold = 0.45;
}
assert.ok(!hasArgmaxFinding(withThreshold), "a declared threshold must clear the finding");

// A threshold of 0 is still a deliberate choice, not a missing field.
const zero = clone(q7);
for (const h of zero.helper_dag) if (h.confidence_signal) h.confidence_signal.threshold = 0;
assert.ok(!hasArgmaxFinding(zero), "threshold 0 is explicit, not absent");

// Non-numeric junk in the field is not a threshold.
const junk = clone(q7);
for (const h of junk.helper_dag) if (h.confidence_signal) h.confidence_signal.threshold = "high";
assert.ok(hasArgmaxFinding(junk), "a non-numeric threshold must not satisfy the rule");

// A projection site is judged by a different metric — argmax there is correct.
const projection = clone(q7);
for (const s of projection.semantic_sites) s.output_type = "string";
assert.ok(!hasArgmaxFinding(projection), "only row-deciding sites are linted");

// best_ocr_match* genuinely returns "none" on a miss, so it needs no threshold.
const ocrOnly = clone(q7);
for (const h of ocrOnly.helper_dag) {
  h.primitive_steps = (h.primitive_steps || [])
    .filter((s) => String(s.primitive).startsWith("best_ocr_match"));
}
assert.ok(!hasArgmaxFinding(ocrOnly), "OCR matching can abstain on its own");

// classify_or_none is the other sanctioned fix.
const abstaining = clone(q7);
for (const h of abstaining.helper_dag) {
  h.primitive_steps = (h.primitive_steps || []).map((s) => (
    String(s.primitive) === "classify_detail" ? { ...s, primitive: "classify_or_none" } : s));
}
assert.ok(!hasArgmaxFinding(abstaining), "classify_or_none needs no separate threshold");

// A text-modality plan is out of scope for the image linter.
const asText = clone(q7);
asText.modality = "text";
assert.equal(lintImagePlan(asText).length, 0);

// --- classify_multi on a filter site -----------------------------------------------
// mmqa q2a bound classify_multi over the Track value space; every admitted image then
// matched every track (65 rows out, 0 correct). Independent per-label scoring is for
// attributes that co-occur, not for "which one of these".
const q2a = Q2A_MULTILABEL_FILTER;
const hasMultilabelFinding = (p) =>
  lintImagePlan(p).some((f) => /scores every option INDEPENDENTLY/.test(f));

assert.ok(hasMultilabelFinding(q2a), "classify_multi on a filter site must be flagged");

// Swapping to the abstaining argmax clears it.
const argmaxed = clone(q2a);
for (const h of argmaxed.helper_dag) {
  h.primitive_steps = (h.primitive_steps || []).map((s) => (
    String(s.primitive) === "classify_multi" ? { ...s, primitive: "classify_or_none" } : s));
}
assert.ok(!hasMultilabelFinding(argmaxed), "classify_or_none clears it");

// A projection site may legitimately extract a multi-valued attribute.
const multiProjection = clone(q2a);
for (const s of multiProjection.semantic_sites) s.output_type = "list[string]";
assert.ok(!hasMultilabelFinding(multiProjection), "only row-deciding sites are linted");

console.log("test_argmax_filter_lint (multilabel) OK");
