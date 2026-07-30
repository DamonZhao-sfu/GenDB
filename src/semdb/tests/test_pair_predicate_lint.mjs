/** A pair site must keep its predicate a function of BOTH sides.
 *
 *  mmqa q2a's ground truth is ONE Santa Anita wordmark that must match two different
 *  Track strings — "Santa Anita Park" and "Oak Tree at Santa Anita Park" (5 GT pairs,
 *  all the same image). The shipped plan declared
 *  `racetrack_logo_name(image, track_names: list[string]) -> string` and compared it with
 *  `==` in the solver, so a single argmax label could satisfy at most one of the two:
 *  recall was capped at 3/5 before any image was read. The run that kept the predicate
 *  binary, with containment instead of equality, scored recall 1.0.
 */
import assert from "node:assert";
import { lintImagePlan } from "../agent-runtime/contracts.mjs";
import { Q2A_LABEL_COLLAPSE, clonePlan as clone } from "./fixtures/plans.mjs";

const bad = Q2A_LABEL_COLLAPSE;
const flagged = (p) => lintImagePlan(p).some((f) => /function of BOTH/.test(f));

// The shipped image→label plan must be flagged.
assert.ok(flagged(bad), "the label-collapsing plan must be flagged");

// A helper that decides one (image, value) pair clears it.
const binary = clone(bad);
binary.helper_dag.push({
  helper_id: "helper_1",
  name: "logo_matches_track",
  args: [{ name: "image", type: "ImagePatch" }, { name: "track", type: "string" }],
  return_type: "boolean",
  depends_on: [],
  primitive_steps: [],
  confidence_signal: null,
});
assert.ok(!flagged(binary), "a binary (image, value) -> bool helper clears the finding");

// The evidence-then-decide split is exactly what the good solver did, and is fine.
const evidenceSplit = clone(bad);
evidenceSplit.helper_dag = [
  { helper_id: "h0", name: "infer_logo_evidence",
    args: [{ name: "image", type: "ImagePatch" }, { name: "tracks", type: "list[string]" }],
    return_type: "dict", depends_on: [], primitive_steps: [], confidence_signal: null },
  { helper_id: "h1", name: "logo_matches",
    args: [{ name: "evidence", type: "dict" }, { name: "track", type: "string" }],
    return_type: "bool", depends_on: ["h0"], primitive_steps: [], confidence_signal: null },
];
assert.ok(!flagged(evidenceSplit), "evidence-once + decide-per-pair is the sanctioned shape");

// Taking only the whole value space is not taking the other side.
const valueSpaceOnly = clone(bad);
valueSpaceOnly.helper_dag = [{
  helper_id: "h0", name: "decide",
  args: [{ name: "image", type: "ImagePatch" }, { name: "tracks", type: "list[string]" }],
  return_type: "boolean", depends_on: [], primitive_steps: [], confidence_signal: null,
}];
assert.ok(flagged(valueSpaceOnly), "a list[] argument is the value space, not the pair's other side");

// A row site is a different shape — one input, one decision. Not this rule's business.
const rowSite = clone(bad);
for (const s of rowSite.semantic_sites) s.sampling_unit = "row";
assert.ok(!flagged(rowSite), "row sites are out of scope");

// A projection site (extract an attribute) legitimately returns a label.
const projection = clone(bad);
for (const s of projection.semantic_sites) s.output_type = "string";
assert.ok(!flagged(projection), "only boolean pair sites are linted");

// ordered_pair is the same shape as pair.
const ordered = clone(bad);
for (const s of ordered.semantic_sites) s.sampling_unit = "ordered_pair";
assert.ok(flagged(ordered), "ordered_pair must be linted like pair");

console.log("test_pair_predicate_lint OK");
