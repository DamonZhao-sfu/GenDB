import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import { countCsvDataRows, formatQueryMetric, valDesignDirName } from "../orchestrator.mjs";
import { lintImagePlan } from "../agent-runtime/contracts.mjs";

// --- formatQueryMetric -------------------------------------------------------
// The workload summary used to print a hardcoded `P=… R=… F1=…` for every query, so a
// grouping query (scored by ARI) reported `P=undefined R=undefined F1=undefined` even
// though telemetry held its real score.

test("a retrieval query reports precision, recall and F1", () => {
  const shown = formatQueryMetric({
    metric: "f1-score",
    metric_family: "SingleAccuracyScoreWithRetrievalDetails",
    precision: 0.96, recall: 0.5161, f1: 0.6713, tp: 48, fp: 2, fn: 45,
  });
  assert.equal(shown.key, "f1");
  assert.equal(shown.value, 0.6713);
  assert.match(shown.label, /P=0\.96 R=0\.5161 F1=0\.6713 {2}\(tp=48 fp=2 fn=45\)/);
});

test("a grouping query reports ARI instead of undefined P/R/F1", () => {
  const shown = formatQueryMetric({
    metric: "adjusted-rand-index", metric_family: "SingleAccuracyScore",
    accuracy: 0.0709, ari: 0.0709, adjusted_rand_index: 0.0709,
  });
  assert.equal(shown.key, "ari");
  assert.equal(shown.value, 0.0709);
  assert.equal(shown.label, "ARI=0.0709");
  assert.doesNotMatch(shown.label, /undefined/);
});

test("a negative ARI is reported rather than dropped", () => {
  const shown = formatQueryMetric({ metric: "adjusted-rand-index", ari: -0.038 });
  assert.equal(shown.value, -0.038);
  assert.equal(shown.label, "ARI=-0.038");
});

test("an aggregation query reports relative error and MAPE", () => {
  const shown = formatQueryMetric({
    metric: "aggregation", metric_family: "QueryMetricAggregation",
    relative_error: 0.12, mape: 12, absolute_error: 3,
  });
  assert.equal(shown.key, "relative_error");
  assert.equal(shown.value, 0.12);
  assert.match(shown.label, /rel_err=0\.12 MAPE=12% abs_err=3/);
});

test("a ranking query reports Spearman and Kendall", () => {
  const shown = formatQueryMetric({
    metric: "ranking", metric_family: "QueryMetricRank",
    spearman_correlation: 0.83, kendall_tau: 0.7,
  });
  assert.equal(shown.key, "spearman");
  assert.equal(shown.value, 0.83);
  assert.match(shown.label, /spearman=0\.83 kendall=0\.7/);
});

test("the macro_classification variant is labelled macroF1, not F1", () => {
  const shown = formatQueryMetric({
    metric: "retrieval_f1", metric_variant: "macro_classification",
    precision: 0.5, recall: 0.5, f1: 0.5,
  });
  assert.equal(shown.key, "macro_f1");
  assert.match(shown.label, /macroF1=0\.5/);
});

test("an unscored query says so instead of printing undefined", () => {
  const shown = formatQueryMetric({});
  assert.equal(shown.value, null);
  assert.equal(shown.label, "no metric recorded");
});

test("a zero score is reported, not treated as missing", () => {
  const shown = formatQueryMetric({
    metric: "f1-score", precision: 0, recall: 0, f1: 0, tp: 0, fp: 65, fn: 12,
  });
  assert.equal(shown.value, 0);
  assert.match(shown.label, /P=0 R=0 F1=0/);
});

// --- lintImagePlan -----------------------------------------------------------
// mmqa q2a/q7 both bound a whole named-entity predicate to one `verify_property`, which
// compares `prop` against `not prop` and is positively biased: recall 1.0, precision
// ~0.02. No rewording fixes that, so it has to be caught at plan time.

const imagePlan = (helperDag) => ({
  schema_version: "1.0",
  query_id: "q",
  plan_version: 1,
  parent_plan_version: null,
  modality: "image",
  compilability: { class: "bounded_approximation", obligations: [], unresolved: [] },
  semantic_sites: [{ site_id: "s", operator: "verify_property", output_type: "boolean" }],
  helper_dag: helperDag,
  relational_plan: [],
  trace_contract: {},
  runtime_contract: {},
  validation_contract: {},
  assumptions: [],
  invariants: [],
});

test("a data-parameterized verify_property with no discriminative step is flagged", () => {
  const findings = lintImagePlan(imagePlan([{
    helper_id: "h", name: "is_track_logo",
    args: [{ name: "track_name", type: "string" }, { name: "image", type: "ImagePatch" }],
    return_type: "boolean", depends_on: [], confidence_signal: null,
    primitive_steps: [{
      primitive: "verify_property",
      inputs: ["image: ImagePatch", "prop: 'the logo of ' + track_name"],
      output_type: "boolean",
    }],
  }]));
  assert.equal(findings.length, 2);
  assert.match(findings[0], /built from a data value/);
  assert.match(findings[1], /confidence_signal/);
});

test("a property_template placeholder counts as parameterization", () => {
  const findings = lintImagePlan(imagePlan([{
    helper_id: "h", name: "shows_logo",
    args: [{ name: "image" }, { name: "airline" }],
    return_type: "boolean", depends_on: [], confidence_signal: null,
    primitive_steps: [{
      primitive: "verify_property", inputs: ["image", "prop"],
      property_template: "the logo of {airline}", output_type: "boolean",
    }],
  }]));
  assert.ok(findings.length >= 1);
  assert.match(findings[0], /built from a data value/);
});

test("a CONSTANT verify_property property is left alone", () => {
  // "a damaged car" is a real generic visual question and scores fine; flagging it
  // would cost a planner call and push a working plan off a correct binding.
  const findings = lintImagePlan(imagePlan([{
    helper_id: "h", name: "is_undamaged",
    args: [{ name: "image" }],
    return_type: "boolean", depends_on: [], confidence_signal: null,
    primitive_steps: [{
      primitive: "verify_property", inputs: ["image", "a car that is not damaged"],
      output_type: "boolean",
    }],
  }]));
  assert.deepEqual(findings, []);
});

test("a discriminative sibling primitive clears the plan", () => {
  const findings = lintImagePlan(imagePlan([{
    helper_id: "h", name: "is_track_logo",
    args: [{ name: "track_name" }, { name: "image" }],
    return_type: "boolean", depends_on: [], confidence_signal: "ocr_conf >= 0.55",
    primitive_steps: [
      { primitive: "verify_property", inputs: ["image", "prop: 'the logo of ' + track_name"] },
      { primitive: "best_ocr_match_detail", inputs: ["image", "tracks"] },
    ],
  }]));
  assert.deepEqual(findings, []);
});

test("a text plan is never linted for image primitives", () => {
  const plan = imagePlan([]);
  plan.modality = "text";
  assert.deepEqual(lintImagePlan(plan), []);
});

test("a not_compilable plan is not linted", () => {
  const plan = imagePlan([{
    helper_id: "h", name: "f", args: [{ name: "x" }], return_type: "boolean",
    depends_on: [], confidence_signal: null,
    primitive_steps: [{ primitive: "verify_property", inputs: ["image", "'a ' + x"] }],
  }]);
  plan.compilability.class = "not_compilable";
  assert.deepEqual(lintImagePlan(plan), []);
});

test("a non-boolean extraction site is not required to discriminate", () => {
  const plan = imagePlan([{
    helper_id: "h", name: "f", args: [{ name: "x" }], return_type: "string",
    depends_on: [], confidence_signal: null,
    primitive_steps: [{ primitive: "verify_property", inputs: ["image", "'a ' + x"] }],
  }]);
  plan.semantic_sites[0].output_type = "string";
  assert.deepEqual(lintImagePlan(plan), []);
});

// --- valDesignDirName --------------------------------------------------------
// mmqa q2b's oracle choices are the 14-colour value space, which pushed the validation
// design key past the 255-byte filename limit: build_valset.py died with ENAMETOOLONG
// and the query failed outright, because --val-rate refuses ground-truth fallback.

test("a short design key is used verbatim so existing caches still hit", () => {
  const key = "stratified_0.05_0_7_5_2_Qwen_auto_oracleframes-v5";
  assert.equal(valDesignDirName(key), key);
});

test("an over-long design key is shortened below the filename limit", () => {
  const key = "stratified_0.05_0_7_5_2_QwenQwen3-VL-30B-A3B-Instruct_auto_oracleframes-v5"
    + "_corpus-7c2d0947e4198a73_columnpair_score_text1_pairwise_full-frame_joint-oracle-v1"
    + "_text_no_match_matchblack_matchwhite_matchred_matchorange_matchyellow_matchgreen"
    + "_matchblue_matchpurple_matchpink_matchbrown_matchgray_matchsilver_matchgold";
  assert.ok(key.length > 255, "fixture must reproduce the overflow");
  const name = valDesignDirName(key);
  assert.ok(name.length <= 150, `got ${name.length}`);
  assert.ok(Buffer.byteLength(name) < 255);
});

test("shortening is deterministic and keeps distinct designs distinct", () => {
  const long = "x".repeat(400);
  assert.equal(valDesignDirName(long), valDesignDirName(long));
  assert.notEqual(valDesignDirName(long), valDesignDirName(`${long}_other`));
});

// --- countCsvDataRows --------------------------------------------------------
// Zero selected rows freezes the optimizer's branch counters, so it has to be a
// first-class signal rather than something parsed out of stderr.

const csvRoot = await mkdtemp(resolve(tmpdir(), "semdb-csv-"));

test("a header-only result CSV counts as zero selected rows", async () => {
  const path = resolve(csvRoot, "empty.csv");
  await writeFile(path, "ID,uri\n");
  assert.equal(countCsvDataRows(path), 0);
});

test("data rows are counted without the header", async () => {
  const path = resolve(csvRoot, "rows.csv");
  await writeFile(path, "ID,uri\n1,a.png\n2,b.png\n");
  assert.equal(countCsvDataRows(path), 2);
});

test("a missing file is unknown, not zero — that is a crash, not an empty predicate", () => {
  assert.equal(countCsvDataRows(resolve(csvRoot, "nope.csv")), null);
});

test("a trailing blank line does not inflate the count", async () => {
  const path = resolve(csvRoot, "trailing.csv");
  await writeFile(path, "ID,uri\n1,a.png\n\n\n");
  assert.equal(countCsvDataRows(path), 1);
});
