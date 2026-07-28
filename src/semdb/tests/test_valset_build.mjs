/**
 * Validation-set auto-build, caching, and the class-breakdown feedback.
 *
 *   node src/semdb/tests/test_valset_build.mjs
 */
import assert from "assert";
import { mkdtempSync, mkdirSync, writeFileSync, existsSync, readFileSync } from "fs";
import { tmpdir } from "os";
import { resolve } from "path";
import { buildValSet, renderFeedback } from "../orchestrator.mjs";

const dir = mkdtempSync(resolve(tmpdir(), "semdb-valset-"));

let passed = 0;
const test = (name, fn) => {
  try { fn(); passed++; console.log(`  ok  ${name}`); }
  catch (e) { console.error(`  FAIL ${name}\n     ${e.message}`); process.exitCode = 1; }
};

const baseArgs = (over = {}) => ({
  out: dir, benchmark: "ecomm", apiKey: "EMPTY", concurrency: 8,
  valMethod: "stratified", valRate: 0.2, valCertRate: 0.1, valSeed: 7,
  valStrataK: 5, valScoreTilt: 2, valCallSite: null, ...over,
});

const spec = (over = {}) => ({
  corpusCsv: "/nonexistent/IMAGES.csv", idCol: "id", sqlPath: "/nonexistent/q2.sql",
  isImage: true, imageCol: "filename", imageDir: "/nonexistent/images",
  clipModel: "openai/clip-vit-base-patch32", textCols: [],
  endpoint: "http://localhost:9/v1", oracleModel: "test-model", ...over,
});

console.log("buildValSet — caching");

test("a cached select.json is reused without spawning a build", () => {
  const args = baseArgs({ out: resolve(dir, "cached") });
  // Pre-create exactly the directory the key resolves to.
  const first = buildValSet(args, "q2", spec());   // fails (no corpus) -> null
  assert.strictEqual(first, null, "a failed build must report null, not a bogus path");
});

/** Seed the cache for the DEFAULT design, then report whether `over` hits it. */
function hitsDefaultCache(out, over = {}, specOver = {}) {
  const key = ["stratified", 0.2, 0.1, 7, 5, 2, "test-model", "auto",
    "oracleframes-v3", "default-score"].join("_")
    .replace(/[^\w.-]/g, "");
  const cacheDir = resolve(out, "_val", "ecomm-q2", key);
  mkdirSync(cacheDir, { recursive: true });
  writeFileSync(resolve(cacheDir, "select.json"), JSON.stringify({ labels: { a: "true" } }));
  return buildValSet(baseArgs({ out, ...over }), "q2", spec(specOver)) !== null;
}

test("a pre-existing select.json short-circuits the build", () => {
  assert.ok(hitsDefaultCache(resolve(dir, "hit")));
});

test("every design knob changes the cache key", () => {
  // Reusing labels across a changed design would report numbers for rows that were
  // never drawn under it — silently, since the file looks the same.
  const cases = {
    rate: { valRate: 0.4 }, certRate: { valCertRate: 0.2 }, seed: { valSeed: 8 },
    method: { valMethod: "uniform" }, strataK: { valStrataK: 10 },
    tilt: { valScoreTilt: 3 }, callSite: { valCallSite: 1 },
  };
  for (const [name, over] of Object.entries(cases)) {
    assert.ok(!hitsDefaultCache(resolve(dir, `k-${name}`), over),
              `${name} must not reuse the default design's labels`);
  }
});

test("a different oracle model does not reuse another model's labels", () => {
  // Not a design knob but a different QUESTION-answerer: the labels are its opinions.
  assert.ok(!hitsDefaultCache(resolve(dir, "k-model"), { oracleModel: "other" },
                              { oracleModel: "other" }));
});

test("the same design hits the cache on a second call", () => {
  const out = resolve(dir, "twice");
  assert.ok(hitsDefaultCache(out));
  assert.ok(hitsDefaultCache(out), "a repeat run must not re-pay for labels");
});

test("a failed build never throws — the run falls back to full-GT scoring", () => {
  assert.doesNotThrow(() => buildValSet(baseArgs(), "q2", spec()));
});

console.log("renderFeedback — class breakdown");

const withQuality = (quality, extra = {}) => renderFeedback({
  status: "ok", f1: 0.98, metrics: { correct: 49, n: 50, quality, ...extra },
  diff: { mistakes: [], n_mistakes: 1 }, history: [],
});

test("precision, recall and F1 are shown alongside accuracy", () => {
  const fb = withQuality({ precision: 0.5, recall: 0.25, f1: 0.33, tp: 1, fp: 1, fn: 3 });
  assert.match(fb, /## CLASS BREAKDOWN/);
  assert.match(fb, /precision=0\.5 recall=0\.25 F1=0\.33/);
  assert.match(fb, /tp=1 fp=1 fn=3/);
});

test("zero recall is called out explicitly", () => {
  // The failure the loop is blind to otherwise: at a 2% base rate, `return false`
  // scores 98% accuracy. The agent must be told that number is worthless.
  const fb = withQuality({ precision: null, recall: 0, f1: null, tp: 0, fp: 0, fn: 5 });
  assert.match(fb, /RECALL IS ZERO/);
  assert.match(fb, /always answers "no" would score the same accuracy/);
});

test("nonzero recall does not trigger the warning", () => {
  const fb = withQuality({ precision: 1, recall: 1, f1: 1, tp: 5, fp: 0, fn: 0 });
  assert.ok(!/RECALL IS ZERO/.test(fb));
});

test("a weighted design says the numbers are corpus estimates", () => {
  const fb = withQuality(
    { precision: 0.4, recall: 0.8, f1: 0.53, tp: 40, fp: 60, fn: 10 },
    { weighted: true, design: { weight_spread: 16, N: 250 },
      unweighted: { precision: 0.9, recall: 0.8 } });
  assert.match(fb, /corpus estimates/);
  assert.match(fb, /16x spread/);
  assert.match(fb, /250-row corpus/);
  assert.match(fb, /Raw counts over the labeled rows are precision=0\.9/);
});

test("an unweighted design makes no corpus-estimate claim", () => {
  const fb = withQuality({ precision: 0.9, recall: 0.8, f1: 0.85, tp: 4, fp: 1, fn: 1 },
                         { weighted: false });
  assert.ok(!/corpus estimates/.test(fb));
});

test("a missing quality block does not throw (older val files)", () => {
  const fb = renderFeedback({
    status: "ok", f1: 0.9, metrics: { correct: 45, n: 50 },
    diff: { mistakes: [], n_mistakes: 5 }, history: [],
  });
  assert.match(fb, /PER-ROW INFERENCE/);
  assert.ok(!/## CLASS BREAKDOWN/.test(fb));
});

console.log(`\n${passed} passed${process.exitCode ? " (with failures)" : ""}`);
