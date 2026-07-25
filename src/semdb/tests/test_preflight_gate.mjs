/**
 * The orchestrator side of the compile gate: runPreflight must fail closed on
 * broken code, fail OPEN when the checker itself is unusable, and renderFeedback
 * must not describe a program that never ran as having crashed.
 *
 *   node src/semdb/tests/test_preflight_gate.mjs
 */
import { mkdtempSync, writeFileSync, existsSync, readFileSync } from "fs";
import { tmpdir } from "os";
import { resolve } from "path";
import assert from "assert";

import { runPreflight, renderPreflightText, renderFeedback } from "../orchestrator.mjs";

const dir = mkdtempSync(resolve(tmpdir(), "semdb-preflight-"));
const write = (name, src) => { const p = resolve(dir, name); writeFileSync(p, src); return p; };

let passed = 0;
const test = (name, fn) => {
  try { fn(); passed++; console.log(`  ok  ${name}`); }
  catch (e) { console.error(`  FAIL ${name}\n     ${e.message}`); process.exitCode = 1; }
};

console.log("runPreflight");

test("clean code passes and writes a report", () => {
  const code = write("clean.py", "def solve(rows):\n    return [r for r in rows]\n");
  const out = resolve(dir, "clean.json");
  const pre = runPreflight([code], out);
  assert.strictEqual(pre.ok, true);
  assert.ok(existsSync(out), "report must be written even on success");
  assert.strictEqual(JSON.parse(readFileSync(out, "utf-8")).ok, true);
});

test("a syntax error fails the gate with a located, contextual message", () => {
  const code = write("broken.py", "def solve(rows):\n    if rows == 1\n        return []\n");
  const pre = runPreflight([code], resolve(dir, "broken.json"));
  assert.strictEqual(pre.ok, false);
  assert.strictEqual(pre.stage, "syntax");
  assert.match(pre.text, /COMPILE FAILED/);
  assert.match(pre.text, /broken\.py:2/);
  assert.match(pre.text, />> +2 \|/, "the offending line must be marked in context");
});

test("missing files are skipped rather than treated as failures", () => {
  const pre = runPreflight([resolve(dir, "nope.py"), null], resolve(dir, "none.json"));
  assert.strictEqual(pre.ok, true);
  assert.strictEqual(pre.stage, "skipped");
});

test("an undefined name fails the gate", () => {
  const code = write("undef.py", "def solve(rows):\n    return [r for r in rows if r in TABLE]\n");
  const pre = runPreflight([code], resolve(dir, "undef.json"));
  // Only meaningful when pyflakes is installed; otherwise the gate must pass.
  const report = JSON.parse(readFileSync(resolve(dir, "undef.json"), "utf-8"));
  if (report.static_checker === "unavailable") {
    assert.strictEqual(pre.ok, true, "a missing linter must never block generation");
  } else {
    assert.strictEqual(pre.ok, false);
    assert.strictEqual(pre.stage, "static");
    assert.match(pre.text, /TABLE/);
  }
});

console.log("renderPreflightText");

test("a passing report renders as empty (nothing to tell the agent)", () => {
  assert.strictEqual(renderPreflightText({ ok: true }), "");
  assert.strictEqual(renderPreflightText(null), "");
});

test("warnings are listed but capped", () => {
  const text = renderPreflightText({
    ok: false, stage: "static", errors: [],
    warnings: Array.from({ length: 9 }, (_, i) => ({
      error_class: "UnusedImport", file: "a.py", line: i + 1, message: `w${i}` })),
  });
  assert.match(text, /non-fatal, 9/);
  assert.strictEqual((text.match(/UnusedImport/g) || []).length, 5, "cap the list at 5");
});

console.log("renderFeedback — compile stage");

test("a compile failure is not described as a crash", () => {
  const fb = renderFeedback({ status: "crash", stage: "compile",
    stderrTail: "COMPILE FAILED at stage `syntax`", history: [] });
  assert.match(fb, /DID NOT COMPILE/);
  assert.match(fb, /was NOT executed/);
  assert.ok(!/crashed/.test(fb), "must not claim a runtime crash");
  assert.match(fb, /Do not change anything else/);
});

test("a real runtime crash still reads as a crash", () => {
  const fb = renderFeedback({ status: "crash", stage: null,
    stderrTail: "KeyError: 'genre'", history: [] });
  assert.match(fb, /crashed/);
  assert.ok(!/DID NOT COMPILE/.test(fb));
});

test("an empty run is still reported as producing no rows", () => {
  const fb = renderFeedback({ status: "empty", stage: null, stderrTail: "", history: [] });
  assert.match(fb, /produced no output rows/);
});

test("the success path is untouched by the stage field", () => {
  const fb = renderFeedback({ status: "ok", f1: 0.8, metrics: { correct: 8, n: 10 },
    diff: { mistakes: [], n_mistakes: 0 }, history: [] });
  assert.match(fb, /PER-ROW INFERENCE accuracy=0.8/);
});

console.log(`\n${passed} passed${process.exitCode ? " (with failures)" : ""}`);
