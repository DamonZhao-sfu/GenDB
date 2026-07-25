/**
 * Run-log capture and reduction. A raw run log is both too long to paste into a
 * prompt and mostly repetition, so filterRunLog has to keep the actionable lines,
 * collapse the repeats, and never present a truncated view as a complete one.
 *
 *   node src/semdb/tests/test_run_log.mjs
 */
import assert from "assert";
import { mkdtempSync, readFileSync, existsSync } from "fs";
import { tmpdir } from "os";
import { resolve } from "path";
import { filterRunLog, renderFeedback, runPythonLogged } from "../orchestrator.mjs";

const dir = mkdtempSync(resolve(tmpdir(), "semdb-runlog-"));

let passed = 0;
const test = (name, fn) => {
  try { fn(); passed++; console.log(`  ok  ${name}`); }
  catch (e) { console.error(`  FAIL ${name}\n     ${e.message}`); process.exitCode = 1; }
};

console.log("filterRunLog");

test("empty input yields nothing to show", () => {
  for (const input of ["", null, undefined, "   \n\n  "]) {
    const r = filterRunLog(input);
    assert.deepStrictEqual(r.lines, []);
    assert.strictEqual(r.total, 0);
  }
});

test("signal lines are kept and progress noise is dropped", () => {
  const log = [
    "processing row 1", "processing row 2", "processing row 3",
    "WARN no genre keyword matched for row 7",
    "processing row 4", "processing row 5",
  ].join("\n");
  const r = filterRunLog(log, { tailLines: 0 });
  assert.strictEqual(r.lines.length, 1);
  assert.match(r.lines[0], /WARN no genre keyword/);
  assert.strictEqual(r.total, 6, "total must count the whole log, not the kept lines");
});

test("the tail is kept even when it matches no keyword", () => {
  const log = [...Array(30).keys()].map((i) => `step ${i}`).join("\n");
  const r = filterRunLog(log, { tailLines: 3 });
  assert.deepStrictEqual(r.lines, ["step 27", "step 28", "step 29"]);
});

test("a traceback is caught by both its header and its frame lines", () => {
  const log = 'ok\nTraceback (most recent call last):\n  File "solve.py", line 42, in main\nKeyError: \'genre\'';
  const r = filterRunLog(log, { tailLines: 0 });
  assert.ok(r.lines.some((l) => /Traceback/.test(l)));
  assert.ok(r.lines.some((l) => /File "solve\.py", line 42/.test(l)));
  assert.ok(r.lines.some((l) => /KeyError/.test(l)));
});

test("identical lines collapse into one with a count", () => {
  const log = Array(180).fill("WARN no keyword matched, using fallback").join("\n");
  const r = filterRunLog(log);
  assert.strictEqual(r.lines.length, 1, "180 identical warnings must not cost 180 lines");
  assert.match(r.lines[0], /\(× 180\)/);
  assert.strictEqual(r.total, 180);
});

test("a single occurrence carries no count suffix", () => {
  const r = filterRunLog("WARN something happened", { tailLines: 0 });
  assert.strictEqual(r.lines[0], "WARN something happened");
});

test("output is capped and the omission is reported", () => {
  const log = [...Array(100).keys()].map((i) => `ERROR distinct problem ${i}`).join("\n");
  const r = filterRunLog(log, { maxLines: 25 });
  assert.strictEqual(r.lines.length, 25);
  assert.strictEqual(r.omitted, 75, "a truncated view must say how much it dropped");
});

test("very long lines are truncated so one line cannot eat the budget", () => {
  const r = filterRunLog(`ERROR ${"x".repeat(5000)}`, { maxLineChars: 200 });
  assert.ok(r.lines[0].length <= 210, `line was ${r.lines[0].length} chars`);
  assert.ok(r.lines[0].endsWith("…"));
});

test("counting happens after truncation, so near-identical long lines still collapse", () => {
  const base = `ERROR ${"y".repeat(300)}`;
  const r = filterRunLog([`${base}A`, `${base}B`].join("\n"), { maxLineChars: 100, tailLines: 0 });
  assert.strictEqual(r.lines.length, 1);
  assert.match(r.lines[0], /\(× 2\)/);
});

test("contracted [solve] lines survive a flood of other warnings", () => {
  // The contract puts branch counts AFTER the row loop, so in a real log they sit
  // behind thousands of warnings — exactly where a cap or a tail window loses them.
  const log = [
    ...[...Array(400).keys()].map((i) => `[solve] WARN no cue id=${i}`),
    ...[...Array(60).keys()].map((i) => `ERROR distinct problem ${i}`),
    "[solve] branch=regex_hit n=52",
    "[solve] branch=fallback n=8",
    "[solve] rows_in=60 rows_out=8 errors=0 elapsed=0.1s",
  ].join("\n");
  const r = filterRunLog(log, { maxLines: 25 });
  assert.ok(r.lines.some((l) => /branch=regex_hit n=52/.test(l)), "branch counts must survive");
  assert.ok(r.lines.some((l) => /branch=fallback n=8/.test(l)));
  assert.ok(r.lines.some((l) => /rows_in=60 rows_out=8/.test(l)), "the summary must survive");
});

test("repeated [solve] warnings collapse rather than crowding out the summary", () => {
  const log = [
    ...Array(400).fill("[solve] WARN no cue"),
    "[solve] rows_in=60 rows_out=8 errors=0 elapsed=0.1s",
  ].join("\n");
  const r = filterRunLog(log, { maxLines: 25 });
  assert.strictEqual(r.lines.length, 2);
  assert.match(r.lines[0], /\(× 400\)/);
  assert.match(r.lines[1], /rows_in=60/);
});

test("a solver emitting nothing but structured lines still reports totals honestly", () => {
  const r = filterRunLog("[solve] branch=a n=1\n[solve] branch=b n=2");
  assert.strictEqual(r.lines.length, 2);
  assert.strictEqual(r.total, 2);
  assert.strictEqual(r.omitted, 0);
});

console.log("runPythonLogged — capture");

test("both streams are captured and persisted", () => {
  const log = resolve(dir, "both.log");
  const r = runPythonLogged(
    ["-c", "import sys; print('to stdout'); print('to stderr', file=sys.stderr)"], log, "t");
  assert.strictEqual(r.proc.status, 0);
  assert.match(r.stdout, /to stdout/);
  assert.match(r.stderr, /to stderr/);
  const written = readFileSync(log, "utf-8");
  assert.match(written, /to stdout/);
  assert.match(written, /to stderr/);
  assert.match(written, /--- stderr ---/, "the two streams stay distinguishable on disk");
  assert.match(written, /^\$ python3 -c/, "the log records the command that produced it");
});

test("stdout survives a nonzero exit — the crash context is the point", () => {
  const log = resolve(dir, "crash.log");
  const r = runPythonLogged(
    ["-c", "print('progress 1'); print('progress 2'); raise KeyError('genre')"], log, "t");
  assert.notStrictEqual(r.proc.status, 0);
  assert.match(r.stdout, /progress 2/);
  assert.match(r.stderr, /KeyError/);
  const filtered = filterRunLog([r.stdout, r.stderr].join("\n"));
  assert.ok(filtered.lines.some((l) => /KeyError/.test(l)));
});

test("execution time is measured", () => {
  const r = runPythonLogged(["-c", "import time; time.sleep(0.25)"], resolve(dir, "t.log"), "t");
  assert.ok(r.execMs >= 200 && r.execMs < 15000, `execMs was ${r.execMs}`);
});

test("a failed spawn is reported rather than looking like an empty run", () => {
  const log = resolve(dir, "nofile.log");
  const r = runPythonLogged(["/definitely/not/here.py"], log, "solver");
  assert.notStrictEqual(r.proc.status, 0);
  assert.ok(existsSync(log), "even a failed run leaves a log behind");
  assert.ok(r.stderr.length > 0, "a failure must say something");
});

console.log("renderFeedback — runtime log section");

const okPrev = (extra) => ({
  status: "ok", f1: 0.9, metrics: { correct: 54, n: 60 },
  diff: { mistakes: [{ id: "26", predicted: "true", expected: "false", text: "t" }], n_mistakes: 1 },
  history: [], ...extra,
});

test("the log section appears with its counts", () => {
  const fb = renderFeedback(okPrev({
    runLog: { lines: ["WARN fallback used   (× 18)", "ERROR KeyError"], total: 220, omitted: 3 },
  }));
  assert.match(fb, /## RUNTIME LOG/);
  assert.match(fb, /220 lines, showing 2, 3 more omitted/);
  assert.match(fb, /WARN fallback used {3}\(× 18\)/);
});

test("no log means no empty section", () => {
  const fb = renderFeedback(okPrev({ runLog: { lines: [], total: 0, omitted: 0 } }));
  assert.ok(!/## RUNTIME LOG/.test(fb));
});

test("a missing runLog field does not throw (older callers)", () => {
  const fb = renderFeedback(okPrev({}));
  assert.match(fb, /PER-ROW INFERENCE/);
  assert.ok(!/## RUNTIME LOG/.test(fb));
});

test("execution time is reported when measured, omitted when not", () => {
  assert.match(renderFeedback(okPrev({ execMs: 8432 })), /in 8\.4s/);
  assert.ok(!/ in .*s\b.*labeled rows/.test(renderFeedback(okPrev({ execMs: null })).split("\n")[1] || ""));
});

test("the F1 branch carries the log and timing too", () => {
  const fb = renderFeedback({
    status: "ok", f1: 0.88, metrics: { precision: 0.9, recall: 0.85, tp: 11, fp: 1, fn: 2 },
    diff: { false_positives: [], false_negatives: [] }, history: [],
    execMs: 3200, runLog: { lines: ["WARN x"], total: 9, omitted: 0 },
  });
  assert.match(fb, /## LAST RUN — F1=0\.88/);
  assert.match(fb, /in 3\.2s/);
  assert.match(fb, /## RUNTIME LOG/);
});

test("a compile failure still shows only the compile block", () => {
  const fb = renderFeedback({ status: "crash", stage: "compile", stderrTail: "COMPILE FAILED",
    runLog: { lines: ["WARN x"], total: 1, omitted: 0 }, history: [] });
  assert.match(fb, /DID NOT COMPILE/);
  assert.ok(!/## RUNTIME LOG/.test(fb), "a program that never ran has no runtime log to show");
});

console.log(`\n${passed} passed${process.exitCode ? " (with failures)" : ""}`);
