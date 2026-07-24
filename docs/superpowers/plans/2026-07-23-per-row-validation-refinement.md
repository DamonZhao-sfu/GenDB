# Per-Row Validation-Driven DIRECT Solver Refinement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user hand-label a small per-row validation set (`val.json`) and use it as the only ground truth to drive the existing refinement loop over a DIRECT-mode `solve_<q>.py`, scoring per-row inference accuracy, then apply the frozen solver to the full unlabeled corpus.

**Architecture:** Reuse the existing `refineLoop`/`checkSemdbImprovement`/`shouldContinueSemdb` unchanged. Add (1) a per-row scorer `score_inference` + `--score-inference` CLI in `evaluate.py`; (2) a per-row feedback branch in `renderFeedback`; (3) a `--val-file` code path in `runQueryDirect`'s text branch that runs iterations on the labeled sub-corpus only (via a best-effort `--only-ids`), scores per-row accuracy, freezes the best solver, then runs it once over the full corpus; (4) a solver-prompt contract requiring a `trace_<q>.json` per-row decision dump. The loop's generic `f1` field carries accuracy in val mode — no loop schema change.

**Tech Stack:** Node.js ESM (`orchestrator.mjs`), Python 3.10+ stdlib (`evaluate.py`), pytest, node's built-in test via plain `assert`. OpenAI-compatible endpoint reused via the existing solver run path.

## Global Constraints

- The val loop engages ONLY when `--val-file <path>` is set AND the corpus is text (`!isImage`) AND `--no-refine` is not set. Absent `--val-file`, behavior is byte-for-byte today's (F1 loop if GT present, else single-shot).
- Precedence: `--val-file` present → per-row val loop. Else GT present + `--no-refine` unset → F1 loop. Else single-shot. `--no-refine` forces single-shot in all cases.
- Per-row scoring restricts to the labeled ids ONLY. `--only-ids` is a best-effort COST bound on iteration runs; correctness of scoring must NOT depend on the solver honoring it (a solver that ignores it still scores correctly, just slower).
- A labeled id absent from the trace scores as WRONG with `predicted: null` (the solver dropped a row it should have inferred).
- Normalization matches `evaluate.py`: compare `str(x).strip().lower()`.
- Sample cap reuses `defaults.refineSampleCap` (15). Copy this value; do not introduce a new default.
- The FINAL full-corpus run MUST omit `--only-ids` (no-leakage: the reported output is produced over all rows, not just labeled ones).
- Reuse `defaults.maxRefineIterations` (5) and `defaults.refineStallThreshold` (2) via the existing loop — do not fork them.
- Python: PEP 8, type annotations on new signatures, `print` for CLI user output (match the file), no new HTTP client.
- Run Python under sembench conda for any endpoint-backed check: `source $HOME/anaconda3/etc/profile.d/conda.sh && conda activate sembench`. Unit tests here need no endpoint.

---

## File Structure

**New**
- `src/semdb/tests/test_score_inference.py` — pytest for `score_inference` (accuracy math, missing-id=wrong, cap, normalization, corpus text lookup).
- `src/semdb/tests/test_val_feedback.mjs` — node assertions for the per-row `renderFeedback` branch.
- `src/semdb/tests/fixtures/val_q3a.json` — example hand-labeled validation file.

**Modified**
- `src/semdb/evaluate.py` — add `score_inference(...)`; add `--score-inference` CLI branch (`--trace`, `--val-file`, `--corpus-csv`, `--id-col`, `--text-col`, reuse `--emit-diff` as output path, `--diff-cap`); make `--csv` optional with a guard.
- `src/semdb/orchestrator.mjs` — `parseArgs` gains `--val-file`; `renderFeedback` gains a per-row branch; new `scoreInference(...)` helper; `runQueryDirect` text branch: val-subset iteration scope, per-row `scoreIter`, final full-corpus run, telemetry `mode`.
- `src/semdb/agents/vadar-solver/prompt-text.md` — require `trace_<q>.json` output + honor `--only-ids`.

---

## Task 1: `score_inference` + `--score-inference` CLI in `evaluate.py`

**Files:**
- Modify: `src/semdb/evaluate.py`
- Test: `src/semdb/tests/test_score_inference.py`

**Interfaces:**
- Consumes: nothing new (pure stdlib).
- Produces:
  - `score_inference(trace: dict, val: dict, corpus_rows: list[dict] | None, cap: int, id_col: str | None = None, text_col: str | None = None) -> dict`
    returning keys: `query, attr, n, correct, accuracy, mistakes, n_mistakes, sampled`.
    Each `mistakes` item: `{ "id": str, "text": str, "predicted": str | None, "expected": str }`.
  - `_corpus_text_map(rows, id_col, text_col) -> dict[str, str]` — id→text snippet map (best-effort col inference).
  - CLI: `--score-inference` flag + `--trace`, `--val-file`, `--corpus-csv`, `--id-col`, `--text-col`; writes the scorer JSON to `--emit-diff <path>`; prints accuracy; returns before the CSV-append block. `--csv` becomes optional.

- [ ] **Step 1: Write the failing test**

```python
# src/semdb/tests/test_score_inference.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import evaluate as E


def _val():
    return {"query": "q3a", "attr": "genre",
            "labels": {"m1": "comedy", "m2": "drama", "m3": "Comedy", "m4": "drama"}}


def test_accuracy_and_normalization():
    # m1 correct; m2 correct; m3 correct (case/strip normalized); m4 wrong.
    trace = {"attr": "genre", "rows": {"m1": "comedy", "m2": "drama", "m3": " COMEDY ", "m4": "comedy"}}
    d = E.score_inference(trace, _val(), corpus_rows=None, cap=15)
    assert d["n"] == 4 and d["correct"] == 3
    assert abs(d["accuracy"] - 0.75) < 1e-9
    assert d["n_mistakes"] == 1
    assert d["mistakes"][0]["id"] == "m4"
    assert d["mistakes"][0]["predicted"] == "comedy" and d["mistakes"][0]["expected"] == "drama"


def test_missing_id_scores_wrong_with_null_predicted():
    trace = {"attr": "genre", "rows": {"m1": "comedy", "m2": "drama", "m3": "comedy"}}  # m4 absent
    d = E.score_inference(trace, _val(), corpus_rows=None, cap=15)
    assert d["correct"] == 3 and d["n"] == 4
    miss = [m for m in d["mistakes"] if m["id"] == "m4"]
    assert miss and miss[0]["predicted"] is None


def test_mistakes_capped():
    val = {"query": "q", "attr": "g", "labels": {f"m{i}": "a" for i in range(20)}}
    trace = {"attr": "g", "rows": {f"m{i}": "b" for i in range(20)}}  # all wrong
    d = E.score_inference(trace, val, corpus_rows=None, cap=5)
    assert d["n_mistakes"] == 20 and len(d["mistakes"]) == 5 and d["sampled"] is True


def test_corpus_text_snippet_lookup():
    trace = {"attr": "genre", "rows": {"m1": "drama"}}
    val = {"query": "q", "attr": "genre", "labels": {"m1": "comedy"}}
    rows = [{"id": "m1", "overview": "a hilarious comedy about " + "x" * 500}]
    d = E.score_inference(trace, val, corpus_rows=rows, cap=15)
    assert d["mistakes"][0]["id"] == "m1"
    assert d["mistakes"][0]["text"].startswith("a hilarious comedy")
    assert len(d["mistakes"][0]["text"]) <= 220   # truncated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/semdb && python3 -m pytest tests/test_score_inference.py -v`
Expected: FAIL — `AttributeError: module 'evaluate' has no attribute 'score_inference'`.

- [ ] **Step 3: Implement `score_inference` + `_corpus_text_map`**

In `src/semdb/evaluate.py`, add after `diff_pair` (near line 210, before `handler_id`):

```python
_TEXT_RE = re.compile(r"text|description|overview|body|content|summary|symptoms|review|complaint|plot|display", re.I)


def _corpus_text_map(rows, id_col=None, text_col=None):
    """id -> a short text snippet, best-effort. id_col defaults to 'id' or the first
    column; text_col defaults to the first text-ish column or the last column."""
    if not rows:
        return {}
    keys = list(rows[0].keys())
    idc = id_col or ("id" if "id" in keys else keys[0])
    txc = text_col or next((k for k in keys if _TEXT_RE.search(k)), keys[-1])
    out = {}
    for r in rows:
        rid = str(r.get(idc, "")).strip()
        if rid:
            out[rid] = str(r.get(txc, "")).strip()
    return out


def _norm(v):
    """Match evaluate.py membership normalization: str, stripped, lowercased."""
    return None if v is None else str(v).strip().lower()


def score_inference(trace, val, corpus_rows, cap, id_col=None, text_col=None):
    """Per-row inference accuracy of a DIRECT solver's trace_<q>.json against a
    hand-labeled val.json. Compares trace.rows[id] to val.labels[id] over the LABELED
    ids only (normalized). A labeled id absent from the trace counts as wrong with
    predicted=None. Returns accuracy + capped mistake samples (with a text snippet)."""
    attr = val.get("attr", trace.get("attr", ""))
    labels = val.get("labels", {})
    pred_rows = trace.get("rows", {}) if isinstance(trace.get("rows"), dict) else {}
    text_map = _corpus_text_map(corpus_rows, id_col, text_col)
    correct = 0
    mistakes = []
    for rid, expected in labels.items():
        sid = str(rid)
        raw_pred = pred_rows.get(sid, pred_rows.get(rid, None))
        if raw_pred is not None and _norm(raw_pred) == _norm(expected):
            correct += 1
        else:
            snippet = text_map.get(sid, "")[:200]
            mistakes.append({"id": sid, "text": snippet,
                             "predicted": None if raw_pred is None else str(raw_pred),
                             "expected": str(expected)})
    n = len(labels)
    return {
        "query": val.get("query", trace.get("query", "")),
        "attr": attr, "n": n, "correct": correct,
        "accuracy": round(correct / n, 4) if n else None,
        "mistakes": mistakes[:cap], "n_mistakes": len(mistakes),
        "sampled": len(mistakes) > cap,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src/semdb && python3 -m pytest tests/test_score_inference.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Wire the `--score-inference` CLI branch**

In `main()`, make `--csv` optional and add the new args. Change the existing line:

```python
    ap.add_argument("--csv", required=True, help="output CSV (row appended; header written if new)")
```
to:
```python
    ap.add_argument("--csv", help="output CSV (row appended; header written if new)")
    ap.add_argument("--score-inference", action="store_true",
                    help="per-row inference scoring mode: score --trace vs --val-file")
    ap.add_argument("--trace", help="trace_<q>.json from a DIRECT solver (id -> inferred attr value)")
    ap.add_argument("--val-file", help="hand-labeled val.json (id -> expected attr value)")
    ap.add_argument("--corpus-csv", help="text corpus CSV, for mistake text snippets")
    ap.add_argument("--id-col", help="corpus id column (default: 'id' or first column)")
    ap.add_argument("--text-col", help="corpus text column (default: first text-ish or last column)")
```

Immediately after `args = ap.parse_args()`, add the early branch + the guard:

```python
    if args.score_inference:
        if not (args.trace and args.val_file):
            ap.error("--score-inference requires --trace and --val-file")
        trace = json.load(open(args.trace)) if os.path.exists(args.trace) else {"rows": {}}
        val = json.load(open(args.val_file))
        corpus_rows = None
        if args.corpus_csv and os.path.exists(args.corpus_csv):
            corpus_rows, _ = load_pred_rows(args.corpus_csv)
        out = score_inference(trace, val, corpus_rows, args.diff_cap, args.id_col, args.text_col)
        if args.emit_diff:
            json.dump(out, open(args.emit_diff, "w"), indent=2)
        print(f"[eval] inference accuracy {out['accuracy']} "
              f"({out['correct']}/{out['n']}, {out['n_mistakes']} wrong) -> {args.emit_diff}")
        return
    if not args.csv:
        ap.error("--csv is required unless --score-inference is set")
```

(The existing body below is unchanged and still requires `--csv` for the scoring/CSV path.)

- [ ] **Step 6: Add a CLI round-trip test**

Append to `tests/test_score_inference.py`:

```python
def test_cli_score_inference_writes_json(tmp_path):
    import json, subprocess, sys as _sys
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"attr": "genre", "rows": {"m1": "comedy", "m2": "comedy"}}))
    val = tmp_path / "val.json"
    val.write_text(json.dumps({"query": "q3a", "attr": "genre",
                               "labels": {"m1": "comedy", "m2": "drama"}}))
    out = tmp_path / "score.json"
    here = os.path.dirname(__file__)
    r = subprocess.run([_sys.executable, os.path.join(here, "..", "evaluate.py"),
                        "--score-inference", "--trace", str(trace), "--val-file", str(val),
                        "--emit-diff", str(out), "--diff-cap", "15"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = json.load(open(out))
    assert d["n"] == 2 and d["correct"] == 1 and d["n_mistakes"] == 1
    assert d["mistakes"][0]["id"] == "m2"
```

- [ ] **Step 7: Run tests**

Run: `cd src/semdb && python3 -m pytest tests/test_score_inference.py -v`
Expected: PASS (5 tests).

- [ ] **Step 8: Commit**

```bash
git add src/semdb/evaluate.py src/semdb/tests/test_score_inference.py
git commit -m "feat(semdb): evaluate.py score_inference + --score-inference per-row scorer"
```

---

## Task 2: `--val-file` flag + per-row `renderFeedback` branch

**Files:**
- Modify: `src/semdb/orchestrator.mjs` (`parseArgs`, `renderFeedback`)
- Test: `src/semdb/tests/test_val_feedback.mjs`

**Interfaces:**
- Consumes: existing `renderFeedback(prev)` shape `{ status, f1, metrics, diff, stderrTail, history }`.
- Produces:
  - `args.valFile` (string | null) on the parsed args object.
  - `renderFeedback` renders a per-row block when `prev.diff` has a `mistakes` array (else the existing FP/FN block). Exported for testing (add `renderFeedback` to the `export`-ed functions, or export it explicitly).

- [ ] **Step 1: Add `--val-file` to `parseArgs`**

In the `args` object literal in `parseArgs`, add:

```js
    valFile: null,
```

In the arg-parsing loop (next to `--max-iterations`):

```js
    else if (a === "--val-file" && argv[i + 1]) args.valFile = resolve(argv[++i]);
```

- [ ] **Step 2: Write the failing test**

```js
// src/semdb/tests/test_val_feedback.mjs
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
console.log("test_val_feedback OK");
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd src/semdb && node tests/test_val_feedback.mjs`
Expected: FAIL — either `renderFeedback` is not exported (`does not provide an export named 'renderFeedback'`) or the per-row assertions fail.

- [ ] **Step 4: Export `renderFeedback` and add the per-row branch**

Change the declaration `function renderFeedback(prev) {` to `export function renderFeedback(prev) {`.

Then, immediately after the `if (prev.status !== "ok") { ... }` block (before `const m = prev.metrics || {};`), insert the per-row branch:

```js
  // Per-row validation mode: diff carries `mistakes` (id/text/predicted/expected).
  if (prev.diff && Array.isArray(prev.diff.mistakes)) {
    const pm = prev.metrics || {};
    const rows = (prev.diff.mistakes || [])
      .map((r) => `  - id=${r.id} predicted=${r.predicted == null ? "MISSING" : r.predicted} expected=${r.expected}  text="${(r.text || "").slice(0, 160)}"`)
      .join("\n") || "  (none)";
    return [
      `\n## LAST RUN — PER-ROW INFERENCE accuracy=${prev.f1} (${pm.correct ?? "?"}/${pm.n ?? "?"} labeled rows correct)`,
      `## MISLABELED ROWS — ${prev.diff.n_mistakes ?? 0} total, showing ${(prev.diff.mistakes || []).length}:`,
      rows,
      histLines ? `## HISTORY\n${histLines}` : "",
      "Each row above was inferred WRONG for the query's key attribute. Diagnose WHY:",
      "wrong value-space mapping, an over/under-broad judge/classify prompt, a bad",
      "threshold, or the wrong attribute entirely. Revise the judge/classify/extract",
      "call in the solver. Edit the existing program in place. Keep writing trace_<q>.json.",
    ].join("\n");
  }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd src/semdb && node tests/test_val_feedback.mjs`
Expected: `test_val_feedback OK`.

- [ ] **Step 6: Sanity-check the module still imports**

Run: `cd src/semdb && node --check orchestrator.mjs && node -e "import('./orchestrator.mjs').then(()=>console.log('import OK'))"`
Expected: `import OK` with no `[SemDB]` output.

- [ ] **Step 7: Commit**

```bash
git add src/semdb/orchestrator.mjs src/semdb/tests/test_val_feedback.mjs
git commit -m "feat(semdb): --val-file flag + per-row renderFeedback branch"
```

---

## Task 3: Solver-prompt contract — `trace_<q>.json` + `--only-ids`

**Files:**
- Modify: `src/semdb/agents/vadar-solver/prompt-text.md`

**Interfaces:**
- Consumes: nothing (prompt text only).
- Produces: generated text `solve_<q>.py` now (a) writes `trace_<q>.json` mapping every processed row id → inferred key-attribute value, and (b) honors an optional `--only-ids <file>` CLI arg (newline-separated ids) restricting which rows it infers.

- [ ] **Step 1: Add the trace + `--only-ids` contract to the solver text system prompt**

Open `src/semdb/agents/vadar-solver/prompt-text.md`. Find the section that lists what the program must write (it currently documents the result CSV + `solve_<q>.meta.json`). Add, in the same list/section, this block verbatim:

````markdown
### ADDITIONAL OUTPUT (per-row validation support) — REQUIRED

Besides the result CSV and `solve_<q>.meta.json`, the program MUST also write
`trace_<q>.json` next to the result CSV. It records, for EVERY row you ran
semantic inference on, the value you inferred for the query's KEY semantic
attribute (the attribute the WHERE/label depends on) — BEFORE any relational
filter drops the row:

```json
{ "attr": "<key attribute name>", "rows": { "<row_id>": "<inferred value>", ... } }
```

- `<row_id>` is the corpus primary-key value (a string) for that row.
- For a boolean predicate (AI.IF / judge), the value is the string `"true"` or
  `"false"`.
- For a classify/extract attribute, the value is the inferred label / field value.
- Accumulate into a dict as you iterate; `json.dump` it once at the end.

The program MUST also accept an OPTIONAL CLI arg `--only-ids <path>`: when given,
`<path>` is a newline-separated list of row ids; restrict semantic inference (and
the trace + result rows) to ONLY those ids. When absent, process the whole corpus.
Implement it as a simple membership filter right after you load the corpus rows:

```python
import argparse
ap = argparse.ArgumentParser()
# ... existing args (results_csv, --data-dir, ...) ...
ap.add_argument("--only-ids")
a = ap.parse_args()
only = None
if a.only_ids:
    with open(a.only_ids) as f:
        only = {ln.strip() for ln in f if ln.strip()}
# after loading corpus rows:
if only is not None:
    rows = [r for r in rows if str(r[<id_col>]).strip() in only]
```
````

- [ ] **Step 2: Verify the prompt still renders (no broken template vars)**

Run:
```bash
cd src/semdb && node orchestrator.mjs --query q3a --direct \
  --query-dir /localhome/hza214/SemBench/files/movie/query/bigquery \
  --data-dir /localhome/hza214/SemBench/files/movie/data \
  --dry-run 2>&1 | grep -c "trace_" || true
```
Expected: a count ≥ 1 (the dry-run prints the solver prompt, which now contains `trace_`). If the query/data paths differ on this machine, substitute a real text scenario; the assertion is only that the printed prompt contains `trace_`.

- [ ] **Step 3: Commit**

```bash
git add src/semdb/agents/vadar-solver/prompt-text.md
git commit -m "feat(semdb): text solver contract — emit trace_<q>.json + honor --only-ids"
```

---

## Task 4: Wire the val loop into `runQueryDirect` (text branch)

**Files:**
- Modify: `src/semdb/orchestrator.mjs` (`runQueryDirect`, new `scoreInference` helper)

**Interfaces:**
- Consumes: `renderFeedback` (per-row branch, Task 2), `refineLoop`, `defaults.refineSampleCap`, `evaluate.py --score-inference` (Task 1), the solver `--only-ids`/`trace_<q>.json` contract (Task 3).
- Produces: `runQueryDirect` runs a per-row val loop when `args.valFile && !isImage`, then a final full-corpus run; telemetry `refine.mode`.
- Adds helper `scoreInference(args, query, iterDir, traceName, corpusCsv, diffPath, valFile) -> { status, f1, metrics, diff }`.

- [ ] **Step 1: Add the `scoreInference` helper**

In `src/semdb/orchestrator.mjs`, add directly above `scoreWithDiff` (search for `async function scoreWithDiff`):

```js
/** Per-row validation scoring for one iteration: run evaluate.py --score-inference on
 *  the solver's trace_<q>.json vs val.json, read accuracy + mistakes back. The loop's
 *  generic `f1` field carries accuracy here. status "ok" requires a trace file. */
async function scoreInference(args, query, iterDir, corpusCsv, valFile, diffPath) {
  const tracePath = resolve(iterDir, `trace_${query}.json`);
  if (!existsSync(tracePath)) return { status: "empty", f1: null, metrics: null, diff: null };
  const evArgs = [resolve(__dirname, "evaluate.py"), "--score-inference",
    "--trace", tracePath, "--val-file", valFile,
    "--emit-diff", diffPath, "--diff-cap", String(defaults.refineSampleCap),
    ...(corpusCsv ? ["--corpus-csv", corpusCsv] : [])];
  const ev = spawnSync("python3", evArgs, { stdio: "inherit" });
  if (ev.status !== 0) console.warn(`[SemDB] evaluate.py --score-inference exited ${ev.status}.`);
  const out = await readJSON(diffPath);
  if (!out) return { status: "ok", f1: null, metrics: null, diff: null };
  return {
    status: "ok", f1: out.accuracy,
    metrics: { accuracy: out.accuracy, n: out.n, correct: out.correct },
    diff: { mistakes: out.mistakes || [], n_mistakes: out.n_mistakes ?? 0, sampled: out.sampled },
  };
}
```

- [ ] **Step 2: Compute val-mode state at the top of the text wiring**

In `runQueryDirect`, right after `const telePath = resolve(runDir, "telemetry.json");` (near line 1003), add:

```js
  // Per-row validation mode (text corpora only): a hand-labeled val.json is the only GT.
  const valMode = !!args.valFile && !isImage && !args.noRefine;
  const val = valMode ? await readJSON(args.valFile) : null;
  const valIdsPath = resolve(runDir, "_val_ids.txt");
  if (valMode) {
    if (!val || !val.labels) throw new Error(`[SemDB] --val-file ${args.valFile} has no "labels"`);
    await writeFile(valIdsPath, Object.keys(val.labels).join("\n") + "\n");
    console.log(`[SemDB] [${query}] per-row validation mode: ${Object.keys(val.labels).length} labeled rows from ${args.valFile}`);
  }
```

- [ ] **Step 3: Let `runSolver` take an `--only-ids` file**

Change the `runSolver` signature and text `sArgs` (near line 1015). Replace:

```js
  const runSolver = (iterDir, iterCode, iterCsv) => {
```
with:
```js
  const runSolver = (iterDir, iterCode, iterCsv, onlyIds = null) => {
```

And replace the text-branch `sArgs` line (the `: [iterCode, iterCsv, "--data-dir", dataDir];` case) with:

```js
      : [iterCode, iterCsv, "--data-dir", dataDir,
         ...(onlyIds ? ["--only-ids", onlyIds] : [])];
```

- [ ] **Step 4: Pass `--only-ids` on iteration runs**

In `genFirst` and `regen` (near lines 1080-1087), pass the val ids file so iterations touch only labeled rows. Replace:

```js
  const genFirst = async (iterDir, iterCode, iterCsv) => {
    await gen3Agents(iterDir, iterCode, sql);
    return runSolver(iterDir, iterCode, iterCsv);
  };
  const regen = async (iterDir, iterCode, iterCsv, feedback) => {
    await regenSolver(iterDir, iterCode, feedback);
    return runSolver(iterDir, iterCode, iterCsv);
  };
```
with:
```js
  const genFirst = async (iterDir, iterCode, iterCsv) => {
    await gen3Agents(iterDir, iterCode, sql);
    return runSolver(iterDir, iterCode, iterCsv, valMode ? valIdsPath : null);
  };
  const regen = async (iterDir, iterCode, iterCsv, feedback) => {
    await regenSolver(iterDir, iterCode, feedback);
    return runSolver(iterDir, iterCode, iterCsv, valMode ? valIdsPath : null);
  };
```

- [ ] **Step 5: Branch `scoreIter` on val mode**

Replace the existing `scoreIter` (near line 1088):

```js
  const scoreIter = async (iterDir, iterCode, iterCsv, run) => {
    const diffPath = resolve(iterDir, "diff.json");
    const scored = await scoreWithDiff(args, query, planObj, telePath, iterCsv, diffPath, csvPath, false);
    const status = run.status === "crash" ? "crash" : (existsSync(iterCsv) ? "ok" : "empty");
    return { ...scored, status, stderrTail: (run.stderr || "").split("\n").slice(-40).join("\n") };
  };
```
with:
```js
  const scoreIter = async (iterDir, iterCode, iterCsv, run) => {
    const diffPath = resolve(iterDir, "diff.json");
    if (valMode) {
      const scored = await scoreInference(args, query, iterDir, corpus.path, args.valFile, diffPath);
      // "ok" requires a trace to score; without one the agent must fix-first.
      const traceOk = existsSync(resolve(iterDir, `trace_${query}.json`));
      const status = run.status === "crash" ? "crash" : (traceOk ? "ok" : "empty");
      return { ...scored, status, stderrTail: (run.stderr || "").split("\n").slice(-40).join("\n") };
    }
    const scored = await scoreWithDiff(args, query, planObj, telePath, iterCsv, diffPath, csvPath, false);
    const status = run.status === "crash" ? "crash" : (existsSync(iterCsv) ? "ok" : "empty");
    return { ...scored, status, stderrTail: (run.stderr || "").split("\n").slice(-40).join("\n") };
  };
```

- [ ] **Step 6: Add the final full-corpus run after the loop (val mode)**

After the `refineLoop` call (near line 1097-1099) and BEFORE the telemetry `report` assembly, add:

```js
  // Val mode: the loop scored the frozen solver on the LABELED sub-corpus only. Now run
  // the promoted best solver over the FULL corpus (NO --only-ids) to produce the real
  // result CSV. No-leakage: the reported output covers all rows, not just labeled ones.
  if (valMode && doRun) {
    const bestCode = resolve(runDir, codeBasename);
    console.log(`[SemDB] [${query}] val mode: final full-corpus run of the frozen solver.`);
    runSolver(runDir, bestCode, resultsCsv, null);
  }
```

- [ ] **Step 7: Record the refine mode in telemetry**

In the `report` object (near line 1114), replace the `refine:` block with:

```js
    refine: { mode: valMode ? "per_row_val" : "f1",
              ...(valMode ? { objective: "inference_accuracy", val_file: args.valFile,
                              val_n: Object.keys(val.labels).length } : {}),
              iterations: history.length - 1, best_iteration: bestIter,
              max_iterations: args.noRefine ? 0 : args.maxIterations,
              f1_history: history },
```

- [ ] **Step 8: Syntax check**

Run: `cd src/semdb && node --check orchestrator.mjs`
Expected: no output (exit 0).

- [ ] **Step 9: Dry-run smoke (val mode, no execution)**

Run:
```bash
cd src/semdb && node orchestrator.mjs --query q3a --direct \
  --query-dir /localhome/hza214/SemBench/files/movie/query/bigquery \
  --data-dir /localhome/hza214/SemBench/files/movie/data \
  --val-file tests/fixtures/val_q3a.json --dry-run 2>&1 | tail -20
```
Expected: prints the 3-agent solver prompts (including the `trace_` contract); no crash. (In `--dry-run`, `gen3Agents` runs then returns before the loop, so val execution is skipped — confirm no error is thrown by the val-mode setup.)

> Note: the fixture `tests/fixtures/val_q3a.json` is created in Task 5. If running this step before Task 5, create a minimal `{"query":"q3a","attr":"genre","labels":{"m1":"comedy"}}` first, or run Task 5 Step 1 ahead of this smoke.

- [ ] **Step 10: Commit**

```bash
git add src/semdb/orchestrator.mjs
git commit -m "feat(semdb): per-row val loop wired into DIRECT text branch + final full-corpus run"
```

---

## Task 5: Fixture + end-to-end verification

**Files:**
- Create: `src/semdb/tests/fixtures/val_q3a.json`
- Test: manual end-to-end + regression checks (documented commands)

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: a runnable validation fixture + a documented verification that the full path works and regressions are clean.

- [ ] **Step 1: Create the fixture**

Create `src/semdb/tests/fixtures/val_q3a.json` (adjust ids/attr to the real movie scenario's key column + a genuinely labelable attribute; the shape is what matters):

```json
{
  "query": "q3a",
  "attr": "genre",
  "labels": {
    "m1": "comedy",
    "m2": "drama",
    "m3": "comedy",
    "m4": "documentary",
    "m5": "drama"
  }
}
```

- [ ] **Step 2: Unit + wiring tests all pass**

Run:
```bash
cd src/semdb && python3 -m pytest tests/test_score_inference.py -v \
  && node tests/test_val_feedback.mjs \
  && node tests/test_refine_pure.mjs \
  && node --check orchestrator.mjs
```
Expected: pytest PASS (5), `test_val_feedback OK`, `test_refine_pure OK`, no syntax errors.

- [ ] **Step 3: Scorer integration on a synthetic trace (no endpoint needed)**

Prove the scorer + emit path end-to-end without an LLM:
```bash
cd src/semdb && python3 - <<'PY'
import json, tempfile, os, subprocess, sys
d = tempfile.mkdtemp()
json.dump({"attr":"genre","rows":{"m1":"comedy","m2":"comedy","m3":"comedy"}},
          open(os.path.join(d,"trace.json"),"w"))
json.dump({"query":"q3a","attr":"genre","labels":{"m1":"comedy","m2":"drama","m3":"comedy","m4":"drama"}},
          open(os.path.join(d,"val.json"),"w"))
out=os.path.join(d,"score.json")
r=subprocess.run([sys.executable,"evaluate.py","--score-inference","--trace",os.path.join(d,"trace.json"),
                  "--val-file",os.path.join(d,"val.json"),"--emit-diff",out,"--diff-cap","15"])
s=json.load(open(out))
assert s["n"]==4 and s["correct"]==2, s
assert {m["id"] for m in s["mistakes"]}=={"m2","m4"}, s   # m2 wrong value, m4 missing
assert [m for m in s["mistakes"] if m["id"]=="m4"][0]["predicted"] is None
print("scorer integration OK:", s["accuracy"])
PY
```
Expected: `scorer integration OK: 0.5`.

- [ ] **Step 4: End-to-end (requires a text scenario + endpoint; run under sembench)**

> Run only if a text scenario + LLM endpoint are available on this machine. Substitute real paths/ids in the fixture first.

```bash
source $HOME/anaconda3/etc/profile.d/conda.sh && conda activate sembench
cd src/semdb && node orchestrator.mjs --query q3a --direct \
  --query-dir /localhome/hza214/SemBench/files/movie/query/bigquery \
  --data-dir /localhome/hza214/SemBench/files/movie/data \
  --endpoint http://localhost:8000/v1 --model <model-id> \
  --val-file tests/fixtures/val_q3a.json --out output/val-smoke 2>&1 | tail -40
```
Verify in `output/val-smoke/movie-q3a/`:
- `iter_0/trace_q3a.json` exists and maps labeled ids → inferred values.
- `iter_0/diff.json` has `accuracy`, `mistakes`.
- ≥1 `iter_N/` dir exists (a refinement iteration ran).
- `q3a_results.csv` at the run root has MORE rows than the 5 labeled ids (proves the final full-corpus run happened, not just the sub-corpus).
- `telemetry.json` → `refine.mode == "per_row_val"`, `refine.val_n == 5`, `f1_history` populated with accuracy values.

- [ ] **Step 5: Regression — no `--val-file` is unchanged**

Confirm the F1 loop / single-shot paths are untouched:
```bash
cd src/semdb && node orchestrator.mjs --query q6 \
  --query-dir /localhome/hza214/SemBench/files/mmqa/query/bigquery \
  --data-dir /localhome/hza214/SemBench/files/mmqa/data/sf_200 \
  --dry-run --no-refine 2>&1 | tail -5
```
Expected: prints the Code Generator prompt; no crash; no `per-row validation mode` line (val mode off).

- [ ] **Step 6: Commit**

```bash
git add src/semdb/tests/fixtures/val_q3a.json
git commit -m "test(semdb): val-mode fixture + end-to-end verification notes"
```

---

## Self-Review Notes (coverage vs spec)

- Spec Part 1 (`val.json` format, `--val-file`) → Task 2 Step 1 (flag), Task 5 Step 1 (fixture), consumed in Task 4.
- Spec Part 2 (`trace_<q>.json` contract) → Task 3.
- Spec Part 3 (scorer, accuracy, missing-id=wrong, cap, snippet) → Task 1.
- Spec Part 4 (loop integration, run-scope, precedence, final full run, no-leakage) → Task 4 (Steps 2,4,5,6) + Global Constraints.
- Spec Part 5 (telemetry `mode`) → Task 4 Step 7.
- Out-of-scope items (Extractor/compiled refinement, few-shot injection, multi-attr, image val) → not implemented, by design.
```
