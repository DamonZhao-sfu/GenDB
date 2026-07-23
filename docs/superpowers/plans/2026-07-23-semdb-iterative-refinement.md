# SemDB Iterative Refinement + Text DIRECT Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GenDB-style iterative refinement loop (max 5 iterations) around every per-query SemDB code generator, and extend `--direct` so text-only SemBench queries compile into an end-to-end `solve_<q>.py` that does semantic inference in code by composing a predefined text API.

**Architecture:** `evaluate.py` gains `--emit-diff` to surface the concrete false-positive/false-negative rows that cost precision/recall. `orchestrator.mjs` gains a `refineLoop` helper (shared by the compiled path, image-DIRECT, and text-DIRECT) that re-invokes the same generating agent with a feedback block, re-runs, re-scores, and keeps the change only if F1 improves (else rolls back), stopping on F1==1.0 / 2 non-improving / max 5. Text DIRECT adds `vadar/predefined_text.py` + `semtext.py` (the text analogs of `predefined.py` + `semvision.py`) and modality-aware system prompts for the existing VADAR 3 agents. In the SchemaDesigner→Extractor→CodeGenerator (compiled) mode, the loop already iterates `compiled_<q>.py`; the text **Extractor** is additionally moved to compose the same `predefined_text` API through a new `semtext.run_extraction` engine (the text analog of how the image Extractor composes the vision API via `vadar_engine.run`), so offline text extraction and text DIRECT share one inference surface.

**Tech Stack:** Node.js ESM (`orchestrator.mjs`, agent `index.mjs`), Python 3.10+ stdlib (`evaluate.py`, `semtext.py`, `predefined_text.py`), pytest for Python tests, OpenAI-compatible endpoint via `semextract.gen_endpoint`.

## Global Constraints

- Max refinement iterations default **5** (`defaults.maxRefineIterations`); stall threshold **2** (`defaults.refineStallThreshold`); FP/FN sample cap **15** (`defaults.refineSampleCap`). Copy these exact values.
- The loop engages ONLY when ground truth is available AND `--no-refine` is not set; otherwise behavior is byte-for-byte today's single-shot path (iteration 0 only).
- Corpus-level Schema Designer + Extractor stay one-shot — the loop is per-query only.
- `--emit-diff` is additive: it must NOT change existing metric output, the appended `results.csv` row, or `telemetry.json["metrics"]`.
- The best (highest-F1) iteration's results CSV + telemetry are what get scored into the workload `results.csv`.
- Text semantic inference reuses `semextract.gen_endpoint` + `_Cfg` for endpoint calls — do not write a second HTTP client.
- Value spaces (closed label sets) are read from the CSV at runtime — never hardcoded in generated code.
- Python: PEP 8, type annotations on new function signatures, `logging` not `print` for library diagnostics (existing CLIs use `print` for user output — match the file you edit).
- Run Python under the sembench conda env for scenario scoring: `source $HOME/anaconda3/etc/profile.d/conda.sh && conda activate sembench`.

---

## File Structure

**New**
- `src/semdb/semtext.py` — endpoint-backed text-inference backend: `TextPatch`, a shared `TextCtx`, an in-process cache, a `METER` call counter. ~150 lines.
- `src/semdb/vadar/predefined_text.py` — free-function text API (`judge/classify/extract/generate/score`) + `MODULES_SIGNATURES_TEXT`. ~120 lines.
- `src/semdb/agents/vadar-signature/prompt-text.md`
- `src/semdb/agents/vadar-api/prompt-text.md`
- `src/semdb/agents/vadar-solver/prompt-text.md`
- `src/semdb/tests/test_evaluate_emit_diff.py`
- `src/semdb/tests/test_semtext_fake_endpoint.py`
- `src/semdb/tests/test_predefined_text.py`
- `src/semdb/tests/test_semtext_run_extraction.py` — the offline text-extraction engine (compiled-mode feature 2).
- `src/semdb/tests/test_refine_pure.mjs` — node assertions for the pure loop-control functions.

**Modified**
- `src/semdb/evaluate.py` — refactor mmqa handlers to return `(results, gold)`; add `score_pair`, `diff_pair`; add `--emit-diff`; scenario id-set diff.
- `src/semdb/semdb.config.mjs` — add `maxRefineIterations`, `refineStallThreshold`, `refineSampleCap` to `defaults`.
- `src/semdb/orchestrator.mjs` — `parseArgs` (`--max-iterations`, `--no-refine`); `runPhase` (`systemPromptPath` override); `shouldContinueSemdb`, `checkSemdbImprovement`, `renderFeedback`, `refineLoop`; wire into `runQueryCodegen`, `runQueryDirect` (image + text branches); text routing.
- `src/semdb/agents/vadar-signature/index.mjs`, `.../vadar-api/index.mjs`, `.../vadar-solver/index.mjs` — add `promptPathText`.
- `src/semdb/semtext.py` — add `run_extraction(...)` (offline corpus extraction engine composing `predefined_text`; matches `semextract.run`'s attrs+meta output) and make `METER` increment thread-safe. (compiled-mode feature 2)
- `src/semdb/agents/extractor/prompt.md`, `.../extractor/user-prompt.md` — the TEXT driver composes `predefined_text` over `semtext.TextPatch` and calls `semtext.run_extraction` (the text analog of the image driver's `vadar_engine.run`). (compiled-mode feature 2)

> **Compiled-mode coverage of the two features.** Feature 1 (the refine loop) already iterates `compiled_<q>.py` — Task 4. Feature 2 (in-code text semantic inference) is added to the compiled path by moving the offline **text Extractor** onto `predefined_text` via `semtext.run_extraction` (Tasks 10–11); inference stays offline/amortized, so `compiled_<q>.py` remains pure relational and the loop needs no change.

---

## Task 1: `evaluate.py --emit-diff` — the feedback signal

**Files:**
- Modify: `src/semdb/evaluate.py`
- Test: `src/semdb/tests/test_evaluate_emit_diff.py`

**Interfaces:**
- Consumes: existing `MMQA_HANDLERS`, `score`, `load_pred_rows`, `handler_id`.
- Produces:
  - `score_pair(results, gold) -> dict` — the metric row (same shape as `score`).
  - `diff_pair(results, gold, cap) -> dict` with keys `false_positives`, `false_negatives`, `fp_total`, `fn_total`, `sampled`.
  - each `eval_qN(rows, fields, gt) -> (results, gold)` (was: metric row).
  - `eval_mmqa(query, pred_path, gt_path) -> dict` (unchanged return: metric row).
  - `eval_mmqa_diff(query, pred_path, gt_path, cap) -> dict` (the diff dict).
  - new CLI arg `--emit-diff <path>`; when set, writes the diff JSON described in spec §1.3.

- [ ] **Step 1: Write the failing test**

```python
# src/semdb/tests/test_evaluate_emit_diff.py
import json, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import evaluate as E


def test_score_pair_matches_legacy_score():
    results = ["a", "b", "c"]
    gold = {"a", "b"}
    assert E.score_pair(results, gold) == E.score(results, gold)


def test_diff_pair_lists_fp_and_fn_with_cap():
    results = ["a", "b", "x", "y"]         # x,y are false positives
    gold = {"a", "b", "z"}                 # z is a false negative
    d = E.diff_pair(results, gold, cap=1)
    assert d["fp_total"] == 2 and d["fn_total"] == 1
    assert d["sampled"] is True            # cap=1 < 2 fps
    assert len(d["false_positives"]) == 1 and len(d["false_negatives"]) == 1
    assert d["false_positives"][0] in ("x", "y")
    assert d["false_negatives"] == ["z"]


def test_emit_diff_writes_json_for_mmqa_q6(tmp_path):
    # q6 = list of Airlines vs a set; simplest mmqa handler.
    pred = tmp_path / "q6.csv"
    pred.write_text("Airlines\nDelta\nBogusAir\n")
    gt = tmp_path / "Q6.json"
    gt.write_text(json.dumps({"ground_truth": ["Delta", "United"]}))
    out = tmp_path / "diff.json"
    d = E.eval_mmqa_diff("q6", str(pred), str(gt), cap=15)
    assert "BogusAir" in d["false_positives"]
    assert "United" in d["false_negatives"]
    # round-trip the writer used by main()
    json.dump(d, open(out, "w"))
    assert json.load(open(out))["fp_total"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/semdb && python3 -m pytest tests/test_evaluate_emit_diff.py -v`
Expected: FAIL — `AttributeError: module 'evaluate' has no attribute 'score_pair'`.

- [ ] **Step 3: Refactor handlers + add score_pair/diff_pair/eval_mmqa_diff**

In `src/semdb/evaluate.py`, change each `eval_qN` to return `(results, gold)` instead of `score(results, gold)`. Concretely, replace the trailing `return score(results, gold)` / `return score(results, set(gt))` / `return score([], set())` in `eval_q1`..`eval_q7` with `return results, gold` (build `gold` first where the current code inlines it). Example for `eval_q1`:

```python
def eval_q1(rows, fields, gt):
    results = [str(r.get("director", "")).strip(' "').lower() for r in rows]
    gold = {g.strip().lower() for g in gt}
    return results, gold
```

Apply the same pattern to q2–q7 (each already computes `results` and a gold set/dict; return the pair). For `eval_q3`/`eval_q6` where gold is `set(gt)`, return `results, set(gt)`. For `eval_q4`, return `results, gold` (the built set).

Then add, after `MMQA_HANDLERS`:

```python
def score_pair(results, gold):
    """Metric row from a normalized (results, gold) pair — same as score()."""
    return score(results, gold)


def diff_pair(results, gold, cap):
    """Concrete FP/FN items behind the P/R/F1. FP = predicted items not in gold;
    FN = gold items not predicted. Both capped at `cap` (order-stable, de-duplicated
    for FP over a list so a repeated wrong item is shown once). Renders each item as
    a JSON-safe value (tuples -> lists)."""
    def _jsonable(x):
        return list(x) if isinstance(x, tuple) else x
    seen = set()
    fp = []
    for item in results:
        if item in gold or item in seen:
            continue
        seen.add(item)
        fp.append(item)
    pred_set = set(results)
    fn = [g for g in gold if g not in pred_set]
    return {
        "false_positives": [_jsonable(x) for x in fp[:cap]],
        "false_negatives": [_jsonable(x) for x in fn[:cap]],
        "fp_total": len(fp),
        "fn_total": len(fn),
        "sampled": len(fp) > cap or len(fn) > cap,
    }


def eval_mmqa_diff(query, pred_path, gt_path, cap):
    """Same dispatch as eval_mmqa, but return the FP/FN diff instead of the metric row."""
    hid = handler_id(query)
    if hid not in MMQA_HANDLERS:
        raise ValueError(f"no mmqa handler for query {query!r} (id {hid})")
    rows, fields = load_pred_rows(pred_path)
    gt = json.load(open(gt_path)).get("ground_truth")
    results, gold = MMQA_HANDLERS[hid](rows, fields, gt)
    return diff_pair(results, gold, cap)
```

Finally, update `eval_mmqa` to score the pair:

```python
def eval_mmqa(query, pred_path, gt_path):
    """Dispatch to the mmqa per-query handler and return its metric row."""
    hid = handler_id(query)
    if hid not in MMQA_HANDLERS:
        raise ValueError(f"no mmqa handler for query {query!r} (id {hid})")
    rows, fields = load_pred_rows(pred_path)
    gt = json.load(open(gt_path)).get("ground_truth")
    results, gold = MMQA_HANDLERS[hid](rows, fields, gt)
    return score_pair(results, gold)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src/semdb && python3 -m pytest tests/test_evaluate_emit_diff.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Add `--emit-diff` CLI wiring + scenario id-set diff**

In `main()`, add the arg (near the other `ap.add_argument` calls):

```python
    ap.add_argument("--emit-diff", help="also write FP/FN sample rows JSON to this path")
    ap.add_argument("--diff-cap", type=int, default=15, help="max FP and FN samples to emit")
```

Then, in the mmqa scoring branch (`elif args.pred and gt_path and args.query:`), after the metrics are computed and printed, add:

```python
        if args.emit_diff:
            diff = eval_mmqa_diff(args.query, args.pred, gt_path, args.diff_cap)
            diff["query"] = args.query
            diff.update(f1=metrics["f1"], precision=metrics["precision"],
                        recall=metrics["recall"], tp=metrics["tp"],
                        fp=metrics["fp"], fn=metrics["fn"])
            json.dump(diff, open(args.emit_diff, "w"), indent=2)
            print(f"[eval] wrote FP/FN diff -> {args.emit_diff}")
```

For the non-mmqa scenario branch (`if args.pred and args.query and bench and bench != "mmqa":`), add a generic id-set diff after `row.update(**metrics)`:

```python
            if args.emit_diff:
                diff = _scenario_diff(args.pred, gt_dir, bench, args.query, args.sf, args.diff_cap)
                diff["query"] = args.query
                diff.update(f1=metrics.get("f1"), precision=metrics.get("precision"),
                            recall=metrics.get("recall"))
                json.dump(diff, open(args.emit_diff, "w"), indent=2)
                print(f"[eval] wrote FP/FN diff -> {args.emit_diff}")
```

And define `_scenario_diff` above `main()`:

```python
def _scenario_diff(pred_path, gt_dir, bench, query, sf, cap):
    """Generic id-set FP/FN for a non-mmqa scenario. Reads the predicted CSV and the
    scenario GT CSV, diffs their shared id column ("id" if present, else the first
    column). For non-membership metrics (aggregation/ranking) the id sets may be
    trivial — then fp/fn are empty and pred_sample/gt_sample carry a few raw rows so
    the agent still sees the shape. Faithful scoring stays in scenario_metrics."""
    import csv as _csv, importlib, re as _re
    sm = importlib.import_module("scenario_metrics")
    qid = int(_re.match(r"(\d+)", query.lstrip("qQ")).group(1))     # q3a -> 3
    gt_path = str(sm._gt_path(bench, qid, gt_dir, int(sf) if sf else None))

    def _rows(p):
        try:
            with open(p, newline="") as f:
                return list(_csv.DictReader(f))
        except FileNotFoundError:
            return []

    pred_rows, gt_rows = _rows(pred_path), _rows(gt_path)
    def _idcol(rows):
        if not rows:
            return None
        keys = list(rows[0].keys())
        return "id" if "id" in keys else keys[0]
    pc, gc = _idcol(pred_rows), _idcol(gt_rows)
    if pc and gc:
        pred_ids = [str(r[pc]).strip() for r in pred_rows]
        gt_ids = {str(r[gc]).strip() for r in gt_rows}
        pred_set = set(pred_ids)
        fp = [i for i in dict.fromkeys(pred_ids) if i not in gt_ids]
        fn = [i for i in (str(r[gc]).strip() for r in gt_rows) if i not in pred_set]
        return {"false_positives": fp[:cap], "false_negatives": fn[:cap],
                "fp_total": len(fp), "fn_total": len(fn),
                "sampled": len(fp) > cap or len(fn) > cap}
    return {"false_positives": [], "false_negatives": [], "fp_total": 0, "fn_total": 0,
            "sampled": False, "note": "no shared id column",
            "pred_sample": pred_rows[:cap], "gt_sample": gt_rows[:cap]}
```

(`scenario_metrics._gt_path(scenario, qid, gt_dir, scale_factor)` takes the numeric
`qid` derived above — `q3a` → `3`.)

- [ ] **Step 6: Add a scenario-diff test**

Append to `tests/test_evaluate_emit_diff.py`:

```python
def test_scenario_diff_id_sets(tmp_path, monkeypatch):
    import types, sys as _sys
    fake = types.ModuleType("scenario_metrics")
    gt = tmp_path / "Q1.csv"; gt.write_text("id\n1\n2\n3\n")
    fake._gt_path = lambda scenario, qid, gt_dir, sf: str(gt)
    monkeypatch.setitem(_sys.modules, "scenario_metrics", fake)
    pred = tmp_path / "pred.csv"; pred.write_text("id\n1\n2\n9\n")   # 9 fp, 3 fn
    d = E._scenario_diff(str(pred), str(tmp_path), "movie", "q1", "", 15)
    assert d["false_positives"] == ["9"] and d["false_negatives"] == ["3"]
```

- [ ] **Step 7: Run tests**

Run: `cd src/semdb && python3 -m pytest tests/test_evaluate_emit_diff.py -v`
Expected: PASS (4 tests).

- [ ] **Step 8: Commit**

```bash
git add src/semdb/evaluate.py src/semdb/tests/test_evaluate_emit_diff.py
git commit -m "feat(semdb): evaluate.py --emit-diff emits FP/FN samples for the refine loop"
```

---

## Task 2: Loop-control config + pure functions

**Files:**
- Modify: `src/semdb/semdb.config.mjs`
- Modify: `src/semdb/orchestrator.mjs`
- Test: `src/semdb/tests/test_refine_pure.mjs`

**Interfaces:**
- Consumes: `defaults` from `semdb.config.mjs`.
- Produces (exported from `orchestrator.mjs`):
  - `checkSemdbImprovement(prev, next) -> boolean` where each arg is `{ status: "ok"|"crash"|"empty", f1: number|null }`.
  - `shouldContinueSemdb(history, iteration, maxIter, stallThreshold) -> { action: "stop"|"continue", reason: string }` where `history` is `[{ iter, f1, status, improved }]`.

- [ ] **Step 1: Add config values**

In `src/semdb/semdb.config.mjs`, inside `defaults`, after `agentTimeoutMs: 20 * 60 * 1000,` add:

```js
  // --- Iterative refinement loop (GenDB-style) ---
  maxRefineIterations: 5,     // per-query optimize→run→score iterations (0 = single-shot)
  refineStallThreshold: 2,    // stop after this many consecutive non-improving iterations
  refineSampleCap: 15,        // max FP and FN rows shown to the agent per iteration
```

- [ ] **Step 2: Write the failing test**

```js
// src/semdb/tests/test_refine_pure.mjs
import assert from "node:assert";
import { checkSemdbImprovement, shouldContinueSemdb } from "../orchestrator.mjs";

// improvement rule
assert.equal(checkSemdbImprovement({ status: "ok", f1: 0.4 }, { status: "ok", f1: 0.6 }), true);
assert.equal(checkSemdbImprovement({ status: "ok", f1: 0.6 }, { status: "ok", f1: 0.6 }), false);
assert.equal(checkSemdbImprovement({ status: "ok", f1: 0.6 }, { status: "crash", f1: null }), false);
assert.equal(checkSemdbImprovement({ status: "crash", f1: null }, { status: "ok", f1: 0.1 }), true);

// stop logic
assert.equal(shouldContinueSemdb([{ iter: 0, f1: 1.0, status: "ok", improved: true }], 1, 5, 2).action, "stop");
assert.equal(shouldContinueSemdb([{ iter: 0, f1: 0.5, status: "ok", improved: true }], 6, 5, 2).action, "stop");
assert.equal(shouldContinueSemdb([{ iter: 0, f1: null, status: "crash", improved: false }], 1, 5, 2).action, "continue");
const stalled = [
  { iter: 0, f1: 0.5, status: "ok", improved: true },
  { iter: 1, f1: 0.5, status: "ok", improved: false },
  { iter: 2, f1: 0.5, status: "ok", improved: false },
];
assert.equal(shouldContinueSemdb(stalled, 3, 5, 2).action, "stop");
assert.equal(shouldContinueSemdb([{ iter: 0, f1: 0.5, status: "ok", improved: true }], 1, 5, 2).action, "continue");
console.log("test_refine_pure OK");
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd src/semdb && node tests/test_refine_pure.mjs`
Expected: FAIL — `SyntaxError` / `does not provide an export named 'checkSemdbImprovement'`.

- [ ] **Step 4: Implement the two functions**

In `src/semdb/orchestrator.mjs`, after the `parseArgs` function (before `loadQuery`), add and export:

```js
/**
 * Improvement judge — correctness-first, then F1 (mirrors GenDB checkExecutionImprovement).
 * `prev`/`next` are { status: "ok"|"crash"|"empty", f1: number|null }.
 */
export function checkSemdbImprovement(prev, next) {
  const prevOk = prev && prev.status === "ok";
  const nextOk = next && next.status === "ok";
  if (prevOk && !nextOk) return false;     // regressed to a crash/empty
  if (!prevOk && nextOk) return true;      // fixed a crash/empty
  if (prevOk && nextOk) return (next.f1 ?? -1) > (prev.f1 ?? -1);
  return false;                            // both broken → no improvement
}

/**
 * Stop/continue gate (mirrors GenDB shouldContinue). `history` = [{ iter, f1, status, improved }].
 */
export function shouldContinueSemdb(history, iteration, maxIter, stallThreshold) {
  if (iteration > maxIter) return { action: "stop", reason: "Max iterations reached" };
  const last = history[history.length - 1];
  if (last && last.status !== "ok") return { action: "continue", reason: "Fix runtime failure first" };
  const bestF1 = history.reduce((m, h) => Math.max(m, h.f1 ?? -1), -1);
  if (bestF1 >= 1.0) return { action: "stop", reason: "Perfect F1 reached" };
  const thresh = stallThreshold || 2;
  const recent = history.slice(-thresh);
  if (recent.length >= thresh && recent.every((h) => !h.improved)) {
    return { action: "stop", reason: `Stalled: ${thresh} non-improving iterations` };
  }
  return { action: "continue", reason: "Refinement potential remains" };
}
```

- [ ] **Step 5: Guard `main()` so importing the module does NOT run it**

The test imports `orchestrator.mjs`; today the file ends with an unconditional
`main().catch(...)`, which would execute the orchestrator on import. Change the
tail of the file to only run when invoked as the entry point. `fileURLToPath` is
already imported at the top. Replace:

```js
main().catch((err) => {
  console.error("[SemDB] Fatal:", err.message);
  process.exit(1);
});
```

with:

```js
// Only run when invoked directly (not when imported by tests).
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((err) => {
    console.error("[SemDB] Fatal:", err.message);
    process.exit(1);
  });
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd src/semdb && node tests/test_refine_pure.mjs`
Expected: `test_refine_pure OK` (and NOT any `[SemDB]` orchestrator output — proving the import guard works).

- [ ] **Step 7: Commit**

```bash
git add src/semdb/semdb.config.mjs src/semdb/orchestrator.mjs src/semdb/tests/test_refine_pure.mjs
git commit -m "feat(semdb): loop-control config + shouldContinueSemdb/checkSemdbImprovement"
```

---

## Task 3: CLI flags + `runPhase` system-prompt override + feedback renderer

**Files:**
- Modify: `src/semdb/orchestrator.mjs` (`parseArgs`, `runPhase`, new `renderFeedback`)

**Interfaces:**
- Consumes: `defaults.maxRefineIterations`, `defaults.refineSampleCap`.
- Produces:
  - `args.maxIterations` (number), `args.noRefine` (boolean) on the parsed args object.
  - `runPhase(agentConfig, vars, runDir, args, opts)` — new optional 5th param `opts = { systemPromptPath }`.
  - `renderFeedback(prev) -> string` where `prev = { status, f1, metrics, diff, stderrTail, history }`.

- [ ] **Step 1: Add CLI flags in `parseArgs`**

In the `args` object literal in `parseArgs`, add:

```js
    maxIterations: defaults.maxRefineIterations,
    noRefine: false,
```

In the arg-parsing loop, add (next to `--direct`):

```js
    else if (a === "--max-iterations" && argv[i + 1]) args.maxIterations = parseInt(argv[++i], 10);
    else if (a === "--no-refine") args.noRefine = true;
```

- [ ] **Step 2: Add the `systemPromptPath` override to `runPhase`**

Change the signature and the system-prompt read:

```js
async function runPhase(agentConfig, vars, runDir, args, opts = {}) {
  const systemPromptPath = opts.systemPromptPath || agentConfig.promptPath;
  const systemPrompt = await readFile(systemPromptPath, "utf-8");
  const template = await readFile(agentConfig.userPromptPath, "utf-8");
  const userPrompt = renderTemplate(template, vars);
  // ... rest unchanged ...
```

(Everything below the `userPrompt` line stays exactly as-is.)

- [ ] **Step 3: Add the `renderFeedback` helper**

Add near `runPhase`:

```js
/** Render the per-iteration feedback block appended to a generating agent's prompt.
 *  Runtime failures short-circuit to a fix-first block; otherwise metrics + FP/FN. */
function renderFeedback(prev) {
  const histLines = (prev.history || [])
    .map((h) => `  iter ${h.iter}: F1=${h.f1 == null ? "n/a" : h.f1} ${h.status.toUpperCase()}${h.improved ? " (improved)" : ""}`)
    .join("\n");
  if (prev.status !== "ok") {
    return [
      "\n## LAST RUN FAILED — FIX THIS FIRST",
      `The program ${prev.status === "empty" ? "produced no output rows" : "crashed"}.`,
      "```",
      (prev.stderrTail || "(no stderr captured)"),
      "```",
      "Diagnose and fix the error before any accuracy work.",
      histLines ? `\n## HISTORY\n${histLines}` : "",
    ].join("\n");
  }
  const m = prev.metrics || {};
  const d = prev.diff || {};
  const fmt = (rows) => (rows && rows.length)
    ? rows.map((r) => `  - ${typeof r === "object" ? JSON.stringify(r) : r}`).join("\n")
    : "  (none)";
  return [
    `\n## LAST RUN — F1=${prev.f1} (P=${m.precision} R=${m.recall}, tp=${m.tp} fp=${m.fp} fn=${m.fn})`,
    `## FALSE POSITIVES (predicted, but wrong) — ${d.fp_total ?? 0} total, showing ${(d.false_positives || []).length}:`,
    fmt(d.false_positives),
    `## FALSE NEGATIVES (missed) — ${d.fn_total ?? 0} total, showing ${(d.false_negatives || []).length}:`,
    fmt(d.false_negatives),
    histLines ? `## HISTORY\n${histLines}` : "",
    "Diagnose WHY these are wrong and revise the code. Common causes: wrong threshold,",
    "wrong label/value mapping, over-broad predicate, wrong join key, CLIP/LLM prompt",
    "too long or off-target. Edit the existing program in place.",
  ].join("\n");
}
```

- [ ] **Step 4: Sanity-check the module still imports and parses**

(The Task 2 Step 5 guard makes importing safe — `main()` no longer runs on import.)

Run: `cd src/semdb && node --check orchestrator.mjs && node -e "import('./orchestrator.mjs').then(()=>console.log('import OK'))"`
Expected: `import OK` with no `[SemDB]` orchestrator output.

- [ ] **Step 5: Commit**

```bash
git add src/semdb/orchestrator.mjs
git commit -m "feat(semdb): --max-iterations/--no-refine flags, runPhase systemPromptPath, renderFeedback"
```

---

## Task 4: `refineLoop` helper + wire into the compiled path (`runQueryCodegen`)

**Files:**
- Modify: `src/semdb/orchestrator.mjs` (`refineLoop`, `runQueryCodegen`)

**Interfaces:**
- Consumes: `checkSemdbImprovement`, `shouldContinueSemdb`, `renderFeedback`, `defaults.refineStallThreshold`, `defaults.refineSampleCap`.
- Produces:
  - `refineLoop({ args, query, runDir, codePath, resultsCsv, genFirst, regen, execAndScore }) -> { bestIter, bestF1, history }`.
  - `scoreWithDiff(args, query, telePath, resultsCsv, diffPath, csvPath) -> { status, f1, metrics, diff, stderrTail }` — runs `evaluate.py --emit-diff`, reads back metrics + diff.

- [ ] **Step 1: Add `refineLoop` + `scoreWithDiff`**

Add above `runQueryCodegen`:

```js
/** Run evaluate.py (with --emit-diff) for one iteration and read the outcome back.
 *  Returns { status, f1, metrics, diff, stderrTail }. status: "ok"|"empty" (crash is
 *  detected by the caller from the run step). Only writes results.csv when finalize. */
async function scoreWithDiff(args, query, planObj, telePath, resultsCsv, diffPath, csvPath, finalize) {
  // Dry-run must never invoke evaluate.py (guarantees --dry-run is a no-op even when a
  // stale iter_0 results CSV is lying around in --out from a prior real run).
  if (args.dryRun) return { status: "ok", f1: null, metrics: null, diff: null };
  if (!existsSync(resultsCsv)) return { status: "empty", f1: null, metrics: null, diff: null };
  const gt = await resolveGroundTruth(args.groundTruthDir, query, args.scaleFactor);
  if (!gt) return { status: "ok", f1: null, metrics: null, diff: null };
  const evArgs = [resolve(__dirname, "evaluate.py"), "--telemetry", telePath,
    "--ground-truth", gt.file, "--pred", resultsCsv, "--pred-cols", args.predCols,
    "--query", query, "--benchmark", args.benchmark,
    "--emit-diff", diffPath, "--diff-cap", String(defaults.refineSampleCap),
    ...(finalize ? ["--csv", csvPath] : ["--csv", resolve(dirname(diffPath), "_scratch_results.csv")]),
    ...(args.groundTruthDir ? ["--ground-truth-dir", args.groundTruthDir] : []),
    ...(args.scaleFactor ? ["--sf", String(args.scaleFactor)] : [])];
  const ev = spawnSync("python3", evArgs, { stdio: "inherit" });
  if (ev.status !== 0) console.warn(`[SemDB] evaluate.py exited ${ev.status}.`);
  // Read metrics from the DIFF json (it carries f1/precision/recall/tp/fp/fn), NOT from
  // telePath: during iterations telemetry.json does not exist yet, so evaluate.py can't
  // persist metrics into it. The diff file is always written when there is ground truth.
  const diff = await readJSON(diffPath);
  if (!diff) return { status: "ok", f1: null, metrics: null, diff: null };
  const metrics = {
    f1: diff.f1 ?? null, precision: diff.precision ?? null, recall: diff.recall ?? null,
    tp: diff.tp ?? null, fp: diff.fp ?? diff.fp_total ?? null, fn: diff.fn ?? diff.fn_total ?? null,
  };
  return { status: "ok", f1: metrics.f1, metrics, diff };
}

/**
 * GenDB-style per-query refinement loop. iter_0 = genFirst(); iters 1..maxIter =
 * regen(feedback) → keep-or-rollback. Best code wins; the best iteration's results
 * CSV is copied back to `resultsCsv`. No-op (single shot) when maxIter is 0.
 *
 * Closure contract (each generates code AND runs it, returning the RUN outcome;
 * scoreIter then scores that run — no shared mutable state between them):
 *   genFirst(iterDir, iterCode, iterCsv) -> { status: "ok"|"crash"|"empty", stderr }
 *   regen(iterDir, iterCode, iterCsv, feedback) -> { status, stderr }
 *   scoreIter(iterDir, iterCode, iterCsv, runOutcome) -> { status, f1, metrics, diff, stderrTail }
 */
async function refineLoop({ args, query, runDir, codeBasename, resultsCsv, genFirst, regen, scoreIter }) {
  const maxIter = args.noRefine ? 0 : (args.maxIterations ?? defaults.maxRefineIterations);
  const iter0Dir = resolve(runDir, "iter_0");
  await mkdir(iter0Dir, { recursive: true });
  const iter0Code = resolve(iter0Dir, codeBasename);
  const iter0Csv = resolve(iter0Dir, basename(resultsCsv));

  const run0 = await genFirst(iter0Dir, iter0Code, iter0Csv);
  let best = { iter: 0, dir: iter0Dir, code: iter0Code, csv: iter0Csv,
               outcome: await scoreIter(iter0Dir, iter0Code, iter0Csv, run0) };
  const history = [{ iter: 0, f1: best.outcome.f1, status: best.outcome.status, improved: true }];

  // GLOBAL CONSTRAINT: refinement engages ONLY with a measurable F1 signal. "No signal"
  // means iter_0 RAN OK but could not be scored (no ground truth) → single-shot. A crash or
  // empty output WITH ground truth is NOT "no signal" — it must keep iterating (fix-first),
  // matching shouldContinueSemdb, which returns "continue" when the last run is not "ok".
  // args.dryRun is included: in dry-run the closures are no-ops so iter_0 status is "empty"
  // (never "ok"); without this the loop would advance to iter 1 and crash seeding code.
  const noSignal = args.dryRun || (best.outcome.status === "ok" && best.outcome.f1 == null);
  const effectiveMaxIter = noSignal ? 0 : maxIter;
  if (noSignal && maxIter > 0) {
    console.log(`[SemDB] [${query}] no F1 signal — single-shot, skipping refinement.`);
  }

  for (let iteration = 1; iteration <= effectiveMaxIter; iteration++) {
    const decision = shouldContinueSemdb(history, iteration, maxIter, defaults.refineStallThreshold);
    console.log(`[SemDB] [${query}] --- refine ${iteration}/${maxIter} --- ${decision.action}: ${decision.reason}`);
    if (decision.action === "stop") break;

    const itDir = resolve(runDir, `iter_${iteration}`);
    await mkdir(itDir, { recursive: true });
    const itCode = resolve(itDir, codeBasename);
    const itCsv = resolve(itDir, basename(resultsCsv));
    // seed from the best code so far
    await writeFile(itCode, await readFile(best.code, "utf-8"));

    const feedback = renderFeedback({
      status: best.outcome.status, f1: best.outcome.f1, metrics: best.outcome.metrics,
      diff: best.outcome.diff, stderrTail: best.outcome.stderrTail, history,
    });
    const run = await regen(itDir, itCode, itCsv, feedback);
    const outcome = await scoreIter(itDir, itCode, itCsv, run);
    const improved = checkSemdbImprovement(best.outcome, outcome);
    history.push({ iter: iteration, f1: outcome.f1, status: outcome.status, improved });
    if (improved) {
      console.log(`[SemDB] [${query}] iter ${iteration} improved (F1 ${best.outcome.f1} → ${outcome.f1}). Keeping.`);
      best = { iter: iteration, dir: itDir, code: itCode, csv: itCsv, outcome };
    } else {
      console.log(`[SemDB] [${query}] iter ${iteration} did not improve. Rolling back.`);
    }
  }
  // Promote the best iteration's artifacts to the run root.
  if (existsSync(best.code)) await writeFile(resolve(runDir, codeBasename), await readFile(best.code, "utf-8"));
  if (existsSync(best.csv)) await writeFile(resultsCsv, await readFile(best.csv, "utf-8"));
  return { bestIter: best.iter, bestF1: best.outcome.f1, stopReason: history, history };
}
```

- [ ] **Step 2: Refactor `runQueryCodegen` to drive the loop**

Replace the body of `runQueryCodegen` from the `// Phase C — Code Generator` comment through the compiled-query `spawnSync` block with a `refineLoop` call. The Code Generator invocation and the compiled-query run move into `genFirst`/`regen`/`scoreIter` closures. Keep telemetry assembly after the loop, reading the best iteration.

Concretely, inside `runQueryCodegen`, after the existing setup (`runDir`, `codePath`, `resultsCsv`, `art`, `doRun`, `phases`, `record`) and the `console.log` header, replace the single Code-Generator `record(...)` + compiled-query `spawnSync` with the closures below. Each closure follows the `refineLoop` contract: `genFirst`/`regen` generate the code AND run it, returning `{ status, stderr }`; `scoreIter` scores the resulting CSV. The only difference between `genFirst` and `regen` is the `query_sql` value (base SQL vs SQL + feedback) — factor the shared `runPhase` into one helper.

```js
  const codeBasename = `compiled_${query}.py`;
  const telePath = resolve(runDir, "telemetry.json");

  const cgVars = (iterCode, querySql) => ({
    query_id: query, query_sql: querySql,
    schema_json: schema ? JSON.stringify(schema, null, 2) : "{{corpus schema.json}}",
    attrs_path: attrsPath, attrs_columns: "(schema attributes + conf)",
    structured_path: structured.path || "(structured table path)",
    structured_columns: "(see table headers)",
    code_path: iterCode, code_basename: codeBasename,
  });

  const runCompiled = (iterDir, iterCode, iterCsv) => {
    if (!(doRun && existsSync(iterCode) && existsSync(attrsPath))) return { status: "empty", stderr: "" };
    const cqArgs = [iterCode, structured.path, attrsPath, iterCsv,
      ...(args.endpoint ? ["--endpoint", args.endpoint, "--api-key", args.apiKey, "--model", extractModel] : [])];
    console.log(`\n[SemDB] Running compiled query: python3 ${cqArgs.join(" ")}`);
    const cq = spawnSync("python3", cqArgs, { stdio: ["inherit", "inherit", "pipe"] });
    const stderr = (cq.stderr || "").toString();
    if (cq.status !== 0) { console.warn(`[SemDB] compiled query exited ${cq.status}.`); return { status: "crash", stderr }; }
    return { status: "ok", stderr };
  };

  const genFirst = async (iterDir, iterCode, iterCsv) => {
    record("code_generator", await runPhase(codeGeneratorConfig, cgVars(iterCode, sql), iterDir, args));
    return runCompiled(iterDir, iterCode, iterCsv);
  };
  const regen = async (iterDir, iterCode, iterCsv, feedback) => {
    record("code_generator", await runPhase(codeGeneratorConfig, cgVars(iterCode, sql + "\n\n" + feedback), iterDir, args));
    return runCompiled(iterDir, iterCode, iterCsv);
  };
  const scoreIter = async (iterDir, iterCode, iterCsv, run) => {
    const diffPath = resolve(iterDir, "diff.json");
    const scored = await scoreWithDiff(args, query, planObj, telePath, iterCsv, diffPath, csvPath, false);
    const status = run.status === "crash" ? "crash" : (existsSync(iterCsv) ? "ok" : "empty");
    return { ...scored, status, stderrTail: (run.stderr || "").split("\n").slice(-40).join("\n") };
  };
```

Then invoke the loop:

```js
  const { bestIter, bestF1, history } = await refineLoop({
    args, query, runDir, codeBasename, resultsCsv,
    genFirst, regen, scoreIter,
  });
```

After the loop, keep the existing telemetry-assembly code, but add the `refine`
block to the `report` object (locate where `report` is assembled):

```js
    refine: { iterations: history.length - 1, best_iteration: bestIter,
              max_iterations: args.noRefine ? 0 : args.maxIterations,
              f1_history: history },
```

TWO telemetry-source fixes are REQUIRED because the loop now runs the
Code Generator and the compiled query in per-iteration `iter_<N>/` dirs, and
promotes the BEST iteration:

1. **Read the compiled-query meta from the best iter dir.** The compiled query
   writes `compiled_<query>.meta.json` next to the code it ran — i.e. in
   `iter_<bestIter>/`, not the run root. Change the existing `cqMeta` read to:
   ```js
   const cqMeta = await readJSON(resolve(runDir, `iter_${bestIter}`, `compiled_${query}.meta.json`));
   ```
   (Falls through to the `?.` guards already in the telemetry code if absent.)
2. **Sum the Code Generator cost across ALL iterations, not just iter_0.**
   `record("code_generator", ...)` pushes one phase entry PER iteration, so
   `phases.find(p => p.phase === "code_generator")` undercounts (returns iter_0
   only). Replace that single-phase lookup with a sum over every codegen phase:
   ```js
   const cgPhases = phases.filter((p) => p.phase === "code_generator");
   const cg = {
     duration_ms: cgPhases.reduce((s, p) => s + (p.duration_ms || 0), 0),
     cost_usd:    cgPhases.reduce((s, p) => s + (p.cost_usd || 0), 0),
     llm_calls:   cgPhases.reduce((s, p) => s + (p.llm_calls || 0), 0),
     tokens: cgPhases.reduce((t, p) => ({ input: (t.input || 0) + (p.tokens?.input || 0),
                                          output: (t.output || 0) + (p.tokens?.output || 0) }), {}),
   };
   ```
   Use this `cg` everywhere the old single `cg` phase object was used.

ORDER MATTERS: `report` must be written to `telePath` BEFORE the finalize scoring,
so `evaluate.py` can merge the metrics into telemetry.json and the metrics-print
block can read them back. So the tail of `runQueryCodegen` becomes: (1) assemble
`report`; (2) `await writeFile(telePath, JSON.stringify(report, null, 2))`; (3) the
finalize scoring below; (4) the existing metrics-print block, re-reading `telePath`.

```js
  // (3) finalize: score the promoted best CSV → merge metrics into telemetry + append results.csv row
  if (doRun) {
    const gt = await resolveGroundTruth(args.groundTruthDir, query, args.scaleFactor);
    if (gt && existsSync(resultsCsv)) {
      await scoreWithDiff(args, query, planObj, telePath, resultsCsv, resolve(runDir, "diff.json"), csvPath, true);
    }
  }
```

> Because `runQueryCodegen` currently builds `report` and calls evaluate.py inline at the end, remove that now-duplicated final `evaluate.py` spawn (the loop + the finalize call replace it). Keep the metrics-print block, but read metrics from the re-read `telePath`.

- [ ] **Step 3: Syntax check**

Run: `cd src/semdb && node --check orchestrator.mjs`
Expected: no output (exit 0).

- [ ] **Step 4: Dry-run smoke (no execution, no GT)**

Run:
```bash
cd src/semdb && node orchestrator.mjs --query q6 \
  --query-dir /localhome/hza214/SemBench/files/mmqa/query/bigquery \
  --data-dir /localhome/hza214/SemBench/files/mmqa/data/sf_200 \
  --dry-run --no-refine 2>&1 | tail -20
```
Expected: prints the Code Generator prompt; no crash. (`--no-refine` + `--dry-run` ⇒ single shot.)

- [ ] **Step 5: Commit**

```bash
git add src/semdb/orchestrator.mjs
git commit -m "feat(semdb): refineLoop + scoreWithDiff wired into the compiled path"
```

---

## Task 5: Wire `refineLoop` into image DIRECT (`runQueryDirect`)

**Files:**
- Modify: `src/semdb/orchestrator.mjs` (`runQueryDirect`)

**Interfaces:**
- Consumes: `refineLoop`, `scoreWithDiff`, existing VADAR Signature/API/Solver configs.
- Produces: `runQueryDirect` runs the 3 agents for iter_0, then refines `solve_<q>.py`.

- [ ] **Step 1: Restructure `runQueryDirect` around `refineLoop`**

Keep the existing setup (runDir, solvePath, resultsCsv, doRun, table docs, image cols). Wrap the 3-agent generation + solver run in the loop closures.

Replace the block from `if (!existsSync(solvePath) || args.force) { ... }` through the solver `spawnSync` with:

```js
  const codeBasename = `solve_${query}.py`;
  const clipModel = args.clipModel || defaults.extraction.clipModel;
  const dataDir = args.tableDir || args.dataDir;
  const telePath = resolve(runDir, "telemetry.json");

  const runSolver = (iterDir, iterCode, iterCsv) => {
    if (!(doRun && existsSync(iterCode))) return { status: "empty", stderr: "" };
    const sArgs = [iterCode, iterCsv, "--data-dir", dataDir,
      ...(imageDir ? ["--image-dir", imageDir] : []), "--clip-model", clipModel];
    console.log(`\n[SemDB] Running direct solver: python3 ${sArgs.join(" ")}`);
    const s = spawnSync("python3", sArgs, { stdio: ["inherit", "inherit", "pipe"] });
    const stderr = (s.stderr || "").toString();
    if (s.status !== 0) { console.warn(`[SemDB] direct solver exited ${s.status}.`); return { status: "crash", stderr }; }
    return { status: "ok", stderr };
  };

  // The 3 agents (Signature → API → Solver) generate iter_0's solve_<q>.py.
  const gen3Agents = async (iterDir, iterCode, querySql) => {
    const sigPath = resolve(iterDir, `_vadar_signatures_${query}.txt`);
    const helpersPath = resolve(iterDir, `_vadar_helpers_${query}.py`);
    const common = { corpus_name: corpus.table, semdb_dir: __dirname };
    record("vadar_signature", await runPhase(vadarSignatureConfig, {
      ...common, query_sql: querySql, schema_json: "(DIRECT mode: no schema; read value spaces from the CSVs at runtime)",
      sig_path: sigPath,
    }, iterDir, args));
    record("vadar_api", await runPhase(vadarApiConfig, {
      ...common, sig_path: sigPath, helpers_path: helpersPath,
    }, iterDir, args));
    record("vadar_solver", await runPhase(vadarSolverConfig, {
      query_id: query, query_sql: querySql, query_nl: nl || "(none)", semdb_dir: __dirname,
      tables_doc: tableLines.join("\n"),
      image_table: corpus.table, image_path: corpus.path,
      image_filename_col: imgFilenameCol, image_filepath_col: imgFilepathCol,
      image_dir: imageDir, helpers_path: helpersPath, solve_path: iterCode,
    }, iterDir, args));
  };

  // Refinement iterations re-invoke ONLY the Solver, editing the seeded code with feedback.
  // Helpers were generated ONCE into iter_0 (refineLoop seeds only the code forward), so read
  // them from iter_0 — NOT from the current iterDir (which has no helpers file).
  const regenSolver = async (iterDir, iterCode, feedback) => {
    const helpersPath = resolve(runDir, "iter_0", `_vadar_helpers_${query}.py`);
    record("vadar_solver", await runPhase(vadarSolverConfig, {
      query_id: query, query_sql: sql + "\n\n" + feedback, query_nl: nl || "(none)", semdb_dir: __dirname,
      tables_doc: tableLines.join("\n"),
      image_table: corpus.table, image_path: corpus.path,
      image_filename_col: imgFilenameCol, image_filepath_col: imgFilepathCol,
      image_dir: imageDir, helpers_path: existsSync(helpersPath) ? helpersPath : "(seed helpers from iter_0)", solve_path: iterCode,
    }, iterDir, args));
  };

  const genFirst = async (iterDir, iterCode, iterCsv) => {
    await gen3Agents(iterDir, iterCode, sql);
    return runSolver(iterDir, iterCode, iterCsv);
  };
  const regen = async (iterDir, iterCode, iterCsv, feedback) => {
    await regenSolver(iterDir, iterCode, feedback);
    return runSolver(iterDir, iterCode, iterCsv);
  };
  const scoreIter = async (iterDir, iterCode, iterCsv, run) => {
    const diffPath = resolve(iterDir, "diff.json");
    const scored = await scoreWithDiff(args, query, planObj, telePath, iterCsv, diffPath, csvPath, false);
    const status = run.status === "crash" ? "crash" : (existsSync(iterCsv) ? "ok" : "empty");
    return { ...scored, status, stderrTail: (run.stderr || "").split("\n").slice(-40).join("\n") };
  };

  if (args.dryRun) { await gen3Agents(runDir, solvePath, sql); return null; }

  const { bestIter, bestF1, history } = await refineLoop({
    args, query, runDir, codeBasename, resultsCsv, genFirst, regen, scoreIter,
  });
```

> The closure signatures match Task 4 exactly: `genFirst(iterDir, iterCode, iterCsv)` and `regen(iterDir, iterCode, iterCsv, feedback)` both return `{ status, stderr }`; `scoreIter(iterDir, iterCode, iterCsv, run)` returns the scored outcome. `gen3Agents` gains a `querySql` param here so Task 9 can reuse it for text.

Keep the DIRECT telemetry assembly after the loop; add the `refine` block to the
DIRECT `report` object (same shape as Task 4):

```js
    refine: { iterations: history.length - 1, best_iteration: bestIter,
              max_iterations: args.noRefine ? 0 : args.maxIterations,
              f1_history: history },
```

Same ORDER as Task 4: (1) assemble `report`; (2) `await writeFile(telePath, JSON.stringify(report, null, 2))`;
(3) the finalize scoring below; (4) the existing metrics-print block re-reading `telePath`:

```js
  if (doRun) {
    const gt = await resolveGroundTruth(args.groundTruthDir, query, args.scaleFactor);
    if (gt && existsSync(resultsCsv)) {
      await scoreWithDiff(args, query, planObj, telePath, resultsCsv, resolve(runDir, "diff.json"), csvPath, true);
    }
  }
```

Remove the now-duplicated inline `evaluate.py` spawn at the end of `runQueryDirect`.

- [ ] **Step 2: Syntax check**

Run: `cd src/semdb && node --check orchestrator.mjs`
Expected: no output.

- [ ] **Step 3: Dry-run smoke on an image query**

Run:
```bash
cd src/semdb && node orchestrator.mjs --query q7 --direct \
  --query-dir /localhome/hza214/SemBench/files/mmqa/query/bigquery \
  --data-dir /localhome/hza214/SemBench/files/mmqa/data/sf_200 \
  --dry-run 2>&1 | tail -20
```
Expected: prints the 3-agent solver prompts; no crash.

- [ ] **Step 4: Commit**

```bash
git add src/semdb/orchestrator.mjs
git commit -m "feat(semdb): refineLoop wired into image DIRECT solver"
```

---

## Task 6: `semtext.py` — endpoint-backed text inference backend

**Files:**
- Create: `src/semdb/semtext.py`
- Test: `src/semdb/tests/test_semtext_fake_endpoint.py`

**Interfaces:**
- Consumes: `semextract.gen_endpoint`, `semextract._Cfg`.
- Produces:
  - `get_ctx(model, endpoint, api_key="EMPTY", timeout=120) -> TextCtx`.
  - `class TextPatch(text, ctx)` with methods `judge(question)->bool`, `classify(options)->str`, `extract(field)->str`, `generate(instruction)->str`, `score(query)->float`.
  - `METER` with `.calls` (int) and `.reset()`.

- [ ] **Step 1: Write the failing test (fake endpoint via monkeypatched gen_endpoint)**

```python
# src/semdb/tests/test_semtext_fake_endpoint.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import semtext


def _fake_gen(monkeypatch, reply):
    import semextract
    calls = {"n": 0}
    def fake(cfg, json_schema, prompt, modality, image_path=None, text=None):
        calls["n"] += 1
        return reply(prompt, text)
    monkeypatch.setattr(semextract, "gen_endpoint", fake)
    return calls


def test_judge_true_false(monkeypatch):
    _fake_gen(monkeypatch, lambda p, t: '{"answer": true}' if "good" in (t or "") else '{"answer": false}')
    semtext.METER.reset()
    ctx = semtext.get_ctx("m", "http://x/v1")
    assert semtext.TextPatch("a good movie", ctx).judge("Is it positive?") is True
    assert semtext.TextPatch("a bad movie", ctx).judge("Is it positive?") is False
    assert semtext.METER.calls == 2


def test_classify_returns_option(monkeypatch):
    _fake_gen(monkeypatch, lambda p, t: '{"value": "comedy"}')
    ctx = semtext.get_ctx("m", "http://x/v1")
    assert semtext.TextPatch("funny", ctx).classify(["comedy", "drama"]) == "comedy"


def test_cache_dedups_identical_calls(monkeypatch):
    calls = _fake_gen(monkeypatch, lambda p, t: '{"answer": true}')
    semtext.METER.reset()
    ctx = semtext.get_ctx("m", "http://x/v1")
    tp = semtext.TextPatch("same text", ctx)
    tp.judge("Q?"); tp.judge("Q?")
    assert calls["n"] == 1        # second call served from cache
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/semdb && python3 -m pytest tests/test_semtext_fake_endpoint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'semtext'`.

- [ ] **Step 3: Implement `semtext.py`**

```python
#!/usr/bin/env python3
"""
semtext.py — endpoint-backed TEXT inference backend for text DIRECT mode. The text
analog of semvision.py: predefined_text.py wraps these primitives, generated
solve_<q>.py programs compose them. All inference goes through an OpenAI-compatible
endpoint via semextract.gen_endpoint (guided-JSON), so there is ONE HTTP client.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/semdb
import json
import semextract


class _Meter:
    def __init__(self) -> None:
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0


METER = _Meter()


@dataclass
class TextCtx:
    cfg: Any                       # semextract._Cfg
    cache: dict


def get_ctx(model: str, endpoint: str, api_key: str = "EMPTY", timeout: int = 120) -> TextCtx:
    cfg = semextract._Cfg(model=model, endpoint=endpoint, api_key=api_key,
                          max_new_tokens=64, timeout=timeout)
    return TextCtx(cfg=cfg, cache={})


def _ask(ctx: TextCtx, schema: dict, prompt: str, text: str) -> dict:
    """One guided-JSON call, memoized on (prompt, text)."""
    key = (prompt, text)
    if key in ctx.cache:
        return ctx.cache[key]
    METER.calls += 1
    raw = semextract.gen_endpoint(ctx.cfg, schema, prompt, "text", text=text)
    try:
        obj = json.loads(raw)
    except Exception:
        obj = {}
    ctx.cache[key] = obj
    return obj


class TextPatch:
    """A row's text, plus the shared endpoint ctx. Mirrors ImagePatch."""

    def __init__(self, text: str, ctx: TextCtx) -> None:
        self.text = text or ""
        self.ctx = ctx

    def judge(self, question: str) -> bool:
        schema = {"type": "object", "properties": {"answer": {"type": "boolean"}},
                  "required": ["answer"]}
        prompt = f"Answer the yes/no question about the INPUT text. Question: {question}"
        return bool(_ask(self.ctx, schema, prompt, self.text).get("answer", False))

    def classify(self, options: list[str]) -> str:
        schema = {"type": "object",
                  "properties": {"value": {"type": "string", "enum": list(options)}},
                  "required": ["value"]}
        prompt = ("Classify the INPUT text into exactly one of these options: "
                  + ", ".join(map(str, options)))
        v = _ask(self.ctx, schema, prompt, self.text).get("value", "")
        return v if v in options else (options[0] if options else "")

    def extract(self, field: str) -> str:
        schema = {"type": "object", "properties": {"value": {"type": "string"}},
                  "required": ["value"]}
        prompt = f"Extract the value of '{field}' from the INPUT text. If absent, return 'none'."
        return str(_ask(self.ctx, schema, prompt, self.text).get("value", "none"))

    def generate(self, instruction: str) -> str:
        schema = {"type": "object", "properties": {"value": {"type": "string"}},
                  "required": ["value"]}
        prompt = f"Follow this instruction over the INPUT text: {instruction}"
        return str(_ask(self.ctx, schema, prompt, self.text).get("value", ""))

    def score(self, query: str) -> float:
        schema = {"type": "object",
                  "properties": {"score": {"type": "number", "minimum": 0, "maximum": 1}},
                  "required": ["score"]}
        prompt = (f"Rate 0.0–1.0 how well the INPUT text matches: {query}. "
                  "Return only the number in JSON.")
        try:
            return float(_ask(self.ctx, schema, prompt, self.text).get("score", 0.0))
        except (TypeError, ValueError):
            return 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src/semdb && python3 -m pytest tests/test_semtext_fake_endpoint.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/semdb/semtext.py src/semdb/tests/test_semtext_fake_endpoint.py
git commit -m "feat(semdb): semtext.py endpoint-backed text inference backend (TextPatch)"
```

---

## Task 7: `vadar/predefined_text.py` — the text API surface

**Files:**
- Create: `src/semdb/vadar/predefined_text.py`
- Test: `src/semdb/tests/test_predefined_text.py`

**Interfaces:**
- Consumes: `semtext.TextPatch`.
- Produces: free functions `judge(text_patch, question)`, `classify(text_patch, options)`, `extract(text_patch, field)`, `generate(text_patch, instruction)`, `score(text_patch, query)`; module constant `MODULES_SIGNATURES_TEXT` (str).

- [ ] **Step 1: Write the failing test**

```python
# src/semdb/tests/test_predefined_text.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vadar import predefined_text as PT


class _FakePatch:
    def judge(self, q): return True
    def classify(self, opts): return opts[0]
    def extract(self, field): return "x"
    def generate(self, instr): return "y"
    def score(self, q): return 0.5


def test_free_functions_delegate_to_patch():
    p = _FakePatch()
    assert PT.judge(p, "q?") is True
    assert PT.classify(p, ["a", "b"]) == "a"
    assert PT.extract(p, "genre") == "x"
    assert PT.generate(p, "summarize") == "y"
    assert PT.score(p, "romance") == 0.5


def test_signatures_doc_present():
    assert "judge(" in PT.MODULES_SIGNATURES_TEXT
    assert "classify(" in PT.MODULES_SIGNATURES_TEXT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/semdb && python3 -m pytest tests/test_predefined_text.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vadar.predefined_text'`.

- [ ] **Step 3: Implement `predefined_text.py`**

```python
#!/usr/bin/env python3
"""
vadar/predefined_text.py — the PREDEFINED TEXT API (the text analog of predefined.py).
Free functions taking a `TextPatch` `text` (semtext-backed), mirroring the vision API's
`classify(image, ...)` call style. The generated solve_<q>.py and helper functions
compose ONLY these. `MODULES_SIGNATURES_TEXT` is the docstring block shown to the
Signature/API/Program agents.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/semdb


def judge(text, question):
    """True/False LLM judgement over the text (AI.IF semantics)."""
    return text.judge(question)


def classify(text, options):
    """Pick the single best VALUE from a CLOSED value space (enum / DB column values)."""
    return text.classify(options)


def extract(text, field):
    """Extract one attribute value named `field` from the text (AI.GENERATE field)."""
    return text.extract(field)


def generate(text, instruction):
    """Free-form generation over the text following `instruction`."""
    return text.generate(instruction)


def score(text, query):
    """Relevance of the text to `query` in [0, 1] (soft filter / ranking)."""
    return text.score(query)


MODULES_SIGNATURES_TEXT = '''
"""
Answers a yes/no question about the text and returns a bool. Use for AI.IF predicates.
Args:
    text (TextPatch): the row's text.
    question (string): a yes/no question.
Returns:
    bool: True iff the answer is yes.
"""
def judge(text, question) -> bool

"""
Classifies the text into the single best option from a CLOSED value space and returns
that VALUE (a real field). Use for enum categories or a DB column's values. Read the
value space from the structured CSV column AT RUNTIME — never hardcode it.
Args:
    text (TextPatch): the row's text.
    options (list): candidate string values.
Returns:
    string: the best-matching option value.
"""
def classify(text, options) -> str

"""
Extracts the value of a named attribute from the text (AI.GENERATE of one field).
Returns 'none' when absent.
Args:
    text (TextPatch): the row's text.
    field (string): the attribute name to extract.
Returns:
    string: the extracted value (or 'none').
"""
def extract(text, field) -> str

"""
Free-form generation over the text following an instruction (AI.GENERATE text).
Args:
    text (TextPatch): the row's text.
    instruction (string): what to produce.
Returns:
    string: the generated text.
"""
def generate(text, instruction) -> str

"""
Relevance of the text to a short query in [0,1]. Use for ranking / soft filters.
Args:
    text (TextPatch): the row's text.
    query (string): a short phrase to match.
Returns:
    float: similarity in [0,1].
"""
def score(text, query) -> float
'''
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src/semdb && python3 -m pytest tests/test_predefined_text.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/semdb/vadar/predefined_text.py src/semdb/tests/test_predefined_text.py
git commit -m "feat(semdb): vadar/predefined_text.py text inference API + signatures"
```

---

## Task 8: Modality-aware system prompts + `promptPathText`

**Files:**
- Create: `src/semdb/agents/vadar-signature/prompt-text.md`
- Create: `src/semdb/agents/vadar-api/prompt-text.md`
- Create: `src/semdb/agents/vadar-solver/prompt-text.md`
- Modify: `src/semdb/agents/vadar-signature/index.mjs`
- Modify: `src/semdb/agents/vadar-api/index.mjs`
- Modify: `src/semdb/agents/vadar-solver/index.mjs`

**Interfaces:**
- Produces: `config.promptPathText` on each of the three VADAR agent configs.

- [ ] **Step 1: Add `promptPathText` to each index.mjs**

In each of the three files, add one line inside the `config` object after `promptPath`:

`vadar-signature/index.mjs`:
```js
  promptPath: resolve(__dirname, "prompt.md"),
  promptPathText: resolve(__dirname, "prompt-text.md"),
```
Apply the identical edit to `vadar-api/index.mjs` and `vadar-solver/index.mjs`.

- [ ] **Step 2: Write `vadar-signature/prompt-text.md`**

```markdown
You are the **VADAR Signature agent (TEXT mode)**. Given a SQL query over TEXT rows, propose
the minimal set of helper-function SIGNATURES (name + args + one-line docstring, NO bodies)
that a program would compose to evaluate the query's semantic predicate over each row's text.
Compose ONLY the predefined text API — read `MODULES_SIGNATURES_TEXT` in
`{{semdb_dir}}/vadar/predefined_text.py` (judge / classify / extract / generate / score).
Write the signatures to `{{sig_path}}`. Prefer `classify` over a closed value space (read
from the CSV at runtime) and `judge` for boolean AI.IF predicates. Keep it to 1–3 helpers.
```

- [ ] **Step 3: Write `vadar-api/prompt-text.md`**

```markdown
You are the **VADAR API agent (TEXT mode)**. IMPLEMENT each proposed helper signature by
composing the predefined TEXT API (and already-implemented helpers) — nothing else. `text`
is a `TextPatch` already; call the predefined free functions directly (judge, classify,
extract, generate, score). Read the API in `{{semdb_dir}}/vadar/predefined_text.py`. Read the
proposed signatures at `{{sig_path}}` and write the implementations to `{{helpers_path}}`.
```

- [ ] **Step 4: Write `vadar-solver/prompt-text.md`**

```markdown
You are the **VADAR Program agent in TEXT DIRECT mode**. You write ONE end-to-end Python
program `solve_<query>.py` that answers the WHOLE SQL query by composing the predefined LOCAL
TEXT API (`judge / classify / extract / generate / score`, endpoint-backed) plus the
generated helpers.

The program:
1. reads the structured CSV(s) and the text CSV from `--data-dir` by filename;
2. builds one shared `semtext` ctx from `--endpoint --model --api-key`;
3. wraps each row's text in `semtext.TextPatch(text, ctx)`;
4. calls the text API / helpers to evaluate the query's semantic predicate per row,
   returning a REAL field value (a label / name / bool), not a raw score;
5. does the relational join / filter / projection / aggregation in plain Python;
6. writes the result CSV whose columns EXACTLY match the query's SELECT list.

Rules:
- Get any closed value space (e.g. the set of genres) by reading the structured column
  AT RUNTIME — do NOT hardcode it.
- Boolean AI.IF predicate → `judge`; a value from a closed space → `classify`; a single
  attribute → `extract`; a ranking/soft filter → `score`.
- Read the predefined API in `{{semdb_dir}}/vadar/predefined_text.py` and the generated
  helpers at `{{helpers_path}}`.
- The program must be runnable EXACTLY as the orchestrator invokes it:
  `python3 {{solve_path}} <out.csv> --data-dir D --endpoint URL --model M --api-key K`.

Write the program to `{{solve_path}}` with EXACTLY this shape:
```python
import sys, os, csv, argparse
sys.path.insert(0, "{{semdb_dir}}")
import semtext
from vadar.predefined_text import judge, classify, extract, generate, score
# <paste the generated helper implementations here>

def _read(data_dir, name):
    with open(os.path.join(data_dir, name), newline="") as f:
        return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    a = ap.parse_args()
    ctx = semtext.get_ctx(a.model, a.endpoint, a.api_key)
    P = lambda t: semtext.TextPatch(t, ctx)
    # 1) read structured + text CSVs from a.data_dir
    # 2) closed value space(s) from the structured column(s) at runtime
    # 3) for each text row: field = <compose text API + helpers over P(row[<text_col>])>
    # 4) relational join/filter/aggregate → out_rows (tuples matching the SELECT columns)
    out_rows = []
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([ ... ])   # SELECT column headers
        w.writerows(out_rows)
    import json
    json.dump({"elapsed_sec": 0, "rows": len(out_rows), "llm_calls": semtext.METER.calls},
              open(a.out.replace(".csv", ".meta.json"), "w"))

if __name__ == "__main__":
    main()
```
```

- [ ] **Step 5: Syntax check the three index.mjs**

Run: `cd src/semdb && node --check agents/vadar-signature/index.mjs && node --check agents/vadar-api/index.mjs && node --check agents/vadar-solver/index.mjs`
Expected: no output (exit 0).

- [ ] **Step 6: Commit**

```bash
git add src/semdb/agents/vadar-signature src/semdb/agents/vadar-api src/semdb/agents/vadar-solver
git commit -m "feat(semdb): modality-aware text system prompts + promptPathText for VADAR agents"
```

---

## Task 9: Route text corpora through DIRECT mode (text branch of `runQueryDirect`)

**Files:**
- Modify: `src/semdb/orchestrator.mjs` (`runQueryDirect` — branch on `corpus.isImage`)

**Interfaces:**
- Consumes: `vadarSignatureConfig.promptPathText` etc., `refineLoop`, `scoreWithDiff`.
- Produces: for a text corpus, `runQueryDirect` runs the 3 agents with the text system prompts and runs `solve_<q>.py` with `--endpoint --model --api-key`.

- [ ] **Step 1: Branch the generation + run on modality**

This task REPLACES three closures from Task 5 with modality-aware versions —
`runSolver`, `gen3Agents`, and `regenSolver`. The `genFirst`/`regen`/`scoreIter`
closures, the `if (args.dryRun)` line, and the `refineLoop` invocation from Task 5
stay exactly as they are (they already call `gen3Agents(iterDir, iterCode, sql)` /
`regenSolver(iterDir, iterCode, feedback)`).

First, guard the image-only setup lines (`imgHeader`, `pickImageCols`, `imageDir`)
so they only compute for an image corpus (a text corpus has no image manifest).
Replace the current unconditional image-col block near the top of `runQueryDirect`
with:

```js
  const isImage = corpus.isImage || corpus.modality === "image";
  let imgFilenameCol = "", imgFilepathCol = "", imageDir = "";
  if (isImage) {
    const imgHeader = await headerOf(corpus.path);
    ({ filename: imgFilenameCol, filepath: imgFilepathCol } = pickImageCols(imgHeader));
    imageDir = args.imageDir || (corpus.path ? resolve(dirname(corpus.path), "images") : "");
  }
```

Then replace the Task 5 `runSolver`, `gen3Agents`, and `regenSolver` with these
modality-aware versions (everything else in the closure block is unchanged):

```js
  const textModel = args.extractModel || defaults.extraction.smallTextModel;
  const sysPrompts = isImage
    ? {}
    : { sig: vadarSignatureConfig.promptPathText, api: vadarApiConfig.promptPathText, solver: vadarSolverConfig.promptPathText };

  const runSolver = (iterDir, iterCode, iterCsv) => {
    if (!(doRun && existsSync(iterCode))) return { status: "empty", stderr: "" };
    const sArgs = isImage
      ? [iterCode, iterCsv, "--data-dir", dataDir, ...(imageDir ? ["--image-dir", imageDir] : []),
         "--clip-model", (args.clipModel || defaults.extraction.clipModel)]
      : [iterCode, iterCsv, "--data-dir", dataDir,
         "--endpoint", args.endpoint || "", "--model", textModel, "--api-key", args.apiKey];
    console.log(`\n[SemDB] Running direct solver: python3 ${sArgs.join(" ")}`);
    const s = spawnSync("python3", sArgs, { stdio: ["inherit", "inherit", "pipe"] });
    const stderr = (s.stderr || "").toString();
    if (s.status !== 0) { console.warn(`[SemDB] direct solver exited ${s.status}.`); return { status: "crash", stderr }; }
    return { status: "ok", stderr };
  };

  // Solver template vars differ by modality: image gets manifest cols, text does not.
  const solverVars = (iterCode, querySql, helpersPath) => isImage
    ? { query_id: query, query_sql: querySql, query_nl: nl || "(none)", semdb_dir: __dirname,
        tables_doc: tableLines.join("\n"),
        image_table: corpus.table, image_path: corpus.path,
        image_filename_col: imgFilenameCol, image_filepath_col: imgFilepathCol,
        image_dir: imageDir, helpers_path: helpersPath, solve_path: iterCode }
    : { query_id: query, query_sql: querySql, query_nl: nl || "(none)", semdb_dir: __dirname,
        tables_doc: tableLines.join("\n"), helpers_path: helpersPath, solve_path: iterCode };

  const gen3Agents = async (iterDir, iterCode, querySql) => {
    const sigPath = resolve(iterDir, `_vadar_signatures_${query}.txt`);
    const helpersPath = resolve(iterDir, `_vadar_helpers_${query}.py`);
    const common = { corpus_name: corpus.table, semdb_dir: __dirname };
    record("vadar_signature", await runPhase(vadarSignatureConfig, {
      ...common, query_sql: querySql,
      schema_json: "(DIRECT mode: no schema; read value spaces from the CSVs at runtime)",
      sig_path: sigPath,
    }, iterDir, args, { systemPromptPath: sysPrompts.sig }));
    record("vadar_api", await runPhase(vadarApiConfig, {
      ...common, sig_path: sigPath, helpers_path: helpersPath,
    }, iterDir, args, { systemPromptPath: sysPrompts.api }));
    record("vadar_solver", await runPhase(vadarSolverConfig,
      solverVars(iterCode, querySql, helpersPath),
      iterDir, args, { systemPromptPath: sysPrompts.solver }));
  };

  const regenSolver = async (iterDir, iterCode, feedback) => {
    // Helpers were generated ONCE into iter_0 (refineLoop seeds only code forward) — read
    // from iter_0, not the current iterDir which has no helpers file.
    const helpersPath = resolve(runDir, "iter_0", `_vadar_helpers_${query}.py`);
    const vars = solverVars(iterCode, sql + "\n\n" + feedback,
      existsSync(helpersPath) ? helpersPath : "(seed helpers from iter_0)");
    record("vadar_solver", await runPhase(vadarSolverConfig, vars, iterDir, args,
      { systemPromptPath: sysPrompts.solver }));
  };
```

(For an image corpus `sysPrompts.*` are all `undefined`, so `runPhase` falls back
to `agentConfig.promptPath` — the vision prompts — exactly as before.)

- [ ] **Step 2: Syntax check**

Run: `cd src/semdb && node --check orchestrator.mjs`
Expected: no output.

- [ ] **Step 3: Dry-run a text query (movie q3a) in DIRECT mode**

Run:
```bash
cd src/semdb && node orchestrator.mjs --query q3a --direct \
  --benchmark movie \
  --query-dir /localhome/hza214/SemBench/files/movie/query/bigquery \
  --data-dir /localhome/hza214/SemBench/files/movie/data/sf_9 \
  --dry-run 2>&1 | tail -30
```
Expected: prints the TEXT-mode Signature/API/Solver prompts (mentions `predefined_text.py`, `TextPatch`, `--endpoint`); no crash. (If `sf_9` doesn't exist, substitute an existing `movie/data/sf_*` dir — list with `ls /localhome/hza214/SemBench/files/movie/data`.)

- [ ] **Step 4: Commit**

```bash
git add src/semdb/orchestrator.mjs
git commit -m "feat(semdb): route text corpora through DIRECT mode (text solver + endpoint run)"
```

---

## Task 10: `semtext.run_extraction` — offline text-extraction engine (compiled mode)

**Files:**
- Modify: `src/semdb/semtext.py`
- Test: `src/semdb/tests/test_semtext_run_extraction.py`

**Interfaces:**
- Consumes: `TextPatch`, `get_ctx`, `METER`, `semextract.gen_endpoint` (via `TextPatch`).
- Produces: `run_extraction(driver, schema, table_path, out_path, *, model, endpoint=None, api_key="EMPTY", concurrency=8, theta=None, text_col=None, limit=0, timeout=120) -> dict`. `driver` has `map_columns(header) -> {"id", "text", "context"}` and `extract(patch) -> {field: value}`. Writes `out_path` (JSON list of `{**fields, "conf", <id_col>}`) + `out_path + ".meta.json"` (same keys as `semextract.run`'s meta: `rows, extracted, none, llm_calls, elapsed_sec, ...`). Returns the meta dict.

The output contract MUST match `semextract.run` so `compiled_<q>.py` and the corpus telemetry read the attrs + meta unchanged: attrs is a JSON list of per-row records, each record has every `schema.attributes[].name` key, a `conf` float, and the id column; meta has at least `rows`, `llm_calls`, `elapsed_sec`.

- [ ] **Step 1: Make `METER` increment thread-safe**

`run_extraction` dispatches rows concurrently (endpoint calls are HTTP/thread-safe), so `METER.calls += 1` inside `_ask` races. Add a lock. In `src/semdb/semtext.py`, change `_Meter` and the increment in `_ask`:

```python
import threading

class _Meter:
    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.calls = 0

    def incr(self) -> None:
        with self._lock:
            self.calls += 1
```

In `_ask`, replace `METER.calls += 1` with `METER.incr()`.

- [ ] **Step 2: Write the failing test**

```python
# src/semdb/tests/test_semtext_run_extraction.py
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import semtext


def _fake_gen(monkeypatch):
    import semextract
    def fake(cfg, json_schema, prompt, modality, image_path=None, text=None):
        # classify → return the label that appears in the text; else first option.
        return json.dumps({"value": "positive" if "great" in (text or "") else "negative"})
    monkeypatch.setattr(semextract, "gen_endpoint", fake)


class _Driver:
    def map_columns(self, header):
        return {"id": "id", "text": "review", "context": []}
    def extract(self, patch):
        return {"sentiment": patch.classify(["positive", "negative"])}


def test_run_extraction_writes_attrs_and_meta(tmp_path, monkeypatch):
    _fake_gen(monkeypatch)
    semtext.METER.reset()
    table = tmp_path / "reviews.csv"
    table.write_text("id,review\n1,a great film\n2,a dull film\n")
    schema = {"attributes": [{"name": "sentiment", "type": "string"}]}
    out = tmp_path / "attrs.json"
    meta = semtext.run_extraction(_Driver(), schema, str(table), str(out),
                                  model="m", endpoint="http://x/v1", concurrency=2)
    recs = json.load(open(out))
    assert {r["id"]: r["sentiment"] for r in recs} == {"1": "positive", "2": "negative"}
    assert all("conf" in r for r in recs)
    m = json.load(open(str(out) + ".meta.json"))
    assert m["rows"] == 2 and m["llm_calls"] == meta["llm_calls"] == 2
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd src/semdb && python3 -m pytest tests/test_semtext_run_extraction.py -v`
Expected: FAIL — `AttributeError: module 'semtext' has no attribute 'run_extraction'`.

- [ ] **Step 4: Implement `run_extraction`**

Append to `src/semdb/semtext.py`:

```python
import csv as _csv
import json as _json
import os as _os
import sys as _sys
import time as _time
from concurrent.futures import ThreadPoolExecutor


def _none_record(schema, id_col, id_val):
    rec = {a["name"]: ([] if "array" in a.get("type", "") or a.get("multi") else "none")
           for a in schema.get("attributes", [])}
    rec["conf"] = 0.0
    rec[id_col] = id_val
    return rec


def run_extraction(driver, schema, table_path, out_path, *, model, endpoint=None,
                   api_key="EMPTY", concurrency=8, theta=None, text_col=None,
                   limit=0, timeout=120):
    """Offline TEXT extraction engine (text analog of semextract.run / vadar_engine.run).
    For each corpus row it builds a TextPatch and calls driver.extract(patch), which
    composes predefined_text primitives. Writes attrs JSON + <out>.meta.json with the
    SAME contract as semextract.run. Returns the meta dict. Aborts (exit 3) without
    writing out_path if most rows error, so the orchestrator re-runs rather than caching
    a broken corpus."""
    ctx = get_ctx(model, endpoint or "", api_key, timeout)
    rows = list(_csv.DictReader(open(table_path)))
    if limit:
        rows = rows[:limit]
    cols = driver.map_columns(list(rows[0].keys()) if rows else [])
    id_col = cols["id"]
    tcol = text_col or cols.get("text")
    METER.reset()
    t_start = _time.time()
    attrs = [None] * len(rows)
    n_none = n_error = 0

    def process_row(i, r):
        id_val = r[id_col]
        try:
            patch = TextPatch(r.get(tcol, "") if tcol else "", ctx)
            fields = driver.extract(patch) or {}
            rec = dict(fields)
            primary = schema.get("attributes", [{}])[0].get("name")
            val = rec.get(primary)
            rec["conf"] = 0.0 if val in (None, "none", "", []) else 1.0
            rec[id_col] = id_val
            return {"i": i, "rec": rec, "error": None}
        except Exception as e:  # noqa: BLE001 — one bad row must not kill the batch
            return {"i": i, "rec": _none_record(schema, id_col, id_val), "error": str(e)}

    def tally(res):
        nonlocal n_none, n_error
        attrs[res["i"]] = res["rec"]
        if res["error"]:
            n_error += 1
            print(f"[semtext] {res['i']+1}/{len(rows)} ERROR: {res['error']}")
        elif res["rec"].get("conf", 0.0) == 0.0:
            n_none += 1

    workers = max(1, concurrency)
    if workers > 1 and endpoint:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(process_row, range(len(rows)), rows):
                tally(res)
    else:
        for i, r in enumerate(rows):
            tally(process_row(i, r))

    attrs = [a for a in attrs if a is not None]
    elapsed = _time.time() - t_start
    if len(rows) > 0 and n_error >= max(1, len(rows) // 2):
        _sys.stderr.write(f"[semtext] ABORT: {n_error}/{len(rows)} rows errored — likely a "
                          f"bad endpoint/model/creds. Not writing {out_path}.\n")
        _sys.exit(3)

    _json.dump(attrs, open(out_path, "w"), indent=2)
    meta = {
        "model": model, "endpoint": endpoint, "modality": "text",
        "rows": len(attrs), "extracted": len(attrs) - n_none, "none": n_none,
        "errors": n_error, "llm_calls": METER.calls, "theta": theta,
        "elapsed_sec": round(elapsed, 2),
        "sec_per_row": round(elapsed / max(1, len(attrs)), 3),
        "concurrency": workers,
    }
    _json.dump(meta, open(out_path + ".meta.json", "w"), indent=2)
    print(f"[semtext] wrote {len(attrs)} rows -> {out_path} ({elapsed:.1f}s, "
          f"{meta['llm_calls']} llm calls, {n_none} none)")
    return meta
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd src/semdb && python3 -m pytest tests/test_semtext_run_extraction.py tests/test_semtext_fake_endpoint.py -v`
Expected: PASS (the new test + the Task 6 tests still green after the METER lock change).

- [ ] **Step 6: Commit**

```bash
git add src/semdb/semtext.py src/semdb/tests/test_semtext_run_extraction.py
git commit -m "feat(semdb): semtext.run_extraction offline text-extraction engine (compiled-mode text inference)"
```

---

## Task 11: Text Extractor composes `predefined_text`

**Files:**
- Modify: `src/semdb/agents/extractor/prompt.md`
- Modify: `src/semdb/agents/extractor/user-prompt.md`

**Interfaces:**
- Consumes: `semtext.run_extraction`, `vadar/predefined_text.py`.
- Produces: for a TEXT corpus, the generated `extract_<corpus>.py` composes `predefined_text` over `semtext.TextPatch` and calls `semtext.run_extraction` (instead of `semextract.run`). The orchestrator's extraction invocation is UNCHANGED (same positional `table out` + `--schema --model --endpoint --api-key --concurrency --theta` flags).

> No orchestrator or `ensureCorpus` change: the text-extraction invocation already passes `--endpoint --api-key --concurrency --model` for text corpora (see `orchestrator.mjs` `ensureCorpus`), and `run_extraction` accepts exactly those. Only the generated driver's body changes.

- [ ] **Step 1: Update `agents/extractor/prompt.md` — add the TEXT composition path**

The prompt currently says (near the end of the Image section) "TEXT corpora still use `semextract.run`." Replace that sentence and add a TEXT section mirroring the IMAGE one. Find:

```
`--model` is the CLIP id; NO `--endpoint`. TEXT corpora still use `semextract.run`.
(Reference compositions + engine: `src/semdb/vadar_run.py`, `src/semdb/vadar_engine.py`.)
```

Replace with:

```
`--model` is the CLIP id; NO `--endpoint`.
(Reference compositions + engine: `src/semdb/vadar_run.py`, `src/semdb/vadar_engine.py`.)

## Text corpora — compose the predefined TEXT API (in-code semantic inference)
When the corpus modality is TEXT, do NOT hand-roll endpoint/JSON plumbing. You WRITE a
`Driver.extract(patch) -> {field: value, ...}` that composes the predefined TEXT API
(`judge / classify / extract / generate / score` from `vadar/predefined_text.py`, backed
by `semtext.TextPatch`) — the text analog of the image `Driver.extract(patch)` that
composes ImagePatch. Pick the lightest primitive per schema attribute:

- boolean AI.IF predicate → `judge(patch, "<yes/no question>")`
- a value from a CLOSED value space (enum / DB column) → `classify(patch, LABELS)`
- one named attribute → `extract(patch, "<field>")`
- a soft/relevance score → `score(patch, "<short phrase>")`

Value-space lists come from the schema's `labels` (already filled from `labels_from`).
The returned value IS the field value (joins/filters downstream). Emit exactly this shape
(the orchestrator runs it with `<table> <attrs> --schema S --model M --endpoint U --api-key K
--concurrency N`):

```python
import sys, os, json, argparse
sys.path.insert(0, "<dir containing semtext.py>")   # given to you (semdb_dir)
import semtext
from vadar.predefined_text import judge, classify, extract, score

LABELS = [...]                      # e.g. from schema.attributes[].labels

class Driver:
    def map_columns(self, header):
        return {"id": "<id col>", "text": "<text col>", "context": ["<extra cols>"]}
    def extract(self, patch):
        # ONE entry per schema attribute, composing predefined_text over `patch`.
        return {"sentiment": classify(patch, LABELS)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("table"); ap.add_argument("out")
    ap.add_argument("--schema", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint"); ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--image-dir"); ap.add_argument("--theta")   # accepted, ignored
    a = ap.parse_args()
    semtext.run_extraction(Driver(), json.load(open(a.schema)), a.table, a.out,
                           model=a.model, endpoint=a.endpoint, api_key=a.api_key,
                           concurrency=a.concurrency, theta=a.theta)
```
`--model` is the endpoint LLM id; `--endpoint` is required for text. Accept `--image-dir`
and ignore it. (Reference API: `src/semdb/vadar/predefined_text.py`, engine: `semtext.run_extraction`.)
```

- [ ] **Step 2: Update `agents/extractor/user-prompt.md` — route text to the new engine**

Find the Output section:

```
- If Modality is `image`, target `semvision.run` (tiered non-VLM proxies driven by each
  attribute's `extractor` spec); `--model` is the CLIP model id, no `--endpoint` needed.
  If Modality is `text`, target `semextract.run` as before.
```

Replace with:

```
- If Modality is `image`, target `vadar_engine.run` (tiered non-VLM proxies driven by each
  attribute's `extractor` spec); `--model` is the CLIP model id, no `--endpoint` needed.
  If Modality is `text`, compose the predefined TEXT API (`judge/classify/extract/score`
  from `vadar/predefined_text.py`) in `Driver.extract(patch)` and call
  `semtext.run_extraction(...)`; `--model` is the endpoint LLM id and `--endpoint` is required.
```

And in the "Engine to import" section, add a text bullet after the `semextract.run` line:

```
- For TEXT corpora import `semtext` instead: `semtext.run_extraction(driver, schema,
  table_path, out_path, *, model, endpoint, api_key="EMPTY", concurrency=8, theta=None)`.
  `driver` implements `map_columns(header)` and `extract(patch) -> {field: value}` composing
  `vadar.predefined_text` over the `semtext.TextPatch` it is handed. The engine owns
  concurrency / meta / checkpoint / abort — do not re-implement them.
```

- [ ] **Step 3: Dry-run the extractor prompt for a text corpus**

Run:
```bash
cd src/semdb && node orchestrator.mjs --query q3a --benchmark movie \
  --query-dir /localhome/hza214/SemBench/files/movie/query/bigquery \
  --data-dir /localhome/hza214/SemBench/files/movie/data/<sf> \
  --dry-run 2>&1 | grep -i "predefined_text\|run_extraction\|TextPatch" | head
```
Expected: the rendered Extractor prompt mentions `predefined_text` / `run_extraction` / `TextPatch`. (Substitute an existing `movie/data/sf_*` for `<sf>`.)

- [ ] **Step 4: Commit**

```bash
git add src/semdb/agents/extractor/prompt.md src/semdb/agents/extractor/user-prompt.md
git commit -m "feat(semdb): text Extractor composes predefined_text via semtext.run_extraction"
```

---

## Task 12: End-to-end verification + regression

**Files:** none (verification only). If a defect is found, fix it in the owning task's files and re-commit.

- [ ] **Step 1: Confirm the endpoint is up (text inference needs it)**

Run: `curl -s http://localhost:8000/v1/models | head -c 200 || echo "NO ENDPOINT"`
Expected: a JSON model list. If `NO ENDPOINT`, start vLLM per CLAUDE.md before Step 3, or skip Step 3 and run only the image E2E in Step 4.

- [ ] **Step 2: Run the full Python test suite (no regressions)**

Run: `cd src/semdb && source $HOME/anaconda3/etc/profile.d/conda.sh && conda activate sembench && python3 -m pytest tests/test_evaluate_emit_diff.py tests/test_semtext_fake_endpoint.py tests/test_predefined_text.py tests/test_semtext_run_extraction.py -v`
Expected: all PASS.

- [ ] **Step 3: Text DIRECT end-to-end on movie with ground truth + refinement**

Run:
```bash
cd src/semdb && source $HOME/anaconda3/etc/profile.d/conda.sh && conda activate sembench && \
node orchestrator.mjs --query q3a --direct --benchmark movie \
  --query-dir /localhome/hza214/SemBench/files/movie/query/bigquery \
  --data-dir /localhome/hza214/SemBench/files/movie/data/<sf> \
  --ground-truth-dir /localhome/hza214/SemBench/files/movie/raw_results/ground_truth \
  --endpoint http://localhost:8000/v1 --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-iterations 5 2>&1 | tail -40
```
Expected:
- `runs/movie-q3a/iter_0/solve_q3a.py` exists and ran;
- at least `runs/movie-q3a/iter_0/diff.json` exists with `false_positives`/`false_negatives`;
- console shows `--- refine 1/5 ---` and a keep/rollback line;
- `runs/movie-q3a/telemetry.json` has a `refine.f1_history` array and `metrics`;
- a row for `movie q3a` is appended to `runs/results.csv`.

Verify: `cat runs/movie-q3a/telemetry.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['refine']); print(d.get('metrics'))"`

- [ ] **Step 4: Regression — image DIRECT + `--no-refine` reproduces single-shot**

Run:
```bash
cd src/semdb && node orchestrator.mjs --query q7 --direct --no-refine \
  --query-dir /localhome/hza214/SemBench/files/mmqa/query/bigquery \
  --data-dir /localhome/hza214/SemBench/files/mmqa/data/sf_200 \
  --ground-truth-dir /localhome/hza214/SemBench/files/mmqa/raw_results/ground_truth 2>&1 | tail -20
```
Expected: runs iter_0 only (no `--- refine 1/5 ---` line), scores, appends a `mmqa q7` row. `telemetry.json.refine.iterations == 0`.

- [ ] **Step 5: Regression — compiled path still scores**

Run:
```bash
cd src/semdb && node orchestrator.mjs --query q2a \
  --query-dir /localhome/hza214/SemBench/files/mmqa/query/bigquery \
  --data-dir /localhome/hza214/SemBench/files/mmqa/data/sf_200 \
  --ground-truth-dir /localhome/hza214/SemBench/files/mmqa/raw_results/ground_truth \
  --max-iterations 2 2>&1 | tail -25
```
Expected: corpus schema/extract run once, compiled_q2a.py generated + refined (≤2 iters), a `mmqa q2a` row appended, telemetry has `refine`.

- [ ] **Step 6: Compiled-mode TEXT extraction composes `predefined_text` (feature 2 in compiled mode)**

Run (needs the endpoint from Step 1):
```bash
cd src/semdb && source $HOME/anaconda3/etc/profile.d/conda.sh && conda activate sembench && \
node orchestrator.mjs --query q3a --benchmark movie \
  --query-dir /localhome/hza214/SemBench/files/movie/query/bigquery \
  --data-dir /localhome/hza214/SemBench/files/movie/data/<sf> \
  --ground-truth-dir /localhome/hza214/SemBench/files/movie/raw_results/ground_truth \
  --endpoint http://localhost:8000/v1 --extract-model Qwen/Qwen2.5-0.5B-Instruct \
  --max-iterations 2 --force 2>&1 | tail -40
```
Expected:
- the generated `runs/_corpus/reviews/extract_reviews.py` imports `semtext` and `vadar.predefined_text` (verify: `grep -l predefined_text runs/_corpus/reviews/extract_reviews.py`);
- `runs/_corpus/reviews/reviews_attrs.json` + `.meta.json` produced (meta has `llm_calls`);
- `compiled_q3a.py` generated + refined (≤2 iters), a `movie q3a` row appended, telemetry has `refine`.

- [ ] **Step 7: Final commit (if any fixes were made) + summary**

```bash
git add -A && git commit -m "test(semdb): verify iterative refinement + text DIRECT end-to-end"
```

---

## Self-Review

**Spec coverage:**
- §1.1 contract → Tasks 2 (`checkSemdbImprovement`/`shouldContinueSemdb`), 4 (`refineLoop`). ✓
- §1.2 plug-in points (both per-query fns) → Tasks 4 (compiled), 5 (image DIRECT), 9 (text DIRECT). ✓
- §1.3 `--emit-diff` → Task 1. ✓
- §1.4 feedback prompt → Task 3 (`renderFeedback`). ✓
- §1.5 stop logic → Task 2. ✓
- §1.6 config + CLI → Tasks 2 (config), 3 (flags). ✓
- §1.7 telemetry `refine` block → Tasks 4, 5, 9 (added per path). ✓
- §2.2 `predefined_text.py` + `semtext.py` → Tasks 6, 7. ✓
- §2.3 modality-aware 3-agent routing → Tasks 8 (prompts + promptPathText), 9 (branch). ✓
- §2.4 text solver contract → Task 8 (solver prompt skeleton). ✓
- §2.5 loop over text solver → Task 9 (uses `refineLoop`). ✓
- Compiled-mode feature 1 (loop over `compiled_<q>.py`) → Task 4. ✓
- Compiled-mode feature 2 (in-code text inference = text Extractor composes `predefined_text`) → Tasks 10 (`semtext.run_extraction`), 11 (Extractor prompt). ✓

**Placeholder scan:** All steps carry complete code. The `[ ... ]` / `[...]` inside generated *solver and driver skeletons* (Tasks 8, 11) are template markers the agent fills at run time, not plan placeholders. No "TBD/TODO" or "similar to Task N" references remain.

**Type consistency:** `checkSemdbImprovement({status,f1})` and `shouldContinueSemdb(history,...)` signatures are identical across Tasks 2, 4, 5, 9. `refineLoop` closures use the uniform contract — `genFirst(iterDir,iterCode,iterCsv)` / `regen(iterDir,iterCode,iterCsv,feedback)` return `{status,stderr}`, `scoreIter(iterDir,iterCode,iterCsv,run)` returns the scored outcome — in Tasks 4, 5, 9. `scoreWithDiff(args, query, planObj, telePath, resultsCsv, diffPath, csvPath, finalize)` is defined once (Task 4) and reused (Tasks 5, 9), reading metrics from the diff JSON. `runPhase(..., opts)` optional 5th arg added in Task 3, used in Tasks 8/9. `semtext.get_ctx`/`TextPatch`/`METER`/`run_extraction` names match across Tasks 6, 8, 10, 11. `predefined_text` free-function names match Tasks 7, 8, 11. `run_extraction`'s attrs+meta output matches `semextract.run` (Task 10) so `compiled_<q>.py` reads it unchanged. The `gen3Agents(iterDir, iterCode, querySql)` signature is consistent between Tasks 5 and 9.

**Compiled-mode consistency note:** feature 2 keeps `compiled_<q>.py` pure relational (inference stays offline in the Extractor), so the Task 4 loop needs no change — the "unified feedback tunes residual inference" option reduces to Task 4 as-is because there is no in-code residual under the chosen Extractor-composes design.

**Known follow-ups (out of scope, noted):** model escalation to a stronger model on repeated failure (GenDB has it; SemDB reuses the same agent) can be added later; scenario `--emit-diff` for non-membership metrics falls back to sampled raw rows.
