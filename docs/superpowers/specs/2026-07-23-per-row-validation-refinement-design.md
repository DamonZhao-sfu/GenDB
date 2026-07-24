# Per-Row Validation-Driven DIRECT Solver Refinement — Design Spec

**Date:** 2026-07-23
**Branch:** `claude/vadar-strict-3agent`
**Status:** Approved design → implementation
**Builds on:** `2026-07-23-semdb-iterative-refinement-design.md` (the F1-based `refineLoop`)

## Summary

The existing iterative refinement loop scores a query's **final output F1**, which
requires **full ground truth for the whole corpus**. At real query time there is
no such GT, so that loop cannot engage in production.

This design adds a second, GT-light feedback channel: the user **manually labels a
small validation test-case** (per-row semantic-inference labels), and those labels
act as the *only* ground truth available. They drive a refinement loop over the
DIRECT-mode `solve_<q>.py` **inference step** — scoring each row's inferred
attribute against the manual label, feeding back the mislabeled rows, and refining
the solver until per-row accuracy plateaus. The frozen solver then runs over the
**full unlabeled corpus** to produce the final result.

This mirrors a real deployment: you can afford to hand-label a few dozen rows, use
them to tune the generated inference code, and then apply it at scale — with no
train-on-test leakage (validation rows drive iteration only; the full-corpus output
is produced by the frozen code).

### Decisions locked in brainstorming

1. **Label source:** truly manual labeling (not subsampled benchmark GT).
2. **Granularity:** per-row inference (the attribute value each row should infer),
   NOT query-final-output.
3. **Refine target:** DIRECT mode `solve_<q>.py` inference step (Extractor and the
   compiled path are untouched).
4. **Mechanism:** labels are a **scoring signal only** and drive the refine loop.
   Labels are **NOT** injected into the inference prompt as few-shot exemplars.

---

## Part 1 — What the user provides: `val.json`

One validation file per query, hand-authored. It names the single semantic
attribute the query depends on and the correct value for each labeled row:

```json
{
  "query": "q3a",
  "attr": "genre",
  "labels": {
    "m1": "comedy",
    "m2": "drama",
    "m3": "comedy"
  }
}
```

- **`query`** — the query id (matches the orchestrator `--query`).
- **`attr`** — the semantic attribute this query hinges on (e.g. a `genre`
  classification, a boolean predicate outcome, an extracted field).
- **`labels`** — `{ row_id: expected_value }`. `row_id` is the corpus primary key
  (the same id column the solver reads). Values are compared **case-insensitively,
  stripped** (same normalization `evaluate.py` already uses for membership).

The labeled ids define the **validation sub-corpus** — the rows the loop runs the
solver on during iteration. Everything else in the corpus is the unlabeled
"training set" the frozen solver is applied to at the end.

**Location:** passed via a new CLI flag `--val-file <path>` (per query). Absent
flag ⇒ this loop does not engage; behavior is exactly today's (F1 loop if GT
present, else single-shot).

---

## Part 2 — What the solver must expose: `trace_<q>.json`

Today the DIRECT text solver writes only the final result CSV + `solve_<q>.meta.json`
(`elapsed_sec, rows, llm_calls`). Per-row scoring needs the solver's **intermediate
per-row inference decisions**, so the solver contract gains one output.

The generated `solve_<q>.py` must additionally write a trace mapping each processed
row id to the value it inferred for the query's key attribute:

```json
{
  "attr": "genre",
  "rows": {
    "m1": "comedy",
    "m2": "comedy",
    "m3": "comedy"
  }
}
```

- Written to `trace_<q>.json` next to the result CSV.
- `attr` echoes the attribute the solver treated as the query's semantic pivot.
- `rows` maps every processed row id → the inferred value for that attribute
  (the output of the `judge/classify/extract` call the solver made for that row,
  BEFORE the relational filter/join drops it).

The VADAR **Solver** system prompt (`prompt-text.md`) is extended with this
requirement + a one-line skeleton showing how to accumulate and dump the trace.
This is the only generated-code change; the relational logic is unchanged.

**Boolean predicates:** when the query pivots on an `AI.IF`-style boolean
(`judge`), the trace value is `"true"`/`"false"` and labels use the same, so the
scorer stays a plain equality check — no separate code path.

---

## Part 3 — The scorer: per-row accuracy + mislabel feedback

A new scorer (subcommand of `evaluate.py`, `--score-inference`, keeping one eval
entrypoint) consumes `trace_<q>.json` + `val.json` and emits, for the **labeled ids
only**:

```json
{
  "query": "q3a",
  "attr": "genre",
  "n": 20,
  "correct": 14,
  "accuracy": 0.70,
  "mistakes": [
    { "id": "m7",  "text": "<row text snippet, capped>", "predicted": "drama",  "expected": "comedy" },
    { "id": "m12", "text": "<row text snippet, capped>", "predicted": "comedy", "expected": "drama"  }
  ],
  "n_mistakes": 6,
  "sampled": false
}
```

- **`accuracy`** = correct / n over the labeled ids present in the trace. A labeled
  id **missing** from the trace counts as wrong (the solver dropped a row it should
  have inferred), and is surfaced with `predicted: null`.
- **`mistakes`** capped at `refineSampleCap` (reuse the existing default 15).
- The row `text` snippet is looked up from the corpus CSV by id and truncated, so
  the agent sees *why* the inference was wrong (analogous to the FP/FN rows in the
  F1 loop). Snippet length capped to bound prompt size.

This is the per-row analog of `--emit-diff`: the same idea (surface the concrete
rows behind the number), one granularity down.

---

## Part 4 — Loop integration (reusing `refineLoop`)

The loop machinery is the same `refineLoop` / `checkSemdbImprovement` /
`shouldContinueSemdb` from the F1-loop design; only the **objective** and the
**per-iteration run scope** change. A small strategy indirection selects which
scorer + which run scope the loop uses:

| Aspect | F1 loop (existing) | Per-row val loop (new) |
|---|---|---|
| Objective metric | query-output F1 (needs full GT) | per-row inference accuracy (needs `val.json`) |
| Per-iteration run scope | full corpus | **validation sub-corpus only** (cheap) |
| Feedback rows | FP/FN result rows (`--emit-diff`) | mislabeled rows (`--score-inference`) |
| `improved` rule | correctness-first, then F1↑ | correctness-first, then accuracy↑ |
| Stop rule | acc/F1==1.0 / stall 2 / max 5 | identical, on accuracy |
| After freeze | best iter's full-corpus CSV | **run frozen solver on FULL corpus once** → final CSV |

Concretely, in `runQueryDirect`, when `--val-file` is set (text branch):

1. **Load `val.json`**; build the validation sub-corpus = the labeled ids. Write a
   filtered corpus view (or pass `--only-ids` to the solver) so each iteration's
   solver run touches only labeled rows → few LLM calls per iteration.
2. **iter_0..N** via `refineLoop`, but the closures use:
   - `runSolver(..., valSubset)` — run on the sub-corpus, producing the result CSV
     **and** `trace_<q>.json`.
   - `scoreIter` → `evaluate.py --score-inference --trace trace_<q>.json
     --val-file val.json` → `{ status, f1: accuracy, metrics, diff: {mistakes...} }`.
     (The loop's `f1` field carries accuracy; no schema change to the loop.)
   - `renderFeedback` gains a per-row branch: "these labeled rows were inferred
     wrong — predicted X, expected Y — here is the row text; revise the
     judge/classify/extract call, threshold, or value-space mapping."
3. **Freeze** the best-accuracy `solve_<q>.py`.
4. **Final full run:** run the frozen solver on the **entire** corpus (no
   `--only-ids`) → the real result CSV that lands in the workload output. If full
   benchmark GT *also* happens to be available, the existing F1 scoring still runs
   on this final CSV for reporting — but it never drives refinement.

### Precedence when both signals exist

- `--val-file` present → **per-row val loop** drives refinement (this design).
- else, GT present and `--no-refine` unset → **F1 loop** (existing design).
- else → single-shot (iter_0 only), today's behavior.

`--no-refine` still forces single-shot in all cases.

---

## Part 5 — Telemetry

Per-query `telemetry.json` `refine` block gains a `mode` discriminator and records
accuracy history (reusing the `f1` field name to avoid a schema fork — it holds
accuracy in val mode):

```json
"refine": {
  "mode": "per_row_val",
  "objective": "inference_accuracy",
  "val_file": "val/q3a.json",
  "val_n": 20,
  "iterations": 3,
  "best_iteration": 2,
  "max_iterations": 5,
  "stop_reason": "stalled",
  "f1_history": [
    { "iter": 0, "f1": 0.55, "status": "ok", "improved": true },
    { "iter": 1, "f1": 0.70, "status": "ok", "improved": true },
    { "iter": 2, "f1": 0.80, "status": "ok", "improved": true },
    { "iter": 3, "f1": 0.80, "status": "ok", "improved": false }
  ],
  "final_full_corpus_rows": 1834
}
```

`mode: "f1"` (or absent) = the existing loop; `mode: "per_row_val"` = this one.

---

## Files touched

**New**
- `src/semdb/tests/test_score_inference.py` — scorer unit tests (accuracy, missing
  id = wrong, mistake capping, case/strip normalization).
- `src/semdb/tests/test_val_loop.mjs` — node assertions for val-mode loop wiring
  (scope selection, precedence, telemetry `mode`).
- Example `val/<q>.json` fixtures under `src/semdb/tests/fixtures/`.

**Modified**
- `src/semdb/evaluate.py` — add `score_inference(trace, val, corpus_csv, cap) -> dict`
  and a `--score-inference` CLI branch (`--trace`, `--val-file`, `--corpus-csv`,
  `--diff-cap`); reuse existing normalization helpers.
- `src/semdb/orchestrator.mjs` — `parseArgs` gains `--val-file`; `runQueryDirect`
  text branch: val-subset run scope, per-row `scoreIter`, per-row `renderFeedback`
  branch, precedence gate, final full-corpus run; telemetry `mode`.
- `src/semdb/agents/vadar-solver/prompt-text.md` (+ `user-prompt-text.md` if the
  trace path is templated) — require `trace_<q>.json` output + skeleton; document
  the `--only-ids` run arg the solver must honor.
- `src/semdb/semtext.py` and/or the generated solver contract — honor an
  `--only-ids <file>` (or `--val-ids`) arg so iteration runs touch only labeled
  rows. (If simpler, the orchestrator writes a filtered corpus CSV per iteration
  instead — chosen in the plan; either keeps LLM calls bounded.)

---

## Out of scope

- Refining the **Extractor** or the **compiled path** with per-row labels (this
  design targets DIRECT `solve_<q>.py` only, per the locked decision).
- Injecting labels as **few-shot exemplars** into the inference prompt (explicitly
  rejected: labels are a scoring signal only).
- Multi-attribute validation in one `val.json` (one pivot attr per query for now;
  a query needing two can get two val files / two loops later).
- Image DIRECT per-row validation (same mechanism could extend to it; text first).
- Auto-selecting which rows to label (active learning). The user picks the
  validation rows by hand.

---

## Testing / verification

- **Scorer unit tests:** accuracy math; a labeled id absent from the trace scores
  as wrong with `predicted: null`; mistakes capped at `refineSampleCap`;
  case/strip normalization matches `evaluate.py`.
- **Dry-run:** `--direct --val-file val/q3a.json --dry-run` prints the text-solver
  prompt including the trace-output requirement; no execution.
- **End-to-end (text scenario, e.g. movie q3\*):** with a small hand-authored
  `val.json`, confirm: iter_0 runs on the sub-corpus, `trace_<q>.json` is written,
  `--score-inference` produces accuracy + mistakes, ≥1 refinement iteration runs,
  best-accuracy solver is frozen, the **final full-corpus** CSV is produced from
  the frozen solver, telemetry `refine.mode == "per_row_val"` with a populated
  `f1_history` (= accuracy history).
- **Regression:** without `--val-file`, F1-loop and single-shot behavior are
  byte-for-byte unchanged; `--no-refine` still forces single-shot.
- **No-leakage check:** the final full-corpus run must NOT be restricted to labeled
  ids (assert `--only-ids` absent on the final run).
