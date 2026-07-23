# SemDB Iterative Refinement + Text DIRECT Mode — Design Spec

**Date:** 2026-07-23
**Branch:** `claude/vadar-strict-3agent`
**Status:** Approved design → implementation

## Summary

Two related changes to `src/semdb/`:

1. **Iterative refinement loop** (max 5 iterations) around every per-query code
   generator, mimicking GenDB's iterative agent stage: the agent is re-invoked
   with concrete feedback about what its last run got wrong, and each change is
   kept only if the objective metric improves (else rolled back).

2. **Text DIRECT mode**: extend `--direct` so text-only SemBench queries are
   compiled into one end-to-end `solve_<q>.py` that does the semantic inference
   *in the Python code itself*, by composing a predefined **text inference API**
   (the text analog of the vision `vadar/predefined.py`).

The two compose: the text solver is exactly the kind of end-to-end generated
code the loop refines.

---

## Part 1 — Iterative Refinement Loop

### 1.1 The GenDB contract being mimicked

GenDB's loop (`src/gendb/orchestrator.mjs:1466`, iterations `1..maxIter`) is a
strict **input → agent → execute → judge → keep-or-rollback** cycle:

| Stage | GenDB | SemDB analog |
|---|---|---|
| Starting code | best `.cpp` copied into `iter_N/` | best `solve_<q>.py` / `compiled_<q>.py` copied into `iter_N/` |
| Feedback input | `execution_results.json`: validation pass/fail + column-mismatch samples, timing, op-timings; `previousIterationOutcome`; one-line history | runtime status (ran/crashed/empty) + metrics (P/R/F1, tp/fp/fn) + concrete FP/FN rows + one-line F1 history + previous outcome |
| Agent | Query Optimizer → Code Generator re-invoked, edits code | Same agent re-invoked (VADAR Solver for DIRECT / Code Generator for compiled) with an appended feedback section, editing the existing file |
| Judge | `checkExecutionImprovement`: correctness-first, then timing | correctness-first (runtime error/empty → fix first), then F1 improvement |
| Keep/rollback | promote best on improve, else roll back | identical |
| Gate | `shouldContinue`: stop / continue / escalate | `shouldContinueSemdb`: stop on F1==1.0, stall (2 non-improving), or max 5; else continue |

Essence: **the agent is told exactly what it got wrong, and every change is
accepted only if the objective metric improves.** GenDB's metric is
correctness+latency; SemDB's is runtime-success+F1.

### 1.2 Where the loop plugs in

Two per-query functions gain the loop (corpus-level Schema Designer + Extractor
stay one-shot, exactly as GenDB's iteration-0 planning is one-shot):

- `runQueryDirect` (DIRECT mode) → iterates `solve_<q>.py`
- `runQueryCodegen` (compiled path) → iterates `compiled_<q>.py`

Iteration 0 = today's behavior (first generation + run + eval). Iterations 1–5 =
refinement. Shared machinery is extracted into one helper both paths call:

```
refineLoop({ label, codePath, runDir, genFirst, regen, execAndScore, maxIter })
  → { bestF1, bestIter, bestResultsCsv, history }
```

- `genFirst()` — existing agent stage producing iter_0 code (unchanged behavior).
- `execAndScore(iterDir, codePath, resultsCsv)` — run code → results CSV →
  `evaluate.py` (+ `--emit-diff`) → `{ status, metrics, diffPath }`.
- `regen(iterDir, codePath, feedback)` — re-invoke the same agent with the
  feedback section appended, editing `codePath` in place.

Per-iteration artifacts live in `runs/<bench>-<q>/iter_<N>/` (mirrors GenDB's
`iter_N/`). The **best** iteration's results CSV + telemetry are what get scored
into the workload `results.csv`.

### 1.3 Feedback signal (`evaluate.py --emit-diff`)

`evaluate.py` today emits only tp/fp/fn **counts**. Because it holds the
per-query normalization (container/column semantics differ per query), it is the
correct place to also emit the **actual** mismatched rows. Add an optional
`--emit-diff <path>` flag that writes:

```json
{
  "query": "q2a",
  "f1": 0.62, "precision": 0.55, "recall": 0.71,
  "tp": 17, "fp": 14, "fn": 7,
  "false_positives": [ { "predicted rows we should NOT have": "..." } ],
  "false_negatives": [ { "gt rows we missed": "..." } ],
  "fp_total": 14, "fn_total": 7, "sampled": true, "sample_cap": 15
}
```

Each list capped at `sample_cap` (default 15) to bound prompt size. The diff is
computed with the **same normalized sets** `evaluate.py` already builds for
scoring (id-set membership after per-query normalization), so FP/FN are exactly
the rows that cost precision/recall. `--emit-diff` is additive — no change to
existing metric output or the CSV row.

### 1.4 Feedback prompt block

The loop renders a feedback section appended to the agent's normal user prompt.

Runtime failure (crash or empty CSV) short-circuits to a fix-first block, before
any F1 discussion (the two-tier rule):

```
## LAST RUN FAILED — FIX THIS FIRST
The program {crashed | produced no output rows}.
{stderr tail, ~40 lines}
Diagnose and fix the error before any accuracy work.
```

Otherwise, a metric + samples block:

```
## LAST RUN — F1=0.62 (P=0.55 R=0.71, tp=17 fp=14 fn=7)
## FALSE POSITIVES (predicted, but wrong) — 14 total, showing 15:
  - id=img_003 fields={...}
## FALSE NEGATIVES (missed) — 7 total, showing 7:
  - id=img_044 expected={...}
## HISTORY
  iter 0: F1=0.40 OK
  iter 1: F1=0.62 OK (improved)
Diagnose WHY these are wrong and revise the code. Common causes: wrong
threshold, wrong label/value mapping, over-broad predicate, wrong join key,
CLIP/LLM prompt too long or off-target.
```

### 1.5 Stop logic (`shouldContinueSemdb`)

Mirrors GenDB `shouldContinue`:

- `iteration > maxIter` → **stop** ("max iterations").
- last run crashed / empty CSV → **continue** ("fix correctness first").
- best F1 == 1.0 → **stop** ("perfect").
- last `stallThreshold` (default 2) iterations non-improving → **stop** ("stalled").
- else → **continue**.

Improvement rule (`checkSemdbImprovement(prev, new)`):

- previously ran but now crashed/empty → **not improved** (roll back).
- previously crashed/empty, now runs → **improved**.
- both run → `new.f1 > prev.f1` (strict).

### 1.6 Config + CLI

`semdb.config.mjs` `defaults`:
```js
maxRefineIterations: 5,
refineStallThreshold: 2,
refineSampleCap: 15,
```

CLI (`orchestrator.mjs parseArgs`):
- `--max-iterations <N>` → `args.maxIterations` (default `defaults.maxRefineIterations`).
- `--no-refine` → `args.noRefine = true` (iter_0 only; today's behavior).

The loop only engages when **ground truth is available** (F1 requires it) and
`--no-refine` is not set; otherwise iter_0-only, identical to current behavior.

### 1.7 Telemetry

Per-query `telemetry.json` gains:
```json
"refine": {
  "iterations": 3,
  "best_iteration": 1,
  "max_iterations": 5,
  "stop_reason": "stalled",
  "f1_history": [
    { "iter": 0, "f1": 0.40, "status": "ok", "improved": true },
    { "iter": 1, "f1": 0.62, "status": "ok", "improved": true },
    { "iter": 2, "f1": 0.62, "status": "ok", "improved": false },
    { "iter": 3, "f1": 0.58, "status": "ok", "improved": false }
  ]
}
```
The scored metrics written to `results.csv` are the **best** iteration's.

---

## Part 2 — Text DIRECT Mode

### 2.1 Motivation

Today text queries flow only through the **extract-then-compile** path: semantic
inference happens offline in the Extractor, and `compiled_<q>.py` is pure
relational (residual disabled). The requirement is the text analog of image
DIRECT mode: transform SQL + schema into one `solve_<q>.py` that does the
semantic inference **in the code**, by composing a predefined text API.

### 2.2 Text inference API — `vadar/predefined_text.py` + `semtext.py`

New `vadar/predefined_text.py` (analog of `vadar/predefined.py`) + a `semtext.py`
backend (analog of `semvision.py`). A `TextPatch(text, ctx)` wrapper (analog of
`ImagePatch`) holds one shared OpenAI-compatible endpoint client (+ embedding
model). Free-function primitives, each backed by the endpoint:

| Primitive | Semantics | Maps to |
|---|---|---|
| `judge(text, question) -> bool` | LLM yes/no over the text | `AI.IF` predicate |
| `classify(text, options) -> str` | best VALUE from a closed value space | enum/field, join key |
| `extract(text, field) -> str` | one attribute value from the text | `AI.GENERATE` field |
| `generate(text, instruction) -> str` | free-form generation | `AI.GENERATE` text |
| `score(text, query) -> float` | embedding cosine similarity in [0,1] | ranking / soft filter |

Plus `MODULES_SIGNATURES_TEXT` — the docstring+signature block shown to the
agents (mirrors `MODULES_SIGNATURES`). Closed value spaces (e.g. a `Genre`
column's distinct values) are read from the CSV **at runtime**, never hardcoded.

`semtext.py` responsibilities: build the endpoint client from `--endpoint
--model --api-key`; implement `judge/classify/extract/generate` as chat
completions with tight prompts; `score` via an embeddings call (or a chat
fallback if embeddings unavailable); a simple in-process cache keyed by
(fn, text, arg) so re-running is cheap; a `METER`-style call counter surfaced in
the solver's `*.meta.json` (`llm_calls`, `elapsed_sec`, `rows`).

### 2.3 Modality-aware VADAR 3-agent DIRECT path

`runQueryDirect` becomes modality-aware, branching on `corpus.isImage`:

- **image corpus** → today's vision path (predefined.py, ImagePatch,
  `--clip-model --image-dir`), unchanged.
- **text corpus** → text path: the same VADAR 3 agents (Signature → API →
  Solver) write `solve_<q>.py` composing `predefined_text.py`; run with
  `--data-dir --endpoint --model --api-key`.

The 3 agents are made modality-aware by selecting a per-modality **system
prompt** and passing the right API path + solver skeleton + run-args into the
user-prompt template:

- Add `prompt-text.md` beside each existing `prompt.md` in
  `agents/vadar-signature/`, `agents/vadar-api/`, `agents/vadar-solver/`.
- Each agent's `index.mjs` exposes `promptPath` (image, default) and
  `promptPathText`. `runPhase` accepts a `systemPromptPath` override so the
  DIRECT text branch selects `promptPathText`.
- User-prompt template variables generalized: `api_path` (predefined.py vs
  predefined_text.py), `api_doc`, `patch_wrapper` (ImagePatch vs TextPatch),
  `run_args_doc`, and the solver skeleton for the modality.

`chooseCorpus` already picks the text table for text-only queries (movie), so
planning needs no change. `runQueryCodegen`/`ensureCorpus` (the compiled path)
are untouched by Part 2.

### 2.4 Text solver contract (`solve_<q>.py`, text branch)

The program:
1. reads the structured CSV(s) + the text corpus CSV from `--data-dir`;
2. builds one shared `semtext` ctx from `--endpoint --model --api-key`;
3. wraps each row's text in `TextPatch(text, ctx)`;
4. calls the text API / helpers to evaluate the query's semantic predicate per
   row, returning a REAL field value (label/name/bool), not a raw score;
5. does the relational join / filter / projection / aggregation in plain Python;
6. writes the result CSV whose columns EXACTLY match the query's SELECT list;
7. writes `solve_<q>.meta.json` = `{ elapsed_sec, rows, llm_calls }`.

Invocation (text branch of `runQueryDirect`):
```
python3 solve_<q>.py <results.csv> --data-dir <dir> \
        --endpoint <url> --model <id> --api-key <key>
```

### 2.5 Interaction with the loop

The text `solve_<q>.py` is refined by the Part 1 loop like any other producer:
per iteration it receives the FP/FN **text** rows (id + offending text snippet +
expected vs predicted), the Solver revises its API composition, re-runs,
re-scores. One loop, three producers: image solver / text solver / compiled query.

---

## Files touched

**New**
- `src/semdb/vadar/predefined_text.py` — text inference API + `MODULES_SIGNATURES_TEXT`.
- `src/semdb/semtext.py` — endpoint-backed text-inference backend + `TextPatch`, cache, METER.
- `src/semdb/agents/vadar-signature/prompt-text.md`
- `src/semdb/agents/vadar-api/prompt-text.md`
- `src/semdb/agents/vadar-solver/prompt-text.md`

**Modified**
- `src/semdb/orchestrator.mjs` — `refineLoop`, `shouldContinueSemdb`,
  `checkSemdbImprovement`, feedback rendering; loop wired into `runQueryDirect`
  (image + new text branch) and `runQueryCodegen`; `parseArgs` gains
  `--max-iterations`/`--no-refine`; `runPhase` gains `systemPromptPath` override.
- `src/semdb/evaluate.py` — `--emit-diff <path>` writing FP/FN samples from the
  normalized sets.
- `src/semdb/semdb.config.mjs` — `maxRefineIterations`, `refineStallThreshold`,
  `refineSampleCap`.
- `src/semdb/agents/vadar-signature/index.mjs`,
  `src/semdb/agents/vadar-api/index.mjs`,
  `src/semdb/agents/vadar-solver/index.mjs` — expose `promptPathText`.

## Out of scope
- Corpus-level Extractor refinement (shared across queries) — stays one-shot.
- Re-enabling the residual live-model judge in the compiled path.
- Audio modality.
- Model escalation to a stronger model on repeated failure (GenDB has it; SemDB
  reuses the same agent/model — can be added later).

## Testing / verification
- `--dry-run` on a text query prints the text-solver prompts (no execution).
- End-to-end on a text scenario (movie q3*) with `--ground-truth-dir`: confirm
  iter_0 runs, `--emit-diff` JSON is produced, ≥1 refinement iteration executes,
  best-F1 row lands in `results.csv`, telemetry `refine.f1_history` is populated.
- Regression: an image DIRECT query and a compiled query still run and score;
  `--no-refine` reproduces today's single-shot behavior.
