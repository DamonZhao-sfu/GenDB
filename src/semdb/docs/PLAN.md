# SemDB: Semantic Operator Compilation & Execution — Implementation Plan

## Goal
Compile semantic operators (`AI.IF`, `AI.GENERATE`, semantic join/map/filter/
rank/classify) over multimodal data into **vectorized relational programs** that
call a live model only on a residual. Same idea as GenDB — an agent pipeline that
*generates execution code* instead of running a fixed engine — but the "storage"
being designed is an **extracted schema over unstructured data**, and the
expensive primitive is a VLM/LLM call, not a disk scan.

Theory reference: semantic-operator compilation / decomposition (arXiv 2607.13407).
Target workload: **SemBench** (`github.com/SemBench/SemBench`), starting with MMQA.

## Relationship to GenDB (what we reuse vs. add)

| GenDB | SemDB analog | Reuse? |
|-------|--------------|--------|
| Workload Analyzer | (folded into Schema Designer) | pattern |
| Storage/Index Designer | **Schema Designer** — decides decomposability, designs extracted schema | new prompt, same shape |
| ingest / build indexes | **Extractor** — small VLM materializes the attribute table | new |
| Code Generator (C++) | **Code Generator** (Python→C++) — relational program + masked residual | new prompt, same shape |
| Query Optimizer loop | residual-threshold / model-tier tuning loop | phase 4 |
| `providers/` (Claude Agent SDK) | identical | **verbatim** |
| `shared.mjs` runAgent/renderTemplate/telemetry | identical | **verbatim** (imported) |

The three agents live in `src/semdb/agents/*` and follow GenDB's exact
`index.mjs` + `prompt.md` + `user-prompt.md` layout. `orchestrator.mjs` imports
`../gendb/shared.mjs` so the agent runner, templating, and telemetry are shared
code, not copies.

## Architecture (three phases + verify)

```
SemBench query ──▶ [A] Schema Designer ──▶ schema.json
                                              │
corpus (images/text) ──────────────────────▶ [B] Extractor (small VLM, once) ──▶ <corpus>_attrs.json
                                              │            (shared across all queries over the corpus)
                                              ▼
query + schema + attrs ────────────────────▶ [C] Code Generator ──▶ compiled_<q>.py
                                              ▼
                          Verify: compiled result == naive M×N oracle
                                  report model-call reduction
```

Correctness invariant: the compiled plan is **result-equivalent** to the naive
plan because every row the cheap path is unsure about (`none` or `conf < theta`)
is re-judged by the *exact original predicate*. Speed comes from answering the
confident majority relationally.

## Roadmap

### Phase 0 — PoC (DONE, in `poc/`, GPU-free, verifiable)
- Synthetic MMQA-shaped sample (airlines × images).
- `mock_extractor.py` (deterministic small-VLM stand-in) → `img_attrs.json`.
- `compiled_q7.py` (hash join + masked residual) vs `baseline_q7.py` (M×N oracle).
- `run_poc.sh` asserts compiled == oracle and prints 60→12 call reduction.
- **Exit criteria (met):** byte-identical results; documented cost collapse.

### Phase 1 — Real small-VLM extraction (next)
- `vlm_extractor.py` is written and gated; wire it to a real MMQA image subset.
- **Experiment:** SmolVLM-256M vs Qwen3-VL-2B — measure extraction accuracy of
  `logo_brand` vs a hand-labeled key, and end-to-end join precision/recall vs the
  BigQuery ground truth in `files/mmqa/raw_results`.
- **Hypothesis to validate:** a 256M–2B model extracts a *reusable* schema whose
  join F1 matches the naive per-pair VLM at a fraction of the calls. This is the
  crux the user asked to verify first.
- Decide `theta` empirically from the conf/accuracy curve.

### Phase 2 — Live agent pipeline over SemBench
- Point `orchestrator.mjs --sembench-dir <checkout>` at real queries.
- Schema Designer emits schemas for q1, q2b, q3a–g, q4; verify each against
  `files/mmqa/raw_results` (see `SEMBENCH_ANALYSIS.md`).
- Persist schemas per corpus so sibling queries (q3 family, q6 family) reuse one
  extraction — implement the amortization the analysis predicts.

### Phase 3 — Cross-scenario coverage
- E-Commerce (product attrs), Wildlife (species/behavior tags, image+audio),
  Cars (damage report), Movie (sentiment/genre/entities). One shared attribute
  table per corpus; queries compile to predicates/joins/aggregations over it.

### Phase 4 — Optimizer loop (GenDB-style)
- Tune residual `theta`, model tier per attribute, and normalization coverage
  against a cost/quality target. Reuse GenDB's `shouldContinue`/telemetry shape.
- Lower the Python plan to C++ for the hot relational path (the residual stays a
  service call).

## Integration points with the existing repo
- Imports `src/gendb/shared.mjs` (`runAgent`, `renderTemplate`, `readJSON`,
  `setAgentProvider`) and `src/gendb/providers/` unchanged.
- `package.json` deps already include `@anthropic-ai/claude-agent-sdk`; add
  `transformers`/`torch` only for the extraction workers (Python side, optional).
- No change to any GenDB file — SemDB is additive under `src/semdb/`.

## Risks & mitigations
- **Confidently-wrong extraction** (high conf, wrong brand) → a false join the
  residual won't catch. Mitigate: conf calibration, `ocr_text` cross-check, and
  optional verify-sample of high-conf joins with the strong model.
- **Normalization gaps** (aliases/subsidiaries) → missed joins. Mitigate:
  Designer builds the synonym map at compile time; log join keys that never match
  for map expansion.
- **Non-decomposable queries** → Designer returns `decomposable:false`; the
  orchestrator falls back to naive execution (still correct, just not cheaper).
- **Schema drift across a query family** → version the attribute table; extend
  rather than re-extract.

## Open questions
- Where to draw the decomposable/irreducible line for fuzzy predicates
  ("is this a *good* photo")? Start conservative; measure.
- Batch extraction vs. per-item; caching key = content hash of the image/text.
- How much of the residual can be pushed to the strong model vs. dropped by a
  precision/recall SLA per query.

## How to verify today
```bash
bash src/semdb/poc/run_poc.sh          # GPU-free, asserts correctness + prints savings
node src/semdb/orchestrator.mjs --query q7 --dry-run   # shows the 3 rendered agent prompts
```
