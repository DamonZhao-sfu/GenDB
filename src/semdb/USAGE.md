# Using SemDB with your SemBench data

You provide two paths — the query folder and the data folder — plus a query id.

```
query folder :  /localhome/hza214/SemBench/files/mmqa/query/bigquery      (*.sql)
data folder  :  /localhome/hza214/SemBench/files/mmqa/data/sf_200         (*.csv + images/)
```

The pipeline has three agent phases and then execution:

```
[A] Schema Designer  →  schema.json           (which slot to extract, normalization, residual θ)
[B] Extractor        →  <table>_attrs.json     (a SMALL model reads each row/image, ONCE)
[C] Code Generator   →  compiled_<q>.py        (the semantic operator becomes relational code)
    Execute          →  run compiled_<q>.py on your real CSVs; verify vs raw_results
```

## 0. Install

```bash
# repo root — the agent side (Claude Agent SDK)
npm install
# the extraction side (only where you run the small model; needs GPU + HuggingFace)
pip install torch transformers pillow accelerate
```

The agent phases need credentials only when using a hosted provider. The default
`vllm` provider uses the local endpoint at `http://localhost:8000/v1`.
The extraction phase needs `huggingface.co` reachable to fetch the small model.

### Choosing the agent provider (local vLLM, Claude, or Codex)
The compiler agents can run on any provider — set it in `semdb.config.mjs`
(`defaults.agentProvider`) or per run with `--agent-provider`:

```bash
# Qwen/Qwen3.8-27B-FP8 served by local vLLM (the default)
node src/semdb/orchestrator.mjs --query q3a --query-dir <...> --data-dir <...> \
     --agent-provider vllm --base-url http://localhost:8000/v1
```

The local provider uses `@openai/codex-sdk` as its agent runtime but does not require
OpenAI authentication. The model per agent lives in
`semdb.config.mjs → defaults.providers.vllm`; `VLLM_BASE_URL` is equivalent to
`--base-url`. The vLLM server must expose the Responses API. If
`--served-model-name` uses an alias (for example `qwen3.8`), the provider resolves it
through the `/models` entry whose `root` is `Qwen/Qwen3.8-27B-FP8`.
`--model <id>` forces one model for all agents (testing). The *extraction* small
models (`defaults.extraction.*`) are independent of this choice.

For local vLLM, Planner and Optimizer default to the full agent runtime; the Generator
is also a coding agent. The tool-free structured path remains available for A/B:

```bash
# Default: full Codex agent for every role
--agent-execution agent

# Opt-in fast path: JSON Schema, no shell tools for Planner/Optimizer
--agent-execution structured
```

When using `run_image_queries.sh`, the wrapper likewise defaults to `agent`. Set
`AGENT_EXECUTION=structured` to opt into the tool-free path. This switch affects
Planner and Optimizer; Generator uses the full coding-agent runtime in both modes.

The full-agent path keeps unrestricted diagnostic tools but avoids the former prompt/read
loop. Planner and Optimizer use their role `SKILL.md` directly as the sole system
procedure, then read one immutable-on-entry `_agent_context_<role>.md` bundle containing
the current query, artifacts, full-file profile, primitive catalog, memory evidence, and
JSON schema. Their role skill is not advertised for discovery a second time. The skill
catalog contains only retrieval-selected learned skills; when memory has no relevant
skill, no catalog is injected. Agents may still inspect other files or run diagnostics
when the bundle exposes a concrete missing or inconsistent fact.

Structured defaults are Planner `max_output_tokens=24000` (32K only on a confirmed
truncation retry), Optimizer `12000` (16K on truncation retry), and
`reasoning_effort=medium`. Qwen3.8 on the supported vLLM deployment accepts `low`, `medium`,
or `xhigh`. Override them for controlled experiments with
`--planner-max-output-tokens`, `--optimizer-max-output-tokens`, and
`--structured-reasoning-effort`. A structured role retries one invalid response once,
then automatically falls back to the full agent.

The structured path remains tool-free but includes the evidence each role needs:

- Planner receives a compact, validation-label-free profile streamed across every row of
  each referenced runtime CSV, including same-name join-column overlap. Distinct tracking
  is memory-bounded, and zero overlap is authoritative only when both the scan and value
  sets are marked complete. Schema/runtime validation and semantic plan lint share the
  existing single correction retry, so a lint failure is repaired before Generator time
  is spent.
- Optimizer receives the exact current helper and solver sources with stable line numbers,
  the validation diff, and an aggregate trace summary. `PATCH_CODE` must cite a real
  `helpers:L<n>` or `solver:L<n>` line; branch names never override the serialized trace.
- Compile/preflight failures and a missing required trace are deterministic implementation
  failures and route directly to Generator without an Optimizer model request.
- A binary-F1 validation sample with null precision/recall and zero FP/FN contains no
  positive optimization signal; the loop preserves the current candidate instead of asking
  Optimizer to invent a patch.

No validation labels, CERT data, or final ground truth enter the Planner profile or trace
summary. Candidate sources are read by Node and placed in the single request; neither role
gets shell or filesystem tools in structured mode.

The provider also canonicalizes presentation-only JSON fences and removes
`replan_reason` when an Optimizer action is `PATCH_CODE` or `STOP`, where the schema
forbids that field. It never invents a missing reason, target, id, or semantic value;
those errors still use the correction/fallback path.

---

## Mode 1 — autonomous (agents + extraction + compiled query + scoring)

```bash
# one query, end-to-end with ground-truth scoring:
node src/semdb/orchestrator.mjs \
  --query q7 \
  --query-dir /localhome/hza214/SemBench/files/mmqa/query/bigquery \
  --data-dir  /localhome/hza214/SemBench/files/mmqa/data/sf_200 \
  --ground-truth-dir /localhome/hza214/SemBench/files/mmqa/raw_results/ground_truth \
  --endpoint http://localhost:8000/v1        # vLLM: extraction + residual judge
```

Passing `--ground-truth-dir` auto-runs the whole pipeline (extraction → compiled
query → scoring) and writes **precision/recall/F1** into that run's
`telemetry.json` and appends a row to `runs/results.csv`. `--endpoint` routes both
the extraction and the residual `vlm_judge` calls to your vLLM server.

**Omit `--query` to process every `*.sql` in the query dir** into one shared CSV,
with a mean-F1 summary at the end:

```bash
node src/semdb/orchestrator.mjs \
  --query-dir /localhome/hza214/SemBench/files/mmqa/query/bigquery \
  --data-dir  /localhome/hza214/SemBench/files/mmqa/data/sf_200 \
  --ground-truth-dir /localhome/hza214/SemBench/files/mmqa/raw_results/ground_truth \
  --endpoint http://localhost:8000/v1
```

Naive baseline is computed per operator: **sem_join = |left| × |right|**,
**sem_filter / sem_map = |table|** (detected from which tables the AI predicate
references).

The orchestrator reads the real `q3a.sql` and the real table headers, then runs
the three agents into `src/semdb/runs/mmqa-q3a/` (`schema.json`,
`<table>_attrs.json`, `compiled_q3a.py`). It prints the exact `extract.py` and
run commands to finish.

Preview everything without credentials (renders the three prompts):

```bash
node src/semdb/orchestrator.mjs --query q3a --query-dir <...> --data-dir <...> --dry-run
```

---

## Production VADAR run — no ground truth and no runtime model endpoint

Use `--direct --run --no-ground-truth`. Do not pass `--endpoint` or `--api-key`:

```bash
node src/semdb/orchestrator.mjs \
  --direct --run --no-ground-truth \
  --query q3a \
  --query-dir /path/to/query/bigquery \
  --data-dir /path/to/data/sf_200
```

`--no-ground-truth` prevents the orchestrator from auto-deriving or reading a SemBench
ground-truth directory. A successful unscored run is single-shot; a crash can still be
regenerated from local stderr feedback. VADAR-generated runtime code receives no model
endpoint or API key. The orchestrator rejects generated code that contains endpoint-backed
text wrappers, model/HTTP clients, or a semantic judgement call before executing it.

Text solvers use deterministic string/regex/lexical helpers. Image solvers may use the
local CLIP/OCR/CV/detector stack via `--clip-model`; this is local inference, not an
LLM/VLM endpoint.

---

## Oracle-labeled validation sets in one command (`--val-rate`)

The orchestrator can build the validation set itself: sample the corpus at a rate you
pick, label those rows with a local model over `--endpoint`, and refine against them.

```bash
node src/semdb/orchestrator.mjs --benchmark ecomm --direct --image-only \
  --query-dir /localhome/hza214/SemBench/files/ecomm/queries/dialects/bigquery \
  --data-dir  /localhome/hza214/SemBench/files/ecomm/data/sf_250 \
  --ground-truth-dir /localhome/hza214/SemBench/files/ecomm/raw_results/ground_truth \
  --endpoint http://localhost:8000/v1 \
  --val-rate 0.25 --val-cert-rate 0.1 --max-iterations 3
```

| flag | default | meaning |
|---|---|---|
| `--val-rate` | off | SELECT size as a fraction of the corpus. Requires `--endpoint`. |
| `--val-cert-rate` | 0 | Sealed CERT half; the loop never reads it. |
| `--val-method` | `stratified` | `uniform` / `stratified` / `importance` |
| `--val-strata-k` | 5 | Score-decile bucket count |
| `--val-score-tilt` | 2 | How hard to oversample high-score strata (0 = proportional) |
| `--val-call-site` | auto | Which `AI.IF`/`AI.GENERATE` call to label, for a multi-predicate query |
| `--val-pair-top` | deprecated | Ignored: join validation samples the complete pair population. |
| `--oracle-model` | strong image/text model | The labeling model |

### Why this matters: the loop must not read the ground truth

With a validation set the loop scores each iteration with `scoreInference` against the
oracle-labeled sample, runs the solver over only those rows (`--only-ids`), and does one
full-corpus pass **after** the loop stops. Without one it scores against the full ground
truth, and `renderFeedback` then shows the agent `FALSE POSITIVES` / `FALSE NEGATIVES`
rows taken straight from it — the agent reads the test set and the reported F1 is the
number the loop selected on.

Because that failure is silent and easy to miss, **passing `--val-rate` and failing to
build a val set is a hard error**, not a downgrade: the orchestrator aborts rather than
quietly iterate against the ground truth. Drop `--val-rate` if you want ground-truth
refinement deliberately.

### Join queries are handled automatically

When `predicate.py` reports the query's AI call site is **pairwise** and the two sides
are different tables (mmqa q2a/q2b/q7 and EComm q8), `--val-rate` builds a
cross-table pair frame first, then samples it — no manual `build_pairs.py` step. The
val set keys on
`"<structured_id>-<image_filename>"`, and the solver is told to key `trace_<q>.json` the
same way. The frame is the complete Cartesian product, and its validation size is
`ceil(filtered_left_rows × right_rows × val_rate)`. For q7 at sf=200 and rate 0.1 this means
`ceil(200 × 200 × 0.1) = 4,000` labeled pairs:

```bash
node src/semdb/orchestrator.mjs --benchmark mmqa --direct --query q2a \
  --query-dir /localhome/hza214/SemBench/files/mmqa/query/bigquery \
  --data-dir  /localhome/hza214/SemBench/files/mmqa/data/sf_200 \
  --ground-truth-dir /localhome/hza214/SemBench/files/mmqa/raw_results/ground_truth \
  --endpoint http://localhost:8000/v1 --oracle-model <vlm> \
  --val-rate 0.1 --max-iterations 5
```
| `--val-seed` | 7 | Draw seed |

EComm q8 applies its ordinary
`CHAR_LENGTH(productDescriptors.description.value) >= 3000` CTE before pairing.
At sf250, 4 of 250 products survive, so its candidate domain is `4 × 250 = 1,000`
description-image pairs and `--val-rate 0.05` samples 50 pairs. Inspect that frame
without invoking the Oracle or agents:

```bash
node src/semdb/orchestrator.mjs \
  --benchmark ecomm --sembench-dir /localhome/hza214/SemBench --sf 250 \
  --query q8 --direct --val-rate 0.05 --val-plan-only \
  --out src/semdb/runs/ecomm
```

The question is read from the query's own `AI.IF` / `AI.GENERATE` / `AI.CLASSIFY` call,
so nothing is retyped. Val sets are cached under `runs/_val/<bench>-<query>/<design>/`
and labels under corpus-scoped
`runs/_val/<bench>-<query>/labels-<fingerprint>.json`, so raising the rate re-pays only
for rows never labeled before without reusing ordinal keys across scale factors.

Per-row validation uses a validation-only `_semdb_row_id` equal to the zero-based
source CSV record ordinal for both text and image manifests. This keeps repeated
physical rows distinct even when their logical key (for example Movie `reviewId` or a
car/patient id with multiple media rows) is duplicated. Solvers derive the ordinal
while reading the original CSV; SQL result projection and duplicate multiplicity are
unchanged. Movie Q5–Q7 additionally execute the ordinary movie-id filter and
`r1.reviewId <> r2.reviewId` before sampling ordered semantic pairs.

EComm materialization also exposes `ecomm_products.csv` as the documented offline
adapter for the SQL image-mapping and `EXTERNAL_OBJECT_TRANSFORM` chain. Planner can
therefore compile q4/q6 against local image files. Offline text primitives expose
bounded, confidence-bearing classification and candidate extraction, allowing q3/q5
to compile as explicit `bounded_approximation` plans instead of falsely reporting that
no local implementation exists. Without `--endpoint`, an explicitly requested
`--val-rate` run fails closed rather than exposing full-ground-truth feedback.

Audit only the typed PGO semantic plan (without Generator, Optimizer, or execution):

```bash
node src/semdb/orchestrator.mjs --benchmark ecomm --direct \
  --semantic-plan-only --query q5 ...
```

Audit the validation capability of every configured SemBench SQL file:

```bash
node src/semdb/validation_matrix.mjs \
  --sembench-dir /localhome/hza214/SemBench \
  --out /tmp/semdb-validation-matrix.json
```

Whole-query validation for a typed tuple or multi-site composition is never silently
reduced to one predicate. The supported `filter_then_extract` shape (MMQA q2b) uses
one joint pair label—`no_match` or `match:<value>`—and query-specific tuple F1, so
both semantic sites participate in optimization. Other tuple/multi-site compositions
still write `validation_capability.json` and report `NOT_COMPILABLE` with reason
`joint_multi_site_oracle_not_implemented` until they have an equivalent typed
composition. This is a validation-frame capability boundary; it does not claim that
the semantic SQL itself lacks an offline solver.

**Why `--val-score-tilt` defaults to 2.** These predicates are highly selective — ecomm
Q2 has 5 positives in 250 rows. A uniform 20% draw expects ~1 positive, and a val set
with one positive scores `return false` at 98%: it cannot rank programs. CLIP ranks Q2's
five positives at 1, 2, 3, 4 and 6 of 250, and the tilt spends labels there — capturing
4 of 5 instead of 1 of 5. The cost is a ~16× spread in the Horvitz–Thompson weights,
which `evaluate.py` divides back out, so the reported precision/recall/F1 are corpus
estimates rather than sample averages.

Inspect a query's call sites before committing labels:

```bash
python3 src/semdb/predicate.py /path/to/q10.sql
```

Join queries such as q8/q9/q10/q11/q14 contain **pairwise** call sites, whose label is
over a pair of rows. Those need a pair frame (below); their **per-row** call sites —
q10 and q11 have 3 and 4 of them — get ordinary val sets.

### Pairwise (join) validation sets

EComm q7/q9 self-join frames are now automatic. The orchestrator first materializes a
flat `ecomm_products.csv`, executes the query's deterministic CTE/filter in an
in-memory DuckDB connection, and only then forms the ordered pair population:

```bash
# No Oracle or agents: inspect population/sample cardinalities and build cached frames.
node src/semdb/orchestrator.mjs \
  --benchmark ecomm --sembench-dir /localhome/hza214/SemBench --sf 250 \
  --query q7,q9 --direct --val-rate 0.05 --val-plan-only \
  --out src/semdb/runs/ecomm
```

At sf250, q7 has 41 filtered rows and includes diagonal pairs, so its population is
`41² = 1,681`; q9 has 18 filtered rows and explicitly requires `p1 != p2`, so its
ordered population is `18 × 17 = 306`. At 5%, the validation samples contain 85 and
16 pair rows respectively.

For a manual frame, build it **after** the query's deterministic predicates and treat
it as an ordinary corpus whose ids are `"<id1>-<id2>"`:

```bash
# ids surviving q9's CTE (baseColour IN ... AND colour1='' AND price<800) -> keep.txt
python3 src/semdb/build_pairs.py --corpus IMAGES.csv --id-col id --image-col filename \
  --image-dir .../images --only-ids keep.txt --ordered --out pairs_q9.csv

python3 src/semdb/build_valset.py --corpus pairs_q9.csv --id-col pair_id --pairwise \
  --query q9 --attr same_outfit --sql .../q9.sql \
  --method stratified --strata-by score-decile --importance-by column:pair_score \
  --rate 0.6 --cert-rate 0.2 --label-source oracle \
  --endpoint http://localhost:8000/v1 --oracle-model <vlm> --out runs/_val/ecomm-q9
```

`--only-ids` is not a tuning knob. Pairing all 250 ecomm images gives 62,250 ordered
non-self pairs. q9's own filter leaves 18 rows → 306 ordered pairs containing the
same six symmetric matches in both directions (12 positive output ids). q7 does not
exclude equal aliases and therefore additionally needs `--include-diagonal`.

`--val-plan-only` also reports q10/q11 multi-site root cardinalities without paying for
labels. At sf250 these are `60³ = 216,000` and `250⁴ = 3,906,250,000`, making the
cost boundary visible before the multi-site Oracle bundle is built.

### Cross-table pair frames (a structured table joined to images)

mmqa q2a/q2b/q7 join a structured table to the image table with the AI predicate itself
as the join condition, so the pair is (structured row, image) rather than (image,
image). `--right` switches `build_pairs.py` to that mode and scores the frame by CLIP
**text↔image** similarity:

```bash
python3 src/semdb/build_pairs.py \
  --corpus .../ap_warrior.csv --id-col ID --text-col Track \
  --right .../images.csv --right-id-col image_filename --right-image-col image_filepath \
  --out pairs_q2a.csv

python3 src/semdb/build_valset.py --corpus pairs_q2a.csv --id-col pair_id \
  --pairwise --pair-image-cols file2 --text-col text1 \
  --query q2a --attr answer --sql .../q2a.sql \
  --method stratified --strata-by score-decile --importance-by column:pair_score \
  --rate 0.1 --label-source oracle --endpoint http://localhost:8000/v1 \
  --oracle-model <vlm> --out runs/_val/mmqa-q2a
```

The left id must be the column the query SELECTs (q2a → `ID`, q7 → `Airlines`), because
that is what SemBench's join ground truth lists.

**Text↔image ranks far better than image↔image.** On mmqa q7 the five ground-truth
pairs rank 18, 31, 95, 107 and 187 of 40,000 — all inside the top 0.47%, a ~200×
lift. Logo recognition is what CLIP is strongest at, unlike ecomm q9's "same category
and colour" where the image↔image lift is only ~2×.

Similarity is used only for stratification and score-ranked certainty anchors. It no
longer prunes the validation population: every pair retains non-zero inclusion
probability, and the requested rate is applied to the complete join.

`--max-iterations N` is an exact refinement budget whenever a validation signal
exists: the system runs `iter_0` followed by `iter_1` through `iter_N`. Perfect or
stalled validation F1 selects the best candidate but does not stop the run early.

## Refining against a pre-built validation set (`--val-file`)

Instead of scoring each iteration against the full ground truth, draw a probability
sample of the corpus, label it, and refine against **that**. Two steps.

### Step 1 — build the validation set

```bash
python3 src/semdb/build_valset.py \
  --corpus /path/to/data/sf_200/lizzy_caplan_text_data.csv \
  --id-col row_id --text-col text \
  --query q3a --attr is_comedy --query-nl "Which movies are comedies?" \
  --method importance --importance-by query-similarity --epsilon 0.2 \
  --n 60 --cert-n 30 --seed 7 \
  --label-source gt \
  --gt-file /path/to/raw_results/ground_truth/Q3a.json --gt-match-col title \
  --out src/semdb/runs/_val/mmqa-q3a
```

Writes `select.json` (the loop reads this), `cert.json` (sealed — the orchestrator
never reads it), and `split_manifest.json`. The two are disjoint by construction.

| flag | meaning |
|---|---|
| `--method` | `uniform` (SRSWOR) · `stratified` · `importance` (Pareto πps) |
| `--strata-by` | `length` · `kmeans` · `column:<name>` — must be a signal available *before* any program exists |
| `--importance-by` | `query-similarity` (TF-IDF vs `--query-nl`) · `column:<name>` |
| `--epsilon` | uniform mixing in the importance proposal; bounds weights at `N/(n·ε)` |
| `--n` / `--cert-n` | SELECT and CERT sizes, drawn as one sample then partitioned |
| `--label-source` | `gt` (plumbing only — prints a warning) · `none` (ids only) · `oracle` (not yet) |

On a 200-row corpus with 13 positives (6.5%), a 60-row budget catches 4 positives
under `uniform` and 7 under `importance` — that gap is the point of the design.

> `--label-source gt` has the refinement loop reading the benchmark answer key. It
> exists to exercise the plumbing without an oracle model and supports no claim
> about oracle-free operation. It warns on every run.

### Step 2 — refine against it

```bash
node src/semdb/orchestrator.mjs \
  --direct --run --benchmark mmqa --query q3a \
  --query-dir /path/to/query/bigquery \
  --data-dir  /path/to/data/sf_200 \
  --val-file  src/semdb/runs/_val/mmqa-q3a/select.json \
  --max-iterations 3
```

Each iteration runs the solver over the labeled rows only (via the solver's
`--only-ids` contract), scores per-row accuracy against `select.json`, and feeds the
mislabeled rows back to the solver agent. After the loop, the promoted best solver is
run once more over the **full** corpus to produce the real result CSV.

## The compile gate

Before any generated solver is executed, `preflight.py` runs `compile()` for syntax
and pyflakes for undefined names. A failure costs a static pass instead of a full
corpus run, and the agent gets an exact line plus a marked source window rather than
a truncated stderr tail. The report lands in `iter_N/preflight.json`.

```bash
python3 src/semdb/preflight.py src/semdb/runs/mmqa-q3a/iter_0/solve_q3a.py --out /tmp/pf.json
```

pyflakes is optional (`pip install pyflakes`); without it the name check reports
`unavailable` and the gate passes — a missing linter never blocks generation.

---

## Mode 2 — step by step (explicit control of the small model)

### A. Design the schema (once per query family)
Run Phase A of Mode 1, or hand-write `schema.json` (see
`examples/mmqa_q3/schema.json` and `poc/expected/schema.json` for the shape).

### B. Extract with a small model (once per corpus, reused by every query)

**Text queries** — q1, q3a–g, q4, q5, q6 (a small text LLM reads a text column):

```bash
python3 src/semdb/extract.py \
  --schema   src/semdb/runs/mmqa-q3a/schema.json \
  --table    /localhome/hza214/SemBench/files/mmqa/data/sf_200/lizzy_caplan_text_data.csv \
  --modality text --id-col title --text-col text \
  --model    Qwen/Qwen2.5-0.5B-Instruct \
  --out      movie_attrs.json
```

**Image queries** — q2a, q2b, q7 (a small VLM reads each image):

```bash
python3 src/semdb/extract.py \
  --schema    src/semdb/runs/mmqa-q7/schema.json \
  --table     /localhome/hza214/SemBench/files/mmqa/data/sf_200/thalamusdb_images.csv \
  --modality  image --id-col uri --image-col uri \
  --image-dir /localhome/hza214/SemBench/files/mmqa/data/sf_200/images \
  --model     HuggingFaceTB/SmolVLM-256M-Instruct \
  --out       img_attrs.json
```

`--image-dir` lets the driver resolve `uri` → `images/<basename>`; if the column
already holds a local path, omit it. Use `--limit N` to smoke-test on a few rows.

**Recommended: serve the model with vLLM + guided JSON decoding.** This constrains
generation to the schema, so output is *always* valid JSON (zero parse failures)
and requests are batched/fast:

```bash
# 1) serve the small VLM (separate terminal)
vllm serve Qwen/Qwen3-VL-2B-Instruct --port 8000

# 2) point the extractor at it — same flags, plus --endpoint
python3 src/semdb/extract.py \
  --schema    src/semdb/runs/mmqa-q2a/schema.json \
  --table     .../sf_200/thalamusdb_images.csv \
  --modality  image --id-col image_filepath --image-col image_filepath \
  --image-dir .../sf_200/images \
  --model     Qwen/Qwen3-VL-2B-Instruct \
  --endpoint  http://localhost:8000/v1 \
  --out       img_attrs.json
```

`--endpoint` builds a JSON Schema from `schema.json` and sends it as `guided_json`
(also `response_format: json_schema`), so the server can't emit malformed output.
`--api-key` defaults to `EMPTY` (vLLM). Local (no-endpoint) runs still work; for a
weak local model without guided decoding, add `--prompt-style simple`.

### C. Run the compiled query on your real data

The Code Generator writes `compiled_<q>.py` referencing the attribute table.
Run it, then compare against SemBench's ground truth in
`files/mmqa/raw_results/`. See the two worked, committed examples:

- `examples/mmqa_q3/` — text, genre family, run for real (`bash run_q3.sh`).
- `poc/` — the image logo-join compiled plan vs the naive M×N oracle (`bash run_poc.sh`).

---

### D. Score against ground truth + write the results CSV
SemBench ground truth lives at `.../raw_results/ground_truth/<Query>.json`
(`{"ground_truth": [[0,"117d...png"], ...]}` — the expected result rows). Pass the
dir to the orchestrator so it records it, then score the compiled output:

```bash
python3 src/semdb/evaluate.py \
  --pred runs/mmqa-q2a/q2a_results.csv --pred-cols 0,1 \
  --ground-truth-dir /localhome/hza214/SemBench/files/mmqa/raw_results/ground_truth \
  --query q2a \
  --telemetry runs/mmqa-q2a/telemetry.json \
  --csv runs/results.csv
```

This prints precision / recall / F1 and appends ONE row to `results.csv` with the
full telemetry (time, cost, LLM-call breakdown) **and** the metrics. Cells are
normalized (image path → basename, lowercased) so a full path matches a bare
filename. `--pred-cols` selects/reorders the predicted columns to match the GT
tuple order (`[t.ID, i.uri]` → `0,1`). Omit `--pred` for a telemetry-only row.
Run it once per query to build a full benchmark CSV.

Add `--ground-truth-dir <dir>` (and optionally `--telemetry-csv <path>`) to the
orchestrator and it records the GT file in `telemetry.json` and prints the exact
`evaluate.py` command to run.

## Setting theta (the residual confidence floor)

`theta` decides which rows the compiled query re-checks with a live model call:
a row is **residual** if its extracted join key is `none`/empty **or** its
`conf < theta`. Higher theta → more residual calls → higher recall but more cost;
lower theta → fewer calls, trusts the extractor more.

Three places, in priority order:
1. **`--theta` on `extract.py`** — overrides everything for that run.
2. **`schema.json → residual.theta`** — what the Schema Designer emits; the
   compiled query reads this. Edit it to change the compiled query's behavior.
3. **`semdb.config.mjs → defaults.extraction.theta`** (0.5) — the fallback default.

Tune it against F1: sweep `--theta 0.3 0.5 0.7`, run `evaluate.py` each time, and
pick the knee where F1 stops improving — that's the fewest residual calls for the
recall you want.

## Original vs compiled call count

`telemetry.json` now reports the naive baseline so you can see the reduction:
- `naive_llm_calls` — the original plan: **M × N** for a join (e.g. q2a: racetracks
  × images) or **N** for a filter/map (one `AI.IF`/`AI.GENERATE` per row).
- `compiled_execution_calls` — `extraction (N, shared) + residual (k)`.
- `call_reduction` — `naive / compiled_execution` (e.g. `5.9×`).
- `llm_calls.total` still includes the one-time agent (compile) stage.

These are also columns in the results CSV.

## Which file backs each SemBench table

| SQL identifier | CSV in `data/sf_200/` | queries |
|----------------|-----------------------|---------|
| `mmqa.tampa_international_airport` | `tampa_international_airport.csv` (Airlines, Destinations, …) | q5, q6, q7 |
| `mmqa.ap_warrior` | `ap_warrior.csv` (ID, Track, …) | q2a, q2b |
| `mmqa.images` | `thalamusdb_images.csv` (uri, ref) + `images/` dir | q2a, q2b, q7 |
| `mmqa.lizzy_caplan_text_data` | `lizzy_caplan_text_data.csv` (title, text) | q3a–g, q4 |
| `mmqa.ben_piazza_text_data` / `ben_piazza` | `ben_piazza_text_data.csv`, `ben_piazza.csv` | q1 |

> `mmqa.images` in the SQL maps to `thalamusdb_images.csv` on disk — pass that
> path to `--table`.

## What runs where

| Step | Needs |
|------|-------|
| `--dry-run` (see prompts) | nothing |
| Phase A/B/C agents (Mode 1) | Claude credentials + `npm install` |
| `extract.py` real small model | GPU-friendly host + `pip install` + HuggingFace access |
| compiled `.py` execution + validate | plain Python 3 |
| offline demos (`poc/`, `examples/mmqa_q3/`) | plain Python 3 (no GPU, no keys) |

If HuggingFace is blocked on your host (as in this sandbox), the compiled-query
logic and validation still run; only Phase B needs the model. `examples/mmqa_q3/`
shows the full loop already executed with a small model.
