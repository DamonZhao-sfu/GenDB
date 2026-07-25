# Aligning SemDB's Image-Modality Semantic Operators with CADENZA — Gap Analysis & Modification Plan

> **Reference paper:** J. Ha, Y. Park, W.-S. Han. *CADENZA: Compiling Natural-Language Intent into
> Task-Specific Operator DAGs for Semantic Query Processing.* arXiv:2606.29151 (v2, 2026-07-01).
> **Note on sourcing:** `arxiv.org` is blocked by this environment's egress proxy (HTTP 403 on
> CONNECT, an org-policy denial that must not be routed around), so the paper model below is
> reconstructed from the public abstract/summaries **and** from the CADENZA references already
> embedded in this codebase (`semvqa.py`, `semcaption.py`, `semvision.py`). Re-verify the exact
> operator/template names against the PDF before relying on them for a paper submission.

## 1. What CADENZA actually asks for (the target)

CADENZA compiles each **semantic operator instance** (a template + a natural-language intent, e.g.
`AI.IF("does this image show the logo of {airline}?")`) into a **plan space of typed task DAGs**, and
selects one executable plan under a user-specified **quality–latency–cost** trade-off. Its core pieces:

1. **TxRA (task-extended relational algebra).** A conservative extension of relational algebra with
   **task-specific operators** that invoke inference over unstructured attributes and return *typed
   structured relations*. Crucially, every task op returns a value **and a calibrated confidence
   score** as ordinary relational attributes — e.g. `OCR → text`, `VQA/TxtQA → (Answer, Score)`.
2. **Intermediate task outputs are exposed as relational optimization objects.** This is the central
   thesis: because each task's `(value, score)` is a first-class relation, the optimizer can
   **filter, reorder, route, threshold, and jointly tune** them — capabilities prior SQPEs lack.
3. **Logical planner.** Synthesizes seed TxRA plans, applies structural rewrites, and enumerates
   semantics-guided alternatives from **alternative-generation templates** (one of which,
   *Cross-Modal Proxying* `Apply_t_img ⇝ Apply_t_txt ∘ Apply_c`, this repo already implements in
   `semcaption.py`). Staged because synthesized plans are open-ended and can violate optimal-
   substructure (e.g. noisy OCR is corrected downstream).
4. **Physical planner.** A **data-aware router** over **family-specific implementations** (multiple
   model backends per task), tuned with **Bayesian optimization** under the multi-objective target.

## 2. Current SemDB image-operator inventory

SemDB is already partway to CADENZA — it names its operators in TxRA style (`OpImg*`) and materializes
intermediate outputs as joinable tables (`img_attrs`, captions). Current state:

| TxRA-style op | Backend | File | Returns | Calibrated score? |
|---|---|---|---|---|
| `OpImgClassify` (classify / multilabel / match / verify_property / score) | CLIP ViT-B/32 | `semvision.clip_*`, `imagepatch.classify` | value / bool / float | **No** — `softmax(temp=0.01)` ≈ argmax→~1.0 |
| `OpImgOCR` (read_text / read_text_boxes) | easyocr | `imagepatch.read_text*` | text (+box, +score) | Partly (easyocr per-token conf, uncalibrated) |
| `OpImgObj` / detect (find / detect_boxes) | YOLOv8n **COCO-80 closed vocab** | `semvision.detect*`, `imagepatch.find` | instances (+conf, +box, +counts) | Raw YOLO conf |
| `OpImgEmbed` / `OpImgPairScore` | CLIP image–image cosine | `semvision.img_pair_score`, `embed_corpus` | similarity [0,1] | cosine, uncalibrated |
| `OpImgCap` | generative VLM (corpus pass) | `semcaption.py` | caption (+conf) | self-reported |
| `OpImgVQA` / `OpTxtQA` | generative VLM, guided decoding | `semvqa.py` | **(Answer, Score)** | **Yes** — token logprobs over the answer span |
| `dominant_colors` | pure CV (HSV) | `semvision.cv_dominant_colors` | color list | **No** — constant `conf=1.0` |
| region ops: crop / regions_grid / regions_center | deterministic | `imagepatch.py` | sub-patches | n/a |

**Runtime/optimizer pieces present:** a one-level **residual/escalation** (`semvqa.escalate`) that
re-runs only rows with `value∈{none,∅}` or `score<theta`; a `_Meter` collecting residual scores for
threshold tuning; a fixed per-schema `theta` (default 0.5). The generator (`code-generator`,
`vadar-*` agents) emits **one** program guided by hand-written prompt "Rules" in `imagepatch_prompt.md`.

## 3. Gap analysis (prioritized)

**G1 — Non-uniform, uncalibrated `Score` half of the TxRA contract (highest priority).**
CADENZA's whole optimization surface depends on every op emitting a *comparable* calibrated score.
In SemDB only `OpImgVQA` is calibrated (logprobs). CLIP classify emits a near-1.0 argmax pseudo-prob,
`dominant_colors` emits a constant `1.0`, OCR/YOLO emit backend-native uncalibrated scores. Therefore
the `theta` cut that drives escalation is only meaningful on VQA, **not** on the cheap-proxy score that
actually decides whether to escalate. Filter/route/threshold across ops is unsound until fixed.

**G2 — Missing image task operators vs. CADENZA/VADAR task vocabulary.**
- **`OpImgGround` (open-vocabulary localization).** `find()` is COCO-80 only and returns `[]` + a warning
  for any out-of-vocab class. VADAR's base (which `predefined.py` explicitly contrasts against) uses
  open-vocab grounding (GroundingDINO/SAM). This is the single biggest *coverage* hole.
- **`OpImgCount`** — no typed count op with a score; count is implicit `len(find(...))` and inherits the
  closed vocab.
- **`OpImgRank`/top-k** — semantic *rank* is a declared semantic operator (README) but there is no typed
  image ranker; only raw `pair_score`.
- **`OpImgRelate`/spatial, `OpImgSegment`, `OpImgDepth`** — absent (acknowledged in `predefined.py`).
  Lower priority for SemBench-MMQA; needed for spatial-reasoning workloads.

**G3 — Single plan, not a plan space.** The generator picks *one* decomposition via heuristic prompt
rules. Only one alternative-generation template is realized (Cross-Modal Proxying via `semcaption`).
Missing as *searched, costed* rewrites: operator reorder / predicate pushdown, decomposition
alternatives (classify vs OCR vs detect+classify vs caption→text vs VQA for the same field), and
multi-tier model routing. CADENZA enumerates and selects; SemDB hard-codes one choice.

**G4 — No general physical router / no family-per-task.** `escalate` is an all-or-nothing 2-tier jump
(proxy → one VLM), not a per-row data-aware router across *multiple* families. classify/detect/OCR each
have exactly one fixed backend; model IDs are config constants. Nothing routes among CLIP-B/32 vs
CLIP-L, or SmolVLM vs Qwen3-VL vs a frontier model, per datum.

**G5 — No joint tuning / cost model / multi-objective.** `theta` is a static schema constant; `_Meter`
gathers scores but nothing *tunes* thresholds, routing fractions, or model choice against a
quality–latency–cost objective. There is no cost model and no Bayesian/grid optimizer — CADENZA's
physical-planner contribution is unimplemented.

**G6 — Intermediate outputs exposed as values but not fully as scored relations.** `img_attrs` and
caption columns already realize "expose intermediate outputs as relations" (good, and CADENZA-aligned).
But per-op **scores** and detected **boxes/regions** are not uniformly materialized as joinable
sidecar relations, so the optimizer/tuner can't filter/threshold on them or amortize them across the
query family the way values already are.

> Aside: `src/semdb/README.md` still cites arXiv 2607.13407 (semantic-operator *decomposition*) as "the
> theory," while the image operators are in fact modeled on CADENZA (2606.29151). Update the pointer.

## 4. Modification plan

Phased so each phase is independently landable and testable. Phases A→E map onto G1→G6. Keep the
existing offline guard (no network in generated programs) and the "residual is the only op that spends
a generative call" invariant unless a phase explicitly revises it.

### Phase A — Unify the typed operator contract (fixes G1) — *foundation, do first*
- **A1. Operator registry.** Add `src/semdb/semops.py`: one entry per `OpImg*` declaring `name`,
  `modality`, input/output types, the **families** available (see D3), and a static **cost estimate**
  (latency/$). Each op wraps its `semvision`/`imagepatch`/`semvqa` backend behind a uniform
  `(value, score, detail)` return. This makes every op a first-class TxRA node.
- **A2. Calibration layer.** Add `src/semdb/calibrate.py`: fit per-op temperature / Platt / isotonic
  maps from cheap-proxy raw scores → calibrated `[0,1]`, using the **per-row val fixtures the repo
  already has** (`tests/fixtures/val_q3a.json`, and the per-row validation loop from
  `2026-07-23-per-row-validation-refinement.md`). Wire calibrated scores back through `imagepatch`
  primitives so `classify`/`detect`/`dominant_colors` stop returning argmax-1.0/constant scores.
  **Acceptance:** on a labeled sample, a single `theta` produces a monotone precision/coverage trade
  across *all* ops, not just VQA.

### Phase B — Fill operator gaps (fixes G2)
- **B1. `OpImgGround`** — add an open-vocabulary localization family (GroundingDINO or OWLv2). Keep
  YOLOv8n as the cheap family; grounding as the escalation family so `find()` stops returning `[]` on
  OOV classes. Reuse the `detect_boxes` typed shape `(label, score, box)`.
- **B2. `OpImgCount`** — typed `(int, score)` over detect/ground, escalating to a VQA-count for crowded
  scenes.
- **B3. `OpImgRank`/top-k** — typed ranker over `OpImgEmbed`/`pair_score` returning `(rank, score)`, so
  semantic rank on images is first-class.
- **B4. (later) `OpImgRelate` / `OpImgSegment` / `OpImgDepth`** — only if a spatial-reasoning workload
  lands; note SemBench-MMQA largely does not need them.

### Phase C — Plan space & alternative-generation templates (fixes G3)
- **C1.** Convert the generator's hand-written "Rules" into an explicit template set the agent (or a
  small logical planner) **enumerates**: `{classify | best_ocr_match | detect+classify | caption→text
  (Cross-Modal Proxying) | VQA}` per field, plus region-decomposition and reorder/pushdown. Emit **K**
  candidate programs, not one.
- **C2.** Score candidates on the labeled val sample by a quality–cost objective and keep the winner —
  CADENZA plan-space selection in miniature. **Reuse the existing per-row validation loop as the plan
  scorer** (no new eval harness).

### Phase D — Physical router + joint tuning (fixes G4, G5)
- **D1.** Generalize `semvqa.escalate` into a per-row **router**: from a datum's calibrated proxy score
  + the cost/quality target, choose `skip | cheap | mid-VLM | large-VLM`. Replace the single `theta`
  with a per-family threshold policy.
- **D2.** Add a small **tuner** (`src/semdb/tune.py`): grid or Bayesian search over `{theta per op,
  routing fractions, model choice}` against the val set + a **cost model**, optimizing a scalarized
  quality–latency–cost score. `_Meter` already collects the score distribution needed.
- **D3.** Register **multiple families per task** in `semdb.config.mjs` (CLIP-B/32 ↔ CLIP-L; SmolVLM ↔
  Qwen3-VL ↔ frontier) so the router has alternatives to route among.

### Phase E — Materialize intermediate task outputs as scored relations (fixes G6)
- **E1.** Persist per-op **score** columns (not just values) and detected **boxes/regions** as sidecar
  relations keyed by row id, amortized across the query family exactly like `img_attrs`/captions today.
  This is the concrete realization of "intermediate outputs as relational optimization objects."

## 5. Suggested sequencing & smallest first cut

`A1 → A2` unblock everything (comparable scores). The highest *user-visible* win is **B1
(`OpImgGround`)** because `find()` silently degrades to `[]` today. A credible v1 that demonstrably
"aligns image-modality operators with CADENZA" = **A1 + A2 + B1 + D1**: a uniform typed
`(value, calibrated-score)` contract across ops, open-vocab localization, and a real per-row router —
i.e. the TxRA contract plus the data-aware routing that is CADENZA's headline. C/D2/E then turn the
hand-tuned heuristic into an actual costed plan-space optimizer.
