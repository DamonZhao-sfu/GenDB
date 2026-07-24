# Generated-Code Visual Extraction (ViperGPT + VISPROG + VADAR) — Plan

- **Date:** 2026-07-22
- **Status:** Plan (no code changes yet)
- **References (local repos to mirror):**
  - ViperGPT — arXiv 2303.08128 — `/localhome/hza214/GenDB/viper`
  - VISPROG — arXiv 2211.11559 — `/localhome/hza214/GenDB/visprog`
  - **VADAR — "Visual Agentic AI for Spatial Reasoning with a Dynamic API"** — arXiv
    2502.06787 — `/localhome/hza214/GenDB/VADAR` (Marsili, Agrawal, Yue, Gkioxari)

## Goal
Replace the **fixed per-attribute tier dispatch** in `semvision` with **agent-generated,
compositional Python code over a vision-primitive API**. The three papers give a ladder
of increasing power, and we adopt the strongest that fits our multi-agent pipeline:

| Paper | Idea | What we take |
|---|---|---|
| VISPROG | LLM emits a line-by-line module program; interpreter runs it → visual rationale | step-wise, auditable program = the rationale |
| ViperGPT | LLM writes `execute_command(image)` composing a **fixed** `ImagePatch` API; `exec` it | the API-as-prompt + exec engine |
| **VADAR** | **3 agents SYNTHESIZE a DYNAMIC API** (new helper fns) over a small predefined base, then a program | **dynamic API synthesis mapped onto our 3 agents** |

VADAR is the target: its "signature → api → program" agent trio and minimal predefined
base (`loc`/`vqa`/`depth`) correspond directly to our pipeline — we swap its base for our
**non-VLM tiers** and keep the VLM off by default.

## The agent correspondence (why VADAR fits us)
VADAR (`VADAR/agents/agents.py`): `SignatureAgent` → (API agent) → `ProgramAgent`, over
`engine/predefined_modules.py` base modules, executed by `engine/engine.py`.

| VADAR | Ours (existing) | Role in the new design |
|---|---|---|
| Signature agent (`prompts/signature_prompt.py`) | **Schema Designer** | decides the fields AND proposes helper-predicate *signatures* the corpus needs (dynamic API surface) |
| API agent (`prompts/api_prompt.py`) | **new API-synth agent** (or Extractor's first pass) | *implements* those helpers by composing predefined base primitives |
| Program agent (`prompts/program_prompt.py`) | **Extractor / Code Generator** | writes `extract(patch)->{fields, conf}` using base + synthesized helpers |
| Engine (`engine/engine.py`, `predefined_modules.py`) | **`semvision.run`** | loads models once, wraps each image, `exec`s the program, assembles the attribute table |

## Predefined base API for SemBench multimodal — REUSE existing implementations
Scope = SemBench's IMAGE queries only (mmqa/ecomm/cars/medical/animals). These are 2D
attribute/category/presence/logo predicates — so we **drop VADAR's `depth`/UniDepth and
3D-spatial base entirely** (not needed), and reuse the vision-and-language primitives from
the three repos verbatim where possible. Base `ImagePatch` (shape from `viper/image_patch.py`):

| Primitive (API the generated code calls) | Backed by | REUSE from |
|---|---|---|
| `find(name)` / `exists(name)` — open-vocab detection (species, objects, logos) | **OWL-ViT** or **GroundingDINO** (open-vocab > YOLO's fixed COCO) | `visprog/engine/step_interpreters.py::LocInterpreter` (OWL-ViT) or `VADAR` GroundingDINO `loc` |
| `classify(options)` / `verify_property` / `best_text_match(options)` — CLIP → **real field VALUE** | **CLIP** (score/classify) | `viper/vision_models.py::CLIPModel` (already have transformers CLIP in `semvision`) |
| `dominant_colors()` — colors incl. pale accents | **HSV CV** | ours (`semvision.cv_dominant_colors`) — no paper has this |
| `read_text()` — OCR (printed text; NOT stylized logos) | **easyocr/paddleocr** | ours (available) |
| `domain_label(model, positives)` — medical | **torchxrayvision** DenseNet | ours (`semvision`) |
| `crop(l,low,r,up)` — region | PIL geometry | `viper/image_patch.py::crop` |
| `simple_query(q)` — VQA **escape hatch, DEFAULT OFF** | BLIP / a VLM | `viper` BLIP or `visprog` BLIP or `VADAR` VLM |

Implementation note (per your steer): each primitive's body directly reuses the
corresponding repo wrapper — e.g. `find` = OWL-ViT `LocInterpreter.predict(...)`, `classify`
= viper `CLIPModel.forward(task='classify')` — so we write thin adapters, not new models.
`semvision`'s current proxies remain the fallback implementations.

## SemBench image queries → what the generated `extract()` composes
Every image-only SemBench query reduces to a short composition over the base — this is the
concrete target set (Phase-1/2 validation):

| Query | Generated `extract(patch)` composes | Base primitives |
|---|---|---|
| mmqa q2a/q7 (logo↔named table) | `logo_name = patch.classify(track_or_airline_names, "the logo of {}")` → field, join | `classify` (CLIP) over `labels_from` value space |
| mmqa q2b | q2a + `logo_color = patch.dominant_colors()` | `classify` + CV |
| ecomm q2 | `is_shoe = patch.classify([...])=='sports_shoes'`; `cols = patch.dominant_colors()`; return shoe ∧ {yellow,silver}⊆cols | `classify` + CV |
| ecomm q4 | `patch.dominant_colors()[0]` | CV |
| ecomm q6 | `patch.classify([Dress,Bottomwear,Socks,Topwear,Innerwear])` | CLIP |
| cars q3/q5/q6/q9 | `patch.classify(["damaged car","undamaged car"])` (or `find`/`verify_property`) | CLIP / detector |
| cars q8 | `dmg = [d for d in ["puncture","paint scratch"] if patch.verify_property(d)]` | CLIP verify |
| medical q3/q5/q6/q9/q11 (x-ray) | `patch.domain_label("torchxrayvision:...", ["Pneumonia",...])` | domain classifier |
| animals q1/q3/q5/q6/q7/q10 | `patch.exists("zebra")` / `patch.find("impala")` | open-vocab detector |

None need `depth`, segmentation, or (by default) a VLM — confirming the base is small.

## Dynamic-API synthesis (the VADAR step we add)
Per corpus (seeing all its image queries), the pipeline synthesizes helper predicates
grounded on the base — e.g. for ecomm: `is_sports_shoe(patch)`, `has_colors(patch, cols)`;
for mmqa: `logo_name(patch, names)` (classify over the `Track`/`Airlines` value space);
for cars: `is_damaged(patch)`; for medical: `xray_abnormal(patch)`. The Extractor's
`extract(patch)` then composes these. Signatures + docstrings are generated first
(SignatureAgent), then implemented (API agent), exactly as VADAR does — so novel
predicates beyond the fixed tier map are handled (VADAR's whole point).

## Execution engine (mirror `viper` `execute_code` / VADAR `engine.py`)
Extend `semvision.run`: (1) load backing models once; (2) build the synthesized API module
+ base `ImagePatch` in a **restricted namespace** (only the API in scope — no os/net);
(3) for each image, `exec` `extract(ImagePatch(img))` → `{fields, conf}`; (4) assemble the
SAME attribute-table output (compiled query + evaluator unchanged). Per-row try/except +
systemic guard as in `semextract`. Keep the generated code on disk = the audit rationale
(VISPROG's visual rationale; VADAR writes a trace too — see `engine/*trace*`).

## What we deliberately drop vs the papers
- VADAR's `depth`/3D-spatial base (UniDepth), VISPROG's Stable-Diffusion editing, ViperGPT
  video/`VideoSegment` — out of scope (we do structured attribute extraction, not 3D or
  editing).
- The VLM `vqa`/`simple_query` base is **default-off** (our residual-off policy); the
  grounded base is CLIP/YOLO/CV/OCR/domain. `simple_query` stays as an explicit,
  opt-in escape hatch only.

## Rollout
- **Phase 1 — fixed API + exec engine (ViperGPT/VISPROG baseline):** build `imagepatch.py`
  (wrap current proxies) + `imagepatch.prompt`; **hand-write** `extract()` for ecomm q2 and
  run through the engine; assert it recovers 4/5 GT shoes. Proves API + `exec` engine, no
  agent, no dynamic synthesis.
- **Phase 2 — program agent (ViperGPT full):** wire the Extractor/Code-Gen agent to write
  `extract()` from the API prompt + schema (+ DB value spaces via `labels_from`). Validate
  ecomm q2, mmqa q2a (classify over `Track`), animals (`find("zebra")`).
- **Phase 3 — VADAR dynamic API:** add the Signature + API-synth agents (fold Signature
  into the Schema Designer) so helper predicates are synthesized per corpus over the base;
  validate on the compositional queries the flat/fixed versions miss (ecomm q9/q10 same-
  color/same-brand joins, cars q8 puncture∧scratch).

## Relationship to existing code
- `semvision.py` proxies → become the **implementation** behind `ImagePatch` (no rewrite).
- The current fixed `_run_attr` dispatch → kept as a **fallback** when code-gen fails.
- Extractor prompt for image corpora → gains the `imagepatch` API spec (mirror
  `viper/prompts/*` / `VADAR/prompts/api_prompt.py`); text corpora unchanged.
- Schema Designer → optionally emits helper-predicate signatures (VADAR SignatureAgent).

## Risks / mitigations
- **No extraction oracle:** cache human-readable generated code; smoke-run on a 2-image
  sample at generation time; fall back to fixed dispatch on failure (VADAR uses an oracle
  in `engine/oracle.py` for eval — we substitute the GT-scored sample).
- **`exec` safety:** restricted namespace exposing only the API; models are the sole side
  effect. Timeout per program (VADAR `engine_utils.TimeoutException`).
- **Dynamic-API drift/cost:** synthesize once per corpus (amortized across its queries);
  validate synthesized helpers compile + run before use.

## Non-goals
- Not reimplementing GLIP/SAM2/UniDepth — our base is YOLO/CLIP/CV/OCR/domain.
- Not per-query VQA answers — we produce the reusable structured attribute table that the
  compiled relational query consumes.
