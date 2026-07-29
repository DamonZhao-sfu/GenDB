---
name: generate-semantic-program
description: Generate or revise a GenDB SemDB offline semantic program from a validated typed plan. Use when creating helper functions and an end-to-end solver, applying a structured optimizer patch, regenerating after a replan, preserving trace and result contracts, or repairing compile/runtime defects without changing the query's planned semantics.
---

# Generate Semantic Program

Implement the validated plan as a complete candidate. Do not redesign the query.

You are the legacy VADAR Program/Solver agent working from a typed plan: ONE end-to-end
program that answers the WHOLE query by composing the predefined LOCAL API (CLIP / OCR /
CV / detector) plus the plan's helpers. No VLM, no LLM, no endpoint. The generated code must
not contain a model-service SDK, HTTP/network client, API key, `semtext`, `TextPatch`, or a
semantic judgement API — the orchestrator enforces this before it runs.

## Read required inputs

Read the validated `plan.json`, exact local primitive source files, table metadata, required output paths, and any parent candidate manifest. When revising code, also read the structured optimizer action and the parent helper and solver files.

If the plan is `not_compilable`, stop without generating a misleading program.

## Generate a complete candidate

1. Map every helper DAG node to one Python function with matching argument and return types.
2. Generate the helper module at the requested path.
3. Generate the end-to-end solver at the requested path.
4. Implement deterministic relational operations in the plan's stated order.
5. Implement the exact SQL result projection and CSV shape.
6. Implement `--only-ids` at the plan-specified sampling-unit boundary.
7. Write `trace_<q>.json` with one entry for every evaluated row, pair, or tuple.
8. Add concise stderr branch diagnostics without printing secrets or complete validation data.
9. Handle individual row/pair failures without corrupting unrelated results.
10. Leave candidate metadata for the orchestrator to hash and finalize.

## Implement the vis-operator runtime patterns

For image plans:

- Create one shared `semvision` encoder context and wrap resolved paths with
  `imagepatch.ImagePatch`. Resolve filenames/URIs with
  `semextract.resolve_image_path`; do not assume the process working directory.
- Implement small enums with `classify`, multi-valued fields with `classify_multi`,
  wordmarks with `best_ocr_match`, colors with `dominant_colors`, closed-vocabulary
  objects with `detect`, and open-vocabulary objects with `detect_open`. Treat
  `detect` hits as sub-images and preserve `bbox` coordinates when requested.
- Implement whole-image-to-region fallback exactly when the plan requests it:
  `regions_center`, `regions_grid`, `regions_propose`, or `crop`. Do not substitute
  blind grids for content proposals.
- Use `pair_score` for image pairs, `topk_similar` for one-to-many image retrieval,
  `topk_text` before verifying a large text label space, `score` for
  image-to-short-text relevance, and `embed` for reusable vectors.
- Use `classify_detail`, `verify_detail`, `detect_detail`, `ocr_detail`, and
  `best_ocr_match_detail` confidence only within the primitive that produced it.
  Never compare confidence values from different primitive families.
- Preserve a planned cheap gate before expensive OCR, open-vocabulary detection, or
  region decomposition.
- Keep CLIP phrases short — a few words. CLIP reads only ~77 tokens and does not process
  negation, so never build a prompt by concatenating the query sentence or a "reject
  X, Y, Z" clause; that lowers precision. Realize a long predicate as the plan's explicit
  helper DAG.
- Implement each planned primitive as planned. Do not collapse a planned OCR or
  closed-set `classify` step into a single `verify_property`, and do not drop a planned
  gate, threshold, confidence, region decomposition, or assignment/dedup step because a
  simpler form runs.
- Return a REAL field value from each helper — a name, a label, colours — not a score, and
  do the relational comparison in plain Python outside the helper.
- Implement the plan's acceptance paths as a DISJUNCTION with distinct branch names, not as
  one AND-chain. Never invent an absolute cutoff the plan did not specify: an unplanned
  `>= 0.5` on an uncalibrated CLIP score typically rejects every row.
- A candidate that selects ZERO rows is a failure, not a strict predicate. It scores F1 0
  and leaves the next iteration nothing to learn from, so loosen the most arbitrary
  rejection rule and regenerate before writing the manifest.
- Read every closed value space (e.g. the set of `Track` names) from the structured column
  AT RUNTIME — never hardcode it.
- `detect` returns SUB-IMAGES (len = count) and covers only COCO-80 names; anything else
  returns [] and warns, so use `detect_open`, `classify`, or `verify_property` there.

For text plans, use ordinary strings plus the exact functions exported by
`vadar.predefined_text` and Python standard-library regex/numeric/date operations.
Do not import `semtext`, `TextPatch`, a model SDK, or a semantic service.

Every solver must:

- accept `<out.csv> --data-dir <dir>`, optional `--only-ids`, and the documented image
  arguments when applicable;
- form the correct row/pair/tuple validation unit before applying `--only-ids`;
- emit a trace entry for both positive and negative decisions;
- preserve duplicate rows and ordered-pair/diagonal semantics from the plan;
- keep branch counters to at most eight short names, cap repeated per-row warnings,
  print aggregate branch/warning counts and elapsed time to stderr, and isolate one
  bad row/pair without aborting unrelated work.

## Apply optimizer actions

For `PATCH_CODE`, edit only the named artifact or symbol and preserve every listed invariant. Use the parent candidate as the starting point.

For `REPLAN`, regenerate the complete helper and solver from the new plan. Do not mix helper code from an older plan version.

Do not implement requests unsupported by the plan. If an action conflicts with the plan, fail with a clear `NEEDS_REPLAN` diagnostic.

## Enforce runtime constraints

- Do not call a network service, remote LLM, endpoint, shell download, or external API.
- Do not read CERT or final ground truth.
- Do not hardcode validation ids, labels, expected outputs, or mistake rows.
- Do not invent repository APIs.
- Do not silently weaken the semantic predicate.
- Do not change pair direction, diagonal inclusion, trace key format, or result projection.
- Keep confidence values aligned with the primitive's documented score semantics.

## Verify before finishing

Run syntax and static preflight checks that do not execute the full corpus. Confirm imports and referenced names resolve. Confirm helper and solver plan versions match.

Write only the requested helper, solver, and manifest draft paths. The orchestrator owns execution, scoring, hashes, best-candidate selection, and promotion.
