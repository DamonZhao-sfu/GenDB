---
name: plan-semantic-query
description: Plan GenDB SemDB semantic SQL operators into a typed, offline-executable physical plan. Use for initial planning or evidence-driven replanning when identifying AI call sites, determining row/pair/tuple sampling units, selecting local VADAR primitives, defining helper DAGs, preserving relational semantics, specifying trace contracts, or deciding that a query is not compilable with the available local runtime.
---

# Plan Semantic Query

Produce a typed semantic plan. Do not write implementation code.

## Read required inputs

Read the query SQL and natural-language description, table metadata, local primitive implementations, trace contract, and output schema. On a replan, also read the previous plan and the optimizer action.

Treat the primitive implementation files as authoritative. Never invent a function, parameter, return type, score meaning, or model capability.

## Build the plan

1. Identify every semantic call site in the SQL.
2. Normalize each call site into its input columns, predicate, output type, and sampling unit.
3. Distinguish row, cross-table pair, ordered self-pair, grouped, and tuple domains.
4. Preserve deterministic relational operations before and after semantic evaluation.
5. Classify compilability as `exact`, `bounded_approximation`, or `not_compilable`.
6. List every compilability obligation and unresolved semantic requirement.
7. Build a typed helper DAG. Give every helper explicit arguments, return type, dependencies, primitive bindings, and confidence signal.
8. Define the relational plan in execution order.
9. Define runtime inputs, result projection, trace key/value semantics, and `--only-ids` filtering location.
10. State assumptions and invariants explicitly.

## Apply planning rules

- Use only local, offline primitives available in the repository.
- Keep value spaces explicit. Read closed value spaces from database columns when allowed; do not infer them from validation labels.
- A same-input/same-output local primitive may be planned as a declared
  `bounded_approximation` even when it cannot reproduce remote connection or model
  controls. Record that limitation; those controls alone are not a reason to reject
  the plan.
- Treat supplied normalized-view and external-object adapter contracts as authoritative
  relational equivalences.
- Preserve ordered-pair direction and diagonal rules for self-joins.
- Apply `--only-ids` after forming the correct validation unit and before semantic inference.
- Require a trace entry for every evaluated validation unit, including negative decisions.
- Keep physical trace identity separate from the SQL result projection.
- Reject silent fallback to a different semantic predicate.
- Mark a plan `not_compilable` when required information or capability is unavailable.

## Preserve the vis-operator physical-operator repertoire

For image sites, select primitives using the same rules as the legacy VADAR agents:

- Use `classify` for a small visual enum, `classify_multi` when one image can contain
  several enum values, and `domain_classify` when a documented domain specialist fits.
- Use `best_ocr_match` or `read_text` for legible names and wordmarks. A large value
  space of names PRINTED in the image — airlines, brands, venues, logos — is an OCR
  problem, not a CLIP problem. For a large runtime value space, use `topk_text` to
  shortlist candidates before verification.
- Use `dominant_colors` for colors. Use `detect` only for its documented closed
  vocabulary and `detect_open` for an open-vocabulary object. A `detect` result is a
  sub-image that may be classified, scored, cropped, or OCRed; use `bbox` only when
  coordinates are part of the physical plan.
- Use `verify_property` ONLY as a coarse gate on a generic visual property, always
  beside a discriminative step. It asks CLIP whether the image matches `prop` better
  than `not prop` and is positively biased: bound to a named-entity property such as
  `"the logo of " + X` it answers true for nearly every image, scoring recall 1.0 at
  precision near zero. Never bind a whole named-entity, multi-clause, or
  identity-of-a-specific-thing predicate to a single `verify_property` call.
- Give every row-filtering image site a discriminative step (OCR or closed-set
  `classify` over the runtime value space), a cheap gate evaluated first, and a
  comparable confidence. Bind the `classify_detail`, `verify_detail`, `detect_detail`,
  `ocr_detail`, or `best_ocr_match_detail` variant and record the gating threshold in
  `confidence_signal`; a filter site with `confidence_signal: null` leaves the
  optimizer nothing to tune.
- When the SQL implies a near one-to-one correspondence — one logo per airline, one
  portrait per person — plan the assignment or dedup step explicitly. Independent
  per-pair matching multiplies false positives.
- Prefer several small, generally named helpers over one helper that hides the whole
  predicate, so the optimizer can patch one factor at a time.
- Record in `compilability.obligations` which primitive you rejected for each image
  site and why, so a replan does not silently reintroduce it.
- Use `regions_center` for product-photo margins, `regions_grid` for a systematic
  scan, and `regions_propose` for several distinct foreground objects. Use `crop` for
  a known layout. Plan region work only when whole-image evidence is too coarse.
- Use `pair_score` for an image-to-image predicate. Use `topk_similar` for one-to-many
  ranking rather than repeated pair calls. Use `score` for image-to-short-text
  relevance and `embed` when the plan benefits from reusable image vectors.
- When one target is rare among many images, plan a cheap kind/property gate before
  expensive OCR, open-vocabulary detection, or region decomposition.
- Keep CLIP prompts short and visual. CLIP reads only ~77 tokens, compares phrases
  rather than sentences, and does not process negation, so appending exclusions or a
  "reject X, Y, Z" clause lowers precision instead of raising it. Decompose long
  natural-language predicates into category, color, presence, OCR, and relational
  checks instead of passing the whole query to one primitive.
- Resolve image references through the supplied offline image adapter and reuse one
  encoder context per process.

For text sites, compose `normalize`, phrase/all/any predicates, lexical matching,
`classify_detail`, `classify_multi_detail`, `classify_movie_genres`,
`has_movie_genres`,
`destination_in_region`, regex extraction,
`extract_person_names`, value splitting, numeric/date parsing, and Python
standard-library logic.
`classify_multi_detail` preserves an explicit multi-label SQL/runtime value space;
`classify_movie_genres` supplies a repository-local general movie-genre taxonomy;
`has_movie_genres` evaluates conjunctive genre predicates such as romantic comedy;
`destination_in_region` supplies repository-local airport geography for Germany and
Europe and fails closed for unknown regions. Read other closed value spaces from
runtime CSV columns. Do not replace an implicit semantic predicate with an unrelated
keyword rule; declare a bounded approximation and its limitation, or mark the site
not compilable.

For a cast/person extraction followed by a strict relational intersection,
`extract_person_names` may produce conservative proper-name candidates and let the
SQL `COUNT(DISTINCT ...)`/intersection disambiguate them. Do not require a literal
"cast:" marker when the description explicitly names performers in prose.

## Replan from evidence

Preserve all unaffected plan sections. Increment `plan_version`, set `parent_plan_version`, and change only what the optimizer evidence supports.

Do not replan merely because one sample is difficult. Require evidence of a plan-level problem such as a wrong sampling unit, missing input, invalid primitive capability, wrong value space, or impossible trace contract.

## Validate before finishing

Confirm that:

- every semantic site is represented;
- every helper dependency resolves;
- every primitive exists;
- every type edge is compatible;
- the relational plan preserves SQL projection and ordering semantics;
- trace keys match the validation unit;
- the runtime remains offline;
- no CERT, final ground truth, or validation label was used for planning.

Write only the requested `plan.json`. Make it conform to `src/semdb/contracts/semantic-plan.schema.json`.
