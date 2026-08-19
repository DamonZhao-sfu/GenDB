---
name: plan-semantic-query
description: Plan GenDB SemDB semantic SQL operators into a typed, offline-executable physical plan. Use for initial planning or evidence-driven replanning when identifying AI call sites, determining row/pair/tuple sampling units, selecting local VADAR primitives, defining helper DAGs, preserving relational semantics, specifying trace contracts, or deciding that a query is not compilable with the available local runtime.
---

# Plan Semantic Query

Produce a typed semantic plan. Do not write implementation code.

Design the minimum reusable helper interfaces over the fixed predefined API, then specify
each helper's implementation as an exact composition of predefined primitives. The helper
DAG you emit is the implementation spec — the Generator transcribes it literally.

## Read required inputs

Read the query SQL and natural-language description, table metadata, local primitive implementations, trace contract, and output schema. On a replan, also read the previous plan and the optimizer action.

Treat the operator library as authoritative — `MODULES_SIGNATURES` in
`vadar/predefined.py` documents BOTH families: the VISION functions (first argument an
ImagePatch) and the TEXT functions (first argument a string). Never invent a function,
parameter, return type, score meaning, or model capability.

## Propose the helpers, then specify their implementations

- Build MINORLY on the existing API. Add a helper ONLY when a combination of existing
  primitives is not already enough, and propose the FEWEST helpers necessary — for a text
  plan, 1-3 general helpers is typical.
- Every visual helper takes `image` first. Every helper returns a real field VALUE — a name,
  a label, a colour list — not a score and not an opaque verdict. `inferred_name(image, names)`
  and `is_expected_kind(image)` are the right shape; `pair_predicate_holds(value, image)`
  is not, because it buries the inferred value where neither the trace nor the optimizer
  can see it.
- Keep helpers general and reusable across the corpus's queries, not fitted to one row.
- Each helper's `primitive_steps` composes ONLY predefined primitives and helpers already
  defined earlier in the DAG.
- Keep the relational comparison of an inferred value against a row's column outside the
  visual helper: the helper infers, Python compares.
- For a cross-table pair site the visual inference usually depends only on the IMAGE — plan
  one inference per image, cached by image key, then evaluate the relational condition per
  pair. A 13 x 200 frame then costs 200 visual calls, not 2600.

## Build the plan

1. Identify every semantic call site in the SQL.
2. Normalize each call site into its input columns, predicate, output type, and sampling unit.
3. **Split each predicate into conjuncts and route every conjunct to the cheapest layer
   that can decide it**, before binding any primitive:
   - decidable from a STRUCTURED column → plain Python/SQL, never a model call;
   - decidable from TEXT → a text primitive;
   - genuinely visual → an image primitive.
   State in the plan which column each non-visual conjunct reads. A conjunct routed to
   the wrong layer is unfixable downstream: the generator must implement the plan and
   the optimizer can only reword prompts.
   - A conjunct asserting a PROPERTY OF A NAMED ENTITY — its region, category, or any
     fact recorded about it elsewhere — is a fact about the structured column holding
     that name, not about the image. Asking an image "is this entity of kind K" when K
     is a column value wastes the site.
   - When a predicate's conjuncts are all equality tests over columns the corpus already
     carries, the whole predicate is a plain conjunction with zero model calls. Check the
     available columns before binding any of it to inference.
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
- Missing external taxonomy data alone is not a reason for `not_compilable` when an
  available runtime text field contains the evidence. Bind `text_classify_detail` to
  short query-defined positive/negative hypotheses and declare a `bounded_approximation`;
  do not invent or hard-code the missing taxonomy.
- Preserve ordered-pair direction and diagonal rules for self-joins.
- Apply `--only-ids` after forming the correct validation unit and before semantic inference.
- Require a trace entry for every evaluated validation unit, including negative decisions.
- Keep physical trace identity separate from the SQL result projection.
- Reject silent fallback to a different semantic predicate.
- Mark a plan `not_compilable` when required information or capability is unavailable.

## Preserve the vis-operator physical-operator repertoire

For image sites, preserve the local visual runtime's physical-operator repertoire:

- Use `classify` for a small visual enum, `classify_multi` when one image can contain
  several enum values, and `domain_classify` when a documented domain specialist fits.
- Use `best_ocr_match` or `read_text` for legible names and wordmarks. A large value
  space of names PRINTED in the image — brands, venues, wordmarks — is an OCR
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
- Give every row-filtering image site a discriminative step that reads the identity out
  of the image: OCR against the runtime value space, or a closed-set `classify` over
  that space.
- **An argmax step on a FILTER must be able to abstain.** `classify` and
  `classify_detail` always return one of `options` — they can never return "none" — so
  on a predicate where most rows match nothing they label every row and acceptance
  collapses to "true": recall 1.0, precision ≈ the base rate. A `!= "none"` guard on
  their result is always true and does nothing. Bind `classify_or_none(image, options,
  min_conf)`, or bind `classify_detail` and record the numeric cutoff as
  `confidence_signal.threshold`. `best_ocr_match*` is exempt — it genuinely returns
  "none" on a miss.
- Both directions fail, and they fail differently. Weigh them explicitly:
  - An AND-chain of weak signals empties the result: one conjunct that never fires on
    the corpus — a `contains_any` over multi-word phrases that appear in no row, say —
    zeroes the whole query no matter how well the others work.
  - A DISJUNCTION of weak signals accepts everything. Each path added to an OR can only
    widen acceptance, so an optimizer chasing recall widens it monotonically until the
    predicate selects the entire corpus at precision equal to the base rate.
  Prefer a disjunction of paths that are individually THRESHOLDED over either extreme:
  each path must be able to say no on its own, and the site must still be able to select
  nothing when the corpus genuinely holds no matches.
- Sanity-check the plan against selectivity before committing to it: state roughly what
  fraction of rows you expect the predicate to accept, and say which evidence path is
  the one that makes it that selective. A predicate you cannot argue is selective is a
  predicate that will accept the corpus.
- Bind the `classify_detail`, `verify_detail`, `detect_detail`, `ocr_detail`, or
  `best_ocr_match_detail` variant when a path needs a confidence, and record it in
  `confidence_signal` together with what it is compared AGAINST. A `*_detail` score is
  comparable across rows for that one primitive, never across primitives. State a
  threshold as a starting point to be tuned from evidence, and prefer a comparison the
  data defines — an argmax between competing options, or a margin between two
  confidences from the SAME primitive — over a guessed absolute cutoff.
- Plan the FIRST version permissive and let the optimizer tighten it from observed false
  positives. A site that returns zero rows for every input is as broken as one that
  returns every row: both score F1 0, and the empty one also freezes the branch counters
  so every later iteration sees identical evidence and learns nothing.
- When the SQL implies a near one-to-one correspondence — one image per named entity, one
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
`text_classify_detail`, `text_classify_multi_detail`, `any_value_in_set`, regex
extraction, `extract_person_names`, value splitting, numeric/date parsing, and Python
standard-library logic.

**The library carries NO taxonomy.** Every closed value space — genres, regions,
brands, venue names — is an ARGUMENT you must source at runtime from a structured
column or the SQL's own literals, and state in the plan where it comes from. A plan
that assumes a built-in genre list or a built-in geography is not implementable.
`text_classify_multi_detail` preserves an explicit multi-label value space and returns
per-option scores, but it matches by TOKEN OVERLAP, so a multi-word cue can half-match
("love story" scores 0.5 against "...horror story..."). When the predicate needs
WHOLE-PHRASE semantics, specify `contains_any` over a cue list instead.
`any_value_in_set` tests a delimited column against a caller-supplied member set.
Do not replace an implicit semantic predicate with an unrelated keyword rule; declare a
bounded approximation and its limitation, or mark the site not compilable.

For a cast/person extraction followed by a strict relational intersection,
`extract_person_names` may produce conservative proper-name candidates and let the
SQL `COUNT(DISTINCT ...)`/intersection disambiguate them. Do not require a literal
"cast:" marker when the description explicitly names performers in prose.

## Primitive-binding failure modes

Each of these has cost a query its entire iteration budget. They are ordered by how much
they cost. `E` below is the entity a predicate is about and `V` the value space it is
matched against, both supplied by the query at runtime.

### 1. A pair predicate must stay a function of BOTH sides

`AI.IF(value, image)` is binary. Do not collapse one side to a label first:

```python
# WRONG — a classification task wearing a join's clothes
inferred = infer_label(image, all_values)   # -> one string
holds = inferred == row["Col"]
```

One input can then satisfy at most ONE value. Whenever a value space contains a value
that is a qualified form of another — `"X"` alongside `"<qualifier> at X"` — a single
argmax label cannot match both, so every ground-truth row using the other form is
unreachable **before any inference runs**. No model quality recovers it.

Declare a helper that decides one `(input, value)` pair and returns bool, and prefer
normalized containment over string equality when one value can be a qualified form of
another. Extracting per-input evidence once and then deciding per `(evidence, value)`
keeps this cheap: model work stays O(inputs) while the predicate stays O(pairs).

### 2. Veto first, identify second

A closed-set classifier is not a predicate — it assigns a nearest value to EVERY input.
Gate on "is this the KIND of thing the predicate is about" before asking "which one",
otherwise acceptance is unconditional. In a working solver the great majority of
rejections should happen before the identity step is ever reached; if your branch
counters show the identity step deciding most rows, the gates are not doing their job.

### 3. Ask a competing-class question — veto on DISJOINT losers, never require the target to win

`verify_property(image, "<a kind of thing>")` compares `prop` against `not prop` and is
positively biased. Ask which of several *competing* categories fits instead, then reject
**only when a category DISJOINT from the target leads**:

```python
kinds = [TARGET_KIND, DISJOINT_A, DISJOINT_B, BROADER_KIND]
if leading in {DISJOINT_A, DISJOINT_B}:
    return False          # correct: those cannot also be the target
```

Do NOT require the target to rank first:

```python
if ranked[0] != TARGET_KIND:      # WRONG
    return False
```

A category list almost always contains a label that SUBSUMES the target — a specific kind
of logo is also "a company logo"; a specific species is also "an animal" — and the
superset legitimately outranks the specific term. Requiring the target to win therefore
rejects the very inputs the query depends on. Audit your veto list for any label that is
a superset of, or overlaps, what you are looking for, and leave it out of the veto.

Mind the scale too: `topk_text` returns RAW cosines, which sit in a narrow band (six
competing prompts on one image can span less than 0.02 in total), so ORDER there is close
to noise and a small margin on it is meaningless. `classify_detail` returns softmaxed
confidences and is the one to threshold on.

### 4. Every primary path needs a fallback on a DIFFERENT backend

```python
# WRONG — one backend, no floor
match, conf = best_ocr_match_detail(image, values)
return match if conf >= 0.5 else "none"
```

A single-backend path returns the SAME CONSTANT for every row on whatever subset its
backend cannot read, and does so silently. OCR is the right primitive for cleanly printed
names and the wrong one for stylized or decorative renderings, where it returns garbled
text that fuzzy-matches nothing; an embedding classifier often reads the same input with
high confidence. Bind a second path on a different backend and count the branches
separately, so the log shows which one carried the run.

### 5. Compare within a primitive family; never threshold across families

```python
if not is_target or target_conf < competitor_conf + MARGIN:   # both from verify_detail
    return False
```

Two scores from the same primitive are comparable, so a MARGIN between them is
meaningful. An absolute cutoff against a hardcoded constant is not: embedding
probabilities, detector confidences and OCR match strengths are on different scales, and
raw similarity additionally carries a large per-input offset that an absolute threshold
cannot see past.

### 6. Strip the value space's shared words before matching on them

```python
distinctive = [w for w in normalize(value).split()
               if len(w) >= 4 and w not in SHARED_TOKENS]
if sum(w in text for w in distinctive) >= min(2, len(distinctive)):
```

Compute `SHARED_TOKENS` from the runtime value space: any token occurring in several of
its values has zero discriminative power and matches every candidate. Drop those and
require at least two distinctive tokens.

### 7. Crop to the subject before reading an attribute off it

```python
dominant_colors(image)                     # the background, if the subject is small
dominant_colors(image, center_frac=0.6)    # the subject
```

Any attribute read off a subject that does not fill the frame — a color, a pattern, a
material — needs `regions_center` / `crop` / `regions_propose` first, or it reports the
background. The default is the whole image, so cropping has to be a deliberate choice
recorded in the primitive's arguments.

### 8. Never tune a branch against the validation sample

```python
if (value == SOME_SPECIFIC_VALUE and conf_a >= 0.90 and conf_b >= 0.99):   # WRONG
    return True
```

A branch keyed on one hardcoded value, with thresholds reverse-engineered from the rows
you can see, scores on the validation sample and produces false positives everywhere
else. It is one of the reasons validation scores read higher than corpus scores. A rule
you cannot state without naming a specific data value does not belong in the plan.

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
