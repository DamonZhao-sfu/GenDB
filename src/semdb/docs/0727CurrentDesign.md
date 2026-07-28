# SemDB current design

The default DIRECT architecture is now Planner–Generator–Optimizer (PGO):

```text
SQL + table metadata + local primitives
          │
          ▼
  Query Planner → typed plan.json
          │
          ▼
  Code Generator → complete helper + solver + candidate manifest
          │
          ▼
  deterministic preflight → run → SELECT validation score
          │
          ▼
  iteration_feedback.json → Optimizer
          │
          ├── PATCH_CODE → Generator
          ├── REPLAN     → Planner → Generator
          └── STOP       → promote historical best
```

Enable it with `--direct --agent-architecture pgo` (the branch default). Each
role receives one fixed repository-local procedural skill. Planner and Optimizer
write JSON only; Generator writes a complete helper and solver for every
candidate. Runtime schemas validate all four JSON artifacts, and the
orchestrator computes candidate hashes before execution.

PGO optimization requires measurable SELECT-validation feedback. Without
`--val-file` or a successfully built `--val-rate` sample, it performs one
Planner → Generator run and freezes that candidate. CERT and final ground truth
never enter `iteration_feedback.json`; final ground-truth evaluation occurs only
after promotion. Each promoted candidate includes `plan.json`, helper, solver,
and `candidate_manifest.json`.

The non-DIRECT Schema Designer → Extractor → Code Generator pipeline is
unchanged. The previous DIRECT flow remains available with
`--agent-architecture legacy`:

```text
SQL + table metadata
          │
          ▼
  1. Signature Agent
     proposes helper interfaces
          │
          ▼
  2. API Agent
     implements those helpers
          │
          ▼
  3. Solver Agent
     writes solve_<query>.py
          │
          ▼
  static validation → execute → score
                                │
                         mistakes/metrics
                                │
                                └── Solver Agent refines

  ### 1. VADAR Signature Agent

  Input:

  - Original SQL query
  - Corpus modality
  - Available predefined operators
  - No designed semantic schema in direct mode

  Output:

  iter_0/_vadar_signatures_<query>.txt

  For an image query, it might propose:

  def logo_name(image, candidate_names):
      """Identify which candidate name appears in the logo."""

  It only proposes helper interfaces; it does not implement them. It is instructed to propose the smallest number of helpers needed.

  For image workloads it can build on:

  classify
  best_ocr_match
  dominant_colors
  detect
  verify_property
  pair_score
  topk_similar
  ...

  For text workloads it instead uses deterministic primitives such as:

  normalize
  contains_phrase
  contains_any
  best_lexical_match
  regex_extract

  The agent contract is in agents/vadar-signature/prompt.md.

  ### 2. VADAR API Agent

  The API agent reads the signatures and implements them by composing the predefined local operators.

  Output:

  iter_0/_vadar_helpers_<query>.py

  Example:

  from vadar.predefined import best_ocr_match

  def logo_name(image, candidate_names):
      return best_ocr_match(image, candidate_names)

  It is forbidden from introducing:

  - Qwen/VLM calls
  - OpenAI-compatible endpoints
  - HTTP clients
  - API keys
  - semtext or other semantic judge services

  The helper module is generated once during iter_0. Signature and API agents are not called again during refinement.

  ### 3. VADAR Solver Agent

  The solver agent receives:

  - SQL and natural-language query
  - Paths and columns for every referenced table
  - Generated helpers
  - Predefined API documentation
  - Required result schema
  - Validation trace contract

  It writes:

  iter_0/solve_<query>.py

  The generated program performs the entire query:

  load CSV tables
      → execute local semantic operators
      → relational filters and joins in Python
      → projection/aggregation
      → write result CSV
      → write trace_<query>.json

  For example, an image-to-structured-table join can look conceptually like:

  track_names = {row["Track"] for row in tracks}

  for image_row in images:
      predicted_track = logo_name(image, track_names)

      for track_row in tracks:
          if predicted_track == track_row["Track"]:
              output.append((track_row["ID"], image_row["uri"]))

  The solver itself is completely offline. Image semantics use local CLIP/OCR/CV/detector operators; text semantics currently use lexical, regex, numeric and date operations.

  The solver contract is in agents/vadar-solver/prompt.md.

  ## How iterative refinement works

  With your earlier command:

  --val-rate 0.1 \
  --oracle-model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --max-iterations 5

  Qwen is not one of the three code-generation agents. Qwen labels the sampled validation rows before code generation/refinement.

  The loop is:

  1. Qwen labels the sampled 10% validation set.
  2. The three agents generate iter_0.
  3. The solver runs only on validation IDs using --only-ids.
  4. It writes:

     result CSV
     trace_<query>.json
     stderr diagnostic log

  5. SemDB compares the trace against Qwen labels.
  6. Feedback is constructed containing:
      - Precision, recall and F1
      - TP, FP and FN counts
      - Mislabeled validation rows
      - Expected and predicted values
      - Runtime errors or compile errors
      - Solver diagnostic branches and warnings
      - Previous iteration history

  7. Only the Solver Agent is called again.
  8. The best solver is copied into the next iteration.
  9. A new version is retained only if it improves F1 or fixes a broken run.

  Conceptually:

  iter_0: Signature → API → Solver → run → score
                                        │
  iter_1: copy best solver + feedback ──┘
          Solver only → run → score
                           │
  iter_2: copy best solver + feedback
          Solver only → run → score

  It stops only when `--max-iterations` is reached. Validation F1=1.0 and
  consecutive non-improvements still consume the requested iteration budget; they
  affect which candidate is retained, not how many candidates are generated.

  Afterward, the best solver is frozen and executed once on the full corpus without --only-ids.

  The loop implementation is in orchestrator.mjs:1654.

  ## Pairwise join validation

  If the AI predicate is the join condition, validation operates on pairs rather than individual rows:

  structured row × image row → pair ID

  A trace key must be:

  "<structured_id>-<image_filename>"

  For example:

  {
    "attr": "matches",
    "rows": {
      "track_12-logo_41.jpg": "true",
      "track_13-logo_41.jpg": "false"
    }
  }

  This lets SemDB tell whether the semantic join predicate is correct independently of the final projected result.

  ## Current limitations worth knowing

  - The Signature and API agents only run for iter_0; refinement is Solver-only.
  - Join planning and relational execution are generated as plain Python. There is no relational optimizer deciding join order yet.
  - Text direct mode is deterministic and lexical. It cannot call the Qwen oracle at runtime, so genuinely implicit text semantics may be poorly approximated.
  - The direct-agent configuration has no explicit vadar_signature, vadar_api, or vadar_solver model entries. They currently inherit the provider-wide model:
      - Codex provider: gpt-5.6-luna
      - Claude provider: opus

    Their effort level is also currently unspecified, so the provider default is used.

  - Without --val-rate or --val-file, refinement can use full benchmark ground truth. With oracle validation, only the sampled Qwen labels drive refinement.
