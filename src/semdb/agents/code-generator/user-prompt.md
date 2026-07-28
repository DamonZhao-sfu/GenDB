# Task: Compile {{query_id}} into a relational program

## SemBench query
```sql
{{query_sql}}
```

## Schema (from Schema Designer)
```json
{{schema_json}}
```

## Attribute table (from Extractor) — reuse, do not re-extract
- Path: {{attrs_path}}
- Columns: {{attrs_columns}}

## Structured side
- Path: {{structured_path}}
- Columns: {{structured_columns}}

## Runtime library
Import the shared residual runtime for the masked model calls:
`from semruntime import vlm_judge, vlm_answer, METER` (add `sys.path` to `src/semdb`
if needed). Both are typed OpImgVQA calls — guided decoding constrains the answer, and
the returned score comes from token logprobs, so it is comparable across rows.

- `vlm_answer(question, choices, endpoint, model, api_key, image_path=None, text=None)`
  -> `(answer, score)`. **Prefer this** when the query has a closed value space
  (e.g. the set of `Airlines` values): the answer is a real field value that joins.
- `vlm_judge(prompt, endpoint, model, api_key, image_path=None, theta=None)` -> bool.
  Sends the ORIGINAL semantic predicate (the AI.IF question, e.g. "Does the image show
  the logo of {airline}?"); pass `theta` to also require the score to clear it.

Pass `image_path` for image queries, `text=` for text ones. With no `--endpoint` both
are no-ops returning False/"none" and counting `METER.skipped`, so the program must be
correct without them. `METER.judge_calls` is your `residual_calls`.

## Output
- Write the compiled program to: {{code_path}}
- Signature (MUST match — the orchestrator invokes it this way):
  `python3 {{code_basename}} <structured.csv> <attrs.json> <out.csv> [--endpoint URL --api-key KEY --model NAME]`
  Write the result rows to `<out.csv>` with a header (columns in the query's
  SELECT order, e.g. `ID,uri`). Residual rows (extracted key is `none`/empty or
  `conf < theta`) are re-checked by `vlm_answer(...)`/`vlm_judge(...)` against the
  endpoint, in ONE pass after the relational work; if no `--endpoint` is given, skip
  them and log how many were left unresolved (`METER.skipped`).
- Also write `compiled_{{query_id}}.meta.json` = `{"elapsed_sec": ..., "rows": ...,
  "residual_calls": ...}` next to the program (the orchestrator reads it).
- Print rows emitted and residual model calls.

## Validation
Run the compiled program, then compare against the naive oracle:
`python3 baseline_{{query_id}}.py ... <out_baseline.csv>` and `diff` the two
result CSVs. They MUST match. Report the model-call reduction (M×N → residual).
