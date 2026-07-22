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
`from semruntime import vlm_judge, METER` (add `sys.path` to `src/semdb` if needed).
`vlm_judge(prompt, endpoint, model, api_key, image_path=None)` sends the ORIGINAL
semantic predicate (the AI.IF question, e.g. "Determine if the image shows the
logo of {airline}. ") to the vLLM/OpenAI endpoint and returns True/False; pass
`image_path` for image queries, omit it for text. `METER.judge_calls` is your
`residual_calls`.

## Output
- Write the compiled program to: {{code_path}}
- Signature (MUST match — the orchestrator invokes it this way):
  `python3 {{code_basename}} <structured.csv> <attrs.json> <out.csv> [--endpoint URL --api-key KEY --model NAME]`
  Write the result rows to `<out.csv>` with a header (columns in the query's
  SELECT order, e.g. `ID,uri`). Residual rows (extracted key is `none`/empty or
  `conf < theta`) are re-checked by `vlm_judge(...)` against the endpoint; if no
  `--endpoint` is given, skip them and log how many were left unresolved.
- Also write `compiled_{{query_id}}.meta.json` = `{"elapsed_sec": ..., "rows": ...,
  "residual_calls": ...}` next to the program (the orchestrator reads it).
- Print rows emitted and residual model calls.

## Validation
Run the compiled program, then compare against the naive oracle:
`python3 baseline_{{query_id}}.py ... <out_baseline.csv>` and `diff` the two
result CSVs. They MUST match. Report the model-call reduction (M×N → residual).
