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
`semlib.py` provides: `load_*`, `normalize` (with the compile-time synonym map),
`P.vlm_judge` / `P.llm` (the masked residual model call, counted), `write_pairs`.

## Output
- Write the compiled program to: {{code_path}}
- Signature (MUST match — the orchestrator invokes it this way):
  `python3 {{code_basename}} <structured.csv> <attrs.json> <out.csv> [--endpoint URL --api-key KEY]`
  Write the result rows to `<out.csv>` with a header (columns in the query's
  SELECT order, e.g. `ID,uri`). If `--endpoint` is given, route residual
  `P.vlm_judge`/`P.llm` calls to that OpenAI-compatible server; otherwise the
  residual path may be skipped (log how many rows were left unresolved).
- Also write `compiled_{{query_id}}.meta.json` = `{"elapsed_sec": ..., "rows": ...,
  "residual_calls": ...}` next to the program (the orchestrator reads it).
- Print rows emitted and residual model calls.

## Validation
Run the compiled program, then compare against the naive oracle:
`python3 baseline_{{query_id}}.py ... <out_baseline.csv>` and `diff` the two
result CSVs. They MUST match. Report the model-call reduction (M×N → residual).
