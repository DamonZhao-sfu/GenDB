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
- Signature: `python3 {{code_basename}} <structured.csv> <attrs.json> <out.csv>`
- Print rows emitted and residual model calls.

## Validation
Run the compiled program, then compare against the naive oracle:
`python3 baseline_{{query_id}}.py ... <out_baseline.csv>` and `diff` the two
result CSVs. They MUST match. Report the model-call reduction (M×N → residual).
