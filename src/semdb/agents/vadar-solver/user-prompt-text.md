# Write the offline TEXT solver `solve_{{query_id}}.py` for query `{{query_id}}`

## Query
```sql
{{query_sql}}
```
Natural language: {{query_nl}}

## Tables (read these from `--data-dir` by filename)
{{tables_doc}}

## Generated offline helpers
Read: `{{helpers_path}}`

## Offline text API
Read `MODULES_SIGNATURES_TEXT` in `{{semdb_dir}}/vadar/predefined_text.py`.

## Output contract
- Write a CSV whose header columns are exactly the query's SELECT list, in order.
- Read closed value spaces from structured CSV columns at runtime.
- Write the program to `{{solve_path}}`.
- It must run exactly as:
  `python3 {{solve_path}} <out.csv> --data-dir D`

The generated Python may use only ordinary strings, the offline text API, generated
helpers, and Python's standard library. It must not contain model/network clients,
endpoint/API-key plumbing, `semtext`, `TextPatch`, or a semantic judgement API.
