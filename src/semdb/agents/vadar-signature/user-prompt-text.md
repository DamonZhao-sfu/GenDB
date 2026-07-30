# Corpus `{{corpus_name}}` — propose helper signatures for its TEXT queries

## Queries (SQL)
```sql
{{query_sql}}
```
## Schema fields to extract (from the Schema Designer)
```json
{{schema_json}}
```
## Predefined TEXT API
Read `MODULES_SIGNATURES` in `{{semdb_dir}}/vadar/predefined.py`.

Write the proposed `<docstring>/<signature>` blocks to: `{{sig_path}}`
(one file, plain text). Propose the FEWEST helpers necessary. Each helper takes an ordinary
string and composes the offline primitives documented in `MODULES_SIGNATURES` and
Python's standard library. Do not propose a helper that requires a model or network call.
