# Task: Design the extraction schema for {{query_id}}

## SemBench query
```sql
{{query_sql}}
```

## Natural-language intent
{{query_nl}}

## Table schemas
{{table_schemas}}

{{#if existing_schema}}
## Existing schema for this corpus (reuse / extend if possible)
```json
{{existing_schema}}
```
{{/if}}

## Corpus stats
- Unstructured side: {{corpus_name}} ({{corpus_size}} items, modality: {{modality}})
- Structured side: {{structured_name}} ({{structured_size}} rows)

## Output
Apply the decomposition test and write `schema.json` to: {{schema_path}}

If the query is NOT decomposable, still write `schema.json` with
`"decomposable": false` and a rationale so the orchestrator can fall back to
naive per-pair execution.
