# Task: Design the extraction schema for {{query_id}}

## SemBench queries (ALL queries over this corpus — design the UNION of the slots they test)
Each query's semantic predicate tells you which slot(s) of the unstructured item to
extract. Derive the attributes from THESE predicates; do not assume a default field.
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

- Each attribute MUST include an `extractor` spec (tier cv|clip|…|vlm) chosen per the
  decision tree in the system prompt — prefer non-VLM tiers (cv for color, clip for
  closed-enum category / brand / image-text match).

If the query is NOT decomposable, still write `schema.json` with
`"decomposable": false` and a rationale so the orchestrator can fall back to
naive per-pair execution.
