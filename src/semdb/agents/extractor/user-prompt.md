# Task: Materialize the attribute table for corpus `{{corpus_name}}`

## Schema to fill (from Schema Designer)
```json
{{schema_json}}
```

## Corpus
- Items: {{corpus_manifest}}  (columns: id/uri + local path or text)
- Modality: {{modality}}
- Size: {{corpus_size}} items

## Model
- Preferred small model: {{small_model}}
- Escalation model (only for critical low-conf items): {{escalation_model}}

## Output
- Write the attribute table to: {{attrs_path}}
- One row per corpus item, columns = schema attributes + `conf`.
- Then print a one-line summary: rows extracted, and how many returned `none`
  (these become the compiled query's residual set).

## Validation
Run your driver, then verify: `rows == {{corpus_size}}`, all columns present,
JSON parses. If a row failed to parse, record it as `none` with `conf: 0.0`.
