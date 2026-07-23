# Task: Generate the extraction driver for corpus `{{corpus_name}}`

Write a thin, per-corpus extraction driver `extract_{{corpus_name}}.py` that imports
the shared `semextract` engine and supplies only the corpus-specific hooks. Do NOT
re-implement HTTP / concurrency / JSON parsing / checkpointing — the engine owns
those.

## Schema (from Schema Designer — designed over ALL queries on this corpus)
```json
{{schema_json}}
```

## Corpus
- Table CSV: {{corpus_manifest}}
- Header columns: `{{header}}`
- Modality: {{modality}}
- Size: {{corpus_size}}
- Column hints (heuristic — verify against the header): id=`{{id_col}}`, text=`{{text_col}}`, image=`{{image_col}}`
- Image dir (image modality only): {{image_dir}}

## Engine to import
- `semextract.py` lives at: {{semextract_path}}
  Add its directory to `sys.path`, then `import semextract`.
- Public API: `semextract.run(driver, schema, table_path, out_path, *, modality,
  model, endpoint=None, api_key="EMPTY", concurrency=8, theta=None, image_dir=None,
  limit=0, max_new_tokens=128, timeout=120, prompt_style="json")`.
- For TEXT corpora import `semtext` instead: `semtext.run_extraction(driver, schema,
  table_path, out_path, *, model, endpoint, api_key="EMPTY", concurrency=8, theta=None)`.
  `driver` implements `map_columns(header)` and `extract(patch) -> {field: value}` composing
  `vadar.predefined_text` over the `semtext.TextPatch` it is handed. The engine owns
  concurrency / meta / checkpoint / abort — do not re-implement them.
- Helpers you may reuse: `semextract.build_prompt(schema, "json")` and
  `semextract.resolve_image_path(uri, image_dir)`.
- Driver hooks to implement: `map_columns(header)`, `preprocess(row, cols)`,
  `build_prompt(schema, row, cols)` (inject context columns here).

## Required CLI (the orchestrator invokes it EXACTLY like this)
```
python3 {{driver_path}} <table.csv> <attrs.json> --schema S --model M \
    [--image-dir D] [--endpoint URL --api-key K --concurrency N] [--theta T]
```
Accept `--image-dir` even for text (ignore it there). Bake `modality="{{modality}}"`
into the `semextract.run(...)` call.

## Output
- If Modality is `image`, target `vadar_engine.run` (tiered non-VLM proxies driven by each
  attribute's `extractor` spec); `--model` is the CLIP model id, no `--endpoint` needed.
  If Modality is `text`, compose the predefined TEXT API (`judge/classify/extract/score`
  from `vadar/predefined_text.py`) in `Driver.extract(patch)` and call
  `semtext.run_extraction(...)`; `--model` is the endpoint LLM id and `--endpoint` is required.
- Write the driver to: {{driver_path}}
- Do NOT run the full corpus. You MAY smoke-test with `--limit 2` if an endpoint is
  reachable; otherwise just verify `python3 {{driver_path}} --help` works and the
  file imports `semextract` cleanly. The orchestrator runs the full extraction.

## Validation
The engine guarantees `rows == {{corpus_size}}`, all schema columns + `conf`
present, and JSON-valid output. Your job is only correct column mapping, context
injection, and modality preprocessing.
