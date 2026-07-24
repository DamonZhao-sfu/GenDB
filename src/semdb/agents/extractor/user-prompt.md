# Task: Generate the extraction driver for corpus `{{corpus_name}}`

Write a thin, per-corpus extraction driver `extract_{{corpus_name}}.py` that imports
the local modality engine and supplies only corpus-specific hooks. Do not add any
HTTP/network/model-service plumbing.

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
- Add the directory containing `{{semextract_path}}` to `sys.path`.
- For IMAGE corpora import `vadar_engine` and call its local CLIP/CV extraction runner.
- For TEXT corpora import `vadar_text_engine` instead:
  `vadar_text_engine.run(driver, schema, table_path, out_path)`.
  `driver` implements `map_columns(header)` and
  `extract(text) -> {field: value}` using deterministic functions from
  `vadar.predefined_text`. The input is an ordinary string.
- Image-path resolution is owned by `vadar_engine`; text context columns are appended by
  `vadar_text_engine`.
- Driver hooks to implement: `map_columns(header)` and `extract(value)`.

## Required CLI (the orchestrator invokes it EXACTLY like this)
```
python3 {{driver_path}} <table.csv> <attrs.json> --schema S --model M \
    [--image-dir D] [--theta T]
```
Accept `--image-dir` even for text (ignore it there). Select `vadar_engine` or
`vadar_text_engine` according to `modality="{{modality}}"`.

## Output
- If Modality is `image`, target `vadar_engine.run` (tiered non-VLM proxies driven by each
  attribute's `extractor` spec); `--model` is the CLIP model id, no `--endpoint` needed.
  If Modality is `text`, compose the deterministic offline API from
  `vadar/predefined_text.py` in `Driver.extract(text)` and call
  `vadar_text_engine.run(...)`. Accept `--model` for CLI compatibility but ignore it.
  Never emit endpoint/API-key options or model/network calls.
- Write the driver to: {{driver_path}}
- Do NOT run the full corpus. You may smoke-test with a small local sample; otherwise
  verify `python3 {{driver_path}} --help` and imports. The orchestrator runs extraction.

## Validation
The engine guarantees `rows == {{corpus_size}}`, all schema columns + `conf`
present, and JSON-valid output. Your job is correct column mapping, context selection,
and deterministic extraction logic.
