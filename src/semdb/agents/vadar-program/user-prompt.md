# Write the extraction program for corpus `{{corpus_name}}`

## Schema fields + value spaces (labels)
```json
{{schema_json}}
```
## Generated helpers
Read: `{{helpers_path}}`
## Predefined API
Read `MODULES_SIGNATURES` in `{{semdb_dir}}/vadar/predefined.py`.

## Corpus columns (from the actual table header — DO NOT GUESS)
- Header: `{{header}}`
- Identity column (`"id"` in map_columns): `{{id_col}}` — this VALUE is what the compiled
  query joins/emits downstream, so it MUST be a real column above (e.g. a filename/uri), not
  a made-up `id`.
- Image column (`"image"` in map_columns, the path/uri to load): `{{image_col}}`

Write the driver to `{{driver_path}}` with EXACTLY this shape (the orchestrator runs it as
`python3 {{driver_path}} <table.csv> <attrs.json> --schema S --model CLIP --image-dir D`):
```python
import sys, os, json, argparse
sys.path.insert(0, "{{semdb_dir}}")
import vadar_engine
from vadar.predefined import classify, best_ocr_match, dominant_colors, verify_property, detect, score, read_text
# <paste the generated helper implementations here>
# value-space constants from schema.attributes[].labels, e.g. TRACKS = [...]
class Driver:
    def __init__(self): self.image_dir=None
    def map_columns(self, header): return {"id":"{{id_col}}","image":"{{image_col}}","text":None,"context":[]}
    def extract(self, patch):
        # ONE entry per schema field, composing the API + helpers
        return { ... }
if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("table"); ap.add_argument("out")
    ap.add_argument("--schema",required=True); ap.add_argument("--model",required=True)
    ap.add_argument("--image-dir"); ap.add_argument("--theta")
    a=ap.parse_args(); d=Driver(); d.image_dir=a.image_dir
    vadar_engine.run(d, json.load(open(a.schema)), a.table, a.out, model=a.model, image_dir=a.image_dir)
```
The generated driver may use the local CLIP/OCR/CV/detector stack shown above, but must not
create or call an LLM/VLM endpoint, model-service SDK, HTTP client, API key, or semantic
judgement API.
