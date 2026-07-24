# Write the end-to-end solver `solve_{{query_id}}.py` for query `{{query_id}}`

## Query
```sql
{{query_sql}}
```
Natural language: {{query_nl}}

## Tables (read these from `--data-dir` by filename)
{{tables_doc}}

## Image manifest
- Manifest table: `{{image_table}}` (path `{{image_path}}`)
- Filename column: `{{image_filename_col}}`  ·  absolute-path column (if any): `{{image_filepath_col}}`
- Image directory (for filename → path): `{{image_dir}}`

## Generated helpers
Read: `{{helpers_path}}`
## Predefined API
Read `MODULES_SIGNATURES` in `{{semdb_dir}}/vadar/predefined.py`.

## Output contract (so the evaluator can score it)
- Write a CSV whose header columns are EXACTLY the query's SELECT list, in order.
- For an image-identity output column, write the image FILENAME (the evaluator takes its
  basename). e.g. mmqa logo joins emit `ID,uri` where `uri` = the image filename.

Write the program to `{{solve_path}}` with EXACTLY this shape (the orchestrator runs it as
`python3 {{solve_path}} <out.csv> --data-dir D --image-dir I --clip-model M`):
```python
import sys, os, csv, argparse
sys.path.insert(0, "{{semdb_dir}}")
import semvision, imagepatch, semextract
from vadar.predefined import classify, best_ocr_match, dominant_colors, verify_property, detect, score, read_text
# <paste the generated helper implementations here>

def _read(data_dir, name):
    with open(os.path.join(data_dir, name), newline="") as f:
        return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--image-dir")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    a = ap.parse_args()
    ctx = {"encoder": semvision.get_encoder(a.clip_model), "palette": None}
    P = lambda path: imagepatch.ImagePatch(path, ctx)
    # 1) read structured + image-manifest CSVs from a.data_dir
    # 2) closed value space(s) from the structured column(s) at runtime
    # 3) for each image row: path = semextract.resolve_image_path(row[<filename/filepath>], a.image_dir)
    #    field = <compose vision API + helpers over P(path)>
    # 4) relational join/filter → out_rows (tuples matching the SELECT columns)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([ ... ])   # SELECT column headers
        w.writerows(out_rows)

if __name__ == "__main__":
    main()
```
Do not add endpoint/API-key options or any LLM/VLM/network call to this generated program.
