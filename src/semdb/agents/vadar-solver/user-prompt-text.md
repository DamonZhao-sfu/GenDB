# Write the end-to-end TEXT solver `solve_{{query_id}}.py` for query `{{query_id}}`

## Query
```sql
{{query_sql}}
```
Natural language: {{query_nl}}

## Tables (read these from `--data-dir` by filename)
{{tables_doc}}

## Generated helpers
Read: `{{helpers_path}}`
## Predefined TEXT API
Read `MODULES_SIGNATURES_TEXT` in `{{semdb_dir}}/vadar/predefined_text.py`.

## Output contract (so the evaluator can score it)
- Write a CSV whose header columns are EXACTLY the query's SELECT list, in order.
- Read any closed value space (e.g. the set of genres) from the structured CSV column AT
  RUNTIME — do NOT hardcode it.

Write the program to `{{solve_path}}` with EXACTLY this shape (the orchestrator runs it as
`python3 {{solve_path}} <out.csv> --data-dir D --endpoint URL --model M --api-key K`):
```python
import sys, os, csv, argparse
sys.path.insert(0, "{{semdb_dir}}")
import semtext
from vadar.predefined_text import judge, classify, extract, generate, score
# <paste the generated helper implementations here>

def _read(data_dir, name):
    with open(os.path.join(data_dir, name), newline="") as f:
        return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    a = ap.parse_args()
    ctx = semtext.get_ctx(a.model, a.endpoint, a.api_key)
    P = lambda t: semtext.TextPatch(t, ctx)
    # 1) read structured + text CSVs from a.data_dir
    # 2) closed value space(s) from the structured column(s) at runtime
    # 3) for each text row: field = <compose text API + helpers over P(row[<text_col>])>
    # 4) relational join/filter/aggregate -> out_rows (tuples matching the SELECT columns)
    out_rows = []
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([ ... ])   # SELECT column headers
        w.writerows(out_rows)
    import json
    json.dump({"elapsed_sec": 0, "rows": len(out_rows), "llm_calls": semtext.METER.calls},
              open(a.out.replace(".csv", ".meta.json"), "w"))

if __name__ == "__main__":
    main()
```
Do NOT import `imagepatch`, `semvision`, or `vadar.predefined`; do NOT use `--clip-model`
or `--image-dir`. This is a TEXT solver — inference goes through `semtext.TextPatch` only.
