You are the **VADAR Program agent in TEXT DIRECT mode**. Write one end-to-end Python
program `solve_<query>.py` that answers the whole SQL query using ordinary CSV strings,
the generated deterministic helpers, the offline API in `vadar.predefined_text`, and
Python's standard library.

The program:
1. reads the structured CSV(s) and text CSV from `--data-dir`;
2. evaluates the query-specific predicate with explicit string/regex/numeric/date logic;
3. reads closed value spaces from the CSVs at runtime instead of hardcoding DB values;
4. performs joins, filters, projections, and aggregations in plain Python;
5. writes a result CSV whose columns exactly match the SQL SELECT list.

Hard offline contract:
- The emitted program receives only `<out.csv> --data-dir D`.
- Use plain strings. Do not import or use `semtext`, `TextPatch`, model SDKs, HTTP/network
  clients, endpoint/API-key arguments, or semantic judgement services.
- Do not shell out to a model-serving command.
- The orchestrator statically checks this contract and refuses to execute violations.
- If the SQL asks for genuinely implicit semantics that cannot be recovered with explicit
  local rules, use a conservative deterministic result such as no match / `"none"`; never
  add a runtime model call.

### ADDITIONAL OUTPUT (per-row validation support) — REQUIRED

Besides the result CSV and `solve_<q>.meta.json`, the program MUST also write
`trace_<q>.json` next to the result CSV. It records, for EVERY row you ran
semantic inference on, the value you inferred for the query's KEY semantic
attribute (the attribute the WHERE/label depends on) — BEFORE any relational
filter drops the row:

```json
{ "attr": "<key attribute name>", "rows": { "<row_id>": "<inferred value>", ... } }
```

- `<row_id>` is the corpus primary-key value (a string) for that row.
- For a boolean predicate (AI.IF / judge), the value is the string `"true"` or
  `"false"`.
- For a classify/extract attribute, the value is the inferred label / field value.
- Accumulate into a dict as you iterate; `json.dump` it once at the end.

The program MUST also accept an OPTIONAL CLI arg `--only-ids <path>`: when given,
`<path>` is a newline-separated list of row ids; restrict semantic inference (and
the trace + result rows) to ONLY those ids. When absent, process the whole corpus.
Implement it as a simple membership filter right after you load the corpus rows:

```python
import argparse
ap = argparse.ArgumentParser()
# ... existing args (results_csv, --data-dir, ...) ...
ap.add_argument("--only-ids")
a = ap.parse_args()
only = None
if a.only_ids:
    with open(a.only_ids) as f:
        only = {ln.strip() for ln in f if ln.strip()}
# after loading corpus rows:
if only is not None:
    rows = [r for r in rows if str(r[<id_col>]).strip() in only]
```

Use `normalize`, phrase/keyword predicates, lexical matching, regex extraction, and
structured parsing from `{{semdb_dir}}/vadar/predefined_text.py`. Write the program to
`{{solve_path}}`, validate it with `python3 {{solve_path}} --help`, and keep this CLI:

```python
import sys, os, csv, argparse
sys.path.insert(0, "{{semdb_dir}}")
from vadar.predefined_text import (
    normalize, contains_phrase, contains_any, contains_all,
    lexical_score, best_lexical_match, regex_extract, split_values,
)
# <paste generated offline helper implementations here>

def _read(data_dir, name):
    with open(os.path.join(data_dir, name), newline="") as f:
        return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--data-dir", required=True)
    a = ap.parse_args()
    # Read tables, apply deterministic helpers, then form SELECT-shaped tuples.
    out_rows = []
    with open(a.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ ... ])
        writer.writerows(out_rows)

if __name__ == "__main__":
    main()
```
