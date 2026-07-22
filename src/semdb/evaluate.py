#!/usr/bin/env python3
"""
evaluate.py — score a compiled SemDB query against SemBench ground truth and
append one telemetry+metrics row to a CSV.

Ground truth (e.g. .../raw_results/ground_truth/Q2a.json):
    { "nl_question": "...", "modalities": ["table","image"],
      "ground_truth": [ [0, "117d....png"], [5, "117d....png"], ... ] }

The `ground_truth` list is the set of expected result rows (each a tuple of the
query's SELECT columns, e.g. [t.ID, i.uri]). We compare it to the compiled
query's predicted rows and report set-based precision / recall / F1. Cells are
normalized (filenames → basename, lowercased) so a full image path matches a
bare filename.

Usage
-----
Metrics + CSV row:
    python3 evaluate.py \
        --pred runs/mmqa-q2a/q2a_results.csv --pred-cols 0,1 \
        --ground-truth-dir /localhome/hza214/SemBench/files/mmqa/raw_results/ground_truth \
        --query q2a \
        --telemetry runs/mmqa-q2a/telemetry.json \
        --csv runs/results.csv

Telemetry-only row (no metrics):
    python3 evaluate.py --telemetry runs/mmqa-q2a/telemetry.json --csv runs/results.csv
"""

import argparse
import csv
import json
import os
import re

IMG_RE = re.compile(r"\.(png|jpe?g|gif|webp|bmp|tiff?)$", re.I)


def norm_cell(x):
    s = str(x).strip()
    if "/" in s or "\\" in s or IMG_RE.search(s):
        s = os.path.basename(s.replace("\\", "/"))
    return s.lower()


def to_tuple(item, cols=None):
    row = list(item) if isinstance(item, (list, tuple)) else [item]
    if cols:
        row = [row[i] for i in cols if i < len(row)]
    return tuple(norm_cell(c) for c in row)


def resolve_gt_file(gt_dir, query):
    """SemBench names files like Q2a.json; our query ids are q2a. Try variants."""
    q = query.lstrip("qQ")
    for name in (f"{query}.json", f"Q{q}.json", f"q{q}.json",
                 f"{query.upper()}.json", f"{query.capitalize()}.json"):
        p = os.path.join(gt_dir, name)
        if os.path.exists(p):
            return p
    return None


def load_ground_truth(path, cols=None):
    d = json.load(open(path))
    gt = d.get("ground_truth", d) if isinstance(d, dict) else d
    return set(to_tuple(x, cols) for x in gt)


def load_pred(path, cols, no_header):
    if path.endswith(".json"):
        data = json.load(open(path))
        out = set()
        for r in data:
            vals = list(r.values()) if isinstance(r, dict) else r
            out.add(to_tuple(vals, cols))
        return out
    rows = list(csv.reader(open(path)))
    if rows and not no_header:
        rows = rows[1:]                      # our writers emit a header row
    return set(to_tuple(r, cols) for r in rows if r)


def prf(pred, gold):
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if tp + fp else (1.0 if not gold else 0.0)
    r = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4))


def model_of(tele, phase):
    for p in tele.get("phases", []):
        if p.get("phase") == phase:
            return p.get("model", "")
    return ""


CSV_COLS = [
    "query", "provider", "designer_model", "extractor_model", "codegen_model",
    "wall_clock_ms", "agent_stage_ms", "code_execution_ms",
    "total_estimated_cost_usd", "total_agent_tokens",
    "agent_calls", "extraction_calls", "residual_calls", "total_llm_calls",
    "gt_count", "pred_count", "tp", "fp", "fn", "precision", "recall", "f1",
]


def main():
    ap = argparse.ArgumentParser(description="Score a compiled query vs ground truth; append to CSV")
    ap.add_argument("--pred", help="compiled query output (.csv or .json). Omit for telemetry-only row.")
    ap.add_argument("--pred-cols", help="comma-separated column indices to compare, in GT order (e.g. 0,1)")
    ap.add_argument("--pred-no-header", action="store_true", help="pred CSV has no header row")
    ap.add_argument("--ground-truth", help="ground-truth JSON file")
    ap.add_argument("--ground-truth-dir", help="ground-truth dir; resolves <query>.json")
    ap.add_argument("--query", default="")
    ap.add_argument("--telemetry", help="telemetry.json from the orchestrator")
    ap.add_argument("--csv", required=True, help="output CSV (row appended; header written if new)")
    args = ap.parse_args()

    cols = [int(c) for c in args.pred_cols.split(",")] if args.pred_cols else None
    tele = json.load(open(args.telemetry)) if args.telemetry and os.path.exists(args.telemetry) else {}
    llm = tele.get("llm_calls", {}) if isinstance(tele.get("llm_calls"), dict) else {}

    row = {c: "" for c in CSV_COLS}
    row.update(
        query=args.query or tele.get("query", ""),
        provider=tele.get("provider", ""),
        designer_model=model_of(tele, "schema_designer"),
        extractor_model=model_of(tele, "extractor"),
        codegen_model=model_of(tele, "code_generator"),
        wall_clock_ms=tele.get("wall_clock_ms", ""),
        agent_stage_ms=tele.get("agent_stage_ms", ""),
        code_execution_ms=tele.get("code_execution_ms", ""),
        total_estimated_cost_usd=tele.get("total_estimated_cost_usd", ""),
        total_agent_tokens=tele.get("total_agent_tokens", ""),
        agent_calls=llm.get("agent_stage", ""),
        extraction_calls=llm.get("extraction", ""),
        residual_calls=llm.get("residual", ""),
        total_llm_calls=llm.get("total", ""),
    )

    gt_path = args.ground_truth or (
        resolve_gt_file(args.ground_truth_dir, args.query)
        if args.ground_truth_dir and args.query else None)

    if args.pred and gt_path:
        gold = load_ground_truth(gt_path, cols)
        pred = load_pred(args.pred, cols, args.pred_no_header)
        m = prf(pred, gold)
        row.update(gt_count=len(gold), pred_count=len(pred), **m)
        print(f"[eval] {args.query}: GT={len(gold)} pred={len(pred)}  "
              f"P={m['precision']} R={m['recall']} F1={m['f1']}  (tp={m['tp']} fp={m['fp']} fn={m['fn']})")
        print(f"[eval] ground truth: {gt_path}")
    elif args.pred or args.ground_truth or args.ground_truth_dir:
        print("[eval] need BOTH --pred and a ground truth to compute metrics; writing telemetry-only row.")
    else:
        print("[eval] telemetry-only row (no --pred/ground truth).")

    new = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[eval] appended row -> {args.csv}")


if __name__ == "__main__":
    main()
