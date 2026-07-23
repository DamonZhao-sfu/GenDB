#!/usr/bin/env python3
"""
evaluate.py — score a compiled SemDB query against SemBench ground truth and
append one telemetry+metrics row to a CSV.

Metric = a FAITHFUL replication of SemBench's mmqa evaluator
(`SemBench/src/scenario/mmqa/evaluation/evaluate.py`). Every mmqa query uses the
same membership-based precision/recall/F1 (`compute_metrics`), but the prediction
container and per-column normalization differ PER QUERY:

    q1  list of director        (.strip(' "').lower())
    q2  set of (ID, image_id[, color])   image_id = basename, %2e -> "."
    q3  list of title           (raw)
    q4  list of (genre, movie)  (comma-split movies, lowercased)
    q5  list of actor/_output   (lowercased, blanks skipped)
    q6  list of Airlines        (raw, empty-safe)
    q7  set of (Airlines, image_id)      image_id = basename, %2e -> "."

Sub-lettered ids collapse to their numeric handler (q3a..q3g -> q3), exactly like
SemBench (`int("3a"[:-1])`). Membership is whole-item exact match (list keeps
duplicates → q1/q3/q4/q5/q6 recall can exceed 1.0; q2/q7 dedup via set). Double
empty → F1 = 0.0 (NOT 1.0), matching `compute_metrics`.

Prediction columns are read BY NAME (our compiled outputs already emit SemBench's
column names: director / ID,uri / ID,uri,color / Airlines,uri), so `--pred-cols`
is accepted for backward-compat but ignored.

Usage
-----
Metrics + CSV row:
    python3 evaluate.py \
        --pred runs/mmqa-q2a/q2a_results.csv \
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


# ---------------------------------------------------------------------------
# SemBench mmqa metric — faithful port of compute_metrics + per-query handlers.
# ---------------------------------------------------------------------------

def score(results, ground_truth):
    """Port of mmqa `compute_metrics`: iterate `results` (list OR set), a hit is
    membership in `ground_truth`. precision = tp/len(results); recall = tp/len(gt);
    both-empty → 0.0. Returns the metric row (tp/fp/fn + P/R/F1)."""
    tp = sum(1 for item in results if item in ground_truth)
    fp = len(results) - tp
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / len(ground_truth) if len(ground_truth) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return dict(gt_count=len(ground_truth), pred_count=len(results),
                tp=tp, fp=fp, fn=len(ground_truth) - tp,
                precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4))


def _isna(v):
    """Mirror pandas' read_csv: a blank/missing cell reads as NaN. csv gives us '',
    so treat None/empty/whitespace as NaN (used for the pd.isna skips in q4/q5)."""
    return v is None or (isinstance(v, str) and v.strip() == "")


def _basename_image_id(v):
    """SemBench q2/q7: image_id = last path segment, with %2e decoded to '.'."""
    return str(v).split("/")[-1].replace("%2e", ".")


def _column_is_int(rows, col):
    """Mirror pandas' per-column int64 inference: True iff EVERY value in `col`
    parses as an int (so q2 IDs compare equal to the integer ground truth)."""
    vals = [r.get(col) for r in rows]
    return bool(vals) and all(re.fullmatch(r"-?\d+", (str(v).strip())) for v in vals)


def eval_q1(rows, fields, gt):
    results = [str(r.get("director", "")).strip(' "').lower() for r in rows]
    gold = {g.strip().lower() for g in gt}
    return results, gold


def eval_q2(rows, fields, gt):
    id_is_int = _column_is_int(rows, "ID")
    results = set()
    for r in rows:
        r = dict(r)
        if "uri" in r:
            r["image_id"] = r.pop("uri")          # BigQuery
        if "filename" in r:
            r["image_id"] = r.pop("filename")     # Palimpzest
        image_id = _basename_image_id(r["image_id"])
        rid = int(str(r["ID"]).strip()) if id_is_int else r["ID"]
        ncols = len(r)
        if ncols == 2:
            results.add((rid, image_id))
        elif ncols == 3:
            results.add((rid, image_id, str(r["color"]).strip().lower()))
        else:
            raise ValueError(f"Unexpected number of columns: {ncols} in the results.")
    gold = set(tuple(g) for g in gt)
    return results, gold


def eval_q3(rows, fields, gt):
    results = [r["title"] for r in rows]
    return results, set(gt)


def eval_q4(rows, fields, gt):
    if not rows or "genre" not in (fields or []):
        return [], set()
    results = []
    for r in rows:
        genre = r.get("genre")
        if _isna(genre):
            continue
        genre = str(genre).strip().lower()
        movies = r.get("movies_in_genre")
        if _isna(movies):
            continue
        for movie in str(movies).split(","):
            results.append((genre, movie.strip().lower()))
    gold = set()
    for genre, movies in gt.items():                # gt is a dict here
        for m in movies:
            gold.add((genre.strip().lower(), m.strip().lower()))
    return results, gold


def eval_q5(rows, fields, gt):
    results = []
    for r in rows:
        if "_output" in r:
            value = r["_output"]
        elif "actor" in r:
            value = r["actor"]
        else:
            raise ValueError("Expected either '_output' or 'actor' column in the results.")
        if _isna(value) or not isinstance(value, str):
            continue
        results.append(value.strip().lower())
    gold = {g.strip().lower() for g in gt}
    return results, gold


def eval_q6(rows, fields, gt):
    if not rows or "Airlines" not in (fields or []):
        results = []
    else:
        results = [r["Airlines"] for r in rows]
    return results, set(gt)


def eval_q7(rows, fields, gt):
    results = set()
    for r in rows:
        r = dict(r)
        if "uri" in r:
            r["image_id"] = r.pop("uri")
        if "filename" in r:
            r["image_id"] = r.pop("filename")
        image_id = _basename_image_id(r["image_id"])
        results.add((r["Airlines"], image_id))
    gold = set(tuple(g) for g in gt)
    return results, gold


MMQA_HANDLERS = {1: eval_q1, 2: eval_q2, 3: eval_q3, 4: eval_q4,
                 5: eval_q5, 6: eval_q6, 7: eval_q7}


def score_pair(results, gold):
    """Metric row from a normalized (results, gold) pair — same as score()."""
    return score(results, gold)


def diff_pair(results, gold, cap):
    """Concrete FP/FN items behind the P/R/F1. FP = predicted items not in gold;
    FN = gold items not predicted. Both capped at `cap` (order-stable, de-duplicated
    for FP over a list so a repeated wrong item is shown once). Renders each item as
    a JSON-safe value (tuples -> lists)."""
    def _jsonable(x):
        return list(x) if isinstance(x, tuple) else x
    seen = set()
    fp = []
    for item in results:
        if item in gold or item in seen:
            continue
        seen.add(item)
        fp.append(item)
    pred_set = set(results)
    fn = [g for g in gold if g not in pred_set]
    return {
        "false_positives": [_jsonable(x) for x in fp[:cap]],
        "false_negatives": [_jsonable(x) for x in fn[:cap]],
        "fp_total": len(fp),
        "fn_total": len(fn),
        "sampled": len(fp) > cap or len(fn) > cap,
    }


def handler_id(query):
    """q3a -> 3, q10 -> 10 (SemBench: int(query_id[:-1]) when a letter trails)."""
    m = re.match(r"(\d+)", str(query).lower().lstrip("q"))
    if not m:
        raise ValueError(f"cannot derive a numeric mmqa handler from query {query!r}")
    return int(m.group(1))


def load_pred_rows(path):
    """Read the compiled output as (rows, fieldnames). rows = list of dict keyed by
    column name — the shape SemBench's per-query handlers expect."""
    if path.endswith(".json"):
        data = json.load(open(path))
        if data and not isinstance(data[0], dict):
            raise ValueError(
                "JSON predictions must be a list of OBJECTS with named columns — the mmqa "
                "handlers read columns by name (e.g. 'director'/'Airlines'), not by position.")
        rows = [dict(r) for r in data]
        fields = list(rows[0].keys()) if rows else []
        return rows, fields
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        return rows, list(reader.fieldnames or [])


def eval_mmqa(query, pred_path, gt_path):
    """Dispatch to the mmqa per-query handler and return its metric row."""
    hid = handler_id(query)
    if hid not in MMQA_HANDLERS:
        raise ValueError(f"no mmqa handler for query {query!r} (id {hid})")
    rows, fields = load_pred_rows(pred_path)
    gt = json.load(open(gt_path)).get("ground_truth")
    results, gold = MMQA_HANDLERS[hid](rows, fields, gt)
    return score_pair(results, gold)


def eval_mmqa_diff(query, pred_path, gt_path, cap):
    """Same dispatch as eval_mmqa, but return the FP/FN diff instead of the metric row."""
    hid = handler_id(query)
    if hid not in MMQA_HANDLERS:
        raise ValueError(f"no mmqa handler for query {query!r} (id {hid})")
    rows, fields = load_pred_rows(pred_path)
    gt = json.load(open(gt_path)).get("ground_truth")
    results, gold = MMQA_HANDLERS[hid](rows, fields, gt)
    return diff_pair(results, gold, cap)


def resolve_gt_file(gt_dir, query):
    """SemBench names files like Q2a.json; our query ids are q2a. Try variants."""
    q = query.lstrip("qQ")
    for name in (f"{query}.json", f"Q{q}.json", f"q{q}.json",
                 f"{query.upper()}.json", f"{query.capitalize()}.json"):
        p = os.path.join(gt_dir, name)
        if os.path.exists(p):
            return p
    return None


def model_of(tele, phase):
    for p in tele.get("phases", []):
        if p.get("phase") == phase:
            return p.get("model", "")
    return ""


CSV_COLS = [
    "query", "benchmark", "provider", "designer_model", "extractor_model", "codegen_model",
    "wall_clock_ms", "agent_stage_ms", "code_execution_ms",
    "total_estimated_cost_usd", "total_agent_tokens",
    "agent_calls", "extraction_calls", "residual_calls", "total_llm_calls",
    "naive_llm_calls", "compiled_execution_calls", "call_reduction",
    "gt_count", "pred_count", "tp", "fp", "fn", "precision", "recall", "f1",
    # non-F1 metric families (blank unless that metric applies)
    "metric", "relative_error", "mape", "spearman", "kendall", "ari", "covered",
]


def eval_scenario(benchmark, query, pred_path, gt_dir, sf):
    """Score a non-mmqa scenario via scenario_metrics (faithful SemBench port;
    needs pandas/sklearn/scipy → run evaluate.py under the sembench conda env)."""
    import importlib
    try:
        sm = importlib.import_module("scenario_metrics")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"[eval] scenario '{benchmark}' needs scenario_metrics (pandas/sklearn/scipy). "
            f"Run evaluate.py under the sembench conda env. ({e})")
    r = sm.score(benchmark, query, pred_path, gt_dir, int(sf) if sf else None)
    keep = ("precision", "recall", "f1", "tp", "fp", "fn", "gt_count", "pred_count",
            "relative_error", "mape", "spearman", "kendall", "ari", "covered")
    row = {"metric": r.get("metric", "")}
    for k in keep:
        if r.get(k) is not None:
            row[k] = round(r[k], 4) if isinstance(r[k], float) else r[k]
    return row, r


def _scenario_diff(pred_path, gt_dir, bench, query, sf, cap):
    """Generic id-set FP/FN for a non-mmqa scenario. Reads the predicted CSV and the
    scenario GT CSV, diffs their shared id column ("id" if present, else the first
    column). For non-membership metrics (aggregation/ranking) the id sets may be
    trivial — then fp/fn are empty and pred_sample/gt_sample carry a few raw rows so
    the agent still sees the shape. Faithful scoring stays in scenario_metrics."""
    import csv as _csv, importlib, re as _re
    sm = importlib.import_module("scenario_metrics")
    qid = int(_re.match(r"(\d+)", query.lstrip("qQ")).group(1))     # q3a -> 3
    gt_path = str(sm._gt_path(bench, qid, gt_dir, int(sf) if sf else None))

    def _rows(p):
        try:
            with open(p, newline="") as f:
                return list(_csv.DictReader(f))
        except FileNotFoundError:
            return []

    pred_rows, gt_rows = _rows(pred_path), _rows(gt_path)
    def _idcol(rows):
        if not rows:
            return None
        keys = list(rows[0].keys())
        return "id" if "id" in keys else keys[0]
    pc, gc = _idcol(pred_rows), _idcol(gt_rows)
    if pc and gc:
        pred_ids = [str(r[pc]).strip() for r in pred_rows]
        gt_ids = {str(r[gc]).strip() for r in gt_rows}
        pred_set = set(pred_ids)
        fp = [i for i in dict.fromkeys(pred_ids) if i not in gt_ids]
        fn = [i for i in (str(r[gc]).strip() for r in gt_rows) if i not in pred_set]
        return {"false_positives": fp[:cap], "false_negatives": fn[:cap],
                "fp_total": len(fp), "fn_total": len(fn),
                "sampled": len(fp) > cap or len(fn) > cap}
    return {"false_positives": [], "false_negatives": [], "fp_total": 0, "fn_total": 0,
            "sampled": False, "note": "no shared id column",
            "pred_sample": pred_rows[:cap], "gt_sample": gt_rows[:cap]}


def main():
    ap = argparse.ArgumentParser(description="Score a compiled mmqa query vs ground truth; append to CSV")
    ap.add_argument("--pred", help="compiled query output (.csv or .json). Omit for telemetry-only row.")
    ap.add_argument("--pred-cols", help="(ignored) mmqa handlers read prediction columns by name")
    ap.add_argument("--ground-truth", help="ground-truth JSON file")
    ap.add_argument("--ground-truth-dir", help="ground-truth dir; resolves <query>.json")
    ap.add_argument("--query", default="", help="query id (mmqa q1..q7 sub-letters; else Q1..Qn)")
    ap.add_argument("--benchmark", default="", help="scenario: mmqa (stdlib) | movie|cars|medical|animals|ecomm (scenario_metrics)")
    ap.add_argument("--sf", default="", help="scale factor (cars/medical GT suffix resolution)")
    ap.add_argument("--telemetry", help="telemetry.json from the orchestrator")
    ap.add_argument("--csv", required=True, help="output CSV (row appended; header written if new)")
    ap.add_argument("--emit-diff", help="also write FP/FN sample rows JSON to this path")
    ap.add_argument("--diff-cap", type=int, default=15, help="max FP and FN samples to emit")
    args = ap.parse_args()

    tele = json.load(open(args.telemetry)) if args.telemetry and os.path.exists(args.telemetry) else {}
    llm = tele.get("llm_calls", {}) if isinstance(tele.get("llm_calls"), dict) else {}

    row = {c: "" for c in CSV_COLS}
    row.update(
        query=args.query or tele.get("query", ""),
        benchmark=args.benchmark or tele.get("benchmark", ""),
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
        naive_llm_calls=tele.get("naive_llm_calls", ""),
        compiled_execution_calls=tele.get("compiled_execution_calls", ""),
        call_reduction=tele.get("call_reduction", ""),
    )

    gt_path = args.ground_truth or (
        resolve_gt_file(args.ground_truth_dir, args.query)
        if args.ground_truth_dir and args.query else None)

    bench = (args.benchmark or "").lower()
    if args.pred and args.query and bench and bench != "mmqa":
        # Non-mmqa scenario → faithful SemBench-parity scoring from the GT dir.
        gt_dir = args.ground_truth_dir or (os.path.dirname(gt_path) if gt_path else None)
        if not gt_dir:
            print("[eval] need --ground-truth-dir for non-mmqa scoring; writing telemetry-only row.")
        else:
            metrics, raw = eval_scenario(bench, args.query, args.pred, gt_dir, args.sf)
            row.update(**metrics)
            if args.emit_diff:
                diff = _scenario_diff(args.pred, gt_dir, bench, args.query, args.sf, args.diff_cap)
                diff["query"] = args.query
                diff.update(f1=metrics.get("f1"), precision=metrics.get("precision"),
                            recall=metrics.get("recall"))
                json.dump(diff, open(args.emit_diff, "w"), indent=2)
                print(f"[eval] wrote FP/FN diff -> {args.emit_diff}")
            head = " ".join(f"{k}={v}" for k, v in metrics.items() if k != "metric")
            print(f"[eval] {bench}/{args.query} [{metrics.get('metric','')}]: {head}")
            if raw.get("audio_only"):
                print(f"[eval] NOTE: {args.query} is audio-only (unsupported extraction) — score is informational.")
            if args.telemetry and os.path.exists(args.telemetry):
                tele["metrics"] = metrics
                tele["ground_truth_dir"] = gt_dir
                json.dump(tele, open(args.telemetry, "w"), indent=2)
                print(f"[eval] wrote metrics into {args.telemetry}")
    elif args.pred and gt_path and args.query:
        metrics = eval_mmqa(args.query, args.pred, gt_path)
        row.update(**metrics)
        print(f"[eval] {args.query}: GT={metrics['gt_count']} pred={metrics['pred_count']}  "
              f"P={metrics['precision']} R={metrics['recall']} F1={metrics['f1']}  "
              f"(tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']})")
        print(f"[eval] ground truth: {gt_path}")
        if args.emit_diff:
            diff = eval_mmqa_diff(args.query, args.pred, gt_path, args.diff_cap)
            diff["query"] = args.query
            diff.update(f1=metrics["f1"], precision=metrics["precision"],
                        recall=metrics["recall"], tp=metrics["tp"],
                        fp=metrics["fp"], fn=metrics["fn"])
            json.dump(diff, open(args.emit_diff, "w"), indent=2)
            print(f"[eval] wrote FP/FN diff -> {args.emit_diff}")
        # Persist the metrics back into telemetry.json too (not just the CSV).
        if args.telemetry and os.path.exists(args.telemetry):
            tele["metrics"] = metrics
            tele["ground_truth_file"] = gt_path
            json.dump(tele, open(args.telemetry, "w"), indent=2)
            print(f"[eval] wrote metrics into {args.telemetry}")
    elif args.pred or args.ground_truth or args.ground_truth_dir:
        print("[eval] need --pred, a ground truth, AND --query to compute metrics; writing telemetry-only row.")
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
