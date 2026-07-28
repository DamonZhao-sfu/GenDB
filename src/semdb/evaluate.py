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


_TEXT_RE = re.compile(r"text|description|overview|body|content|summary|symptoms|review|complaint|plot|display", re.I)


def _corpus_text_map(rows, id_col=None, text_col=None):
    """id -> a short text snippet, best-effort. id_col defaults to 'id' or the first
    column; text_col defaults to the first text-ish column or the last column."""
    if not rows:
        return {}
    keys = list(rows[0].keys())
    idc = id_col or ("id" if "id" in keys else keys[0])
    txc = text_col or next((k for k in keys if _TEXT_RE.search(k)), keys[-1])
    out = {}
    for r in rows:
        rid = str(r.get(idc, "")).strip()
        if rid:
            out[rid] = str(r.get(txc, "")).strip()
    return out


def _norm(v):
    """Match evaluate.py membership normalization: str, stripped, lowercased."""
    return None if v is None else str(v).strip().lower()


TRUEISH = {"true", "yes", "1", "t", "y"}


def _is_positive(value):
    return _norm(value) in TRUEISH


def _rate(num, den):
    return round(num / den, 4) if den else None


def weighted_quality(labels, pred_rows, weights):
    """Horvitz-Thompson estimates of corpus accuracy and P/R/F1 from a weighted sample.

    Each labeled row i carries w_i = 1/pi_i, the number of corpus rows it stands for,
    so summing w_i over the rows with some property estimates how many corpus rows
    have it. Under an unequal-probability design the UNWEIGHTED mean is not an
    estimate of anything about the corpus: a validation set that deliberately
    oversamples the score-dense region (which is the only way a 2%-positive predicate
    becomes measurable at all) would otherwise report a number dominated by the
    oversampled slice.

    Precision/recall matter more than accuracy at these base rates. With 2% positives
    a program that answers `false` everywhere scores 98% accuracy and 0 recall, and
    only the second number tells the refinement loop anything.

    Equal weights reduce this to the plain unweighted counts, so a val file without a
    `weights` field keeps behaving exactly as before.
    """
    tp = fp = fn = tn = 0.0
    for rid, expected in labels.items():
        sid = str(rid)
        w = float(weights.get(sid, weights.get(rid, 1.0)))
        pred = pred_rows.get(sid, pred_rows.get(rid, None))
        # An id the program never emitted is a negative prediction, not a skip: the
        # program was asked about it and said nothing.
        got, want = _is_positive(pred), _is_positive(expected)
        if got and want:
            tp += w
        elif got and not want:
            fp += w
        elif want:
            fn += w
        else:
            tn += w
    precision, recall = _rate(tp, tp + fp), _rate(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    return {
        "tp": round(tp, 2), "fp": round(fp, 2), "fn": round(fn, 2), "tn": round(tn, 2),
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": _rate(tp + tn, tp + tn + fp + fn),
        "positive_rate": _rate(tp + fn, tp + tn + fp + fn),
    }


def score_inference(trace, val, corpus_rows, cap, id_col=None, text_col=None):
    """Per-row inference accuracy of a DIRECT solver's trace_<q>.json against a
    labeled val.json. Compares trace.rows[id] to val.labels[id] over the LABELED
    ids only (normalized). A labeled id absent from the trace counts as wrong with
    predicted=None.

    Returns the unweighted accuracy (what the refinement loop has always read) plus,
    under `estimates`, Horvitz-Thompson corpus estimates derived from val["weights"].
    Both are reported because they answer different questions: the unweighted counts
    describe the rows the agent can actually look at, while the weighted numbers are
    the only ones that describe the corpus when the design is unequal-probability.
    """
    attr = val.get("attr", trace.get("attr", ""))
    labels = val.get("labels", {})
    pred_rows = trace.get("rows", {}) if isinstance(trace.get("rows"), dict) else {}
    text_map = _corpus_text_map(corpus_rows, id_col, text_col)
    correct = 0
    mistakes = []
    for rid, expected in labels.items():
        sid = str(rid)
        raw_pred = pred_rows.get(sid, pred_rows.get(rid, None))
        if raw_pred is not None and _norm(raw_pred) == _norm(expected):
            correct += 1
        else:
            snippet = text_map.get(sid, "")[:200]
            mistakes.append({"id": sid, "text": snippet,
                             "predicted": None if raw_pred is None else str(raw_pred),
                             "expected": str(expected)})
    n = len(labels)

    weights = val.get("weights") or {}
    design = val.get("design") or {}
    out = {
        "query": val.get("query", trace.get("query", "")),
        "attr": attr, "n": n, "correct": correct,
        "accuracy": _rate(correct, n),
        "mistakes": mistakes[:cap], "n_mistakes": len(mistakes),
        "sampled": len(mistakes) > cap,
        "unweighted": weighted_quality(labels, pred_rows, {}),
    }
    if weights:
        out["estimates"] = weighted_quality(labels, pred_rows, weights)
        spread = max(weights.values()) / min(weights.values()) if weights else 1.0
        out["design"] = {"method": design.get("method"), "N": design.get("N"),
                         "weight_spread": round(spread, 2),
                         "weighted": spread > 1.0 + 1e-9}
    return out


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


def direct_telemetry(tele):
    """Flatten DIRECT-mode phase telemetry into the run-level CSV fields."""
    phases = tele.get("phases", [])
    direct = tele.get("direct", {}) if isinstance(tele.get("direct"), dict) else {}
    timing = direct.get("timing_breakdown_ms", {})
    timing = timing if isinstance(timing, dict) else {}
    wall_ms = tele.get("wall_clock_ms", "")
    agent_ms = timing.get(
        "agent_stage_ms", direct.get("agent_stage_ms", tele.get("agent_stage_ms", "")))
    validation_ms = timing.get("validation_sampling_llm_ms", "")
    code_ms = timing.get(
        "code_execution_ms", direct.get("code_execution_ms", tele.get("code_execution_ms", "")))
    if code_ms == "" and isinstance(wall_ms, (int, float)) and isinstance(agent_ms, (int, float)):
        # Backward compatibility for telemetry written before DIRECT stages were timed
        # independently. This residual is not exact code execution because old runs
        # also included validation construction, scoring, preflight, and artifact I/O.
        code_ms = wall_ms - agent_ms

    input_tokens = sum((p.get("tokens") or {}).get("input", 0) or 0 for p in phases)
    output_tokens = sum((p.get("tokens") or {}).get("output", 0) or 0 for p in phases)
    models = list(dict.fromkeys(p.get("model", "") for p in phases if p.get("model")))
    calls = direct.get("agent_calls", "")
    return {
        "codegen_model": model_of(tele, "code_generator")
            or model_of(tele, "vadar_solver") or "+".join(models),
        "agent_stage_ms": agent_ms,
        "validation_sampling_llm_ms": validation_ms,
        "code_execution_ms": code_ms,
        "total_agent_tokens": input_tokens + output_tokens,
        "agent_input_tokens": input_tokens,
        "agent_output_tokens": output_tokens,
        "agent_calls": calls,
        "extraction_calls": 0 if direct else "",
        "residual_calls": 0 if direct else "",
        "total_llm_calls": calls,
    }


F1_HISTORY_COLS = [f"val_f1_iter_{i}" for i in range(6)]
CODE_EXECUTION_HISTORY_COLS = [f"code_execution_ms_iter_{i}" for i in range(6)]

CSV_COLS = [
    "query", "benchmark", "provider", "designer_model", "extractor_model", "codegen_model",
    "wall_clock_ms", "agent_stage_ms", "validation_sampling_llm_ms", "code_execution_ms",
    *CODE_EXECUTION_HISTORY_COLS, "code_execution_ms_final", "code_execution_runs",
    "total_estimated_cost_usd", "total_agent_tokens",
    "agent_input_tokens", "agent_output_tokens",
    "agent_calls", "extraction_calls", "residual_calls", "total_llm_calls",
    "naive_llm_calls", "compiled_execution_calls", "call_reduction",
    "val_n", "refine_iterations", "best_iteration", "max_iterations",
    *F1_HISTORY_COLS, "val_f1_history",
    "gt_count", "pred_count", "tp", "fp", "fn", "precision", "recall", "f1",
    # non-F1 metric families (blank unless that metric applies)
    "metric", "relative_error", "mape", "spearman", "kendall", "ari", "covered",
]


def telemetry_row(tele, query="", benchmark=""):
    """Flatten one telemetry.json into the stable results.csv schema.

    `f1` remains the final full-corpus score. `val_f1_iter_N` records the validation
    signal used by the refinement loop, and `val_f1_history` preserves every value
    even when a run uses more than the six conventional iter_0..iter_5 slots.
    """
    tele = tele if isinstance(tele, dict) else {}
    llm = tele.get("llm_calls", {}) if isinstance(tele.get("llm_calls"), dict) else {}
    direct = direct_telemetry(tele)
    refine = tele.get("refine", {}) if isinstance(tele.get("refine"), dict) else {}
    history = refine.get("f1_history", [])
    history = history if isinstance(history, list) else []
    direct_block = tele.get("direct", {}) if isinstance(tele.get("direct"), dict) else {}
    execution_runs = direct_block.get("code_execution_runs")
    execution_runs = execution_runs if isinstance(execution_runs, list) else []

    row = {c: "" for c in CSV_COLS}
    row.update(
        query=query or tele.get("query", ""),
        benchmark=benchmark or tele.get("benchmark", ""),
        provider=tele.get("provider", ""),
        designer_model=model_of(tele, "schema_designer"),
        extractor_model=model_of(tele, "extractor"),
        codegen_model=direct["codegen_model"],
        wall_clock_ms=tele.get("wall_clock_ms", ""),
        agent_stage_ms=direct["agent_stage_ms"],
        validation_sampling_llm_ms=direct["validation_sampling_llm_ms"],
        code_execution_ms=direct["code_execution_ms"],
        total_estimated_cost_usd=tele.get("total_estimated_cost_usd", ""),
        total_agent_tokens=tele.get("total_agent_tokens", direct["total_agent_tokens"]),
        agent_input_tokens=direct["agent_input_tokens"],
        agent_output_tokens=direct["agent_output_tokens"],
        agent_calls=llm.get("agent_stage", direct["agent_calls"]),
        extraction_calls=llm.get("extraction", direct["extraction_calls"]),
        residual_calls=llm.get("residual", direct["residual_calls"]),
        total_llm_calls=llm.get("total", direct["total_llm_calls"]),
        naive_llm_calls=tele.get("naive_llm_calls", ""),
        compiled_execution_calls=tele.get("compiled_execution_calls", ""),
        call_reduction=tele.get("call_reduction", ""),
        val_n=refine.get("val_n", ""),
        refine_iterations=refine.get("iterations", ""),
        best_iteration=refine.get("best_iteration", ""),
        max_iterations=refine.get("max_iterations", ""),
        val_f1_history=json.dumps(
            [h.get("f1") if isinstance(h, dict) else None for h in history],
            separators=(",", ":")),
    )
    if execution_runs:
        row["code_execution_runs"] = json.dumps(execution_runs, separators=(",", ":"))
        for run in execution_runs:
            if not isinstance(run, dict) or run.get("duration_ms") is None:
                continue
            iteration = run.get("iteration")
            try:
                iteration = int(iteration)
            except (TypeError, ValueError):
                iteration = None
            column = f"code_execution_ms_iter_{iteration}"
            if iteration is not None and column in row:
                row[column] = run["duration_ms"]
            elif run.get("scope") == "final_full_corpus":
                row["code_execution_ms_final"] = run["duration_ms"]
    for item in history:
        if not isinstance(item, dict):
            continue
        try:
            iteration = int(item.get("iter"))
        except (TypeError, ValueError):
            continue
        column = f"val_f1_iter_{iteration}"
        if column in row:
            row[column] = item.get("f1", "")

    metrics = tele.get("metrics")
    if isinstance(metrics, dict):
        for column in (
            "gt_count", "pred_count", "tp", "fp", "fn", "precision", "recall", "f1",
            "metric", "relative_error", "mape", "spearman", "kendall", "ari", "covered",
        ):
            if metrics.get(column) is not None:
                row[column] = metrics[column]
    return row


def upsert_result_row(path, row):
    """Write one authoritative row per (benchmark, query), migrating old headers.

    A query rerun updates its telemetry.json, so appending another CSV row leaves a
    stale duplicate. Rewriting also upgrades older results.csv files that predate the
    iteration-history columns.
    """
    existing = []
    if os.path.exists(path) and os.path.getsize(path):
        with open(path, newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    key = (str(row.get("benchmark", "")), str(row.get("query", "")))
    kept = [
        {column: old.get(column, "") for column in CSV_COLS}
        for old in existing
        if (str(old.get("benchmark", "")), str(old.get("query", ""))) != key
    ]
    kept.append({column: row.get(column, "") for column in CSV_COLS})
    temporary = path + ".tmp"
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLS)
        writer.writeheader()
        writer.writerows(kept)
    os.replace(temporary, path)


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
    ap.add_argument("--csv", help="output CSV (row appended; header written if new)")
    ap.add_argument("--score-inference", action="store_true",
                    help="per-row inference scoring mode: score --trace vs --val-file")
    ap.add_argument("--trace", help="trace_<q>.json from a DIRECT solver (id -> inferred attr value)")
    ap.add_argument("--val-file", help="hand-labeled val.json (id -> expected attr value)")
    ap.add_argument("--corpus-csv", help="text corpus CSV, for mistake text snippets")
    ap.add_argument("--id-col", help="corpus id column (default: 'id' or first column)")
    ap.add_argument("--text-col", help="corpus text column (default: first text-ish or last column)")
    ap.add_argument("--emit-diff", help="also write FP/FN sample rows JSON to this path")
    ap.add_argument("--diff-cap", type=int, default=15, help="max FP and FN samples to emit")
    args = ap.parse_args()

    if args.score_inference:
        if not (args.trace and args.val_file):
            ap.error("--score-inference requires --trace and --val-file")
        trace = json.load(open(args.trace)) if os.path.exists(args.trace) else {"rows": {}}
        val = json.load(open(args.val_file))
        corpus_rows = None
        if args.corpus_csv and os.path.exists(args.corpus_csv):
            corpus_rows, _ = load_pred_rows(args.corpus_csv)
        out = score_inference(trace, val, corpus_rows, args.diff_cap, args.id_col, args.text_col)
        if args.emit_diff:
            json.dump(out, open(args.emit_diff, "w"), indent=2)
        print(f"[eval] inference accuracy {out['accuracy']} "
              f"({out['correct']}/{out['n']}, {out['n_mistakes']} wrong) -> {args.emit_diff}")
        return
    if not args.csv:
        ap.error("--csv is required unless --score-inference is set")

    tele = json.load(open(args.telemetry)) if args.telemetry and os.path.exists(args.telemetry) else {}
    row = telemetry_row(tele, query=args.query, benchmark=args.benchmark)

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
                # A diff-generation failure must NOT prevent metric persistence / the CSV row.
                try:
                    diff = _scenario_diff(args.pred, gt_dir, bench, args.query, args.sf, args.diff_cap)
                    diff["query"] = args.query
                    diff.update(f1=metrics.get("f1"), precision=metrics.get("precision"),
                                recall=metrics.get("recall"))
                    json.dump(diff, open(args.emit_diff, "w"), indent=2)
                    print(f"[eval] wrote FP/FN diff -> {args.emit_diff}")
                except Exception as e:  # noqa: BLE001
                    print(f"[eval] WARN: --emit-diff failed ({e}); metrics still written.")
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
            # A diff-generation failure must NOT prevent metric persistence / the CSV row.
            try:
                diff = eval_mmqa_diff(args.query, args.pred, gt_path, args.diff_cap)
                diff["query"] = args.query
                diff.update(f1=metrics["f1"], precision=metrics["precision"],
                            recall=metrics["recall"], tp=metrics["tp"],
                            fp=metrics["fp"], fn=metrics["fn"])
                json.dump(diff, open(args.emit_diff, "w"), indent=2)
                print(f"[eval] wrote FP/FN diff -> {args.emit_diff}")
            except Exception as e:  # noqa: BLE001
                print(f"[eval] WARN: --emit-diff failed ({e}); metrics still written.")
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

    upsert_result_row(args.csv, row)
    print(f"[eval] updated row -> {args.csv}")


if __name__ == "__main__":
    main()
