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
    # SemBench's retrieval scorer defines the no-positive-prediction and
    # double-empty cases as F1=0.0.  Keep precision/recall nullable for diagnostic
    # honesty, but never turn a valid zero-quality query objective into "no signal".
    f1 = 0.0
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    return {
        "tp": round(tp, 2), "fp": round(fp, 2), "fn": round(fn, 2), "tn": round(tn, 2),
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": _rate(tp + tn, tp + tn + fp + fn),
        "positive_rate": _rate(tp + fn, tp + tn + fp + fn),
    }


AGGREGATION_REFINEMENT = {
    ("movie", 3): ("count", None),
    ("movie", 4): ("ratio", None),
    ("animals", 1): ("count", None),
    ("animals", 2): ("count", None),
    ("cars", 4): ("mean_age_2026", "year"),
    ("medical", 4): ("mean", "age"),
}
EXACT_SINGLE_RESULT_RETRIEVAL = {
    ("animals", 3): ("city",),
    ("animals", 4): ("city",),
    ("animals", 10): ("city", "stationid"),
}
ARI_REFINEMENT = {("ecomm", q) for q in (3, 4, 5, 6)}
MACRO_F1_REFINEMENT = {("cars", 10), ("medical", 10)}
RANKING_REFINEMENT = {("movie", 9), ("movie", 10)}
# Queries whose official SemBench metric is retrieval/F1 and whose SELECT
# validation unit provides the corresponding membership decisions.  Keep this
# registry explicit: falling through to a generic predicate score must never make
# an unknown or unsupported query look as though it has an official F1 objective.
F1_REFINEMENT = (
    {("mmqa", q) for q in range(1, 8)}
    | {("movie", q) for q in (1, 2, 5, 6, 7)}
    | {("animals", q) for q in range(5, 10)}
    | {("cars", q) for q in (1, 2, 3, 6, 7, 8, 9)}
    | {("medical", q) for q in (1, 2, 3, 6, 7, 8, 9)}
    | {("ecomm", q) for q in (1, 2, 7, 8, 9, 10, 11, 12, 13, 14)}
)
MULTISITE_QUERY_OBJECTIVE_UNAVAILABLE = {
    ("movie", 8), ("cars", 5), ("medical", 5),
}
UNSUPPORTED_QUERY_METRIC = {
    # The current SemBench MedicalEvaluator dispatches Q1-Q10 only.
    ("medical", 11),
}


def _query_number(query):
    match = re.match(r"(\d+)", str(query).lower().lstrip("q"))
    return int(match.group(1)) if match else None


def _objective(name, value, direction="maximize", **details):
    return {
        "name": name,
        "value": None if value is None else round(float(value), 6),
        "direction": direction,
        **({"details": details} if details else {}),
    }


def _corpus_index(corpus_rows, id_col=None):
    if not corpus_rows:
        return {}
    keys = list(corpus_rows[0].keys())
    idc = id_col or ("id" if "id" in keys else keys[0])
    return {
        str(row.get(idc, "")).strip(): row
        for row in corpus_rows
        if str(row.get(idc, "")).strip()
    }


def _column(row, wanted):
    if not row:
        return None
    actual = next((key for key in row if key.lower() == wanted.lower()), None)
    return row.get(actual) if actual else None


def _sample_weight(weights, rid):
    return float(weights.get(str(rid), weights.get(rid, 1.0)))


def _multi_label_set(value):
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        parts = value
    else:
        text = str(value).strip().strip("[]")
        parts = re.split(r"[,;|]", text)
    aliases = {
        "sci fi": "science fiction",
        "sci-fi": "science fiction",
        "biographical": "biography",
    }
    out = set()
    for part in parts:
        label = str(part).strip().strip("'\"").lower().replace("_", " ")
        label = " ".join(label.split())
        if label:
            out.add(aliases.get(label, label))
    return out


def _top_group(labels, pred_rows, weights, corpus_rows, group_columns, id_col):
    rows = _corpus_index(corpus_rows, id_col)

    def counts(values):
        totals = {}
        for rid in labels:
            value = values.get(str(rid), values.get(rid))
            if not _is_positive(value):
                continue
            row = rows.get(str(rid))
            if not row:
                continue
            key = tuple(str(_column(row, col)).strip().lower() for col in group_columns)
            totals[key] = totals.get(key, 0.0) + _sample_weight(weights, rid)
        return totals

    expected_counts = counts(labels)
    predicted_counts = counts(pred_rows)
    choose = lambda values: (sorted(values, key=lambda k: (-values[k], k))[0]
                             if values else None)
    return choose(expected_counts), choose(predicted_counts), expected_counts, predicted_counts


def _relative_error(predicted, expected):
    if expected == 0:
        return 0.0 if predicted == 0 else 1.0
    return abs(predicted - expected) / abs(expected)


def _aggregation_objective(kind, field, labels, pred_rows, weights, corpus_rows, id_col):
    rows = _corpus_index(corpus_rows, id_col)

    def result(expected, predicted, aggregate):
        absolute_error = abs(predicted - expected)
        relative_error = _relative_error(predicted, expected)
        return _objective(
            "relative_error", relative_error, "minimize",
            expected=round(expected, 6),
            predicted=round(predicted, 6),
            absolute_error=round(absolute_error, 6),
            mean_absolute_percentage_error=round(relative_error * 100.0, 6),
            validation_aggregate=aggregate)

    def positive_total(values):
        return sum(
            _sample_weight(weights, rid)
            for rid in labels
            if _is_positive(values.get(str(rid), values.get(rid)))
        )

    if kind in {"count", "ratio"}:
        expected = positive_total(labels)
        predicted = positive_total(pred_rows)
        if kind == "ratio":
            denominator = sum(_sample_weight(weights, rid) for rid in labels) or 1.0
            expected, predicted = expected / denominator, predicted / denominator
        return result(expected, predicted, kind)

    if kind in {"mean", "mean_age_2026"}:
        def mean(values):
            numerator = denominator = 0.0
            for rid in labels:
                if not _is_positive(values.get(str(rid), values.get(rid))):
                    continue
                try:
                    number = float(_column(rows.get(str(rid)), field))
                except (TypeError, ValueError):
                    continue
                weight = _sample_weight(weights, rid)
                numerator += weight * number
                denominator += weight
            return numerator / denominator if denominator else None

        expected, predicted = mean(labels), mean(pred_rows)
        if expected is None or predicted is None:
            return _objective("relative_error", None, "minimize",
                              validation_aggregate=f"mean({field})")
        if kind == "mean_age_2026":
            expected, predicted = 2026.0 - expected, 2026.0 - predicted
        return result(expected, predicted, f"mean({field})")

    def group_counts(values, by_label=False):
        totals = {}
        for rid, expected_label in labels.items():
            value = values.get(str(rid), values.get(rid))
            if by_label:
                group = _norm(value)
                if group is None:
                    continue
            else:
                if not _is_positive(value):
                    continue
                group = _norm(_column(rows.get(str(rid)), field))
            totals[group] = totals.get(group, 0.0) + _sample_weight(weights, rid)
        return totals

    expected_counts = group_counts(labels, kind == "label_counts")
    predicted_counts = group_counts(pred_rows, kind == "label_counts")
    groups = set(expected_counts) | set(predicted_counts)
    errors = [
        _relative_error(predicted_counts.get(group, 0.0), expected_counts.get(group, 0.0))
        for group in groups
    ]
    return _objective(
        "mape", (sum(errors) / len(errors) * 100.0) if errors else 0.0, "minimize",
        expected=expected_counts, predicted=predicted_counts,
        validation_aggregate=kind)


def inference_objective(benchmark, query, labels, pred_rows, weights,
                        corpus_rows, id_col, quality):
    """Metric-aware SELECT objective, computed only from oracle-labeled rows.

    This deliberately does not use SemBench's held-out full-query ground truth.
    Non-decomposable query metrics are reconstructed on the weighted validation
    population whenever the trace and corpus columns make that possible.
    """
    key = (str(benchmark or "").lower(), _query_number(query))
    query_name = str(query).lower().lstrip("q")
    if key[0] == "mmqa" and query_name == "4":
        tp = fp = fn = 0.0
        for rid, expected in labels.items():
            weight = _sample_weight(weights, rid)
            expected_set = _multi_label_set(expected)
            predicted_set = _multi_label_set(
                pred_rows.get(str(rid), pred_rows.get(rid)))
            tp += weight * len(expected_set & predicted_set)
            fp += weight * len(predicted_set - expected_set)
            fn += weight * len(expected_set - predicted_set)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 0.0
        if precision is not None and recall is not None and precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        return _objective(
            "f1", f1, "maximize",
            metric_family="QueryMetricRetrieval",
            variant="multi_label_genre_tuple_f1",
            precision=None if precision is None else round(precision, 6),
            recall=None if recall is None else round(recall, 6),
            f1_score=round(f1, 6),
            tp=round(tp, 6), fp=round(fp, 6), fn=round(fn, 6),
            validation_scope="per_movie_genre_membership")
    if key[0] == "mmqa" and query_name == "5":
        expected_rows = [_multi_label_set(value) for value in labels.values()]
        predicted_rows = [
            _multi_label_set(pred_rows.get(str(rid), pred_rows.get(rid)))
            for rid in labels
        ]
        expected_common = set.intersection(*expected_rows) if expected_rows else set()
        predicted_common = set.intersection(*predicted_rows) if predicted_rows else set()
        tp = len(expected_common & predicted_common)
        fp = len(predicted_common - expected_common)
        fn = len(expected_common - predicted_common)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 0.0
        if precision is not None and recall is not None and precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        return _objective(
            "f1", f1, "maximize",
            metric_family="QueryMetricRetrieval",
            variant="cross_document_person_intersection_f1",
            precision=None if precision is None else round(precision, 6),
            recall=None if recall is None else round(recall, 6),
            f1_score=round(f1, 6),
            expected=sorted(expected_common), predicted=sorted(predicted_common),
            validation_scope="complete_deterministic_target_movie_frame")
    if key[0] == "mmqa" and query_name == "2b":
        # q2b composes logo-pair membership and color extraction. A wrong color on
        # an actual logo is both a false positive tuple and a missed expected tuple,
        # matching the official tuple-membership F1 rather than treating every
        # non-"false" string as the same positive decision.
        tp = fp = fn = 0.0
        for rid, expected in labels.items():
            weight = _sample_weight(weights, rid)
            predicted = _norm(pred_rows.get(str(rid), pred_rows.get(rid, "no_match")))
            expected = _norm(expected)
            expected_positive = bool(expected and expected != "no_match")
            predicted_positive = bool(predicted and predicted != "no_match")
            if expected_positive and predicted == expected:
                tp += weight
            elif expected_positive and predicted_positive:
                fp += weight
                fn += weight
            elif expected_positive:
                fn += weight
            elif predicted_positive:
                fp += weight
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 0.0
        if precision is not None and recall is not None and precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        return _objective(
            "f1", f1, "maximize",
            metric_family="QueryMetricRetrieval",
            variant="filter_then_extract_tuple_f1",
            precision=None if precision is None else round(precision, 6),
            recall=None if recall is None else round(recall, 6),
            f1_score=round(f1, 6),
            tp=round(tp, 6), fp=round(fp, 6), fn=round(fn, 6),
            validation_scope="joint_pair_sampling_unit")
    if key in UNSUPPORTED_QUERY_METRIC:
        return _objective(
            "query_metric_unavailable", None, "maximize",
            reason="the official SemBench evaluator does not define a metric "
                   "for this query")
    if key in MULTISITE_QUERY_OBJECTIVE_UNAVAILABLE:
        return _objective(
            "query_metric_unavailable", None, "maximize",
            reason="final aggregate depends on multiple semantic call sites, "
                   "but the current validation trace labels only one call site")
    if key in ARI_REFINEMENT:
        from sklearn.metrics import adjusted_rand_score
        expected = [_norm(labels[rid]) for rid in labels]
        predicted = [_norm(pred_rows.get(str(rid), pred_rows.get(rid, "__missing__")))
                     for rid in labels]
        value = adjusted_rand_score(expected, predicted)
        return _objective(
            "adjusted_rand_index", value, "maximize",
            metric_type="adjusted-rand-index", accuracy=round(float(value), 6),
            n=len(expected))
    if key in MACRO_F1_REFINEMENT:
        from sklearn.metrics import precision_recall_fscore_support
        expected = [_norm(labels[rid]) for rid in labels]
        predicted = [_norm(pred_rows.get(str(rid), pred_rows.get(rid, "__missing__")))
                     for rid in labels]
        sample_weights = [_sample_weight(weights, rid) for rid in labels]
        precision, recall, f1, _ = precision_recall_fscore_support(
            expected, predicted, average="macro", zero_division=0,
            sample_weight=sample_weights)
        return _objective(
            "macro_f1", f1, "maximize",
            precision=round(float(precision), 6),
            recall=round(float(recall), 6),
            f1_score=round(float(f1), 6),
            average="macro", n=len(expected))
    if key in RANKING_REFINEMENT:
        from scipy.stats import kendalltau, spearmanr
        pairs = []
        for rid, expected in labels.items():
            try:
                pairs.append((float(expected), float(
                    pred_rows.get(str(rid), pred_rows.get(rid)))))
            except (TypeError, ValueError):
                continue
        if len(pairs) >= 2:
            expected_scores = [pair[0] for pair in pairs]
            predicted_scores = [pair[1] for pair in pairs]
            spearman = spearmanr(expected_scores, predicted_scores).correlation
            kendall = kendalltau(expected_scores, predicted_scores).correlation
        else:
            spearman = kendall = None
        spearman = 0.0 if spearman is None else spearman
        kendall = 0.0 if kendall is None else kendall
        return _objective(
            "spearman_correlation", spearman, "maximize",
            kendall_tau=round(float(kendall), 6), n_common=len(pairs))
    if key in EXACT_SINGLE_RESULT_RETRIEVAL:
        expected, predicted, expected_counts, predicted_counts = _top_group(
            labels, pred_rows, weights, corpus_rows,
            EXACT_SINGLE_RESULT_RETRIEVAL[key], id_col)
        value = None if expected is None else float(predicted == expected)
        return _objective(
            "f1", value, "maximize",
            metric_family="QueryMetricRetrieval",
            variant=("exactly_one_top_city_station"
                     if len(EXACT_SINGLE_RESULT_RETRIEVAL[key]) == 2
                     else "exactly_one_top_city"),
            precision=value, recall=value, f1_score=value,
            expected=list(expected) if expected else None,
            predicted=list(predicted) if predicted else None,
            expected_counts={"|".join(k): v for k, v in expected_counts.items()},
            predicted_counts={"|".join(k): v for k, v in predicted_counts.items()})
    if key in AGGREGATION_REFINEMENT:
        kind, field = AGGREGATION_REFINEMENT[key]
        return _aggregation_objective(
            kind, field, labels, pred_rows, weights, corpus_rows, id_col)
    if key in F1_REFINEMENT:
        return _objective(
            "f1", quality.get("f1"), "maximize",
            metric_family="QueryMetricRetrieval",
            metric_type=("f1-score" if key[0] == "ecomm" else None),
            precision=quality.get("precision"),
            recall=quality.get("recall"),
            f1_score=quality.get("f1"),
            validation_scope="select_sampling_unit")

    # Unknown/custom queries still expose a useful operator diagnostic, but its
    # explicit name prevents it from being mistaken for an official query metric.
    value = quality.get("f1")
    if value is None:
        value = quality.get("accuracy")
    return _objective("predicate_fidelity_f1", value, "maximize")


def score_inference(trace, val, corpus_rows, cap, id_col=None, text_col=None,
                    benchmark=""):
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
    quality = out.get("estimates") or out["unweighted"]
    out["objective"] = inference_objective(
        benchmark, val.get("query", trace.get("query", "")),
        labels, pred_rows, weights, corpus_rows, id_col, quality)
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
    result = score_pair(results, gold)
    result.update(
        metric="retrieval_f1",
        metric_family="QueryMetricRetrieval",
        metric_variant={
            1: "director_membership",
            2: "tuple_membership",
            3: "title_membership",
            4: "genre_movie_membership",
            5: "actor_membership",
            6: "airline_membership",
            7: "airline_image_tuple_membership",
        }[hid],
        f1_score=result["f1"],
    )
    return result


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
OBJECTIVE_HISTORY_COLS = [f"val_objective_iter_{i}" for i in range(6)]
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
    "val_objective_name", "val_objective_direction",
    *OBJECTIVE_HISTORY_COLS, "val_objective_history",
    "gt_count", "pred_count", "tp", "fp", "fn", "precision", "recall", "f1",
    # non-F1 metric families (blank unless that metric applies)
    "metric", "metric_family", "metric_type", "metric_variant",
    "f1_score", "accuracy",
    "relative_error", "absolute_error", "mape", "mean_absolute_percentage_error",
    "spearman", "kendall", "spearman_correlation", "kendall_tau",
    "ari", "adjusted_rand_index", "covered",
]


def telemetry_row(tele, query="", benchmark=""):
    """Flatten one telemetry.json into the stable results.csv schema.

    `f1` remains the final full-corpus score. `val_f1_iter_N` is retained as the
    operator-fidelity diagnostic. The typed `val_objective_*` fields record the
    metric that actually selected iterations, including its optimization direction.
    """
    tele = tele if isinstance(tele, dict) else {}
    llm = tele.get("llm_calls", {}) if isinstance(tele.get("llm_calls"), dict) else {}
    direct = direct_telemetry(tele)
    refine = tele.get("refine", {}) if isinstance(tele.get("refine"), dict) else {}
    history = refine.get("f1_history", [])
    history = history if isinstance(history, list) else []
    objective_history = refine.get("objective_history", [])
    if not isinstance(objective_history, list) or not objective_history:
        objective_history = [
            item.get("objective") if isinstance(item, dict) else None
            for item in history
        ]
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
        val_objective_name=refine.get("objective", ""),
        val_objective_direction=refine.get("objective_direction", ""),
        val_objective_history=json.dumps(objective_history, separators=(",", ":")),
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
    for iteration, objective in enumerate(objective_history):
        column = f"val_objective_iter_{iteration}"
        if column in row and isinstance(objective, dict):
            row[column] = objective.get("value", "")

    metrics = tele.get("metrics")
    if isinstance(metrics, dict):
        for column in (
            "gt_count", "pred_count", "tp", "fp", "fn", "precision", "recall", "f1",
            "metric", "metric_family", "metric_type", "f1_score", "accuracy",
            "relative_error", "absolute_error", "mape",
            "mean_absolute_percentage_error",
            "spearman", "kendall", "spearman_correlation", "kendall_tau",
            "ari", "adjusted_rand_index", "covered",
        ):
            if metrics.get(column) is not None:
                row[column] = metrics[column]
        if metrics.get("variant") is not None:
            row["metric_variant"] = metrics["variant"]
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
    keep = (
        "metric_family", "metric_type", "variant",
        "precision", "recall", "f1", "f1_score", "tp", "fp", "fn",
        "gt_count", "pred_count", "accuracy",
        "relative_error", "absolute_error", "mape",
        "mean_absolute_percentage_error",
        "spearman", "kendall", "spearman_correlation", "kendall_tau",
        "ari", "adjusted_rand_index", "covered",
    )
    row = {"metric": r.get("metric", "")}
    for k in keep:
        if r.get(k) is not None:
            row[k] = round(r[k], 4) if isinstance(r[k], float) else r[k]
    return row, r


def metric_objective(metrics):
    """Translate a final SemBench metric dict into a typed optimization objective."""
    metric = str((metrics or {}).get("metric", "")).lower()
    if metric == "aggregation":
        return _objective(
            "relative_error", metrics.get("relative_error"), "minimize",
            absolute_error=metrics.get("absolute_error"),
            mean_absolute_percentage_error=metrics.get(
                "mean_absolute_percentage_error", metrics.get("mape")))
    if metric == "ranking":
        return _objective(
            "spearman_correlation",
            metrics.get("spearman_correlation", metrics.get("spearman")),
            "maximize",
            kendall_tau=metrics.get("kendall_tau", metrics.get("kendall")))
    if metric in {"ari", "adjusted-rand-index"}:
        return _objective(
            "adjusted_rand_index",
            metrics.get("adjusted_rand_index", metrics.get("ari")),
            "maximize",
            metric_type="adjusted-rand-index",
            accuracy=metrics.get("accuracy"))
    if metric == "top1":
        # Compatibility with telemetry produced before animals Q3/Q4/Q10 were
        # correctly identified as QueryMetricRetrieval.
        return _objective("f1", metrics.get("f1", metrics.get("precision")), "maximize")
    return _objective(
        "macro_f1" if metrics.get("variant") == "macro_classification"
        or metric == "macro_f1" else "f1",
        metrics.get("f1_score", metrics.get("f1")), "maximize",
        precision=metrics.get("precision"), recall=metrics.get("recall"),
        f1_score=metrics.get("f1_score", metrics.get("f1")))


def _scenario_diff(pred_path, gt_dir, bench, query, sf, cap):
    """Generic id-set FP/FN for a non-mmqa scenario. Reads the predicted CSV and the
    scenario GT CSV, diffs their shared id column ("id" if present, else the first
    column). For non-membership metrics (aggregation/ranking) the id sets may be
    trivial — then fp/fn are empty and pred_sample/gt_sample carry a few raw rows so
    the agent still sees the shape. Faithful scoring stays in scenario_metrics."""
    import csv as _csv, importlib, re as _re
    sm = importlib.import_module("scenario_metrics")
    qid = int(_re.match(r"(\d+)", query.lstrip("qQ")).group(1))     # q3a -> 3
    scale = int(sf) if sf else None
    gt_path = str(sm._gt_path(bench, qid, gt_dir, scale))

    def _rows(p):
        try:
            with open(p, newline="") as f:
                return list(_csv.DictReader(f))
        except FileNotFoundError:
            return []

    pred_rows = _rows(pred_path)
    if hasattr(sm, "_load_gt"):
        gt_rows = sm._load_gt(bench, qid, gt_dir, scale).to_dict("records")
    else:
        gt_rows = _rows(gt_path)
    def _shared_idcols(predicted, expected):
        if not predicted or not expected:
            return None, None
        pred_map = {key.lower(): key for key in predicted[0]}
        gt_map = {key.lower(): key for key in expected[0]}
        shared = set(pred_map) & set(gt_map)
        for wanted in ("id", "car_id", "patient_id", "vin", "image_id",
                       "audio_id", "reviewid"):
            if wanted in shared:
                return pred_map[wanted], gt_map[wanted]
        if shared:
            first = next(key.lower() for key in predicted[0] if key.lower() in shared)
            return pred_map[first], gt_map[first]
        return None, None
    pc, gc = _shared_idcols(pred_rows, gt_rows)
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
        out = score_inference(
            trace, val, corpus_rows, args.diff_cap, args.id_col, args.text_col,
            benchmark=args.benchmark)
        if args.emit_diff:
            json.dump(out, open(args.emit_diff, "w"), indent=2)
        objective = out.get("objective") or {}
        print(f"[eval] inference accuracy {out['accuracy']} "
              f"objective={objective.get('name')}:{objective.get('value')} "
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
                    diff["objective"] = metric_objective(metrics)
                    diff["query_metrics"] = metrics
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
                diff["objective"] = metric_objective(metrics)
                diff["query_metrics"] = metrics
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
