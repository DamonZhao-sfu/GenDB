"""
scenario_metrics.py
===================

Self-contained, faithful port of SemBench's per-scenario, per-query evaluation
metrics for the scenarios: ``movie``, ``cars``, ``medical``, ``animals`` and
``ecomm`` (``mmqa`` is intentionally out of scope and handled elsewhere).

The single public entry point is :func:`score`.  It reproduces -- as closely as
possible, using the *same* libraries (pandas / scikit-learn / scipy) -- the
exact logic found in:

    SemBench/src/evaluator/generic_evaluator.py
    SemBench/src/scenario/<scenario>/evaluation/evaluate.py
    SemBench/src/scenario/ecomm/ecomm_scenario.py

Every metric handler below is a line-by-line transcription of the corresponding
``_evaluate_qN`` method (or generic helper) so that numbers are bit-for-bit
comparable with SemBench's real evaluators.  See ``scenario_metrics_validate``
(companion test script) for the MATCH table proving parity.

--------------------------------------------------------------------------------
Ground-truth file rules (gt_dir = files/<scenario>/raw_results/ground_truth)
--------------------------------------------------------------------------------
* movie   : ``Q<id>.csv``                    (no scale suffix)
* animals : ``Q<id>.csv``                    (no scale suffix)
* ecomm   : ``Q<id>.csv``                    (no scale suffix)
* cars    : ``Q<id>_<scale>.csv`` if present else ``Q<id>.csv``
* medical : ``Q<id>.csv`` when scale in {None, 11112};
            otherwise ``Q<id>_<scale>.csv`` if present else ``Q<id>.csv``

--------------------------------------------------------------------------------
Sampling parity
--------------------------------------------------------------------------------
cars Q3/Q8 and medical Q3/Q8 balance the ground truth around a LIMIT by
sampling with ``pandas.DataFrame.sample(random_state=42)``.  This is replicated
here *exactly* (including the subtle difference that cars uses ``min(n, len)``
while medical uses a bare ``n``), so the sampled GT -- and therefore the score
-- matches SemBench for identical inputs.

--------------------------------------------------------------------------------
Audio-only queries (won't be produced by a text/image system)
--------------------------------------------------------------------------------
The metric computation itself is modality-agnostic (it only compares CSVs), so
every query is fully portable and validated.  Queries whose semantic predicate
is *purely audio* are recorded in :data:`AUDIO_ONLY_QUERIES` and flagged in the
returned dict via ``"audio_only": True`` (never hard-failed, so the scoring core
stays usable if such a result ever appears).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import adjusted_rand_score, precision_recall_fscore_support


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

SCENARIOS = ("movie", "cars", "medical", "animals", "ecomm")

#: Queries whose semantic predicate is *purely* audio (informational flag only).
AUDIO_ONLY_QUERIES: Dict[str, set] = {
    "movie": set(),
    "cars": {2},          # Electric cars w/ dead-battery *audio*
    "medical": {2},       # non-current smokers w/ normal lung *audio*
    "animals": {2, 4},    # elephant *audio* count / city-with-most elephant audio
    "ecomm": set(),
}

#: ecomm accuracy metric per query, encoded from files/ecomm/queries/q*.toml
#: ([definition].accuracy_metric).  q15/q17 are drafts and excluded.
ECOMM_ACCURACY_METRIC: Dict[int, str] = {
    1: "f1-score",
    2: "f1-score",
    3: "adjusted-rand-index",
    4: "adjusted-rand-index",
    5: "adjusted-rand-index",
    6: "adjusted-rand-index",
    7: "f1-score",
    8: "f1-score",
    9: "f1-score",
    10: "f1-score",
    11: "f1-score",
    12: "f1-score",
    13: "f1-score",
    14: "f1-score",
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _read_csv_safe(path: str | Path) -> pd.DataFrame:
    """Read a CSV, returning an empty DataFrame for an empty file.

    Mirrors ``GenericEvaluator._load_system_results`` which catches
    ``pandas.errors.EmptyDataError`` and returns ``pd.DataFrame()``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _gt_path(scenario: str, qid: int, gt_dir: str | Path, scale_factor: Optional[int]) -> Path:
    """Resolve the correct ground-truth CSV path for (scenario, qid, scale)."""
    gt_dir = Path(gt_dir)
    plain = gt_dir / f"Q{qid}.csv"

    if scenario == "cars":
        if scale_factor is not None:
            suff = gt_dir / f"Q{qid}_{int(scale_factor)}.csv"
            if suff.exists():
                return suff
        return plain

    if scenario == "medical":
        if scale_factor is not None and int(scale_factor) != 11112:
            suff = gt_dir / f"Q{qid}_{int(scale_factor)}.csv"
            if suff.exists():
                return suff
        return plain

    # movie / animals / ecomm : no scale suffix
    return plain


def _load_gt(scenario: str, qid: int, gt_dir: str | Path, scale_factor: Optional[int]) -> pd.DataFrame:
    return _read_csv_safe(_gt_path(scenario, qid, gt_dir, scale_factor))


# ---------------------------------------------------------------------------
# Small numeric helpers (ported verbatim in spirit)
# ---------------------------------------------------------------------------


def _f1(precision: float, recall: float) -> float:
    return (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )


def _retrieval_result(precision: float, recall: float, f1: float, **extra) -> Dict[str, Any]:
    out = {
        "metric": "retrieval_f1",
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    out.update(extra)
    return out


def _agg_result(rel: float, abs_: float, mape: float, **extra) -> Dict[str, Any]:
    out = {
        "metric": "aggregation",
        "relative_error": float(rel),
        "absolute_error": float(abs_),
        "mape": float(mape),
    }
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# GenericEvaluator ports
# ---------------------------------------------------------------------------


def _compute_precision(ground_truth: pd.DataFrame, query_result: pd.DataFrame, id_column: str = "id") -> float:
    gt_ids = set(ground_truth[id_column]) if not ground_truth.empty else set()
    res_ids = (
        set(query_result[id_column])
        if query_result is not None and not query_result.empty
        else set()
    )
    if len(gt_ids) == 0:
        return 1.0 if len(res_ids) == 0 else 0.0
    predicted_positives = len(res_ids)
    if predicted_positives == 0:
        return 0.0
    tp = len(res_ids & gt_ids)
    return tp / predicted_positives


def _compute_recall(ground_truth: pd.DataFrame, query_result: pd.DataFrame, id_column: str = "id") -> float:
    gt_ids = set(ground_truth[id_column]) if not ground_truth.empty else set()
    res_ids = (
        set(query_result[id_column])
        if query_result is not None and not query_result.empty
        else set()
    )
    if len(gt_ids) == 0:
        return 1.0 if len(res_ids) == 0 else 0.0
    tp = len(res_ids & gt_ids)
    return tp / len(gt_ids)


def _id_set_extras(ground_truth: pd.DataFrame, query_result: pd.DataFrame, id_column: str) -> Dict[str, int]:
    gt_ids = set(ground_truth[id_column]) if not ground_truth.empty else set()
    res_ids = (
        set(query_result[id_column])
        if query_result is not None and not query_result.empty
        else set()
    )
    tp = len(res_ids & gt_ids)
    return {
        "tp": tp,
        "fp": len(res_ids - gt_ids),
        "fn": len(gt_ids - res_ids),
        "gt_count": len(gt_ids),
        "pred_count": len(res_ids),
    }


def _id_set_f1(ground_truth: pd.DataFrame, system_results: pd.DataFrame, id_column: str) -> Dict[str, Any]:
    """Faithful reproduction of the cars/medical id-set precision/recall/f1 path.

    In SemBench these use ``compute_accuracy_score("precision"/"recall", ...)``
    whose ``.accuracy`` returns precision resp. recall (computed from id sets),
    then ``f1 = 2pr/(p+r)`` is recomputed by the scenario handler.
    """
    precision = _compute_precision(ground_truth, system_results, id_column=id_column)
    recall = _compute_recall(ground_truth, system_results, id_column=id_column)
    f1 = _f1(precision, recall)
    out = _retrieval_result(precision, recall, f1, variant="id_set_f1")
    out["metric"] = "id_set_f1"
    out.update(_id_set_extras(ground_truth, system_results, id_column))
    return out


def _generic_aggregation_evaluation(system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> Dict[str, Any]:
    if len(system_results) != 1 or len(ground_truth) != 1:
        return _agg_result(1.0, float("inf"), 100.0)

    def first_num(df):
        for c in df.columns:
            val = df[c].iloc[0]
            if isinstance(val, str):
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    continue
            if pd.api.types.is_numeric_dtype(type(val)) or isinstance(val, (int, float)):
                return val
        return None

    sys_val, gt_val = first_num(system_results), first_num(ground_truth)
    if sys_val is None or gt_val is None:
        return _agg_result(1.0, float("inf"), 100.0)

    abs_err = float(abs(sys_val - gt_val))
    if gt_val != 0:
        rel = float(abs_err / abs(gt_val))
        mape = float(rel * 100)
    else:
        rel = float("inf") if sys_val != 0 else 0.0
        mape = float(rel * 100)
    return _agg_result(rel, abs_err, mape)


def _generic_retrieval_evaluation(system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> Dict[str, Any]:
    """Row-by-row greedy matching over common columns (no LIMIT)."""
    if len(ground_truth) == 0:
        p = 1.0 if len(system_results) == 0 else 0.0
        return _retrieval_result(p, 0.0, 0.0, variant="generic_retrieval")
    if len(system_results) == 0:
        return _retrieval_result(0.0, 0.0, 0.0, variant="generic_retrieval")

    matches = 0
    matched_gt = set()
    for _, srow in system_results.iterrows():
        for gt_idx, gt_row in ground_truth.iterrows():
            if gt_idx in matched_gt:
                continue
            common = set(srow.index) & set(gt_row.index)
            if all(
                srow[c] == gt_row[c]
                for c in common
                if pd.notna(srow[c]) and pd.notna(gt_row[c])
            ):
                matches += 1
                matched_gt.add(gt_idx)
                break
    precision = matches / len(system_results)
    recall = matches / len(ground_truth)
    f1 = _f1(precision, recall)
    return _retrieval_result(
        precision, recall, f1, variant="generic_retrieval",
        tp=matches, fp=len(system_results) - matches, fn=len(ground_truth) - matches,
        gt_count=len(ground_truth), pred_count=len(system_results),
    )


def _generic_ranking_evaluation(system_results: pd.DataFrame, ground_truth: pd.DataFrame) -> Dict[str, Any]:
    def _empty():
        return {"metric": "ranking", "spearman": 0.0, "kendall": 0.0, "n_common": 0}

    if len(system_results) == 0 or len(ground_truth) == 0:
        return _empty()
    if len(system_results.columns) < 2 or len(ground_truth.columns) < 2:
        return _empty()

    sys_id_col, sys_score_col = system_results.columns[0], system_results.columns[1]
    gt_id_col, gt_score_col = ground_truth.columns[0], ground_truth.columns[1]

    sys_scores: Dict[Any, float] = {}
    for _, row in system_results.iterrows():
        i, s = row[sys_id_col], row[sys_score_col]
        if pd.notna(i) and pd.notna(s):
            try:
                sys_scores[i] = float(s)
            except (ValueError, TypeError):
                continue

    gt_scores: Dict[Any, float] = {}
    for _, row in ground_truth.iterrows():
        i, s = row[gt_id_col], row[gt_score_col]
        if pd.notna(i) and pd.notna(s):
            try:
                gt_scores[i] = float(s)
            except (ValueError, TypeError):
                continue

    common_ids = set(sys_scores.keys()) & set(gt_scores.keys())
    if len(common_ids) < 2:
        return _empty()

    sys_values = [sys_scores[i] for i in common_ids]
    gt_values = [gt_scores[i] for i in common_ids]

    try:
        sp = spearmanr(sys_values, gt_values).correlation
        spearman_corr = sp if not pd.isna(sp) else 0.0
    except Exception:
        spearman_corr = 0.0
    try:
        kt = kendalltau(sys_values, gt_values).correlation
        kendall_corr = kt if not pd.isna(kt) else 0.0
    except Exception:
        kendall_corr = 0.0

    return {
        "metric": "ranking",
        "spearman": float(spearman_corr),
        "kendall": float(kendall_corr),
        "n_common": len(common_ids),
    }


# ---------------------------------------------------------------------------
# MOVIE handlers
# ---------------------------------------------------------------------------


def _movie_retrieval_limit(system_results, ground_truth, limit=5) -> Dict[str, Any]:
    if len(system_results) == 0:
        return _retrieval_result(
            1.0 if len(ground_truth) == 0 else 0.0, 0.0, 0.0, variant="retrieval_limit"
        )
    if len(ground_truth) == 0:
        return _retrieval_result(0.0, 0.0, 0.0, variant="retrieval_limit")

    system_results = system_results.head(limit)
    if len(system_results.columns) > 0 and len(ground_truth.columns) > 0:
        sys_col = system_results.columns[0]
        gt_col = ground_truth.columns[0]
        sys_ids = set(system_results[sys_col].dropna())
        gt_ids = set(ground_truth[gt_col].dropna())
        valid = sys_ids & gt_ids
        precision = len(valid) / len(sys_ids) if sys_ids else 0.0
        recall = (
            (len(valid) if len(valid) <= limit else limit) / min(limit, len(gt_ids))
            if gt_ids
            else 0.0
        )
        f1 = _f1(precision, recall)
        return _retrieval_result(
            precision, recall, f1, variant="retrieval_limit", limit=limit,
            tp=len(valid), fp=len(sys_ids - gt_ids), fn=len(gt_ids - sys_ids),
            gt_count=len(gt_ids), pred_count=len(sys_ids),
        )
    return _generic_retrieval_evaluation(system_results, ground_truth)


def _movie_pairs(system_results, ground_truth, limit=None) -> Dict[str, Any]:
    variant = "pair_retrieval_limit" if limit is not None else "pair_retrieval"
    if len(system_results) == 0:
        return _retrieval_result(
            1.0 if len(ground_truth) == 0 else 0.0, 0.0, 0.0, variant=variant
        )
    if len(ground_truth) == 0:
        return _retrieval_result(0.0, 0.0, 0.0, variant=variant)
    if len(system_results.columns) < 3 or len(ground_truth.columns) < 3:
        return _retrieval_result(0.0, 0.0, 0.0, variant=variant)

    if limit is not None:
        system_results = system_results.head(limit)

    def create_pair_tuple(row, columns):
        if len(columns) >= 3:
            movie_id = row[columns[0]]
            val1 = row[columns[1]]
            val2 = row[columns[2]]
            if pd.notna(val1) and pd.notna(val2) and pd.notna(movie_id):
                return (movie_id, tuple(sorted([val1, val2])))
        return None

    sys_cols = list(system_results.columns)
    gt_cols = list(ground_truth.columns)
    sys_pairs = {
        t for t in system_results.apply(lambda r: create_pair_tuple(r, sys_cols), axis=1) if t is not None
    }
    gt_pairs = {
        t for t in ground_truth.apply(lambda r: create_pair_tuple(r, gt_cols), axis=1) if t is not None
    }
    correct = sys_pairs & gt_pairs
    precision = len(correct) / len(sys_pairs) if sys_pairs else 0.0
    if limit is not None:
        recall = (
            (len(correct) if len(correct) <= limit else limit) / min(limit, len(gt_pairs))
            if gt_pairs
            else 0.0
        )
    else:
        recall = len(correct) / len(gt_pairs) if gt_pairs else 0.0
    f1 = _f1(precision, recall)
    return _retrieval_result(
        precision, recall, f1, variant=variant,
        tp=len(correct), fp=len(sys_pairs - gt_pairs), fn=len(gt_pairs - sys_pairs),
        gt_count=len(gt_pairs), pred_count=len(sys_pairs),
    )


def _movie_sentiment_counts(system_results, ground_truth) -> Dict[str, Any]:
    if len(system_results) == 0 or len(ground_truth) == 0:
        return _agg_result(1.0, float("inf"), 100.0, variant="sentiment_counts")
    if len(system_results.columns) < 2 or len(ground_truth.columns) < 2:
        return _agg_result(1.0, float("inf"), 100.0, variant="sentiment_counts")

    sys_sent_c, sys_cnt_c = system_results.columns[0], system_results.columns[1]
    gt_sent_c, gt_cnt_c = ground_truth.columns[0], ground_truth.columns[1]

    def to_counts(df, sc, cc):
        d = {}
        for _, row in df.iterrows():
            sent, cnt = row[sc], row[cc]
            if pd.notna(sent) and pd.notna(cnt):
                try:
                    d[str(sent).strip().upper()] = float(cnt)
                except (ValueError, TypeError):
                    continue
        return d

    sys_counts = to_counts(system_results, sys_sent_c, sys_cnt_c)
    gt_counts = to_counts(ground_truth, gt_sent_c, gt_cnt_c)
    if not sys_counts or not gt_counts:
        return _agg_result(1.0, float("inf"), 100.0, variant="sentiment_counts")

    total_abs = 0.0
    total_rel = 0.0
    valid = 0
    for sentiment in set(sys_counts) | set(gt_counts):
        s = sys_counts.get(sentiment, 0.0)
        g = gt_counts.get(sentiment, 0.0)
        abs_err = abs(s - g)
        total_abs += abs_err
        if g != 0:
            total_rel += abs_err / abs(g)
            valid += 1
        elif s != 0:
            total_rel += 1.0
            valid += 1

    if valid > 0:
        rel = total_rel / valid
        mape = rel * 100
    else:
        rel = 0.0
        mape = 0.0
    return _agg_result(rel, total_abs, mape, variant="sentiment_counts")


# ---------------------------------------------------------------------------
# ANIMALS handlers
# ---------------------------------------------------------------------------


def _animals_top_city(system_results, ground_truth) -> Dict[str, Any]:
    if len(ground_truth) == 0:
        p = 1.0 if len(system_results) == 0 else 0.0
        return {"metric": "top1", "precision": p, "recall": 0.0, "f1": 0.0, "variant": "top1_city"}
    if len(system_results) == 0:
        return {"metric": "top1", "precision": 0.0, "recall": 0.0, "f1": 0.0, "variant": "top1_city"}

    gt_cities = set(ground_truth.iloc[:, 0]) if len(ground_truth.columns) > 0 else set()
    if len(system_results) == 1:
        sys_city = system_results.iloc[0, 0] if len(system_results.columns) > 0 else None
        if sys_city in gt_cities:
            return {"metric": "top1", "precision": 1.0, "recall": 1.0, "f1": 1.0, "variant": "top1_city"}
    return {"metric": "top1", "precision": 0.0, "recall": 0.0, "f1": 0.0, "variant": "top1_city"}


def _animals_top_city_station(system_results, ground_truth) -> Dict[str, Any]:
    if len(ground_truth) == 0:
        p = 1.0 if len(system_results) == 0 else 0.0
        return {"metric": "top1", "precision": p, "recall": 0.0, "f1": 0.0, "variant": "top1_city_station"}
    if len(system_results) == 0:
        return {"metric": "top1", "precision": 0.0, "recall": 0.0, "f1": 0.0, "variant": "top1_city_station"}

    sys_col_map = {col.lower(): col for col in system_results.columns}
    gt_tuples = set()
    for _, row in ground_truth.iterrows():
        if len(ground_truth.columns) >= 2:
            gt_tuples.add((row.iloc[0], row.iloc[1]))

    if len(system_results) == 1 and len(ground_truth.columns) >= 2:
        sys_row = system_results.iloc[0]
        city_col = None
        station_col = None
        for gt_col in ground_truth.columns[:2]:
            gt_col_lower = gt_col.lower()
            if gt_col_lower in sys_col_map:
                if city_col is None:
                    city_col = sys_col_map[gt_col_lower]
                else:
                    station_col = sys_col_map[gt_col_lower]
        if city_col and station_col:
            sys_tuple = (sys_row[city_col], sys_row[station_col])
            if sys_tuple in gt_tuples:
                return {"metric": "top1", "precision": 1.0, "recall": 1.0, "f1": 1.0, "variant": "top1_city_station"}
    return {"metric": "top1", "precision": 0.0, "recall": 0.0, "f1": 0.0, "variant": "top1_city_station"}


# ---------------------------------------------------------------------------
# cars / medical LIMIT-balanced id-set handlers (with sampling)
# ---------------------------------------------------------------------------


def _limit_balanced_id_set(
    system_results: pd.DataFrame,
    ground_truth: pd.DataFrame,
    id_column: str,
    limit: int,
    use_min_sampling: bool,
) -> Dict[str, Any]:
    """Port of cars/medical Q3/Q8 LIMIT-balancing + random_state=42 sampling.

    ``use_min_sampling`` distinguishes the two families:
      * cars   -> ``min(limit, len)`` / ``min(needed, len(false))`` (safe)
      * medical-> bare ``n`` (may raise if not enough rows, faithful to source)
    """
    correct_ids_ix = ground_truth[id_column].isin(system_results[id_column].to_list())
    correct_ids = ground_truth.loc[correct_ids_ix, :]
    ground_truth_sample = None

    if correct_ids.empty:
        if use_min_sampling:
            n_to_sample = min(limit, len(ground_truth))
            ground_truth_sample = (
                ground_truth.sample(n=n_to_sample, random_state=42)
                if n_to_sample > 0
                else ground_truth
            )
        else:
            ground_truth_sample = ground_truth.sample(n=limit, random_state=42)
    elif correct_ids.shape[0] > limit:
        raise ValueError(
            f"Ground truth should not contain more than {limit} rows. Query contains LIMIT {limit}."
        )
    elif correct_ids.shape[0] < limit:
        false_cases = ground_truth[correct_ids_ix == False]  # noqa: E712
        n_needed = limit - correct_ids.shape[0]
        if use_min_sampling:
            n_to_sample = min(n_needed, len(false_cases))
            if n_to_sample > 0:
                ground_truth_sample = pd.concat(
                    [correct_ids, false_cases.sample(n=n_to_sample, random_state=42)]
                )
            else:
                ground_truth_sample = correct_ids
        else:
            ground_truth_sample = pd.concat(
                [correct_ids, false_cases.sample(n=n_needed, random_state=42)]
            )
    else:  # == limit
        ground_truth_sample = correct_ids

    precision = _compute_precision(ground_truth_sample, system_results, id_column=id_column)
    recall = _compute_recall(ground_truth_sample, system_results, id_column=id_column)
    f1 = _f1(precision, recall)
    out = _retrieval_result(precision, recall, f1, variant="id_set_f1_limit_balanced", limit=limit)
    out["metric"] = "id_set_f1"
    out.update(_id_set_extras(ground_truth_sample, system_results, id_column))
    return out


# ---------------------------------------------------------------------------
# cars / medical macro-F1 handler (Q10)
# ---------------------------------------------------------------------------


def _macro_f1(system_results: pd.DataFrame, ground_truth: pd.DataFrame, id_column: str, result_column: str) -> Dict[str, Any]:
    # SemBench sorts both frames by id and feeds them to sklearn, which REQUIRES
    # equal length. We instead reindex the system's labels onto the GT ids: this is
    # identical to SemBench when the id sets match (the validated case), but robust to
    # a partial/oversized prediction — a GT id with no prediction counts as a wrong
    # label instead of raising `inconsistent numbers of samples`. `covered` surfaces
    # under-production / scale mismatch (few covered ⇒ the GT scale ≠ the data scale).
    gt_s = (ground_truth.assign(**{id_column: ground_truth[id_column].astype(str)})
            .drop_duplicates(id_column).set_index(id_column)[result_column])
    q_s = (system_results.assign(**{id_column: system_results[id_column].astype(str)})
           .drop_duplicates(id_column).set_index(id_column)[result_column])
    y_true = gt_s.tolist()
    y_pred = [q_s.get(i, "__missing__") for i in gt_s.index]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    covered = int(sum(1 for i in gt_s.index if i in q_s.index))
    return {
        "metric": "macro_f1",
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "gt_count": int(len(gt_s)),
        "pred_count": int(len(q_s)),
        "covered": covered,
    }


# ---------------------------------------------------------------------------
# ecomm handlers
# ---------------------------------------------------------------------------


def _ecomm_f1(ground_truth: pd.DataFrame, system_results: pd.DataFrame) -> Dict[str, Any]:
    precision = _compute_precision(ground_truth, system_results, id_column="id")
    recall = _compute_recall(ground_truth, system_results, id_column="id")
    f1 = _f1(precision, recall)  # compute_f1_score
    out = _retrieval_result(f1, recall, f1, variant="ecomm_f1")
    out["metric"] = "id_set_f1"
    # compute_accuracy_score("f1-score") returns accuracy=f1_score, precision, recall, f1_score
    out["precision"] = float(precision)
    out["recall"] = float(recall)
    out["f1"] = float(f1)
    out["accuracy"] = float(f1)
    out.update(_id_set_extras(ground_truth, system_results, id_column="id"))
    return out


def _ecomm_ari(ground_truth: pd.DataFrame, query_result: pd.DataFrame) -> Dict[str, Any]:
    if query_result is None:
        return {"metric": "ari", "ari": 0.0, "accuracy": 0.0, "n_common": 0}

    gt_groups = ground_truth.set_index("id")["category"]
    qr_groups = query_result.set_index("id")["category"]
    common_ids = set(gt_groups.index) & set(qr_groups.index)
    if not common_ids:
        return {"metric": "ari", "ari": 0.0, "accuracy": 0.0, "n_common": 0}
    gt_labels = [gt_groups[i] for i in common_ids]
    qr_labels = [qr_groups[i] for i in common_ids]
    ari = float(adjusted_rand_score(gt_labels, qr_labels))
    return {"metric": "ari", "ari": ari, "accuracy": ari, "n_common": len(common_ids)}


# ---------------------------------------------------------------------------
# Per-scenario dispatch tables
# ---------------------------------------------------------------------------
#
# Each handler takes (system_results, ground_truth) and returns the result dict.
# cars/medical handlers first lowercase the system-result column names exactly
# as SemBench does; the wrappers below encapsulate those transformations.


def _lower_cols(df: pd.DataFrame, guard: bool = True) -> pd.DataFrame:
    """Lowercase column names.

    ``guard=True`` reproduces the cars handlers (``if len(cols) > 0``).
    ``guard=False`` reproduces the medical handlers, which call
    ``.str.lower()`` unconditionally and therefore raise ``AttributeError``
    on a truly-empty (0-column) result -- faithful to SemBench.
    """
    df = df.copy()
    if guard:
        if len(df.columns) > 0:
            df.columns = df.columns.str.lower()
    else:
        df.columns = df.columns.str.lower()
    return df


# ---- movie ----------------------------------------------------------------

_MOVIE: Dict[int, Callable] = {
    1: lambda s, g: _movie_retrieval_limit(s, g, limit=5),
    2: lambda s, g: _movie_retrieval_limit(s, g, limit=5),
    3: lambda s, g: _generic_aggregation_evaluation(s, g),
    4: lambda s, g: _generic_aggregation_evaluation(s, g),
    5: lambda s, g: _movie_pairs(s, g, limit=10),
    6: lambda s, g: _movie_pairs(s, g, limit=10),
    7: lambda s, g: _movie_pairs(s, g, limit=None),
    8: lambda s, g: _movie_sentiment_counts(s, g),
    9: lambda s, g: _generic_ranking_evaluation(s, g),
    10: lambda s, g: _generic_ranking_evaluation(s, g),
}


# ---- animals --------------------------------------------------------------

_ANIMALS: Dict[int, Callable] = {
    1: lambda s, g: _generic_aggregation_evaluation(s, g),
    2: lambda s, g: _generic_aggregation_evaluation(s, g),
    3: lambda s, g: _animals_top_city(s, g),
    4: lambda s, g: _animals_top_city(s, g),
    5: lambda s, g: _generic_retrieval_evaluation(s, g),
    6: lambda s, g: _generic_retrieval_evaluation(s, g),
    7: lambda s, g: _generic_retrieval_evaluation(s, g),
    8: lambda s, g: _generic_retrieval_evaluation(s, g),
    9: lambda s, g: _generic_retrieval_evaluation(s, g),
    10: lambda s, g: _animals_top_city_station(s, g),
}


# ---- cars -----------------------------------------------------------------


def _cars_q1(s, g):
    return _id_set_f1(g, _lower_cols(s), id_column="car_id")


def _cars_q2(s, g):
    s = _lower_cols(s)
    s = s.drop_duplicates()
    return _id_set_f1(g, s, id_column="car_id")


def _cars_q3(s, g):
    s = _lower_cols(s)
    return _limit_balanced_id_set(s, g, id_column="vin", limit=10, use_min_sampling=True)


def _cars_q4(s, g):
    return _generic_aggregation_evaluation(_lower_cols(s), g)


def _cars_q5(s, g):
    return _generic_aggregation_evaluation(_lower_cols(s), g)


def _cars_q6(s, g):
    return _id_set_f1(g, _lower_cols(s), id_column="car_id")


def _cars_q7(s, g):
    return _id_set_f1(g, _lower_cols(s), id_column="car_id")


def _cars_q8(s, g):
    s = _lower_cols(s)
    return _limit_balanced_id_set(s, g, id_column="car_id", limit=100, use_min_sampling=True)


def _cars_q9(s, g):
    return _id_set_f1(g, _lower_cols(s), id_column="car_id")


def _cars_q10(s, g):
    s = _lower_cols(s)
    if "problem_category" in s.columns:
        s = s.copy()
        s["problem_category"] = s["problem_category"].apply(lambda x: str(x).lower().replace("\n", ""))
    return _macro_f1(s, g, id_column="car_id", result_column="problem_category")


_CARS: Dict[int, Callable] = {
    1: _cars_q1, 2: _cars_q2, 3: _cars_q3, 4: _cars_q4, 5: _cars_q5,
    6: _cars_q6, 7: _cars_q7, 8: _cars_q8, 9: _cars_q9, 10: _cars_q10,
}


# ---- medical --------------------------------------------------------------


def _med_q1(s, g):
    return _id_set_f1(g, _lower_cols(s, guard=False), id_column="patient_id")


def _med_q2(s, g):
    s = _lower_cols(s, guard=False)
    s = s.drop_duplicates()
    return _id_set_f1(g, s, id_column="patient_id")


def _med_q3(s, g):
    s = _lower_cols(s, guard=False)
    return _limit_balanced_id_set(s, g, id_column="patient_id", limit=5, use_min_sampling=False)


def _med_q4(s, g):
    return _generic_aggregation_evaluation(_lower_cols(s, guard=False), g)


def _med_q5(s, g):
    return _generic_aggregation_evaluation(_lower_cols(s, guard=False), g)


def _med_q6(s, g):
    return _id_set_f1(g, _lower_cols(s, guard=False), id_column="patient_id")


def _med_q7(s, g):
    return _id_set_f1(g, _lower_cols(s, guard=False), id_column="patient_id")


def _med_q8(s, g):
    s = _lower_cols(s, guard=False)
    return _limit_balanced_id_set(s, g, id_column="patient_id", limit=100, use_min_sampling=False)


def _med_q9(s, g):
    return _id_set_f1(g, _lower_cols(s, guard=False), id_column="patient_id")


def _med_q10(s, g):
    s = _lower_cols(s, guard=False)
    s = s.copy()
    s["text_diagnosis"] = s["text_diagnosis"].apply(lambda x: str(x).lower().replace("\n", ""))
    return _macro_f1(s, g, id_column="patient_id", result_column="text_diagnosis")


_MEDICAL: Dict[int, Callable] = {
    1: _med_q1, 2: _med_q2, 3: _med_q3, 4: _med_q4, 5: _med_q5,
    6: _med_q6, 7: _med_q7, 8: _med_q8, 9: _med_q9, 10: _med_q10,
}


# ---- ecomm ----------------------------------------------------------------


def _ecomm_dispatch(qid: int, s: pd.DataFrame, g: pd.DataFrame) -> Dict[str, Any]:
    metric = ECOMM_ACCURACY_METRIC.get(int(qid))
    if metric is None:
        raise NotImplementedError(f"ecomm Q{qid} has no encoded accuracy_metric (draft/unknown).")
    if metric == "f1-score":
        return _ecomm_f1(g, s)
    if metric == "adjusted-rand-index":
        return _ecomm_ari(g, s)
    raise NotImplementedError(f"ecomm accuracy_metric '{metric}' not supported for Q{qid}.")


_DISPATCH: Dict[str, Dict[int, Callable]] = {
    "movie": _MOVIE,
    "cars": _CARS,
    "medical": _MEDICAL,
    "animals": _ANIMALS,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def score(
    scenario: str,
    query_id: str | int,
    pred_csv: str,
    gt_dir: str,
    scale_factor: Optional[int] = None,
) -> Dict[str, Any]:
    """Score a system's result CSV against SemBench ground truth.

    Parameters
    ----------
    scenario:
        One of ``movie``, ``cars``, ``medical``, ``animals``, ``ecomm``.
    query_id:
        Query id, e.g. ``"Q3"``, ``"3"`` or ``3``.
    pred_csv:
        Path to the system result CSV.  An empty file is treated as an empty
        result (matching SemBench).
    gt_dir:
        ``files/<scenario>/raw_results/ground_truth``.  The correct GT file is
        chosen per the scale/sample suffix rules documented at module level.
    scale_factor:
        Needed only for cars / medical suffix resolution.

    Returns
    -------
    dict
        Always contains ``"metric"`` (one of ``retrieval_f1``, ``id_set_f1``,
        ``aggregation``, ``ranking``, ``ari``, ``macro_f1``, ``top1``) plus the
        metric's numbers.  ``"scenario"``, ``"query_id"`` and ``"audio_only"``
        are always present.
    """
    scenario = scenario.lower().strip()
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown/unsupported scenario '{scenario}'. Expected one of {SCENARIOS}.")

    qid = int(str(query_id).lower().lstrip("q"))

    system_results = _read_csv_safe(pred_csv)
    ground_truth = _load_gt(scenario, qid, gt_dir, scale_factor)

    if scenario == "ecomm":
        result = _ecomm_dispatch(qid, system_results, ground_truth)
    else:
        table = _DISPATCH[scenario]
        if qid not in table:
            raise NotImplementedError(f"{scenario} Q{qid} is not implemented.")
        result = table[qid](system_results, ground_truth)

    result = dict(result)
    result["scenario"] = scenario
    result["query_id"] = f"Q{qid}"
    result["audio_only"] = qid in AUDIO_ONLY_QUERIES.get(scenario, set())
    return result


__all__ = ["score", "SCENARIOS", "AUDIO_ONLY_QUERIES", "ECOMM_ACCURACY_METRIC"]
