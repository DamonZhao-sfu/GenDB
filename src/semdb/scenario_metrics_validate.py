"""
Validation harness: compares scenario_metrics.score() against SemBench's REAL
per-query evaluators on identical synthetic (pred, GT) pairs.

Run with the sembench conda python:
    /localhome/hza214/miniconda3/envs/sembench/bin/python scenario_metrics_validate.py
"""

from __future__ import annotations

import io
import math
import os
import sys
import tempfile
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

SEMBENCH_SRC = "/localhome/hza214/SemBench/src"
sys.path.insert(0, SEMBENCH_SRC)

# Local module under test
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenario_metrics as sm  # noqa: E402

from evaluator.generic_evaluator import (  # noqa: E402
    GenericEvaluator,
    QueryMetricRetrieval,
    QueryMetricAggregation,
    QueryMetricRank,
)
from scenario.movie.evaluation.evaluate import MovieEvaluator  # noqa: E402
from scenario.cars.evaluation.evaluate import CarsEvaluator  # noqa: E402
from scenario.medical.evaluation.evaluate import MedicalEvaluator  # noqa: E402
from scenario.animals.evaluation.evaluate import AnimalsEvaluator  # noqa: E402


def _inst(cls):
    return cls.__new__(cls)


MOVIE = _inst(MovieEvaluator)
CARS = _inst(CarsEvaluator)
MEDICAL = _inst(MedicalEvaluator)
ANIMALS = _inst(AnimalsEvaluator)


# ---------------------------------------------------------------------------
# comparison utilities
# ---------------------------------------------------------------------------


def _num_eq(a, b, tol=1e-9):
    if a is None or b is None:
        return a is b
    fa, fb = float(a), float(b)
    if math.isnan(fa) and math.isnan(fb):
        return True
    if math.isinf(fa) or math.isinf(fb):
        return fa == fb
    return abs(fa - fb) <= tol


def _extract_sembench(obj):
    """Map a SemBench result object to a canonical {field: value} dict."""
    if isinstance(obj, QueryMetricRetrieval):
        return ("retrieval", {"precision": obj.precision, "recall": obj.recall, "f1": obj.f1_score})
    if isinstance(obj, QueryMetricAggregation):
        return ("aggregation", {
            "relative_error": obj.relative_error,
            "absolute_error": obj.absolute_error,
            "mape": obj.mean_absolute_percentage_error,
        })
    if isinstance(obj, QueryMetricRank):
        return ("ranking", {"spearman": obj.spearman_correlation, "kendall": obj.kendall_tau})
    # SingleAccuracyScore / WithRetrievalDetails (ecomm)
    mt = getattr(obj, "metric_type", None)
    if mt in ("f1-score", "precision", "recall"):
        return ("ecomm_f1", {
            "precision": obj.precision,
            "recall": obj.recall,
            "f1": obj.f1_score,
            "accuracy": obj.accuracy,
        })
    if mt == "adjusted-rand-index":
        return ("ari", {"ari": obj.accuracy})
    raise RuntimeError(f"Unknown sembench result type: {obj!r}")


def _compare(kind, ref, mine):
    for k, v in ref.items():
        if k == "ari":
            mk = "ari"
        else:
            mk = k
        if mk not in mine:
            return False, f"missing field {mk} in mine"
        if not _num_eq(v, mine[mk]):
            return False, f"{k}: sembench={v} vs mine={mine[mk]}"
    return True, ""


# ---------------------------------------------------------------------------
# run a single case
# ---------------------------------------------------------------------------

RESULTS = []  # (scenario, qid, case, status, detail)


def run_case(scenario, qid, case, pred_df, gt_df, sembench_call, ecomm_metric=None, scale_factor=None):
    """sembench_call: fn(pred_copy, gt_copy) -> sembench result object."""
    with tempfile.TemporaryDirectory() as td:
        gt_dir = Path(td)
        gt_path = gt_dir / f"Q{qid}.csv"
        gt_df.to_csv(gt_path, index=False)
        # canonical GT (round-tripped) shared by both sides for sampling parity
        gt_canon = pd.read_csv(gt_path)

        pred_path = gt_dir / "pred.csv"
        if pred_df is None or len(pred_df.columns) == 0:
            pred_path.write_text("")  # empty file
        else:
            pred_df.to_csv(pred_path, index=False)

        # --- SemBench reference ---
        try:
            f = io.StringIO()
            with redirect_stdout(f):
                if pred_df is None or len(pred_df.columns) == 0:
                    sys_for_ref = pd.DataFrame()
                else:
                    sys_for_ref = pred_df.copy()
                ref_obj = sembench_call(sys_for_ref, gt_canon.copy())
            kind, ref = _extract_sembench(ref_obj)
            ref_err = None
        except Exception as e:
            kind, ref, ref_err = None, None, f"{type(e).__name__}: {e}"

        # --- mine ---
        try:
            f = io.StringIO()
            with redirect_stdout(f):
                mine = sm.score(scenario, f"Q{qid}", str(pred_path), str(gt_dir), scale_factor=scale_factor)
            mine_err = None
        except Exception as e:
            mine, mine_err = None, f"{type(e).__name__}: {e}"

        if ref_err or mine_err:
            if ref_err and mine_err:
                RESULTS.append((scenario, qid, case, "MATCH(both-raise)", f"ref={ref_err} mine={mine_err}"))
            else:
                RESULTS.append((scenario, qid, case, "MISMATCH", f"ref_err={ref_err} mine_err={mine_err}"))
            return

        ok, detail = _compare(kind, ref, mine)
        RESULTS.append((scenario, qid, case, "MATCH" if ok else "MISMATCH", detail if not ok else f"{ref}"))


# ---------------------------------------------------------------------------
# MOVIE cases
# ---------------------------------------------------------------------------


def movie_cases():
    sc = "movie"
    # Q1/Q2 retrieval limit=5 : GT reviewId
    gt = pd.DataFrame({"reviewId": [1, 2, 3, 4, 5, 6, 7]})
    for qid in (1, 2):
        call = getattr(MOVIE, f"_evaluate_q{qid}")
        run_case(sc, qid, "correct-subset", pd.DataFrame({"reviewId": [1, 2, 3]}), gt, call)
        run_case(sc, qid, "wrong", pd.DataFrame({"reviewId": [100, 200]}), gt, call)
        run_case(sc, qid, "dup", pd.DataFrame({"reviewId": [1, 1, 2, 8]}), gt, call)
        run_case(sc, qid, "empty", None, gt, call)
        run_case(sc, qid, "over-limit", pd.DataFrame({"reviewId": [1, 2, 3, 4, 5, 6, 7]}), gt, call)

    # Q3/Q4 aggregation single value
    gt3 = pd.DataFrame({"positive_review_cnt": [14]})
    for qid in (3, 4):
        call = getattr(MOVIE, f"_evaluate_q{qid}")
        run_case(sc, qid, "exact", pd.DataFrame({"cnt": [14]}), gt3, call)
        run_case(sc, qid, "off", pd.DataFrame({"cnt": [10]}), gt3, call)
        run_case(sc, qid, "empty", None, gt3, call)
        run_case(sc, qid, "multi-row", pd.DataFrame({"cnt": [14, 15]}), gt3, call)

    # Q5/Q6 pairs limit=10 ; Q7 pairs no limit
    gt_pairs = pd.DataFrame({
        "id": ["m", "m", "m"],
        "reviewId1": [10, 20, 30],
        "reviewId2": [11, 21, 31],
    })
    for qid in (5, 6, 7):
        call = getattr(MOVIE, f"_evaluate_q{qid}")
        run_case(sc, qid, "correct", pd.DataFrame({"id": ["m", "m"], "r1": [11, 21], "r2": [10, 20]}), gt_pairs, call)
        run_case(sc, qid, "wrong", pd.DataFrame({"id": ["m"], "r1": [99], "r2": [98]}), gt_pairs, call)
        run_case(sc, qid, "dup", pd.DataFrame({"id": ["m", "m"], "r1": [10, 10], "r2": [11, 11]}), gt_pairs, call)
        run_case(sc, qid, "empty", None, gt_pairs, call)
        run_case(sc, qid, "twocol", pd.DataFrame({"id": ["m"], "r1": [10]}), gt_pairs, call)

    # Q8 sentiment counts
    gt8 = pd.DataFrame({"scoreSentiment": ["NEGATIVE", "POSITIVE"], "count": [106, 14]})
    call = MOVIE._evaluate_q8
    run_case(sc, 8, "exact", pd.DataFrame({"s": ["negative", "positive"], "c": [106, 14]}), gt8, call)
    run_case(sc, 8, "off", pd.DataFrame({"s": ["NEGATIVE", "POSITIVE"], "c": [100, 20]}), gt8, call)
    run_case(sc, 8, "empty", None, gt8, call)
    run_case(sc, 8, "missing-cat", pd.DataFrame({"s": ["NEGATIVE"], "c": [106]}), gt8, call)

    # Q9/Q10 ranking
    gt9 = pd.DataFrame({"reviewId": [1, 2, 3, 4], "reviewScore": [2.5, 3.5, 4.0, 1.0]})
    for qid in (9, 10):
        call = getattr(MOVIE, f"_evaluate_q{qid}")
        run_case(sc, qid, "perfect", pd.DataFrame({"id": [1, 2, 3, 4], "sc": [2.5, 3.5, 4.0, 1.0]}), gt9, call)
        run_case(sc, qid, "reversed", pd.DataFrame({"id": [1, 2, 3, 4], "sc": [4.0, 3.0, 2.0, 5.0]}), gt9, call)
        run_case(sc, qid, "partial", pd.DataFrame({"id": [1, 2, 5], "sc": [2.5, 1.0, 9.0]}), gt9, call)
        run_case(sc, qid, "empty", None, gt9, call)


# ---------------------------------------------------------------------------
# ANIMALS cases
# ---------------------------------------------------------------------------


def animals_cases():
    sc = "animals"
    # Q1/Q2 aggregation
    gt = pd.DataFrame({"count_star()": [5]})
    for qid in (1, 2):
        call = getattr(ANIMALS, f"_evaluate_q{qid}")
        run_case(sc, qid, "exact", pd.DataFrame({"c": [5]}), gt, call)
        run_case(sc, qid, "off", pd.DataFrame({"c": [3]}), gt, call)
        run_case(sc, qid, "empty", None, gt, call)

    # Q3/Q4 top city
    gt34 = pd.DataFrame({"City": ["Eldoret", "Kisumu"]})
    for qid in (3, 4):
        call = getattr(ANIMALS, f"_evaluate_q{qid}")
        run_case(sc, qid, "correct-one", pd.DataFrame({"City": ["Kisumu"]}), gt34, call)
        run_case(sc, qid, "wrong-one", pd.DataFrame({"City": ["Nairobi"]}), gt34, call)
        run_case(sc, qid, "two-rows", pd.DataFrame({"City": ["Eldoret", "Kisumu"]}), gt34, call)
        run_case(sc, qid, "empty", None, gt34, call)

    # Q5-Q9 generic retrieval
    gt5 = pd.DataFrame({"City": ["Mombasa", "Kisumu", "Nakuru"]})
    for qid in (5, 6, 7, 8, 9):
        call = getattr(ANIMALS, f"_evaluate_q{qid}")
        run_case(sc, qid, "correct-subset", pd.DataFrame({"City": ["Mombasa", "Kisumu"]}), gt5, call)
        run_case(sc, qid, "wrong", pd.DataFrame({"City": ["X", "Y"]}), gt5, call)
        run_case(sc, qid, "dup", pd.DataFrame({"City": ["Mombasa", "Mombasa"]}), gt5, call)
        run_case(sc, qid, "empty", None, gt5, call)
        run_case(sc, qid, "exact", pd.DataFrame({"City": ["Mombasa", "Kisumu", "Nakuru"]}), gt5, call)

    # Q10 top city+station
    gt10 = pd.DataFrame({"City": ["Eldoret", "Kisumu"], "StationID": ["Station_A", "Station_A"]})
    call = ANIMALS._evaluate_q10
    run_case(sc, 10, "correct-one", pd.DataFrame({"City": ["Kisumu"], "StationID": ["Station_A"]}), gt10, call)
    run_case(sc, 10, "wrong-one", pd.DataFrame({"City": ["Kisumu"], "StationID": ["Station_Z"]}), gt10, call)
    run_case(sc, 10, "two-rows", pd.DataFrame({"City": ["Eldoret", "Kisumu"], "StationID": ["Station_A", "Station_A"]}), gt10, call)
    run_case(sc, 10, "empty", None, gt10, call)


# ---------------------------------------------------------------------------
# CARS cases
# ---------------------------------------------------------------------------


def cars_cases():
    sc = "cars"
    # id-set queries : Q1,Q2,Q6,Q7,Q9
    gt = pd.DataFrame({"car_id": [1, 2, 3, 4, 5]})
    for qid in (1, 2, 6, 7, 9):
        call = getattr(CARS, f"_evaluate_q{qid}")
        run_case(sc, qid, "correct-subset", pd.DataFrame({"CAR_ID": [1, 2, 3]}), gt, call)
        run_case(sc, qid, "wrong", pd.DataFrame({"CAR_ID": [100, 200]}), gt, call)
        run_case(sc, qid, "dup", pd.DataFrame({"CAR_ID": [1, 1, 2, 99]}), gt, call)
        run_case(sc, qid, "empty", None, gt, call)
        run_case(sc, qid, "exact", pd.DataFrame({"CAR_ID": [1, 2, 3, 4, 5]}), gt, call)

    # Q3 vin, LIMIT 10 balanced (min sampling)
    gt3 = pd.DataFrame({"vin": [f"V{i}" for i in range(15)]})
    call = CARS._evaluate_q3
    run_case(sc, 3, "some-correct", pd.DataFrame({"VIN": ["V0", "V1", "V2"]}), gt3, call)
    run_case(sc, 3, "zero-correct", pd.DataFrame({"VIN": ["Z0", "Z1"]}), gt3, call)
    run_case(sc, 3, "empty", None, gt3, call)
    run_case(sc, 3, "dup", pd.DataFrame({"VIN": ["V0", "V0", "V3"]}), gt3, call)

    # Q4/Q5 aggregation
    gt4 = pd.DataFrame({"average_age": [13.7325]})
    call = CARS._evaluate_q4
    run_case(sc, 4, "exact", pd.DataFrame({"average_age": [13.7325]}), gt4, call)
    run_case(sc, 4, "off", pd.DataFrame({"average_age": [10.0]}), gt4, call)
    run_case(sc, 4, "empty", None, gt4, call)
    gt5 = pd.DataFrame({"transmission": ["Automatic"], "count": [5]})
    call = CARS._evaluate_q5
    run_case(sc, 5, "exact", pd.DataFrame({"transmission": ["Automatic"], "count": [5]}), gt5, call)
    run_case(sc, 5, "off", pd.DataFrame({"transmission": ["Automatic"], "count": [8]}), gt5, call)

    # Q8 car_id LIMIT 100 balanced (min sampling)
    gt8 = pd.DataFrame({"car_id": list(range(150))})
    call = CARS._evaluate_q8
    run_case(sc, 8, "some-correct", pd.DataFrame({"CAR_ID": [0, 1, 2, 3, 4]}), gt8, call)
    run_case(sc, 8, "zero-correct", pd.DataFrame({"CAR_ID": [9000, 9001]}), gt8, call)
    run_case(sc, 8, "empty", None, gt8, call)

    # Q10 macro-f1 : GT index,car_id,problem_category
    gt10 = pd.DataFrame({
        "index": [0, 1, 2, 3],
        "car_id": [10, 20, 30, 40],
        "problem_category": ["electrical system", "fuel system", "engine", "engine"],
    })
    call = CARS._evaluate_q10
    run_case(sc, 10, "perfect", pd.DataFrame({
        "CAR_ID": [10, 20, 30, 40],
        "problem_category": ["Electrical System\n", "fuel system", "engine", "engine"],
    }), gt10, call)
    run_case(sc, 10, "partial", pd.DataFrame({
        "CAR_ID": [10, 20, 30, 40],
        "problem_category": ["engine", "fuel system", "engine", "engine"],
    }), gt10, call)


# ---------------------------------------------------------------------------
# MEDICAL cases
# ---------------------------------------------------------------------------


def medical_cases():
    sc = "medical"
    gt = pd.DataFrame({"patient_id": [1, 2, 3, 4, 5]})
    for qid in (1, 2, 6, 7, 9):
        call = getattr(MEDICAL, f"_evaluate_q{qid}")
        run_case(sc, qid, "correct-subset", pd.DataFrame({"PATIENT_ID": [1, 2, 3]}), gt, call)
        run_case(sc, qid, "wrong", pd.DataFrame({"PATIENT_ID": [100, 200]}), gt, call)
        run_case(sc, qid, "dup", pd.DataFrame({"PATIENT_ID": [1, 1, 2, 99]}), gt, call)
        run_case(sc, qid, "empty", None, gt, call)

    # Q3 patient_id LIMIT 5 (exact n sampling)
    gt3 = pd.DataFrame({"patient_id": list(range(20))})
    call = MEDICAL._evaluate_q3
    run_case(sc, 3, "two-correct", pd.DataFrame({"PATIENT_ID": [0, 1]}), gt3, call)
    run_case(sc, 3, "zero-correct", pd.DataFrame({"PATIENT_ID": [900, 901]}), gt3, call)
    run_case(sc, 3, "empty", None, gt3, call)

    # Q4/Q5 aggregation
    gt4 = pd.DataFrame({"average_acne_age": [30.5]})
    call = MEDICAL._evaluate_q4
    run_case(sc, 4, "exact", pd.DataFrame({"average_acne_age": [30.5]}), gt4, call)
    run_case(sc, 4, "off", pd.DataFrame({"average_acne_age": [25.0]}), gt4, call)
    gt5 = pd.DataFrame({"smoking_history": ["Current"], "count": [1]})
    call = MEDICAL._evaluate_q5
    run_case(sc, 5, "exact", pd.DataFrame({"smoking_history": ["Current"], "count": [1]}), gt5, call)
    run_case(sc, 5, "off", pd.DataFrame({"smoking_history": ["Current"], "count": [3]}), gt5, call)

    # Q8 patient_id LIMIT 100 (exact n sampling)
    gt8 = pd.DataFrame({"patient_id": list(range(120))})
    call = MEDICAL._evaluate_q8
    run_case(sc, 8, "some-correct", pd.DataFrame({"PATIENT_ID": [0, 1, 2]}), gt8, call)
    run_case(sc, 8, "zero-correct", pd.DataFrame({"PATIENT_ID": [9000]}), gt8, call)
    run_case(sc, 8, "empty", None, gt8, call)

    # Q10 macro-f1
    gt10 = pd.DataFrame({
        "index": [0, 1, 2, 3],
        "patient_id": [10, 20, 30, 40],
        "text_diagnosis": ["migraine", "jaundice", "acne", "acne"],
    })
    call = MEDICAL._evaluate_q10
    run_case(sc, 10, "perfect", pd.DataFrame({
        "PATIENT_ID": [10, 20, 30, 40],
        "text_diagnosis": ["Migraine\n", "jaundice", "acne", "acne"],
    }), gt10, call)
    run_case(sc, 10, "partial", pd.DataFrame({
        "PATIENT_ID": [10, 20, 30, 40],
        "text_diagnosis": ["acne", "jaundice", "acne", "acne"],
    }), gt10, call)


# ---------------------------------------------------------------------------
# ECOMM cases (call GenericEvaluator.compute_accuracy_score directly)
# ---------------------------------------------------------------------------


def ecomm_cases():
    sc = "ecomm"
    # f1-score query (id column)
    for qid in (1, 7):
        metric = sm.ECOMM_ACCURACY_METRIC[qid]
        gt = pd.DataFrame({"id": [1, 2, 3, 4]})

        def make_call(metric):
            return lambda pred, g: GenericEvaluator.compute_accuracy_score(metric, g, pred)

        call = make_call(metric)
        run_case(sc, qid, "correct-subset", pd.DataFrame({"id": [1, 2]}), gt, call)
        run_case(sc, qid, "wrong", pd.DataFrame({"id": [90, 91]}), gt, call)
        run_case(sc, qid, "dup", pd.DataFrame({"id": [1, 1, 2, 99]}), gt, call)
        run_case(sc, qid, "empty", None, gt, call)
        run_case(sc, qid, "exact", pd.DataFrame({"id": [1, 2, 3, 4]}), gt, call)

    # adjusted-rand-index query (id, category columns)
    for qid in (3, 4):
        metric = sm.ECOMM_ACCURACY_METRIC[qid]
        gt = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "category": ["A", "A", "B", "B", "C", "C"]})

        def make_call(metric):
            return lambda pred, g: GenericEvaluator.compute_accuracy_score(metric, g, pred)

        call = make_call(metric)
        run_case(sc, qid, "perfect", pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "category": ["A", "A", "B", "B", "C", "C"]}), gt, call)
        run_case(sc, qid, "relabelled", pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "category": ["X", "X", "Y", "Y", "Z", "Z"]}), gt, call)
        run_case(sc, qid, "scrambled", pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "category": ["A", "B", "A", "B", "C", "C"]}), gt, call)
        run_case(sc, qid, "partial-ids", pd.DataFrame({"id": [1, 2, 3, 4], "category": ["A", "A", "B", "B"]}), gt, call)


def main():
    movie_cases()
    animals_cases()
    cars_cases()
    medical_cases()
    ecomm_cases()

    # Report
    print("\n" + "=" * 78)
    print("VALIDATION: scenario_metrics.score()  vs  SemBench real evaluators")
    print("=" * 78)
    n_match = n_mismatch = 0
    cur = None
    for sc, qid, case, status, detail in RESULTS:
        key = (sc, qid)
        if key != cur:
            cur = key
            print(f"\n[{sc}] Q{qid}")
        flag = "OK " if status.startswith("MATCH") else "XX "
        if status.startswith("MATCH"):
            n_match += 1
        else:
            n_mismatch += 1
        shown = detail if status != "MATCH" else ""
        extra = f"   <- {status}" if status != "MATCH" else f"   {status}"
        print(f"    {flag}{case:16s}{extra}" + (f"  {shown}" if shown else ""))

    print("\n" + "-" * 78)
    print(f"TOTAL: {n_match} MATCH, {n_mismatch} MISMATCH  (of {len(RESULTS)} cases)")
    print("-" * 78)
    if n_mismatch:
        print("\nMISMATCH details:")
        for sc, qid, case, status, detail in RESULTS:
            if not status.startswith("MATCH"):
                print(f"  {sc} Q{qid} [{case}]: {detail}")
    return 0 if n_mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
