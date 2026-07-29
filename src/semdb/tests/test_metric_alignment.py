import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import evaluate as E
import scenario_metrics as S


def test_animals_special_match_is_retrieval_not_top1_metric():
    result = S._animals_top_city(
        pd.DataFrame({"city": ["Nairobi"]}),
        pd.DataFrame({"city": ["Nairobi"]}),
    )
    assert result["metric"] == "retrieval_f1"
    assert result["metric_family"] == "QueryMetricRetrieval"
    assert result["variant"] == "exactly_one_top_city"
    assert result["f1"] == result["f1_score"] == 1.0


def test_sembench_exact_aggregation_and_ranking_fields_are_emitted():
    aggregation = S._generic_aggregation_evaluation(
        pd.DataFrame({"count": [8]}), pd.DataFrame({"count": [10]}))
    assert aggregation["metric_family"] == "QueryMetricAggregation"
    assert aggregation["mape"] == aggregation["mean_absolute_percentage_error"] == 20.0
    assert aggregation["absolute_error"] == 2.0

    ranking = S._generic_ranking_evaluation(
        pd.DataFrame({"id": [1, 2, 3], "score": [1, 2, 3]}),
        pd.DataFrame({"id": [1, 2, 3], "score": [3, 2, 1]}),
    )
    assert ranking["metric_family"] == "QueryMetricRank"
    assert ranking["spearman"] == ranking["spearman_correlation"]
    assert ranking["kendall"] == ranking["kendall_tau"]


def test_ecomm_metric_type_matches_toml_vocabulary():
    f1 = S._ecomm_f1(
        pd.DataFrame({"id": [1, 2]}), pd.DataFrame({"id": [1]}))
    assert f1["metric"] == f1["metric_type"] == "f1-score"
    assert f1["metric_family"] == "SingleAccuracyScoreWithRetrievalDetails"
    assert f1["accuracy"] == f1["f1_score"] == f1["f1"]

    ari = S._ecomm_ari(
        pd.DataFrame({"id": [1, 2], "category": ["a", "b"]}),
        pd.DataFrame({"id": [1, 2], "category": ["x", "y"]}),
    )
    assert ari["metric"] == ari["metric_type"] == "adjusted-rand-index"
    assert ari["metric_family"] == "SingleAccuracyScore"
    assert ari["accuracy"] == ari["ari"] == ari["adjusted_rand_index"]


def test_mmqa_emits_query_metric_metadata(tmp_path):
    pred = tmp_path / "q6.csv"
    pred.write_text("Airlines\nDelta\n")
    gt = tmp_path / "Q6.json"
    gt.write_text(json.dumps({"ground_truth": ["Delta"]}))
    result = E.eval_mmqa("q6a", str(pred), str(gt))
    assert result["metric"] == "retrieval_f1"
    assert result["metric_family"] == "QueryMetricRetrieval"
    assert result["metric_variant"] == "airline_membership"
    assert result["f1_score"] == result["f1"] == 1.0


def test_final_metric_objective_keeps_secondary_sembench_metrics():
    objective = E.metric_objective({
        "metric": "ranking",
        "spearman_correlation": 0.8,
        "kendall_tau": 0.6,
    })
    assert objective == {
        "name": "spearman_correlation",
        "value": 0.8,
        "direction": "maximize",
        "details": {"kendall_tau": 0.6},
    }


def test_cars_q10_ground_truth_is_rebuilt_for_requested_scale(tmp_path):
    root = tmp_path / "cars"
    gt_dir = root / "raw_results" / "ground_truth"
    sample = root / "data" / "sf_2"
    full = root / "data" / "full_data"
    for path in (gt_dir, sample, full):
        path.mkdir(parents=True)
    # Deliberately stale plain evaluator output from another scale.
    pd.DataFrame({
        "car_id": [99, 100, 101],
        "problem_category": ["wrong", "wrong", "wrong"],
    }).to_csv(gt_dir / "Q10.csv", index=False)
    pd.DataFrame({"complaint_id": [10, 12], "car_id": [1, 2], "summary": ["a", "b"]}).to_csv(
        sample / "text_complaints_data_2.csv", index=False)
    pd.DataFrame({"car_id": [1, 2]}).to_csv(sample / "car_data_2.csv", index=False)
    pd.DataFrame({
        "complaint_id": [10, 11, 12],
        "car_id": [1, 3, 2],
        "component_class": ["ENGINE", "TIRES", "STEERING"],
    }).to_csv(full / "text_complaints_data_full.csv", index=False)
    pd.DataFrame({"car_id": [1, 2, 3]}).to_csv(
        full / "car_data_full.csv", index=False)

    rebuilt = S._load_gt("cars", 10, gt_dir, 2)
    assert rebuilt["car_id"].tolist() == [1, 2]
    assert rebuilt["problem_category"].tolist() == ["engine", "steering"]


def test_macro_f1_penalizes_missing_ids_without_sklearn_length_crash():
    gt = pd.DataFrame({
        "car_id": [1, 2],
        "problem_category": ["engine", "steering"],
    })
    pred = pd.DataFrame({"car_id": [1], "problem_category": ["engine"]})
    result = S._macro_f1(pred, gt, "car_id", "problem_category")
    assert result["covered"] == 1
    assert result["pred_count"] == 1
    assert 0 < result["f1"] < 1
