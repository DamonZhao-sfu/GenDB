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
