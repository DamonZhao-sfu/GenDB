import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import evaluate as E


def _val():
    return {"query": "q3a", "attr": "genre",
            "labels": {"m1": "comedy", "m2": "drama", "m3": "Comedy", "m4": "drama"}}


def test_accuracy_and_normalization():
    # m1 correct; m2 correct; m3 correct (case/strip normalized); m4 wrong.
    trace = {"attr": "genre", "rows": {"m1": "comedy", "m2": "drama", "m3": " COMEDY ", "m4": "comedy"}}
    d = E.score_inference(trace, _val(), corpus_rows=None, cap=15)
    assert d["n"] == 4 and d["correct"] == 3
    assert abs(d["accuracy"] - 0.75) < 1e-9
    assert d["n_mistakes"] == 1
    assert d["mistakes"][0]["id"] == "m4"
    assert d["mistakes"][0]["predicted"] == "comedy" and d["mistakes"][0]["expected"] == "drama"


def test_missing_id_scores_wrong_with_null_predicted():
    trace = {"attr": "genre", "rows": {"m1": "comedy", "m2": "drama", "m3": "comedy"}}  # m4 absent
    d = E.score_inference(trace, _val(), corpus_rows=None, cap=15)
    assert d["correct"] == 3 and d["n"] == 4
    miss = [m for m in d["mistakes"] if m["id"] == "m4"]
    assert miss and miss[0]["predicted"] is None


def test_mistakes_capped():
    val = {"query": "q", "attr": "g", "labels": {f"m{i}": "a" for i in range(20)}}
    trace = {"attr": "g", "rows": {f"m{i}": "b" for i in range(20)}}  # all wrong
    d = E.score_inference(trace, val, corpus_rows=None, cap=5)
    assert d["n_mistakes"] == 20 and len(d["mistakes"]) == 5 and d["sampled"] is True


def test_corpus_text_snippet_lookup():
    trace = {"attr": "genre", "rows": {"m1": "drama"}}
    val = {"query": "q", "attr": "genre", "labels": {"m1": "comedy"}}
    rows = [{"id": "m1", "overview": "a hilarious comedy about " + "x" * 500}]
    d = E.score_inference(trace, val, corpus_rows=rows, cap=15)
    assert d["mistakes"][0]["id"] == "m1"
    assert d["mistakes"][0]["text"].startswith("a hilarious comedy")
    assert len(d["mistakes"][0]["text"]) <= 220   # truncated


def test_animals_special_retrieval_f1_uses_weighted_group_counts():
    val = {
        "query": "Q3",
        "labels": {"a": "true", "b": "true", "c": "false"},
        "weights": {"a": 5, "b": 1, "c": 1},
    }
    rows = [
        {"ImagePath": "a", "City": "Nairobi"},
        {"ImagePath": "b", "City": "Kisumu"},
        {"ImagePath": "c", "City": "Kisumu"},
    ]
    good = E.score_inference(
        {"rows": {"a": "true", "b": "false", "c": "false"}},
        val, rows, 15, benchmark="animals")
    assert good["objective"]["name"] == "f1"
    assert good["objective"]["value"] == 1.0
    assert good["objective"]["details"]["metric_family"] == "QueryMetricRetrieval"
    assert good["objective"]["details"]["variant"] == "exactly_one_top_city"
    assert good["objective"]["details"]["f1_score"] == 1.0
    assert good["objective"]["details"]["expected"] == ["nairobi"]

    bad = E.score_inference(
        {"rows": {"a": "false", "b": "true", "c": "true"}},
        val, rows, 15, benchmark="animals")
    assert bad["objective"]["value"] == 0.0
    assert bad["objective"]["details"]["predicted"] == ["kisumu"]


def test_ecomm_ari_objective_is_not_row_accuracy_or_f1():
    val = {"query": "q3", "labels": {
        "1": "nike", "2": "nike", "3": "adidas", "4": "adidas"}}
    # Labels are renamed, so exact row accuracy is zero but clustering is identical.
    trace = {"rows": {"1": "a", "2": "a", "3": "b", "4": "b"}}
    result = E.score_inference(trace, val, None, 15, benchmark="ecomm")
    assert result["accuracy"] == 0.0
    assert result["objective"] == {
        "name": "adjusted_rand_index", "value": 1.0, "direction": "maximize",
        "details": {
            "metric_type": "adjusted-rand-index",
            "accuracy": 1.0,
            "n": 4,
        },
    }


def test_aggregation_objective_minimizes_relative_error():
    val = {"query": "Q1", "labels": {
        "a": "true", "b": "true", "c": "false"}}
    result = E.score_inference(
        {"rows": {"a": "true", "b": "false", "c": "false"}},
        val, None, 15, benchmark="animals")
    assert result["objective"]["name"] == "relative_error"
    assert result["objective"]["direction"] == "minimize"
    assert result["objective"]["value"] == 0.5
    assert result["objective"]["details"]["absolute_error"] == 1.0
    assert result["objective"]["details"]["mean_absolute_percentage_error"] == 50.0


def test_multisite_aggregate_does_not_fall_back_to_wrong_f1_objective():
    val = {"query": "Q5", "labels": {"a": "true", "b": "false"}}
    result = E.score_inference(
        {"rows": {"a": "true", "b": "false"}},
        val, None, 15, benchmark="cars")
    assert result["objective"]["name"] == "query_metric_unavailable"
    assert result["objective"]["value"] is None
    assert "multiple semantic call sites" in result["objective"]["details"]["reason"]


def test_cli_score_inference_writes_json(tmp_path):
    import json, subprocess, sys as _sys
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"attr": "genre", "rows": {"m1": "comedy", "m2": "comedy"}}))
    val = tmp_path / "val.json"
    val.write_text(json.dumps({"query": "q3a", "attr": "genre",
                               "labels": {"m1": "comedy", "m2": "drama"}}))
    out = tmp_path / "score.json"
    here = os.path.dirname(__file__)
    r = subprocess.run([_sys.executable, os.path.join(here, "..", "evaluate.py"),
                        "--score-inference", "--trace", str(trace), "--val-file", str(val),
                        "--emit-diff", str(out), "--diff-cap", "15"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = json.load(open(out))
    assert d["n"] == 2 and d["correct"] == 1 and d["n_mistakes"] == 1
    assert d["mistakes"][0]["id"] == "m2"
