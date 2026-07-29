import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import evaluate as E


def test_every_supported_query_has_one_explicit_metric_route():
    expected = {
        "mmqa": set(range(1, 8)),
        "movie": set(range(1, 11)),
        "animals": set(range(1, 11)),
        "cars": set(range(1, 11)),
        "medical": set(range(1, 12)),
        "ecomm": set(range(1, 15)),
    }
    registries = [
        set(E.AGGREGATION_REFINEMENT),
        set(E.EXACT_SINGLE_RESULT_RETRIEVAL),
        set(E.ARI_REFINEMENT),
        set(E.MACRO_F1_REFINEMENT),
        set(E.RANKING_REFINEMENT),
        set(E.F1_REFINEMENT),
        set(E.MULTISITE_QUERY_OBJECTIVE_UNAVAILABLE),
        set(E.UNSUPPORTED_QUERY_METRIC),
    ]
    for benchmark, queries in expected.items():
        for query in queries:
            routes = sum((benchmark, query) in registry for registry in registries)
            assert routes == 1, f"{benchmark}.Q{query} has {routes} metric routes"


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


def test_known_retrieval_query_uses_official_f1_family():
    val = {"query": "Q7", "labels": {"a": "true", "b": "false"}}
    result = E.score_inference(
        {"rows": {"a": "true", "b": "true"}},
        val, None, 15, benchmark="movie")
    assert result["objective"]["name"] == "f1"
    assert result["objective"]["direction"] == "maximize"
    assert result["objective"]["details"]["metric_family"] == "QueryMetricRetrieval"
    assert result["objective"]["details"]["precision"] == 0.5
    assert result["objective"]["details"]["recall"] == 1.0


def test_mmqa_q2b_joint_objective_penalizes_wrong_color_as_fp_and_fn():
    val = {"query": "q2b", "labels": {
        "a-x.png": "match:blue",
        "b-y.png": "no_match",
        "c-z.png": "match:red",
    }}
    result = E.score_inference(
        {"rows": {
            "a-x.png": "match:green",  # wrong positive tuple: FP + FN
            "b-y.png": "match:blue",   # false matching pair: FP
            "c-z.png": "match:red",    # TP
        }},
        val, None, 15, benchmark="mmqa")
    objective = result["objective"]
    assert objective["name"] == "f1"
    assert objective["details"]["variant"] == "filter_then_extract_tuple_f1"
    assert objective["details"]["tp"] == 1
    assert objective["details"]["fp"] == 2
    assert objective["details"]["fn"] == 1
    assert objective["value"] == 0.4


def test_mmqa_q4_objective_scores_genre_memberships_not_list_format():
    val = {"query": "q4", "labels": {
        "a": "comedy, romance",
        "b": "science fiction|horror",
    }}
    result = E.score_inference(
        {"rows": {
            "a": ["Romance", "Comedy"],
            "b": "sci-fi, drama",
        }},
        val, None, 15, benchmark="mmqa")
    objective = result["objective"]
    assert objective["details"]["variant"] == "multi_label_genre_tuple_f1"
    assert objective["details"]["tp"] == 3
    assert objective["details"]["fp"] == 1
    assert objective["details"]["fn"] == 1
    assert objective["value"] == 0.75


def test_mmqa_q5_objective_scores_cross_document_person_intersection():
    val = {"query": "q5", "labels": {
        "a": "Lizzy Caplan, Jesse Bradford",
        "b": "Lizzy Caplan, Brad Pitt",
        "c": "Lizzy Caplan",
    }}
    result = E.score_inference(
        {"rows": {
            "a": ["Lizzy Caplan", "Jesse Bradford"],
            "b": ["Lizzy Caplan", "Brad Pitt"],
            "c": ["Lizzy Caplan"],
        }},
        val, None, 15, benchmark="mmqa")
    objective = result["objective"]
    assert objective["value"] == 1
    assert objective["details"]["variant"] == "cross_document_person_intersection_f1"
    assert objective["details"]["expected"] == ["lizzy caplan"]


def test_zero_retrieval_f1_is_measurable_not_missing():
    val = {"query": "Q7", "labels": {"a": "true", "b": "false"}}
    result = E.score_inference(
        {"rows": {"a": "false", "b": "false"}},
        val, None, 15, benchmark="movie")
    assert result["objective"]["name"] == "f1"
    assert result["objective"]["value"] == 0.0


def test_query_without_official_evaluator_does_not_silently_use_f1():
    val = {"query": "Q11", "labels": {"a": "true", "b": "false"}}
    result = E.score_inference(
        {"rows": {"a": "true", "b": "false"}},
        val, None, 15, benchmark="medical")
    assert result["objective"]["name"] == "query_metric_unavailable"
    assert result["objective"]["value"] is None
    assert "does not define a metric" in result["objective"]["details"]["reason"]


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
