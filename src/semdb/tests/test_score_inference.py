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
