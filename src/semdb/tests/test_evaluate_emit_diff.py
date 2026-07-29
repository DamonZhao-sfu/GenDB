import json, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import evaluate as E


def test_score_pair_matches_legacy_score():
    results = ["a", "b", "c"]
    gold = {"a", "b"}
    assert E.score_pair(results, gold) == E.score(results, gold)


def test_diff_pair_lists_fp_and_fn_with_cap():
    results = ["a", "b", "x", "y"]         # x,y are false positives
    gold = {"a", "b", "z"}                 # z is a false negative
    d = E.diff_pair(results, gold, cap=1)
    assert d["fp_total"] == 2 and d["fn_total"] == 1
    assert d["sampled"] is True            # cap=1 < 2 fps
    assert len(d["false_positives"]) == 1 and len(d["false_negatives"]) == 1
    assert d["false_positives"][0] in ("x", "y")
    assert d["false_negatives"] == ["z"]


def test_emit_diff_writes_json_for_mmqa_q6(tmp_path):
    # q6 = list of Airlines vs a set; simplest mmqa handler.
    pred = tmp_path / "q6.csv"
    pred.write_text("Airlines\nDelta\nBogusAir\n")
    gt = tmp_path / "Q6.json"
    gt.write_text(json.dumps({"ground_truth": ["Delta", "United"]}))
    out = tmp_path / "diff.json"
    d = E.eval_mmqa_diff("q6", str(pred), str(gt), cap=15)
    assert "BogusAir" in d["false_positives"]
    assert "United" in d["false_negatives"]
    # round-trip the writer used by main()
    json.dump(d, open(out, "w"))
    assert json.load(open(out))["fp_total"] == 1


def test_scenario_diff_id_sets(tmp_path, monkeypatch):
    import types, sys as _sys
    fake = types.ModuleType("scenario_metrics")
    gt = tmp_path / "Q1.csv"; gt.write_text("id\n1\n2\n3\n")
    fake._gt_path = lambda scenario, qid, gt_dir, sf: str(gt)
    monkeypatch.setitem(_sys.modules, "scenario_metrics", fake)
    pred = tmp_path / "pred.csv"; pred.write_text("id\n1\n2\n9\n")   # 9 fp, 3 fn
    d = E._scenario_diff(str(pred), str(tmp_path), "movie", "q1", "", 15)
    assert d["false_positives"] == ["9"] and d["false_negatives"] == ["3"]


def test_scenario_diff_uses_shared_named_id_not_ground_truth_index(tmp_path,
                                                                   monkeypatch):
    import types, sys as _sys
    import pandas as pd
    fake = types.ModuleType("scenario_metrics")
    fake._gt_path = lambda *args: str(tmp_path / "unused.csv")
    fake._load_gt = lambda *args: pd.DataFrame({
        "index": [40, 41], "car_id": [1, 2], "label": ["a", "b"]})
    monkeypatch.setitem(_sys.modules, "scenario_metrics", fake)
    pred = tmp_path / "pred-cars.csv"
    pred.write_text("car_id,label\n1,a\n9,x\n")
    d = E._scenario_diff(str(pred), str(tmp_path), "cars", "q10", "2", 15)
    assert d["false_positives"] == ["9"]
    assert d["false_negatives"] == ["2"]
