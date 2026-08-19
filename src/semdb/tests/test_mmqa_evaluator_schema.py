import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import evaluate as E


def evaluate(tmp_path, query, header, row, ground_truth):
    pred = tmp_path / f"{query}.csv"
    pred.write_text(f"{header}\n{row}\n")
    gt = tmp_path / f"{query}.json"
    gt.write_text(json.dumps({"ground_truth": ground_truth}))
    return E.eval_mmqa(query, str(pred), str(gt))


@pytest.mark.parametrize(("query", "header", "row", "ground_truth"), [
    ("q1", "Director", "Jane Doe", ["Jane Doe"]),
    ("q2a", "t.ID,image_filepath", "7,/data/images/logo.png", [[7, "logo.png"]]),
    ("q2b", "id,image_filename,primary_color", "7,logo.png,Blue",
     [[7, "logo.png", "blue"]]),
    ("q3a", "t.title", "Movie", ["Movie"]),
    ("q4", "unnested_genre,movies", "comedy,Movie", {"comedy": ["Movie"]}),
    ("q5", "cast_member", "Actor Name", ["Actor Name"]),
    ("q6c", "airline", "Delta Air Lines", ["Delta Air Lines"]),
    ("q7", "airline,image_filepath", "British Airways,/images/logo.png",
     [["British Airways", "logo.png"]]),
])
def test_all_mmqa_handlers_accept_supported_projection_aliases(
        tmp_path, query, header, row, ground_truth):
    result = evaluate(tmp_path, query, header, row, ground_truth)
    assert result["f1"] == 1.0


def test_q2_ignores_non_projection_diagnostic_columns(tmp_path):
    result = evaluate(
        tmp_path, "q2a", "ID,image_filepath,decision_branch", "7,/images/logo.png,ocr",
        [[7, "logo.png"]],
    )
    assert result["f1"] == 1.0


def test_invalid_projection_reports_schema_instead_of_keyerror(tmp_path):
    pred = tmp_path / "bad.csv"
    pred.write_text("ID,unexpected\n7,value\n")
    gt = tmp_path / "Q2a.json"
    gt.write_text(json.dumps({"ground_truth": [[7, "logo.png"]]}))
    with pytest.raises(ValueError, match=r"MMQA q2.*image identifier.*got columns"):
        E.eval_mmqa("q2a", str(pred), str(gt))


def test_empty_csv_with_valid_header_is_a_valid_empty_prediction(tmp_path):
    pred = tmp_path / "empty.csv"
    pred.write_text("ID,image_filepath\n")
    gt = tmp_path / "Q2a.json"
    gt.write_text(json.dumps({"ground_truth": [[7, "logo.png"]]}))
    result = E.eval_mmqa("q2a", str(pred), str(gt))
    assert result["pred_count"] == 0
    assert result["f1"] == 0.0
