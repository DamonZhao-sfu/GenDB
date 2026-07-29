import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import physical_frame as PF  # noqa: E402


def test_duplicate_logical_ids_receive_distinct_source_ordinals(tmp_path):
    source = tmp_path / "reviews.csv"
    source.write_text(
        "id,reviewId,reviewText\nm,7,same\nm,7,same\nn,8,other\n",
        encoding="utf-8")
    out = tmp_path / "frame.csv"
    meta = PF.build_frame(
        str(source), str(out), text_cols=["reviewText"],
        filter_col="id", filter_value="m")
    with open(out, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["_semdb_row_id"] for row in rows] == ["0", "1"]
    assert [row["reviewId"] for row in rows] == ["7", "7"]
    assert meta["input_rows"] == 3
    assert meta["output_rows"] == 2
    written_ns = out.stat().st_mtime_ns
    cached = PF.build_frame(
        str(source), str(out), text_cols=["reviewText"],
        filter_col="id", filter_value="m")
    assert out.stat().st_mtime_ns == written_ns
    assert cached["spec"] == meta["spec"]
