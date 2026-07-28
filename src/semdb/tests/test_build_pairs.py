"""Pair-frame construction for join predicates.

A join predicate is a function of two rows, so the sampling unit is a pair and the
key is "<id1>-<id2>" -- the composite key SemBench's own join ground truth uses. These
tests pin the frame's shape, its key round-trip, and the frame-restriction behaviour
that decides whether a pairwise validation set is usable at all.
"""
import csv
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import build_pairs as BP  # noqa: E402
import build_valset as BV  # noqa: E402


# --- pair ids --------------------------------------------------------------

def test_pair_id_matches_the_sembench_join_key_format():
    assert BP.pair_id("1645", "2606") == "1645-2606"


def test_pair_ids_round_trip():
    assert BP.split_pair_id(BP.pair_id("1645", "2606")) == ("1645", "2606")


def test_an_id_containing_the_separator_still_round_trips():
    """Splitting on the FIRST '-' would corrupt any id that contains one."""
    assert BP.split_pair_id(BP.pair_id("a-b", "c")) == ("a-b", "c")


def test_a_non_pair_key_is_rejected():
    with pytest.raises(ValueError, match="not a pair id"):
        BP.split_pair_id("1645")


# --- the frame -------------------------------------------------------------

IDS = ["a", "b", "c", "d"]
FILES = [f"{i}.jpg" for i in IDS]


def sims(values=None):
    """Symmetric similarity matrix; `values` overrides specific (i, j) entries."""
    m = np.full((4, 4), 0.5, dtype=np.float32)
    np.fill_diagonal(m, 1.0)
    for (i, j), v in (values or {}).items():
        m[i][j] = m[j][i] = v
    return m


def test_unordered_frame_has_n_choose_2_rows():
    rows = BP.pair_rows(IDS, FILES, sims(), ordered=False, top=None)
    assert len(rows) == 6                       # 4 choose 2
    assert len({r["pair_id"] for r in rows}) == 6


def test_no_row_is_paired_with_itself():
    rows = BP.pair_rows(IDS, FILES, sims(), ordered=False, top=None)
    assert all(r["id1"] != r["id2"] for r in rows)


def test_ordered_frame_emits_both_directions():
    rows = BP.pair_rows(IDS, FILES, sims(), ordered=True, top=None)
    assert len(rows) == 12
    keys = {r["pair_id"] for r in rows}
    assert "a-b" in keys and "b-a" in keys


def test_ordered_diagonal_frame_has_n_squared_rows():
    rows = BP.pair_rows(IDS, FILES, sims(), ordered=True, top=None,
                        include_diagonal=True)
    assert len(rows) == len(IDS) ** 2
    keys = {r["pair_id"] for r in rows}
    assert {"a-a", "a-b", "b-a", "d-d"} <= keys


def test_unordered_diagonal_is_not_duplicated():
    rows = BP.pair_rows(IDS, FILES, sims(), ordered=False, top=None,
                        include_diagonal=True)
    assert len(rows) == 10  # n(n+1)/2
    assert sum(r["id1"] == r["id2"] for r in rows) == 4


def test_both_orderings_carry_the_same_score():
    rows = {r["pair_id"]: r["pair_score"]
            for r in BP.pair_rows(IDS, FILES, sims({(0, 1): 0.9}), ordered=True, top=None)}
    assert rows["a-b"] == rows["b-a"]


def test_the_two_sides_keep_their_order_within_a_row():
    """The question refers to 'the first' and 'the second' product, so file1/file2 must
    follow id1/id2 rather than being normalised."""
    rows = {r["pair_id"]: r for r in BP.pair_rows(IDS, FILES, sims(), ordered=True, top=None)}
    assert rows["a-b"]["file1"].endswith("a.jpg") and rows["a-b"]["file2"].endswith("b.jpg")
    assert rows["b-a"]["file1"].endswith("b.jpg") and rows["b-a"]["file2"].endswith("a.jpg")


def test_the_frame_is_ordered_by_similarity_descending():
    rows = BP.pair_rows(IDS, FILES, sims({(0, 1): 0.99, (2, 3): 0.01}),
                        ordered=False, top=None)
    scores = [float(r["pair_score"]) for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert rows[0]["pair_id"] == "a-b"


def test_top_keeps_the_most_similar_pairs():
    rows = BP.pair_rows(IDS, FILES, sims({(0, 1): 0.99, (0, 2): 0.98}),
                        ordered=False, top=2)
    assert {r["pair_id"] for r in rows} == {"a-b", "a-c"}


def test_top_counts_unordered_pairs_even_when_emitting_both_directions():
    rows = BP.pair_rows(IDS, FILES, sims({(0, 1): 0.99}), ordered=True, top=1)
    assert len(rows) == 2
    assert {r["pair_id"] for r in rows} == {"a-b", "b-a"}


def test_a_two_row_corpus_yields_exactly_one_pair():
    rows = BP.pair_rows(["a", "b"], ["a.jpg", "b.jpg"], sims()[:2, :2],
                        ordered=False, top=None)
    assert len(rows) == 1


def test_a_single_row_corpus_yields_no_pairs():
    assert BP.pair_rows(["a"], ["a.jpg"], np.ones((1, 1), np.float32),
                        ordered=False, top=None) == []


def test_text_pair_rows_keep_role_order_and_diagonal():
    rows = BP.text_pair_rows(
        ["a", "b"], ["alpha", "beta"], np.eye(2, dtype=np.float32),
        ordered=True, top=None, include_diagonal=True)
    by_id = {r["pair_id"]: r for r in rows}
    assert set(by_id) == {"a-a", "a-b", "b-a", "b-b"}
    assert by_id["a-b"]["text1"] == "alpha"
    assert by_id["a-b"]["text2"] == "beta"
    assert by_id["b-a"]["text1"] == "beta"


# --- frame restriction -----------------------------------------------------

def _write_corpus(tmp_path, n):
    path = tmp_path / "IMAGES.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["id", "filename"])
        for i in range(n):
            w.writerow([i, f"{i}.jpg"])
    return str(path)


def test_restricting_the_row_set_shrinks_the_frame_quadratically(tmp_path, monkeypatch):
    """The measurement that makes pairwise val sets viable: on ecomm q9, applying the
    query's own deterministic predicates takes 250 rows -> 18, and the frame 31,125
    pairs -> 153, while keeping all 6 positives. The frame must be built AFTER them."""
    monkeypatch.setattr(BP, "similarity_matrix",
                        lambda paths, model: np.full((len(paths), len(paths)), 0.5,
                                                     dtype=np.float32))
    corpus = _write_corpus(tmp_path, 50)
    keep = tmp_path / "keep.txt"
    keep.write_text("\n".join(str(i) for i in range(10)))
    out = tmp_path / "pairs.csv"
    assert BP.main(["--corpus", corpus, "--id-col", "id", "--image-col", "filename",
                    "--only-ids", str(keep), "--out", str(out)]) == 0
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 45                       # 10 choose 2, not 50 choose 2 = 1225
    assert all(int(r["id1"]) < 10 and int(r["id2"]) < 10 for r in rows)


def test_an_only_ids_file_matching_nothing_is_refused(tmp_path, monkeypatch):
    """Silently producing an empty frame would look like 'this query has no pairs'."""
    monkeypatch.setattr(BP, "similarity_matrix", lambda paths, model: np.zeros((0, 0)))
    corpus = _write_corpus(tmp_path, 10)
    keep = tmp_path / "keep.txt"
    keep.write_text("nope\n")
    with pytest.raises(SystemExit, match="matched none"):
        BP.main(["--corpus", corpus, "--id-col", "id", "--image-col", "filename",
                 "--only-ids", str(keep), "--out", str(tmp_path / "p.csv")])


def test_duplicate_row_ids_are_refused(tmp_path):
    path = tmp_path / "dup.csv"
    path.write_text("id,filename\n1,a.jpg\n1,b.jpg\n")
    with pytest.raises(SystemExit, match="unique"):
        BP.main(["--corpus", str(path), "--id-col", "id", "--image-col", "filename",
                 "--out", str(tmp_path / "p.csv")])


def test_a_missing_column_is_reported_with_what_is_available(tmp_path):
    corpus = _write_corpus(tmp_path, 3)
    with pytest.raises(SystemExit, match="available"):
        BP.main(["--corpus", corpus, "--id-col", "id", "--image-col", "nope",
                 "--out", str(tmp_path / "p.csv")])


def test_a_missing_corpus_reports_a_message_not_a_traceback(tmp_path):
    with pytest.raises(SystemExit, match="cannot read corpus"):
        BP.main(["--corpus", "/nonexistent.csv", "--id-col", "id",
                 "--image-col", "f", "--out", str(tmp_path / "p.csv")])


def test_the_written_frame_is_readable_as_a_build_valset_corpus(tmp_path, monkeypatch):
    """The whole point of materializing pairs as a CSV: everything downstream treats
    it as an ordinary corpus whose id column happens to be a pair key."""
    monkeypatch.setattr(BP, "similarity_matrix",
                        lambda paths, model: np.full((len(paths), len(paths)), 0.5,
                                                     dtype=np.float32))
    corpus = _write_corpus(tmp_path, 6)
    out = tmp_path / "pairs.csv"
    BP.main(["--corpus", corpus, "--id-col", "id", "--image-col", "filename",
             "--out", str(out)])
    rows = list(csv.DictReader(open(out)))
    assert set(rows[0]) == {"pair_id", "id1", "id2", "file1", "file2", "pair_score"}
    assert len({r["pair_id"] for r in rows}) == len(rows), "pair ids must be unique"


# --- cross-table frames (a join between two DIFFERENT tables) --------------
#
# mmqa q2a/q7 join a structured table to the image table with the AI predicate itself
# as the join condition. The sampling unit is (structured row, image), so the frame is
# the full L x R product scored by CLIP TEXT-image similarity -- not the self-join
# upper triangle scored image-image.

def _write_left(tmp_path, n, name="LEFT.csv"):
    path = tmp_path / name
    with open(path, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["id", "Airlines"])
        for i in range(n):
            w.writerow([i, f"airline {i}"])
    return str(path)


def _cross_args(left, right, out, *extra):
    return ["--corpus", left, "--id-col", "id", "--text-col", "Airlines",
            "--right", right, "--right-id-col", "filename",
            "--right-image-col", "filename", "--out", out, *extra]


def _stub_sims(monkeypatch, value=0.5):
    monkeypatch.setattr(BP, "text_image_similarity",
                        lambda texts, paths, model: np.full((len(texts), len(paths)),
                                                            value, dtype=np.float32))


def test_cross_table_frame_is_the_full_cartesian_product(tmp_path, monkeypatch):
    """L x R, not L choose 2: the two sides are different entities, so there is no
    i<j to dedupe on. mmqa q7 is 200 x 200 = 40,000 pairs."""
    _stub_sims(monkeypatch)
    left, right = _write_left(tmp_path, 3), _write_corpus(tmp_path, 5)
    out = tmp_path / "pairs.csv"
    assert BP.main(_cross_args(left, right, str(out))) == 0
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 15


def test_join_validation_rate_is_applied_to_the_full_pair_population(tmp_path,
                                                                    monkeypatch):
    """A 20% join val set over 3x5 rows is ceil(15*0.2)=3, not a percentage
    of a globally pruned candidate head."""
    _stub_sims(monkeypatch)
    left, right = _write_left(tmp_path, 3), _write_corpus(tmp_path, 5)
    frame = tmp_path / "pairs.csv"
    BP.main(_cross_args(left, right, str(frame)))
    out = tmp_path / "val"
    assert BV.main([
        "--corpus", str(frame), "--id-col", "pair_id", "--text-col", "text1",
        "--query", "q", "--attr", "answer", "--query-nl", "does the pair match?",
        "--method", "uniform", "--rate", "0.2", "--label-source", "none",
        "--out", str(out),
    ]) == 0
    import json
    select = json.load(open(out / "select.json"))
    assert select["design"]["N"] == 15
    assert len(select["ids"]) == 3


def test_cross_table_frame_carries_the_left_text_and_the_right_image(tmp_path,
                                                                    monkeypatch):
    """The column shape oracle_label.make_pair_caller consumes: --pair-image-cols
    file2 --text-col text1 asks a one-text-one-image question."""
    _stub_sims(monkeypatch)
    left, right = _write_left(tmp_path, 2), _write_corpus(tmp_path, 2)
    out = tmp_path / "pairs.csv"
    BP.main(_cross_args(left, right, str(out)))
    rows = list(csv.DictReader(open(out)))
    assert set(rows[0]) == {"pair_id", "id1", "id2", "text1", "file2", "pair_score"}
    for r in rows:
        assert r["text1"] == f"airline {r['id1']}"
        assert r["id2"] in r["file2"]


def test_cross_table_pair_id_joins_the_two_sides(tmp_path, monkeypatch):
    """The key must be '<left_id>-<image_id>' -- what SemBench's join ground truth
    lists and what the solver's trace has to key on."""
    _stub_sims(monkeypatch)
    left, right = _write_left(tmp_path, 2), _write_corpus(tmp_path, 3)
    out = tmp_path / "pairs.csv"
    BP.main(_cross_args(left, right, str(out)))
    rows = list(csv.DictReader(open(out)))
    assert len({r["pair_id"] for r in rows}) == len(rows)
    for r in rows:
        assert BP.split_pair_id(r["pair_id"]) == (r["id1"], r["id2"])


def test_cross_table_frame_is_ordered_by_similarity_descending(tmp_path, monkeypatch):
    """score-decile stratification reads pair_score, and --top cuts from the head."""
    monkeypatch.setattr(BP, "text_image_similarity",
                        lambda texts, paths, model: np.arange(
                            len(texts) * len(paths), dtype=np.float32
                        ).reshape(len(texts), len(paths)) / 100.0)
    left, right = _write_left(tmp_path, 2), _write_corpus(tmp_path, 3)
    out = tmp_path / "pairs.csv"
    BP.main(_cross_args(left, right, str(out)))
    scores = [float(r["pair_score"]) for r in csv.DictReader(open(out))]
    assert scores == sorted(scores, reverse=True)


def test_cross_table_top_keeps_the_most_similar_pairs(tmp_path, monkeypatch):
    """Pruning restricts the sampling frame. On mmqa q7 all five GT positives rank
    inside the top 187 of 40,000, so a --top cut keeps them."""
    monkeypatch.setattr(BP, "text_image_similarity",
                        lambda texts, paths, model: np.arange(
                            len(texts) * len(paths), dtype=np.float32
                        ).reshape(len(texts), len(paths)) / 100.0)
    left, right = _write_left(tmp_path, 4), _write_corpus(tmp_path, 5)
    out = tmp_path / "pairs.csv"
    BP.main(_cross_args(left, right, str(out), "--top", "6"))
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 6
    assert float(rows[0]["pair_score"]) == pytest.approx(0.19)


def test_only_ids_restricts_the_left_side_of_a_cross_frame(tmp_path, monkeypatch):
    """--only-ids applies the query's deterministic predicates to the structured
    side; the frame then shrinks linearly rather than quadratically."""
    _stub_sims(monkeypatch)
    left, right = _write_left(tmp_path, 10), _write_corpus(tmp_path, 4)
    keep = tmp_path / "keep.txt"
    keep.write_text("0\n1\n2\n")
    out = tmp_path / "pairs.csv"
    BP.main(_cross_args(left, right, str(out), "--only-ids", str(keep)))
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 12                       # 3 x 4, not 10 x 4
    assert {r["id1"] for r in rows} == {"0", "1", "2"}


def test_cross_table_mode_requires_its_own_columns(tmp_path):
    """--right without the columns that describe it must say which are missing."""
    left, right = _write_left(tmp_path, 2), _write_corpus(tmp_path, 2)
    with pytest.raises(SystemExit, match="--text-col"):
        BP.main(["--corpus", left, "--id-col", "id", "--right", right,
                 "--right-id-col", "id", "--right-image-col", "filename",
                 "--out", str(tmp_path / "p.csv")])


def test_self_join_mode_still_requires_image_col(tmp_path):
    """--image-col became optional for cross-table mode; without --right it is still
    mandatory rather than silently producing an unscored frame."""
    left = _write_left(tmp_path, 2)
    with pytest.raises(SystemExit, match="--image-col"):
        BP.main(["--corpus", left, "--id-col", "id", "--out", str(tmp_path / "p.csv")])


def test_duplicate_right_ids_are_refused(tmp_path, monkeypatch):
    """Pair ids would collide and the val set would silently label the wrong pair."""
    _stub_sims(monkeypatch)
    left = _write_left(tmp_path, 2)
    right = tmp_path / "R.csv"
    right.write_text("filename\na.jpg\na.jpg\n")
    with pytest.raises(SystemExit, match="not unique"):
        BP.main(_cross_args(left, str(right), str(tmp_path / "p.csv")))
