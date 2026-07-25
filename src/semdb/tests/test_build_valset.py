"""build_valset CLI: reproducible draws, a genuinely sealed CERT split, and a
val file the existing --val-file path can still read."""
import csv
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import build_valset as B  # noqa: E402

SCRIPT = os.path.join(HERE, "..", "build_valset.py")
GOLD = ["Movie 3", "Movie 11", "Movie 42"]


@pytest.fixture
def corpus(tmp_path):
    """120 rows; 3 of them are the gold positives (2.5%, the imbalanced shape)."""
    path = tmp_path / "corpus.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        # decade: 6 values (a stratifiable column). pair_id: 60 values x 2 rows
        # (more strata than any sample size we ask for).
        writer.writerow(["row_id", "title", "text", "decade", "pair_id"])
        for i in range(120):
            comedic = f"Movie {i}" in GOLD
            text = ("a hilarious comedy full of jokes " if comedic else "a solemn wartime drama ") * (3 + i % 7)
            writer.writerow([i, f"Movie {i}", text, f"d{i % 6}", f"p{i // 2}"])
    return str(path)


@pytest.fixture
def gt_file(tmp_path):
    path = tmp_path / "Q3a.json"
    path.write_text(json.dumps({"nl_question": "Which movies are comedies?",
                                "ground_truth": GOLD}))
    return str(path)


def _run(corpus, gt_file, out, *extra):
    argv = ["--corpus", corpus, "--id-col", "row_id", "--text-col", "text",
            "--query", "q3a", "--attr", "is_comedy",
            "--label-source", "gt", "--gt-file", gt_file, "--gt-match-col", "title",
            "--out", str(out), *extra]
    assert B.main(argv) == 0
    select = json.load(open(os.path.join(out, "select.json")))
    cert_path = os.path.join(out, "cert.json")
    cert = json.load(open(cert_path)) if os.path.exists(cert_path) else None
    manifest = json.load(open(os.path.join(out, "split_manifest.json")))
    return select, cert, manifest


# --- format / compatibility ------------------------------------------------

def test_select_file_is_readable_by_the_existing_val_path(corpus, gt_file, tmp_path):
    select, _, _ = _run(corpus, gt_file, tmp_path / "v", "--n", "30", "--cert-n", "20")
    # The orchestrator only requires `labels`; evaluate.score_inference only reads
    # `labels`, `attr`, `query`. Everything else must be additive.
    assert select["query"] == "q3a" and select["attr"] == "is_comedy"
    assert len(select["labels"]) == 30
    assert set(select["labels"].values()) <= {"true", "false"}


def test_extra_fields_carry_the_design_and_weights(corpus, gt_file, tmp_path):
    select, _, _ = _run(corpus, gt_file, tmp_path / "v", "--n", "30", "--cert-n", "20")
    assert select["design"]["method"] == "uniform"
    assert select["design"]["N"] == 120 and select["design"]["n"] == 30
    assert len(select["weights"]) == 30
    assert sum(select["weights"].values()) == pytest.approx(120.0, rel=1e-3)


# --- the split -------------------------------------------------------------

def test_select_and_cert_are_disjoint(corpus, gt_file, tmp_path):
    select, cert, manifest = _run(corpus, gt_file, tmp_path / "v", "--n", "30", "--cert-n", "20")
    assert not set(select["labels"]) & set(cert["labels"])
    assert manifest["disjoint"] is True
    assert not set(manifest["select_ids"]) & set(manifest["cert_ids"])
    assert len(manifest["select_ids"]) == 30 and len(manifest["cert_ids"]) == 20


def test_manifest_opens_an_empty_cert_read_log(corpus, gt_file, tmp_path):
    """Any later read of CERT has somewhere to leave a trace."""
    _, _, manifest = _run(corpus, gt_file, tmp_path / "v", "--n", "30", "--cert-n", "20")
    assert manifest["cert_read_log"] == []


def test_zero_cert_writes_no_cert_file(corpus, gt_file, tmp_path):
    out = tmp_path / "v"
    select, cert, manifest = _run(corpus, gt_file, out, "--n", "30", "--cert-n", "0")
    assert cert is None and not os.path.exists(os.path.join(out, "cert.json"))
    assert manifest["cert_ids"] == []


# --- reproducibility -------------------------------------------------------

def test_same_seed_reproduces_the_draw(corpus, gt_file, tmp_path):
    a, _, _ = _run(corpus, gt_file, tmp_path / "a", "--n", "30", "--cert-n", "20", "--seed", "7")
    b, _, _ = _run(corpus, gt_file, tmp_path / "b", "--n", "30", "--cert-n", "20", "--seed", "7")
    assert a["ids"] == b["ids"]


def test_different_seed_changes_the_draw(corpus, gt_file, tmp_path):
    a, _, _ = _run(corpus, gt_file, tmp_path / "a", "--n", "30", "--cert-n", "20", "--seed", "7")
    b, _, _ = _run(corpus, gt_file, tmp_path / "b", "--n", "30", "--cert-n", "20", "--seed", "8")
    assert a["ids"] != b["ids"]


# --- designs ---------------------------------------------------------------

def test_stratified_records_its_allocation(corpus, gt_file, tmp_path):
    select, _, _ = _run(corpus, gt_file, tmp_path / "v", "--n", "30", "--cert-n", "20",
                        "--method", "stratified", "--strata-by", "length", "--strata-k", "4")
    design = select["design"]
    assert design["method"] == "stratified"
    assert sum(v["n_h"] for v in design["strata"].values()) == 50   # parent draw
    assert design["stratum_of"], "stratified design must record each row's stratum"


def test_stratified_by_column_uses_that_column(corpus, gt_file, tmp_path):
    select, _, _ = _run(corpus, gt_file, tmp_path / "v", "--n", "20", "--cert-n", "10",
                        "--method", "stratified", "--strata-by", "column:decade")
    design = select["design"]
    assert design["method"] == "stratified"
    assert set(design["strata"]) == {f"d{i}" for i in range(6)}


def test_tiny_strata_are_pooled_rather_than_refused(corpus, gt_file, tmp_path):
    """One row per stratum cannot survive the SELECT/CERT split; collapsing is the
    survey-standard fix (coarser strata, no bias) and must happen automatically."""
    select, _, _ = _run(corpus, gt_file, tmp_path / "v", "--n", "40", "--cert-n", "20",
                        "--method", "stratified", "--strata-by", "column:title")
    design = select["design"]
    assert all(v["N_h"] >= 2 for v in design["strata"].values())
    assert sum(v["N_h"] for v in design["strata"].values()) == 120   # no rows dropped


def test_a_stratification_finer_than_the_sample_is_refused(corpus, gt_file, tmp_path):
    """60 strata of 2 rows survive pooling but still outnumber n=15+5: unsampled
    strata would get pi=0, for which no unbiased estimator exists. Refuse with an
    actionable message rather than silently sampling a subpopulation."""
    with pytest.raises(SystemExit, match="stratify more coarsely|Raise n"):
        _run(corpus, gt_file, tmp_path / "v", "--n", "15", "--cert-n", "5",
             "--method", "stratified", "--strata-by", "column:pair_id")


def test_importance_oversamples_the_query_relevant_rows(corpus, gt_file, tmp_path):
    """The imbalance case: 3 positives in 120 rows. A uniform draw of 30 expects
    0.75 of them; the similarity proposal should do better."""
    select, cert, _ = _run(corpus, gt_file, tmp_path / "v", "--n", "30", "--cert-n", "20",
                           "--method", "importance", "--importance-by", "query-similarity",
                           "--query-nl", "a hilarious comedy full of jokes",
                           "--epsilon", "0.1")
    drawn = {**select["labels"], **cert["labels"]}
    assert sum(1 for v in drawn.values() if v == "true") >= 2
    assert select["design"]["method"] == "pareto_pps"
    assert select["design"]["w_max"] > 0


def test_importance_warns_that_the_current_scorer_ignores_its_weights(corpus, gt_file, tmp_path):
    """An unequal-probability sample scored unweighted does not estimate corpus
    accuracy. Saying so is the difference between a caveat and a silent bug."""
    select, _, _ = _run(corpus, gt_file, tmp_path / "v", "--n", "30", "--cert-n", "20",
                        "--method", "importance", "--query-nl", "hilarious comedy jokes")
    assert any("unweighted" in w.lower() for w in select["provenance"]["warnings"])


def test_uniform_carries_no_pps_warning(corpus, gt_file, tmp_path):
    select, _, _ = _run(corpus, gt_file, tmp_path / "v", "--n", "30", "--cert-n", "20")
    assert not any("unweighted" in w.lower() for w in select["provenance"]["warnings"])


def test_importance_weights_are_not_all_equal(corpus, gt_file, tmp_path):
    select, _, _ = _run(corpus, gt_file, tmp_path / "v", "--n", "30", "--cert-n", "20",
                        "--method", "importance", "--query-nl", "hilarious comedy jokes")
    assert len(set(round(w, 6) for w in select["weights"].values())) > 1


# --- labels ----------------------------------------------------------------

def test_gt_labels_match_the_ground_truth_set(corpus, gt_file, tmp_path):
    # n + cert-n == 120 == the corpus, so the union is the whole population and
    # the positives must come back exactly.
    select, cert, _ = _run(corpus, gt_file, tmp_path / "v", "--n", "80", "--cert-n", "40")
    every = {**select["labels"], **cert["labels"]}
    positives = {i for i, v in every.items() if v == "true"}
    assert positives == {"3", "11", "42"}


def test_gt_source_is_recorded_with_a_loud_warning(corpus, gt_file, tmp_path):
    select, _, _ = _run(corpus, gt_file, tmp_path / "v", "--n", "30", "--cert-n", "20")
    prov = select["provenance"]
    assert prov["label_source"] == "gt"
    assert prov["warnings"] and "ground truth" in prov["warnings"][0].lower()


def test_label_source_none_emits_ids_without_labels(corpus, tmp_path):
    out = tmp_path / "v"
    assert B.main(["--corpus", corpus, "--id-col", "row_id", "--text-col", "text",
                   "--query", "q3a", "--attr", "is_comedy", "--label-source", "none",
                   "--n", "30", "--cert-n", "10", "--out", str(out)]) == 0
    select = json.load(open(os.path.join(out, "select.json")))
    assert "labels" not in select and len(select["ids"]) == 30


def test_label_source_oracle_is_refused_until_it_exists(corpus, tmp_path):
    with pytest.raises(SystemExit, match="not implemented"):
        B.main(["--corpus", corpus, "--id-col", "row_id", "--text-col", "text",
                "--query", "q3a", "--attr", "a", "--label-source", "oracle",
                "--n", "10", "--out", str(tmp_path / "v")])


# --- input validation ------------------------------------------------------

def test_oversized_request_is_rejected(corpus, gt_file, tmp_path):
    with pytest.raises(SystemExit, match="exceeds the corpus size"):
        _run(corpus, gt_file, tmp_path / "v", "--n", "100", "--cert-n", "50")


def test_unknown_id_column_is_rejected(corpus, gt_file, tmp_path):
    with pytest.raises(SystemExit, match="not in"):
        B.main(["--corpus", corpus, "--id-col", "nope", "--query", "q", "--attr", "a",
                "--n", "5", "--label-source", "none", "--out", str(tmp_path / "v")])


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "dup.csv"
    path.write_text("row_id,text\n1,a\n1,b\n")
    with pytest.raises(SystemExit, match="duplicate id"):
        B.main(["--corpus", str(path), "--id-col", "row_id", "--query", "q", "--attr", "a",
                "--n", "1", "--label-source", "none", "--out", str(tmp_path / "v")])


def test_cli_subprocess_runs_end_to_end(corpus, gt_file, tmp_path):
    out = tmp_path / "v"
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--corpus", corpus, "--id-col", "row_id",
         "--text-col", "text", "--query", "q3a", "--attr", "is_comedy",
         "--method", "stratified", "--n", "30", "--cert-n", "20", "--seed", "7",
         "--label-source", "gt", "--gt-file", gt_file, "--gt-match-col", "title",
         "--out", str(out)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "disjoint" in proc.stdout
    assert "WARNING" in proc.stderr           # the GT-labels caveat must be visible
    assert os.path.exists(os.path.join(out, "select.json"))
