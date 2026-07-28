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
import sampling  # noqa: E402

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


def _argv(corpus, gt_file, out, *extra):
    return ["--corpus", corpus, "--id-col", "row_id", "--text-col", "text",
            "--query", "q3a", "--attr", "is_comedy",
            "--label-source", "gt", "--gt-file", gt_file, "--gt-match-col", "title",
            "--out", str(out), *extra]


def _run(corpus, gt_file, out, *extra):
    # --allow-single-class: this fixture is 2.5% positive on purpose, so a uniform
    # draw of a few dozen rows often contains no positive at all. These tests assert
    # sampling plumbing (seeds, disjointness, file shape), not label usefulness; the
    # guard that refuses such a set is exercised separately below.
    argv = _argv(corpus, gt_file, out, "--allow-single-class", *extra)
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


def test_multiple_text_columns_are_rendered_into_each_sampling_document():
    rows = [{"Airlines": "Discover Airlines", "Destinations": "Frankfurt"}]
    assert B._texts(rows, ["Airlines", "Destinations"]) == [
        "Airlines: Discover Airlines. Destinations: Frankfurt."
    ]


def test_score_ranked_head_is_a_certainty_stratum():
    """A 10% score-stratified draw must not randomly miss the very rows its score
    identified as most informative (the old q6/q7 failure)."""
    ids = [str(i) for i in range(200)]
    scores = [float(i) for i in range(200)]
    strata = B.strata_by_score_decile(scores, 5)
    sample = B.sample_stratified_with_score_anchors(
        ids, strata, scores, 20, 7, gamma=2, min_per_stratum=2)
    assert sample.meta["certainty_n"] == 10
    assert set(ids[-10:]) <= set(sample.ids)
    assert all(sample.pi[i] == 1.0 for i in ids[-10:])
    sample.check()


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


# --- sizing by rate --------------------------------------------------------

def test_rate_resolves_against_the_corpus_size(corpus, gt_file, tmp_path):
    select, cert, _ = _run(corpus, gt_file, tmp_path / "v",
                           "--rate", "0.25", "--cert-rate", "0.1")
    assert len(select["ids"]) == 30      # ceil(0.25 * 120)
    assert len(cert["ids"]) == 12        # ceil(0.10 * 120)


def test_rate_and_count_agree_when_both_describe_the_same_size(corpus, gt_file, tmp_path):
    by_rate, _, _ = _run(corpus, gt_file, tmp_path / "a", "--rate", "0.25", "--seed", "5")
    by_count, _, _ = _run(corpus, gt_file, tmp_path / "b", "--n", "30", "--seed", "5")
    assert by_rate["ids"] == by_count["ids"]


def test_giving_both_a_rate_and_a_count_is_refused(corpus, gt_file, tmp_path):
    """They would disagree the moment the corpus changes size, so resolving one
    silently would make the design depend on which flag won."""
    with pytest.raises(SystemExit, match="one or the other"):
        _run(corpus, gt_file, tmp_path / "v", "--n", "30", "--rate", "0.25")


def test_giving_neither_a_rate_nor_a_count_is_refused(corpus, gt_file, tmp_path):
    with pytest.raises(SystemExit, match="required"):
        _run(corpus, gt_file, tmp_path / "v")


@pytest.mark.parametrize("rate", ["-0.1", "1.5"])
def test_a_rate_outside_the_unit_interval_is_refused(corpus, gt_file, tmp_path, rate):
    with pytest.raises(SystemExit, match="not in"):
        _run(corpus, gt_file, tmp_path / "v", "--rate", rate)


def test_a_tiny_positive_rate_still_draws_one_row(corpus, gt_file, tmp_path):
    """Rounding a requested 0.1% down to zero would produce an empty val set that
    silently scores every program identically."""
    select, _, _ = _run(corpus, gt_file, tmp_path / "v", "--rate", "0.001")
    assert len(select["ids"]) == 1


def test_small_rate_automatically_coarsens_score_strata(corpus, gt_file, tmp_path):
    """cars image tables have 39 rows: rate=.05 gives n=2, which must not fail only
    because the workload-wide default asks for five strata."""
    select, _, _ = _run(
        corpus, gt_file, tmp_path / "v", "--n", "2",
        "--method", "stratified", "--strata-by", "score-decile",
        "--importance-by", "query-similarity", "--query-nl", "comedy",
        "--strata-k", "5")
    assert len(select["ids"]) == 2


def test_two_row_score_sample_keeps_top_candidate_and_population_coverage():
    ids = [str(i) for i in range(39)]
    scores = [float(i) for i in range(39)]
    strata = B.strata_by_score_decile(scores, 1)
    sample = B.sample_stratified_with_score_anchors(
        ids, strata, scores, 2, 7, gamma=2, min_per_stratum=2)
    assert sample.meta["certainty_n"] == 1
    assert "38" in sample.ids and sample.pi["38"] == 1
    assert any(sample.pi[row_id] < 1 for row_id in sample.ids)
    sample.check()


def test_an_oversized_rate_pair_is_refused(corpus, gt_file, tmp_path):
    with pytest.raises(SystemExit, match="exceeds the corpus size"):
        _run(corpus, gt_file, tmp_path / "v", "--rate", "0.8", "--cert-rate", "0.5")


# --- single-class guard ----------------------------------------------------

def test_a_single_class_select_half_is_refused_by_default(corpus, gt_file, tmp_path):
    """The fixture is 2.5% positive, so a uniform 20-row draw usually catches none.
    Such a set scores `return false` perfectly and cannot rank programs at all."""
    with pytest.raises(SystemExit, match="single-class|unusable"):
        assert B.main(_argv(corpus, gt_file, tmp_path / "v", "--n", "20", "--seed", "3")) == 0


def test_the_refusal_names_a_way_out(corpus, gt_file, tmp_path):
    with pytest.raises(SystemExit) as exc:
        B.main(_argv(corpus, gt_file, tmp_path / "v", "--n", "20", "--seed", "3"))
    text = str(exc.value)
    assert "--rate" in text and "score-decile" in text
    assert "--allow-single-class" in text


def test_the_guard_can_be_overridden_and_records_why(corpus, gt_file, tmp_path):
    select, _, _ = _run(corpus, gt_file, tmp_path / "v", "--n", "20", "--seed", "3")
    warnings = " ".join(select["provenance"]["warnings"])
    assert "unusable" in warnings


def test_class_balance_is_recorded_for_every_build(corpus, gt_file, tmp_path):
    select, _, _ = _run(corpus, gt_file, tmp_path / "v", "--n", "30")
    balance = select["provenance"]["class_balance"]
    assert sum(balance.values()) == 30


def test_a_two_class_sample_passes_the_guard_without_the_override(corpus, gt_file,
                                                                  tmp_path):
    """Taking nearly the whole corpus necessarily catches the 3 positives."""
    argv = _argv(corpus, gt_file, tmp_path / "v", "--n", "110")
    assert B.main(argv) == 0
    select = json.load(open(os.path.join(tmp_path / "v", "select.json")))
    assert set(select["labels"].values()) == {"true", "false"}


# --- score-decile strata ---------------------------------------------------

def test_score_decile_strata_concentrate_labels_on_the_query_like_rows(corpus, gt_file,
                                                                       tmp_path):
    """The design that makes a rare-positive predicate measurable: buckets of TF-IDF
    similarity to the query, Neyman-allocated. It must find positives where a
    uniform draw of the same size does not."""
    select, _, _ = _run(corpus, gt_file, tmp_path / "v", "--n", "30", "--seed", "3",
                        "--method", "stratified", "--strata-by", "score-decile",
                        "--importance-by", "query-similarity",
                        "--query-nl", "a hilarious comedy full of jokes")
    assert "true" in set(select["labels"].values())


def test_score_decile_keeps_exact_inclusion_probabilities(corpus, gt_file, tmp_path):
    """Unlike pps, a stratified design's weights sum to N exactly, not in expectation."""
    select, _, _ = _run(corpus, gt_file, tmp_path / "v", "--n", "40",
                        "--method", "stratified", "--strata-by", "score-decile",
                        "--importance-by", "query-similarity", "--query-nl", "comedy")
    assert sum(select["weights"].values()) == pytest.approx(120.0, rel=1e-6)


# --- disproportionate allocation -------------------------------------------

def _rates(scores, k, gamma, n):
    """Realized per-stratum sampling rate for a tilt, lowest stratum first."""
    strata = B.strata_by_score_decile(scores, k)
    sizes = {h: strata.count(h) for h in set(strata)}
    sigma = B.tilt_from_scores(scores, strata, gamma) if gamma > 0 else None
    alloc = sampling.allocate(sizes, n, allocation="neyman" if sigma else "proportional",
                              sigma=sigma, min_per_stratum=2)
    return [alloc[h] / sizes[h] for h in sorted(sizes)]


SCORES = [i / 250 for i in range(250)]          # strictly increasing, no ties


def test_a_tilt_samples_high_score_strata_harder():
    rates = _rates(SCORES, k=5, gamma=2.0, n=70)
    assert rates == sorted(rates), "rate must increase with the score"
    assert rates[-1] > 4 * rates[0]


def test_zero_tilt_is_proportional():
    rates = _rates(SCORES, k=5, gamma=0, n=70)
    assert max(rates) - min(rates) < 1e-9


def test_a_stronger_tilt_concentrates_further():
    weak, strong = _rates(SCORES, k=5, gamma=2.0, n=70), _rates(SCORES, k=5, gamma=3.0, n=70)
    assert strong[-1] >= weak[-1]
    assert strong[0] <= weak[0]


def test_every_stratum_keeps_a_positive_inclusion_probability():
    """A tilted-away stratum still has to be observable, or its units have no
    estimator at all."""
    assert all(r > 0 for r in _rates(SCORES, k=5, gamma=6.0, n=70))


def test_the_tilt_uses_rank_not_raw_score():
    """CLIP cosines sit in a narrow band (~0.20-0.33 on ecomm q2); rescaling raw
    values there compresses every stratum toward the same tilt, which is exactly the
    flat allocation that drew 1 of 5 positives."""
    narrow = [0.20 + 0.13 * i / 249 for i in range(250)]
    wide = [float(i) for i in range(250)]
    assert _rates(narrow, k=5, gamma=2.0, n=70) == _rates(wide, k=5, gamma=2.0, n=70)


def test_tilt_weights_are_recorded_so_the_estimate_can_be_corrected(corpus, gt_file,
                                                                    tmp_path):
    """A tilt is only legitimate because the HT weights undo it. For a stratified
    design those weights are exact, so they must still sum to N."""
    select, cert, _ = _run(corpus, gt_file, tmp_path / "v", "--n", "40", "--cert-n", "20",
                           "--method", "stratified", "--strata-by", "score-decile",
                           "--importance-by", "query-similarity", "--query-nl", "comedy",
                           "--score-tilt", "3")
    assert sum(select["weights"].values()) == pytest.approx(120.0, rel=1e-6)
    assert len(set(round(w, 6) for w in select["weights"].values())) > 1, \
        "a tilted design must produce UNEQUAL weights"


def test_score_decile_without_a_score_source_is_refused(corpus, gt_file, tmp_path):
    with pytest.raises(SystemExit, match="needs --query-nl"):
        _run(corpus, gt_file, tmp_path / "v", "--n", "30",
             "--method", "stratified", "--strata-by", "score-decile",
             "--importance-by", "query-similarity")


def test_score_strata_survive_heavy_ties(corpus, gt_file, tmp_path):
    """Regression: quantile cut points collapse when most scores are equal, which
    produced ONE stratum holding the whole corpus while the run still reported
    itself as stratified. Rank bucketing must always yield k of them."""
    strata = B.strata_by_score_decile([0.4] * 100 + [0.9] * 5, k=4)
    assert len(set(strata)) == 4
    sizes = [strata.count(s) for s in sorted(set(strata))]
    assert max(sizes) - min(sizes) <= 1


def test_score_strata_order_lowest_bucket_first():
    strata = B.strata_by_score_decile([0.1, 0.2, 0.8, 0.9], k=2)
    assert strata[0] == strata[1] == "s0"
    assert strata[2] == strata[3] == "s1"


def test_score_strata_degenerate_inputs_do_not_crash():
    assert B.strata_by_score_decile([], k=4) == []
    assert len(set(B.strata_by_score_decile([1.0] * 10, k=4))) == 4
    assert len(B.strata_by_score_decile([0.5, float("nan")], k=4)) == 2


# --- oracle labeling -------------------------------------------------------

@pytest.fixture
def stub_oracle(monkeypatch):
    """Answer 'true' for the gold titles, 'false' otherwise, with a calibrated score."""
    sys.path.insert(0, os.path.dirname(HERE))
    import semvqa

    def reply(text, question, choices=None, *, cfg, default=0.5):
        comedic = "hilarious" in str(text)
        return {"answer": "true" if comedic else "false", "score": 0.9,
                "score_source": "logprobs", "raw": "", "error": None}

    monkeypatch.setattr(semvqa, "txt_vqa_detail", reply)
    return reply


SQL = """SELECT row_id FROM movies
WHERE AI.IF(('Is this a comedy?', movies.text), connection_id => '<<c>>')"""


def test_oracle_labels_come_from_the_model_not_the_answer_key(corpus, tmp_path,
                                                              stub_oracle):
    sql = tmp_path / "q3a.sql"
    sql.write_text(SQL)
    out = tmp_path / "v"
    assert B.main(["--corpus", corpus, "--id-col", "row_id", "--text-col", "text",
                   "--query", "q3a", "--attr", "is_comedy", "--sql", str(sql),
                   "--label-source", "oracle", "--endpoint", "http://x/v1",
                   "--oracle-model", "m", "--n", "110", "--out", str(out)]) == 0
    select = json.load(open(os.path.join(out, "select.json")))
    assert set(select["labels"].values()) == {"true", "false"}
    prov = select["provenance"]
    assert prov["label_source"] == "oracle"
    assert prov["oracle"]["question"] == "Is this a comedy?"
    assert prov["oracle"]["model"] == "m"


def test_the_oracle_question_is_read_from_the_sql(corpus, tmp_path, stub_oracle):
    sql = tmp_path / "q.sql"
    sql.write_text(SQL)
    out = tmp_path / "v"
    B.main(["--corpus", corpus, "--id-col", "row_id", "--text-col", "text",
            "--query", "q", "--attr", "a", "--sql", str(sql),
            "--label-source", "oracle", "--endpoint", "http://x/v1",
            "--oracle-model", "m", "--n", "110", "--out", str(out)])
    prov = json.load(open(os.path.join(out, "select.json")))["provenance"]
    assert prov["call_site"]["kind"] == "if"
    assert prov["call_site"]["shape"] == "per_row"


JOIN_SQL = ("SELECT a.id FROM images as a, images as b "
            "WHERE AI.IF(('same colour?', a.ref, b.ref), connection_id => '<<c>>')")


def _oracle_argv(corpus, sql, out, *extra):
    return ["--corpus", corpus, "--id-col", "row_id", "--text-col", "text",
            "--query", "q", "--attr", "a", "--sql", str(sql),
            "--label-source", "oracle", "--endpoint", "http://x/v1",
            "--oracle-model", "m", "--n", "10", "--out", str(out), *extra]


def test_a_join_only_query_explains_that_its_call_sites_are_pairwise(corpus, tmp_path):
    """Saying only 'no per_row call site' reads as 'your SQL has no AI call'. The
    message has to distinguish a parse failure from a shape mismatch."""
    sql = tmp_path / "join.sql"
    sql.write_text(JOIN_SQL)
    with pytest.raises(SystemExit) as exc:
        B.main(_oracle_argv(corpus, sql, tmp_path / "v"))
    text = str(exc.value)
    assert "pairwise" in text and "self-join" in text


def test_an_explicitly_chosen_pairwise_call_site_is_refused(corpus, tmp_path):
    sql = tmp_path / "join.sql"
    sql.write_text(JOIN_SQL)
    with pytest.raises(SystemExit, match="PAIRWISE"):
        B.main(_oracle_argv(corpus, sql, tmp_path / "v", "--call-site", "0"))


def test_oracle_without_an_endpoint_is_refused(corpus, tmp_path):
    with pytest.raises(SystemExit, match="requires --endpoint"):
        B.main(["--corpus", corpus, "--id-col", "row_id", "--text-col", "text",
                "--query", "q", "--attr", "a", "--query-nl", "is it a comedy?",
                "--label-source", "oracle", "--n", "10", "--out", str(tmp_path / "v")])


def test_oracle_with_nothing_to_show_the_model_is_refused(corpus, tmp_path):
    with pytest.raises(SystemExit, match="--image-col or --text-col"):
        B.main(["--corpus", corpus, "--id-col", "row_id",
                "--query", "q", "--attr", "a", "--query-nl", "is it a comedy?",
                "--label-source", "oracle", "--endpoint", "http://x/v1",
                "--oracle-model", "m", "--n", "10", "--out", str(tmp_path / "v")])


def test_oracle_abstentions_are_reported_and_left_unlabeled(corpus, tmp_path,
                                                            monkeypatch):
    """An abstained row must keep its slot in ids/weights while carrying no label."""
    sys.path.insert(0, os.path.dirname(HERE))
    import semvqa

    def flaky(text, question, choices=None, *, cfg, default=0.5):
        if "Movie 7 " in str(text) or "hilarious" in str(text):
            return {"answer": "true", "score": 0.9, "score_source": "logprobs",
                    "raw": "", "error": None}
        return {"answer": "false", "score": 0.5, "score_source": "self",
                "raw": "", "error": None}

    monkeypatch.setattr(semvqa, "txt_vqa_detail", flaky)
    out = tmp_path / "v"
    B.main(["--corpus", corpus, "--id-col", "row_id", "--text-col", "text",
            "--query", "q", "--attr", "a", "--query-nl", "is it a comedy?",
            "--label-source", "oracle", "--endpoint", "http://x/v1",
            "--oracle-model", "m", "--n", "110", "--allow-single-class",
            "--out", str(out)])
    select = json.load(open(os.path.join(out, "select.json")))
    assert len(select["ids"]) == 110
    assert len(select["labels"]) < 110, "uncalibrated rows must not be labeled"
    assert select["provenance"]["oracle"]["abstained"]
    assert any("abstained" in w for w in select["provenance"]["warnings"])


def test_oracle_cost_is_recorded(corpus, tmp_path, stub_oracle):
    out = tmp_path / "v"
    B.main(["--corpus", corpus, "--id-col", "row_id", "--text-col", "text",
            "--query", "q", "--attr", "a", "--query-nl", "is it a comedy?",
            "--label-source", "oracle", "--endpoint", "http://x/v1",
            "--oracle-model", "m", "--n", "110", "--out", str(out)])
    cost = json.load(open(os.path.join(out, "select.json")))["provenance"]["oracle"]["cost"]
    assert cost["calls"] == 110 and cost["cache_hits"] == 0


def test_the_label_cache_makes_a_rebuild_free(corpus, tmp_path, stub_oracle):
    cache = str(tmp_path / "labels.json")
    common = ["--corpus", corpus, "--id-col", "row_id", "--text-col", "text",
              "--query", "q", "--attr", "a", "--query-nl", "is it a comedy?",
              "--label-source", "oracle", "--endpoint", "http://x/v1",
              "--oracle-model", "m", "--n", "110", "--label-cache", cache]
    B.main(common + ["--out", str(tmp_path / "a")])
    B.main(common + ["--out", str(tmp_path / "b")])
    cost = json.load(open(os.path.join(tmp_path / "b",
                                       "select.json")))["provenance"]["oracle"]["cost"]
    assert cost["calls"] == 0 and cost["cache_hits"] == 110


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
