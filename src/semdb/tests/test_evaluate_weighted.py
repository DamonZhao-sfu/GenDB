"""Horvitz-Thompson estimates from an unequal-probability validation sample.

The tilted stratified design that makes a rare-positive predicate measurable draws
high-score rows several times harder than low-score ones (measured on ecomm q2: it
turns 1-of-5 positives captured into 4-of-5, at the cost of a ~16x spread in the HT
weights). That tilt is only legitimate if the scorer undoes it. These tests pin the
undoing: an unweighted read of such a sample is not an estimate of anything about the
corpus, and a program that answers `false` everywhere must not look good.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluate as E  # noqa: E402


def val(labels, weights=None, **extra):
    out = {"query": "q", "attr": "a", "labels": labels}
    if weights:
        out["weights"] = weights
    out.update(extra)
    return out


def trace(rows):
    return {"rows": rows}


# --- reduces to the old behaviour -----------------------------------------

def test_without_weights_nothing_changes():
    out = E.score_inference(trace({"1": "true", "2": "false"}),
                            val({"1": "true", "2": "true"}), None, 5)
    assert out["accuracy"] == 0.5 and out["correct"] == 1 and out["n"] == 2
    assert "estimates" not in out, "no weights means no corpus estimate to offer"


def test_equal_weights_match_the_unweighted_counts():
    labels = {str(i): ("true" if i < 3 else "false") for i in range(10)}
    rows = {str(i): ("true" if i < 2 else "false") for i in range(10)}
    weights = {str(i): 4.0 for i in range(10)}
    out = E.score_inference(trace(rows), val(labels, weights), None, 5)
    assert out["estimates"]["precision"] == out["unweighted"]["precision"]
    assert out["estimates"]["recall"] == out["unweighted"]["recall"]
    assert out["estimates"]["accuracy"] == out["unweighted"]["accuracy"]


# --- the tilt is undone ----------------------------------------------------

def test_an_oversampled_slice_does_not_dominate_the_estimate():
    """20 positives drawn at 100% and 20 negatives drawn at 10%. Unweighted this
    looks 50% positive; the corpus is 20/220 = 9%."""
    labels = {**{f"p{i}": "true" for i in range(20)},
              **{f"n{i}": "false" for i in range(20)}}
    weights = {**{f"p{i}": 1.0 for i in range(20)},
               **{f"n{i}": 10.0 for i in range(20)}}
    perfect = {k: v for k, v in labels.items()}
    out = E.score_inference(trace(perfect), val(labels, weights), None, 5)
    assert out["unweighted"]["positive_rate"] == pytest.approx(0.5)
    assert out["estimates"]["positive_rate"] == pytest.approx(20 / 220, abs=1e-3)


def test_recovers_the_true_corpus_precision_under_oversampling():
    """The program has 10 false positives among the heavily-downweighted negatives.
    Unweighted precision overstates the damage; each of those rows stands for 10."""
    labels = {**{f"p{i}": "true" for i in range(20)},
              **{f"n{i}": "false" for i in range(20)}}
    weights = {**{f"p{i}": 1.0 for i in range(20)},
               **{f"n{i}": 10.0 for i in range(20)}}
    rows = {**{f"p{i}": "true" for i in range(20)},
            **{f"n{i}": ("true" if i < 10 else "false") for i in range(20)}}
    out = E.score_inference(trace(rows), val(labels, weights), None, 5)
    assert out["unweighted"]["precision"] == pytest.approx(20 / 30, abs=1e-3)
    # weighted: tp = 20*1 = 20, fp = 10*10 = 100
    assert out["estimates"]["precision"] == pytest.approx(20 / 120, abs=1e-3)


# --- precision/recall beat accuracy at a low base rate ---------------------

def test_the_all_false_program_is_exposed_by_recall_not_accuracy():
    """The whole reason the loop needs per-class numbers: at a 2% base rate
    `return false` scores 98% accuracy while getting every positive wrong."""
    labels = {**{f"p{i}": "true" for i in range(2)},
              **{f"n{i}": "false" for i in range(98)}}
    all_false = {k: "false" for k in labels}
    out = E.score_inference(trace(all_false), val(labels), None, 5)
    assert out["unweighted"]["accuracy"] == pytest.approx(0.98)
    assert out["unweighted"]["recall"] == 0.0
    assert out["unweighted"]["f1"] is None or out["unweighted"]["f1"] == 0.0


def test_a_perfect_program_scores_one_on_every_measure():
    labels = {"1": "true", "2": "false", "3": "true"}
    out = E.score_inference(trace(dict(labels)), val(labels, {"1": 2.0, "2": 5.0, "3": 2.0}),
                            None, 5)
    est = out["estimates"]
    assert est["precision"] == 1.0 and est["recall"] == 1.0 and est["f1"] == 1.0


# --- missing predictions ---------------------------------------------------

def test_an_id_the_program_never_emitted_counts_as_a_negative_prediction():
    """The program was asked about the row and said nothing; for a filter that is a
    'no', not a row to skip. Skipping it would let a program raise its score by
    emitting fewer rows."""
    labels = {"1": "true", "2": "true"}
    out = E.score_inference(trace({"1": "true"}), val(labels, {"1": 1.0, "2": 1.0}), None, 5)
    assert out["estimates"]["fn"] == 1.0
    assert out["estimates"]["recall"] == pytest.approx(0.5)


def test_a_row_with_no_weight_falls_back_to_one():
    out = E.score_inference(trace({"1": "true"}), val({"1": "true", "2": "true"},
                                                      {"1": 3.0}), None, 5)
    assert out["estimates"]["tp"] == 3.0 and out["estimates"]["fn"] == 1.0


# --- design reporting ------------------------------------------------------

def test_the_weight_spread_is_reported_so_a_tilt_is_visible():
    out = E.score_inference(trace({"1": "true"}),
                            val({"1": "true", "2": "false"}, {"1": 1.0, "2": 16.0},
                                design={"method": "stratified", "N": 250}), None, 5)
    assert out["design"]["weight_spread"] == 16.0
    assert out["design"]["weighted"] is True
    assert out["design"]["method"] == "stratified"


def test_an_equal_weight_design_is_flagged_as_unweighted():
    out = E.score_inference(trace({"1": "true"}),
                            val({"1": "true", "2": "false"}, {"1": 2.0, "2": 2.0}), None, 5)
    assert out["design"]["weighted"] is False


# --- label vocabulary ------------------------------------------------------

@pytest.mark.parametrize("yes", ["true", "TRUE", "Yes", "1", " true "])
def test_positive_spellings_are_recognised(yes):
    out = E.score_inference(trace({"1": yes}), val({"1": "true"}), None, 5)
    assert out["unweighted"]["tp"] == 1.0


def test_a_non_boolean_value_space_still_scores_accuracy():
    """AI.GENERATE labels are values, not booleans. Precision/recall over a
    'positive' class are meaningless there, but exact-match accuracy is not."""
    labels = {"1": "Navy Blue", "2": "Red", "3": "Red"}
    out = E.score_inference(trace({"1": "Navy Blue", "2": "Red", "3": "Green"}),
                            val(labels), None, 5)
    assert out["accuracy"] == pytest.approx(2 / 3, abs=1e-4)   # reported to 4 dp


def test_empty_labels_do_not_divide_by_zero():
    out = E.score_inference(trace({}), val({}), None, 5)
    assert out["accuracy"] is None and out["n"] == 0
    assert out["unweighted"]["precision"] is None
