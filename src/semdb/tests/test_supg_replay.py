"""SUPG replay mode: the oracle budget, the leak guard, and target grading.

Replay mode lets the three content-less SUPG datasets run by exposing a cheap
`proxy_score` plus a ground-truth oracle capped at `ORACLE LIMIT`. Every claim that mode
makes is only as good as its enforcement, so these tests pin the three things that would
silently invalidate a result if they broke:

  1. the budget is a hard cap, and repeats are free (else "400 labels" means nothing);
  2. no corpus the generated program reads carries a `label` column (else it can cheat);
  3. an over-budget run cannot report `target_met` (else overspending looks like success).
"""
import csv
import importlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SEMDB = os.path.dirname(HERE)
sys.path.insert(0, SEMDB)
sys.path.insert(0, os.path.join(SEMDB, "data", "supg"))

import build_supg_scenario as BSS  # noqa: E402
import evaluate as EV  # noqa: E402


@pytest.fixture
def oracle(tmp_path):
    """A freshly imported supg_oracle wired to a synthetic 100-row label file."""
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label"])
        for i in range(100):
            writer.writerow([i, int(i % 10 == 0)])
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps({
        "labels_csv": str(labels), "budget": 10,
        "ledger": str(tmp_path / "ledger.json"), "dataset": "t", "query": "Q",
    }))
    os.environ["SUPG_ORACLE_CONFIG"] = str(config)
    sys.modules.pop("supg_oracle", None)
    module = importlib.import_module("supg_oracle")
    yield module
    os.environ.pop("SUPG_ORACLE_CONFIG", None)
    sys.modules.pop("supg_oracle", None)


# --- the budget ------------------------------------------------------------

def test_distinct_ids_are_charged_and_labels_are_correct(oracle):
    assert oracle.oracle([0, 1, 2, 10]) == [True, False, False, True]
    assert oracle.spent() == 4
    assert oracle.remaining() == 6


def test_repeat_lookups_are_free(oracle):
    oracle.oracle(range(5))
    assert oracle.spent() == 5
    # Re-reading the same sample many times must not consume any more budget: an
    # algorithm has to be able to iterate over its labelled sample.
    for _ in range(10):
        oracle.oracle(range(5))
    assert oracle.spent() == 5


def test_a_duplicated_id_within_one_call_is_charged_once(oracle):
    oracle.oracle([7, 7, 7, 7])
    assert oracle.spent() == 1


def test_budget_is_a_hard_cap(oracle):
    oracle.oracle(range(10))
    assert oracle.remaining() == 0
    with pytest.raises(oracle.OracleBudgetExceeded):
        oracle.oracle([42])


def test_an_overshooting_call_is_rejected_whole_not_truncated(oracle):
    """Partial service would leave the program believing it had labelled everything."""
    oracle.oracle(range(8))
    with pytest.raises(oracle.OracleBudgetExceeded):
        oracle.oracle([90, 91, 92, 93, 94])
    assert oracle.spent() == 8


def test_ledger_records_the_spend(oracle, tmp_path):
    oracle.oracle(range(6))
    oracle.oracle(range(6))          # free
    oracle._write_ledger()
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert ledger["distinct_ids"] == 6
    assert ledger["total_requests"] == 12
    assert ledger["exceeded"] is False
    assert ledger["budget"] == 10


def test_ledger_marks_an_exceeded_run(oracle, tmp_path):
    oracle.oracle(range(10))
    with pytest.raises(oracle.OracleBudgetExceeded):
        oracle.oracle([99])
    oracle._write_ledger()
    assert json.loads((tmp_path / "ledger.json").read_text())["exceeded"] is True


# --- the leak guard --------------------------------------------------------

def test_leak_guard_rejects_a_corpus_exposing_labels(tmp_path):
    (tmp_path / "corpus.csv").write_text("id,proxy_score,label\n1,0.5,1\n")
    with pytest.raises(SystemExit, match="labels leaked"):
        BSS.assert_no_labels_in_corpus(tmp_path)


def test_leak_guard_accepts_a_replay_corpus(tmp_path):
    (tmp_path / "corpus.csv").write_text("id,proxy_score\n1,0.5\n")
    BSS.assert_no_labels_in_corpus(tmp_path)          # must not raise


def test_replay_table_carries_no_label_column(tmp_path):
    rows = [(0, True, 0.9), (1, False, 0.1)]

    class DS:
        table = "ontonotes"

    BSS.write_replay_table(tmp_path, DS(), rows)
    with (tmp_path / "ontonotes.csv").open(newline="") as handle:
        header = next(csv.reader(handle))
    assert header == ["id", "proxy_score"]


# --- target grading --------------------------------------------------------

def _tele(metric, value, target=0.9, exceeded=False):
    return {
        "supg": {"kind": "PT" if metric == "precision" else "RT",
                 "graded_metric": metric, "target": target, "oracle_budget": 400,
                 "oracle_calls": 400, "oracle_exceeded": exceeded,
                 "val_oracle_calls": 400},
        "metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0, metric: value},
    }


@pytest.mark.parametrize("metric,value,expected", [
    ("precision", 0.95, "true"),
    ("precision", 0.90, "true"),          # the target is >=, not >
    ("precision", 0.89, "false"),
    ("recall", 0.93, "true"),
    ("recall", 0.42, "false"),
])
def test_target_met_grades_the_class_specific_metric(metric, value, expected):
    row = EV.telemetry_row(_tele(metric, value), "Q5", "supg")
    assert row["graded_metric"] == metric
    assert row["graded_value"] == value
    assert row["target_met"] == expected


def test_an_over_budget_run_cannot_claim_its_target():
    """The number was bought with oracle calls the SUPG contract did not allow."""
    row = EV.telemetry_row(_tele("precision", 0.99, exceeded=True), "Q5", "supg")
    assert row["target_met"] == "false"


def test_grading_survives_the_two_phase_row_build():
    """`main` builds the row from telemetry BEFORE the scenario metrics exist, then
    merges them in. Grading must run after that merge or graded_value reads blank."""
    tele = _tele("precision", 0.95)
    metrics = tele.pop("metrics")
    row = EV.telemetry_row(tele, "Q5", "supg")
    assert row["graded_value"] == ""              # nothing to grade against yet
    row.update(**metrics)
    EV.apply_supg_grading(row, tele)
    assert row["graded_value"] == 0.95
    assert row["target_met"] == "true"


def test_other_benchmarks_keep_the_supg_columns_blank():
    row = EV.telemetry_row({"metrics": {"precision": 0.5, "recall": 0.5}}, "Q1", "cars")
    assert all(row[column] == "" for column in (
        "supg_class", "supg_target", "graded_metric", "graded_value", "target_met",
        "oracle_budget", "oracle_calls", "val_oracle_calls"))
    assert row["precision"] == 0.5
