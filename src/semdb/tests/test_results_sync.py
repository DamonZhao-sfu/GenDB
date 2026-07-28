import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sync_results as S  # noqa: E402
import evaluate as E  # noqa: E402


def telemetry(query, history, final_f1):
    return {
        "query": query,
        "provider": "codex",
        "wall_clock_ms": 100,
        "direct": {"agent_stage_ms": 70, "agent_calls": len(history)},
        "refine": {
            "val_n": 40,
            "iterations": len(history) - 1,
            "best_iteration": 1,
            "max_iterations": 5,
            "f1_history": [
                {"iter": i, "f1": f1, "status": "ok", "improved": True}
                for i, f1 in enumerate(history)
            ],
        },
        "metrics": {"f1": final_f1, "precision": 1, "recall": 1},
    }


def test_sync_writes_every_iteration_f1_and_final_f1_separately(tmp_path):
    q7 = tmp_path / "mmqa-q7"
    q7.mkdir()
    (q7 / "telemetry.json").write_text(
        json.dumps(telemetry("q7", [0.8, 0.9, 1.0, 1.0, 1.0, 0.9], 1.0)))
    out = S.sync(tmp_path)
    row = next(csv.DictReader(out.open()))
    assert row["f1"] == "1.0"
    assert [row[f"val_f1_iter_{i}"] for i in range(6)] == [
        "0.8", "0.9", "1.0", "1.0", "1.0", "0.9",
    ]
    assert json.loads(row["val_f1_history"]) == [0.8, 0.9, 1.0, 1.0, 1.0, 0.9]


def test_sync_has_one_row_per_telemetry_and_natural_query_order(tmp_path):
    for query in ("q10", "q2", "q1"):
        directory = tmp_path / f"mmqa-{query}"
        directory.mkdir()
        (directory / "telemetry.json").write_text(
            json.dumps(telemetry(query, [0.5], 0.6)))
    # A stale results file must not create a duplicate when rebuilt.
    (tmp_path / "results.csv").write_text("query\nq2\nq2\n")
    rows = list(csv.DictReader(S.sync(tmp_path).open()))
    assert [row["query"] for row in rows] == ["q1", "q2", "q10"]
    assert all(row["benchmark"] == "mmqa" for row in rows)


def test_future_writer_migrates_old_header_and_replaces_a_rerun(tmp_path):
    out = tmp_path / "results.csv"
    out.write_text("query,benchmark,f1\nq7,mmqa,0.1\n")
    row = E.telemetry_row(
        telemetry("q7", [0.8, 1.0], 1.0), query="q7", benchmark="mmqa")
    E.upsert_result_row(str(out), row)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    assert rows[0]["f1"] == "1.0"
    assert rows[0]["val_f1_iter_0"] == "0.8"
    assert rows[0]["val_f1_iter_1"] == "1.0"


def test_direct_timing_breakdown_is_flattened_without_wall_residual():
    tele = telemetry("q13", [0.8, 1.0], 0.7)
    tele["wall_clock_ms"] = 1000
    tele["direct"]["timing_breakdown_ms"] = {
        "agent_stage_ms": 700,
        "validation_sampling_llm_ms": 40,
        "code_execution_ms": 200,
        "other_overhead_ms": 60,
    }
    row = E.telemetry_row(tele, query="q13", benchmark="ecomm")
    assert row["agent_stage_ms"] == 700
    assert row["validation_sampling_llm_ms"] == 40
    assert row["code_execution_ms"] == 200


def test_code_execution_runs_are_flattened_per_iteration_and_final_run():
    tele = telemetry("q13", [0.8, 1.0], 0.7)
    runs = [
        {"iteration": 0, "scope": "validation_iteration",
         "duration_ms": 30, "status": "ok"},
        {"iteration": 1, "scope": "validation_iteration",
         "duration_ms": 40, "status": "ok"},
        {"iteration": None, "scope": "final_full_corpus",
         "duration_ms": 90, "status": "ok"},
    ]
    tele["direct"]["code_execution_runs"] = runs
    row = E.telemetry_row(tele, query="q13", benchmark="ecomm")
    assert row["code_execution_ms_iter_0"] == 30
    assert row["code_execution_ms_iter_1"] == 40
    assert row["code_execution_ms_final"] == 90
    assert json.loads(row["code_execution_runs"]) == runs


def test_missing_code_execution_runs_are_not_inferred():
    tele = telemetry("q13", [0.8, 1.0], 0.7)
    row = E.telemetry_row(tele, query="q13", benchmark="ecomm")
    assert row["code_execution_ms_iter_0"] == ""
    assert row["code_execution_ms_iter_1"] == ""
    assert row["code_execution_ms_final"] == ""
    assert row["code_execution_runs"] == ""
