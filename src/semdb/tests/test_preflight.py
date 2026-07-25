"""The compile gate: exact location + source context on failure, and a missing
optional linter must never block code generation."""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import preflight as P  # noqa: E402

CLEAN = '''\
import csv


def solve(rows):
    out = []
    for row in rows:
        if "comedy" in row["text"].lower():
            out.append(row["id"])
    return out
'''

BROKEN_SYNTAX = '''\
def solve(rows):
    out = []
    for row in rows:
        if row["x"] == 1
            out.append(row)
    return out
'''

UNDEFINED_NAME = '''\
def solve(rows):
    return [r for r in rows if r["genre"] in GENRE_MAP]
'''

UNUSED_IMPORT = '''\
import csv
import json


def solve(rows):
    return json.dumps(list(rows))
'''


def _write(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


# --- passing ---------------------------------------------------------------

def test_clean_file_passes(tmp_path):
    report = P.preflight([_write(tmp_path, "solve_q.py", CLEAN)])
    assert report["ok"] is True
    assert report["stage"] is None
    assert report["errors"] == []


def test_clean_file_renders_as_compile_ok(tmp_path):
    report = P.preflight([_write(tmp_path, "solve_q.py", CLEAN)])
    assert P.render_text(report).startswith("COMPILE OK")


# --- syntax ----------------------------------------------------------------

def test_syntax_error_reports_the_exact_line(tmp_path):
    report = P.preflight([_write(tmp_path, "solve_q.py", BROKEN_SYNTAX)])
    assert report["ok"] is False and report["stage"] == "syntax"
    err = report["errors"][0]
    assert err["error_class"] in {"SyntaxError", "IndentationError"}
    assert err["line"] == 4                      # the `if ...` missing its colon


def test_syntax_error_carries_marked_source_context(tmp_path):
    report = P.preflight([_write(tmp_path, "solve_q.py", BROKEN_SYNTAX)])
    context = report["errors"][0]["context"]
    assert context, "a syntax error with no source context is the status quo we are fixing"
    marked = [line for line in context if line.startswith(">>")]
    assert len(marked) == 1 and 'if row["x"] == 1' in marked[0]
    assert len(context) <= 2 * P.CONTEXT_RADIUS + 1


def test_every_file_with_a_syntax_error_is_reported(tmp_path):
    report = P.preflight([_write(tmp_path, "a.py", BROKEN_SYNTAX),
                          _write(tmp_path, "b.py", BROKEN_SYNTAX)])
    assert {os.path.basename(e["file"]) for e in report["errors"]} == {"a.py", "b.py"}


def test_syntax_stage_short_circuits_the_static_stage(tmp_path):
    report = P.preflight([_write(tmp_path, "solve_q.py", BROKEN_SYNTAX)])
    assert report["static_checker"] is None      # never reached


# --- static names ----------------------------------------------------------

def test_undefined_name_fails_the_gate(tmp_path):
    report = P.preflight([_write(tmp_path, "solve_q.py", UNDEFINED_NAME)])
    if report["static_checker"] == "unavailable":
        pytest.skip("pyflakes not installed")
    assert report["ok"] is False and report["stage"] == "static"
    err = report["errors"][0]
    assert err["error_class"] == "UndefinedName"
    assert "GENRE_MAP" in err["message"] and err["line"] == 2
    assert err["context"], "a fatal static finding needs source context too"


def test_undefined_name_can_be_downgraded_to_a_warning(tmp_path):
    path = _write(tmp_path, "solve_q.py", UNDEFINED_NAME)
    report = P.preflight([path], names_are_fatal=False)
    if report["static_checker"] == "unavailable":
        pytest.skip("pyflakes not installed")
    assert report["ok"] is True
    assert any(w["error_class"] == "UndefinedName" for w in report["warnings"])


def test_style_findings_warn_but_do_not_fail(tmp_path):
    report = P.preflight([_write(tmp_path, "solve_q.py", UNUSED_IMPORT)])
    if report["static_checker"] == "unavailable":
        pytest.skip("pyflakes not installed")
    assert report["ok"] is True
    assert any("csv" in w["message"] for w in report["warnings"])


def test_a_missing_linter_passes_the_gate_rather_than_blocking(tmp_path, monkeypatch):
    """A missing optional dependency must never stop code generation."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _no_pyflakes(name, *args, **kwargs):
        if name.startswith("pyflakes"):
            raise ImportError("simulated: pyflakes not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _no_pyflakes)
    report = P.preflight([_write(tmp_path, "solve_q.py", UNDEFINED_NAME)])
    assert report["ok"] is True
    assert report["static_checker"] == "unavailable"
    assert "pyflakes not installed" in P.render_text(report)


# --- io / cli --------------------------------------------------------------

def test_unreadable_file_fails_at_the_read_stage(tmp_path):
    report = P.preflight([str(tmp_path / "does_not_exist.py")])
    assert report["ok"] is False and report["stage"] == "read"
    assert report["errors"][0]["error_class"] == "OSError"


def test_cli_writes_json_and_exits_nonzero_on_failure(tmp_path):
    path = _write(tmp_path, "solve_q.py", BROKEN_SYNTAX)
    out = tmp_path / "preflight.json"
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "..", "preflight.py"), path, "--out", str(out)],
        capture_output=True, text=True)
    assert proc.returncode == 1
    report = json.load(open(out))
    assert report["ok"] is False and report["stage"] == "syntax"
    assert "COMPILE FAILED" in proc.stdout


def test_cli_exits_zero_on_a_clean_file(tmp_path):
    path = _write(tmp_path, "solve_q.py", CLEAN)
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "..", "preflight.py"), path, "--quiet"],
        capture_output=True, text=True)
    assert proc.returncode == 0
