#!/usr/bin/env python3
"""Measure final generated Python solvers and backfill exact execution runtimes.

The original command is recovered from each query's root ``run.log`` so benchmark-
specific data, image, and model arguments remain identical. The existing prediction
and trace artifacts are never overwritten: every solver writes into a temporary
directory. Only successful runs with an output file update telemetry.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from sync_results import sync


def final_solver(run_dir: Path) -> Path | None:
    solvers = sorted(run_dir.glob("solve_*.py"))
    return solvers[0] if len(solvers) == 1 else None


def command_from_log(run_dir: Path, solver: Path, output: Path) -> list[str]:
    log_path = run_dir / "run.log"
    first_line = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    if not first_line.startswith("$ "):
        raise ValueError(f"{log_path}: first line is not a recorded command")
    command = shlex.split(first_line[2:])
    if len(command) < 3 or Path(command[0]).name not in {"python", "python3"}:
        raise ValueError(f"{log_path}: unsupported command")

    # The generated solver and its positional output are the first two arguments.
    command[1] = str(solver.resolve())
    command[2] = str(output.resolve())

    # A runtime backfill always measures the promoted solver on the full corpus.
    cleaned = command[:3]
    index = 3
    while index < len(command):
        if command[index] == "--only-ids":
            index += 2
            continue
        cleaned.append(command[index])
        index += 1
    return cleaned


def atomic_json_write(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def measure_query(run_dir: Path, *, force: bool, timeout: float | None,
                  dry_run: bool) -> tuple[str, int | None]:
    telemetry_path = run_dir / "telemetry.json"
    if not telemetry_path.exists():
        return "skip:no_telemetry", None
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    direct = telemetry.get("direct")
    if not isinstance(direct, dict):
        return "skip:not_direct", None
    if isinstance(direct.get("code_execution_runs"), list) and not force:
        return "skip:already_measured", None

    solver = final_solver(run_dir)
    if solver is None:
        return "skip:no_unique_final_solver", None
    if not (run_dir / "run.log").exists():
        return "skip:no_run_log", None

    with tempfile.TemporaryDirectory(prefix=f"semdb-runtime-{run_dir.name}-") as temp:
        output = Path(temp) / f"{telemetry.get('query', run_dir.name)}_results.csv"
        command = command_from_log(run_dir, solver, output)
        if dry_run:
            print("[dry-run]", shlex.join(command), flush=True)
            return "dry-run", None

        log_path = run_dir / "runtime_backfill.log"
        started = time.perf_counter_ns()
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + shlex.join(command) + "\n\n")
            log.flush()
            try:
                process = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parent,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
                status = process.returncode
            except subprocess.TimeoutExpired:
                duration_ms = (time.perf_counter_ns() - started) // 1_000_000
                log.write(f"\n[backfill] timeout after {duration_ms} ms\n")
                return "failed:timeout", duration_ms
        duration_ms = (time.perf_counter_ns() - started) // 1_000_000
        if status != 0:
            return f"failed:exit_{status}", duration_ms
        if not output.exists():
            return "failed:no_output", duration_ms

        run = {
            "iteration": None,
            "scope": "final_full_corpus",
            "duration_ms": duration_ms,
            "status": "ok",
            "source": "runtime_backfill",
        }
        direct["code_execution_runs"] = [run]
        direct["code_execution_ms"] = duration_ms
        direct["code_execution_backfill"] = {
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "solver": str(solver.resolve()),
            "command_args": command[3:],
            "clock": "time.perf_counter_ns",
        }
        atomic_json_write(telemetry_path, telemetry)
        return "updated", duration_ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root", action="append", required=True,
        help="Root containing <benchmark>-<query>/telemetry.json; repeatable.")
    parser.add_argument("--force", action="store_true",
                        help="Remeasure queries that already have code_execution_runs.")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Per-query timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    failures = 0
    for raw_root in args.run_root:
        root = Path(raw_root).resolve()
        print(f"[backfill] root {root}", flush=True)
        for telemetry_path in sorted(root.glob("*/telemetry.json")):
            run_dir = telemetry_path.parent
            try:
                status, duration_ms = measure_query(
                    run_dir, force=args.force, timeout=args.timeout,
                    dry_run=args.dry_run)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                status, duration_ms = f"failed:{exc}", None
            timing = "" if duration_ms is None else f" {duration_ms / 1000:.3f}s"
            print(f"[backfill] {run_dir.name}: {status}{timing}", flush=True)
            if status.startswith("failed:"):
                failures += 1
        if not args.dry_run:
            result_path = sync(root)
            print(f"[backfill] synced {result_path}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
