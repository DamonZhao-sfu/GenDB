#!/usr/bin/env python3
"""Rebuild a run's results.csv from every child telemetry.json.

One telemetry file is authoritative for one query. Rebuilding instead of appending
also removes stale duplicate rows left by rerunning a query in the same output
directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

from evaluate import CSV_COLS, telemetry_row


def query_sort_key(query: str) -> tuple[int, str]:
    match = re.fullmatch(r"q(\d+)(.*)", str(query).lower())
    return (int(match.group(1)), match.group(2)) if match else (10**9, str(query))


def infer_benchmark(directory: str, query: str) -> str:
    suffix = "-" + query
    return directory[:-len(suffix)] if query and directory.endswith(suffix) else ""


def collect_rows(run_dir: str | Path) -> list[dict]:
    rows = []
    for path in Path(run_dir).glob("*/telemetry.json"):
        try:
            telemetry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read {path}: {exc}") from exc
        query = str(telemetry.get("query") or path.parent.name.rsplit("-", 1)[-1])
        benchmark = str(
            telemetry.get("benchmark") or infer_benchmark(path.parent.name, query)
        )
        rows.append(telemetry_row(telemetry, query=query, benchmark=benchmark))
    return sorted(rows, key=lambda row: query_sort_key(row.get("query", "")))


def sync(run_dir: str | Path, out: str | Path | None = None) -> Path:
    run_dir = Path(run_dir)
    destination = Path(out) if out else run_dir / "results.csv"
    rows = collect_rows(run_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild results.csv from <run-dir>/*/telemetry.json.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", help="Defaults to <run-dir>/results.csv.")
    args = parser.parse_args(argv)
    path = sync(args.run_dir, args.out)
    print(f"[sync_results] wrote {len(collect_rows(args.run_dir))} row(s) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
