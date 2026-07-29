#!/usr/bin/env python3
"""Build a validation-only CSV with collision-free physical row identity.

SemBench occasionally repeats an entire logical row. SQL keeps that multiplicity,
but validation dictionaries cannot key two rows by the same logical id. This frame
adds ``_semdb_row_id`` as the zero-based source-record ordinal. Generated solvers can
derive the same value while reading the original CSV; result projection remains
unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

PHYSICAL_ID = "_semdb_row_id"


def build_frame(corpus: str, out: str, *, text_cols: list[str] | None = None,
                filter_col: str | None = None,
                filter_value: str | None = None,
                filter_values: list[str] | None = None) -> dict:
    accepted_values = list(filter_values or [])
    if filter_value is not None:
        accepted_values.append(str(filter_value))
    source_stat = os.stat(corpus)
    spec = {
        "version": 1,
        "source": {
            "path": os.path.abspath(corpus),
            "size": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
        },
        "text_cols": text_cols or [],
        "filter": ({filter_col: accepted_values} if filter_col else None),
    }
    try:
        with open(out + ".meta.json", encoding="utf-8") as handle:
            cached = json.load(handle)
        if os.path.exists(out) and cached.get("spec") == spec:
            print(f"[physical_frame] inputs unchanged; reusing {out}")
            return cached
    except (OSError, ValueError, TypeError):
        pass
    with open(corpus, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        raise SystemExit(f"corpus {corpus} is empty")
    required = [*(text_cols or []), *([filter_col] if filter_col else [])]
    missing = [col for col in required if col not in fields]
    if missing:
        raise SystemExit(f"{corpus} lacks column(s): {', '.join(missing)}")
    if PHYSICAL_ID in fields:
        raise SystemExit(f"{corpus} already contains reserved column {PHYSICAL_ID}")

    framed = []
    for source_index, source in enumerate(rows):
        if filter_col and str(source.get(filter_col, "")) not in accepted_values:
            continue
        row = {PHYSICAL_ID: str(source_index), **source}
        if text_cols:
            row["semantic_text"] = " ".join(
                str(source.get(col, "") or "").strip() for col in text_cols
            ).strip()
        framed.append(row)
    if not framed:
        raise SystemExit("deterministic row filter produced an empty frame")

    out_fields = [PHYSICAL_ID, *fields]
    if text_cols:
        out_fields.append("semantic_text")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(framed)
    meta = {
        "version": 1,
        "spec": spec,
        "source": os.path.abspath(corpus),
        "input_rows": len(rows),
        "output_rows": len(framed),
        "physical_id_col": PHYSICAL_ID,
        "identity": "zero-based source CSV record ordinal",
        "filter": ({filter_col: accepted_values} if filter_col else None),
        "text_cols": text_cols or [],
    }
    with open(out + ".meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")
    print(f"[physical_frame] {len(rows)} -> {len(framed)} rows; wrote {out}")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--text-col", action="append", default=[])
    parser.add_argument("--filter-col")
    parser.add_argument("--filter-value", action="append", default=[])
    args = parser.parse_args()
    if bool(args.filter_col) != bool(args.filter_value):
        parser.error("--filter-col and --filter-value must be supplied together")
    build_frame(
        args.corpus, args.out, text_cols=args.text_col,
        filter_col=args.filter_col, filter_values=args.filter_value,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
