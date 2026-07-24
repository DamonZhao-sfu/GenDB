#!/usr/bin/env python3
"""Offline execution engine for VADAR-generated text extraction drivers."""
from __future__ import annotations

import csv
import json
import time


def _empty_value(attribute):
    return [] if "array" in attribute.get("type", "").casefold() or attribute.get("multi") else "none"


def run(driver, schema, table_path, out_path, *, limit=0):
    """Run deterministic ``driver.extract(text)`` over a text CSV with zero model calls."""
    started = time.time()
    with open(table_path, newline="") as source:
        rows = list(csv.DictReader(source))
    if limit:
        rows = rows[:limit]

    columns = driver.map_columns(list(rows[0].keys()) if rows else [])
    id_col = columns["id"]
    text_col = columns["text"]
    context_cols = columns.get("context", [])
    attributes = schema.get("attributes", [])
    output = []
    none_count = error_count = 0

    for index, row in enumerate(rows):
        base = row.get(text_col, "")
        context = " ".join(
            f"{column}: {row.get(column, '')}" for column in context_cols if row.get(column)
        )
        text = f"{base}\n{context}".strip() if context else base
        try:
            fields = driver.extract(text) or {}
            record = {
                attribute["name"]: fields.get(attribute["name"], _empty_value(attribute))
                for attribute in attributes
            }
        except Exception as error:  # one malformed row must not stop the corpus
            error_count += 1
            print(f"[vadar-text] row {index} extract error: {error}")
            record = {attribute["name"]: _empty_value(attribute) for attribute in attributes}

        primary = attributes[0]["name"] if attributes else None
        if primary and record.get(primary) in (None, "none", "", []):
            none_count += 1
            record["conf"] = 0.0
        else:
            record["conf"] = 1.0
        record[id_col] = row.get(id_col, "")
        output.append(record)

    with open(out_path, "w") as target:
        json.dump(output, target, indent=2)
    elapsed = time.time() - started
    meta = {
        "engine": "vadar-text-offline",
        "modality": "text",
        "rows": len(output),
        "none": none_count,
        "errors": error_count,
        "llm_calls": 0,
        "elapsed_sec": round(elapsed, 2),
        "rows_per_sec": round(len(output) / max(elapsed, 1e-9), 2),
    }
    with open(out_path + ".meta.json", "w") as target:
        json.dump(meta, target, indent=2)
    print(f"[vadar-text] wrote {len(output)} rows -> {out_path} ({elapsed:.1f}s, 0 model calls)")
    return meta
