#!/usr/bin/env python3
"""
materialize.py — convert a PARQUET SemBench benchmark's tables to CSV (+ build an
image manifest) so the CSV-based extractor/compiler can consume them.

Only ecomm needs this today (its tables are parquet and its "IMAGES" table is really
the images/ directory referenced by id). Idempotent: skips files already written
unless --force.

Usage:
  python3 materialize.py <benchmark> <data_dir> <out_dir> [--force]
Writes into <out_dir>:
  ecomm → styles_details.csv, styles.csv, IMAGES.csv (id,filename from image_mapping)
"""
import os
import sys


def _need(path, force):
    return force or not os.path.exists(path)


def main():
    if len(sys.argv) < 4:
        sys.stderr.write("usage: materialize.py <benchmark> <data_dir> <out_dir> [--force]\n")
        sys.exit(2)
    bench, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    force = "--force" in sys.argv[4:]
    try:
        import pandas as pd
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"materialize.py needs pandas (run under the gendb/sembench conda env): {e}\n")
        sys.exit(2)
    os.makedirs(out_dir, exist_ok=True)

    if bench == "ecomm":
        # structured/text tables → CSV (nested struct columns are stringified; the
        # extractor only reads flat columns like productDisplayName, so that's fine).
        for name in ("styles_details", "styles"):
            src = os.path.join(data_dir, f"{name}.parquet")
            dst = os.path.join(out_dir, f"{name}.csv")
            if os.path.exists(src) and _need(dst, force):
                pd.read_parquet(src).to_csv(dst, index=False)
                print(f"[materialize] {name}.parquet -> {dst}")
        # image manifest: id + filename (image lives at <data_dir>/images/<filename>)
        imp = os.path.join(data_dir, "image_mapping.parquet")
        dst = os.path.join(out_dir, "IMAGES.csv")
        if os.path.exists(imp) and _need(dst, force):
            m = pd.read_parquet(imp)
            cols = [c for c in ("id", "filename") if c in m.columns]
            m[cols].to_csv(dst, index=False)
            print(f"[materialize] image_mapping.parquet -> {dst} ({len(m)} images)")
    else:
        sys.stderr.write(f"[materialize] no parquet materialization needed/known for '{bench}'\n")

    print(f"[materialize] done -> {out_dir}")


if __name__ == "__main__":
    main()
