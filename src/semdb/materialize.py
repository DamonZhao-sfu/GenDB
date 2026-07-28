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
  ecomm → styles_details.csv, styles.csv, IMAGES.csv (id,filename from image_mapping),
          ecomm_products.csv (flat product+image view for deterministic validation)
"""
import json
import os
import sys


def _need(path, force):
    return force or not os.path.exists(path)


def _source_signature(path):
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _nested(value, *path):
    """Read a nested parquet struct without evaluating its CSV string form."""
    cur = value
    for key in path:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(key)
    return "" if cur is None else cur


def _meta_matches(meta_path, signatures):
    try:
        with open(meta_path, encoding="utf-8") as handle:
            return json.load(handle).get("sources") == signatures
    except (OSError, ValueError, TypeError):
        return False


def materialize_ecomm(data_dir, out_dir, *, force=False):
    """Materialize EComm atomically enough to reject stale cross-scale CSVs.

    Older code reused any existing destination without recording which parquet input
    produced it. Source path/size/mtime provenance now controls reuse; a cache without
    provenance is rebuilt once. (Wide descriptions contain embedded newlines, so a
    physical ``wc -l`` is intentionally not used as a CSV-record count.)
    """
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    sources = {
        name: os.path.join(data_dir, f"{name}.parquet")
        for name in ("styles_details", "styles", "image_mapping")
    }
    missing = [path for path in sources.values() if not os.path.exists(path)]
    if missing:
        raise SystemExit(
            f"[materialize] missing required parquet input(s) under {data_dir}: "
            + ", ".join(os.path.basename(path) for path in missing))
    signatures = {name: _source_signature(path) for name, path in sources.items()}
    meta_path = os.path.join(out_dir, "materialize_meta.json")
    outputs = [
        os.path.join(out_dir, name)
        for name in ("styles_details.csv", "styles.csv", "IMAGES.csv",
                     "ecomm_products.csv")
    ]
    if (not force and all(os.path.exists(path) for path in outputs)
            and _meta_matches(meta_path, signatures)):
        print(f"[materialize] inputs unchanged; reusing {out_dir}")
        return outputs

    details = pd.read_parquet(sources["styles_details"])
    styles = pd.read_parquet(sources["styles"])
    mapping = pd.read_parquet(sources["image_mapping"])
    required_mapping = ["id", "filename", "link"]
    missing_cols = [c for c in required_mapping if c not in mapping.columns]
    if missing_cols:
        raise SystemExit(
            f"[materialize] image_mapping.parquet needs id,filename,link; "
            f"missing: {', '.join(missing_cols)}; "
            f"available: {', '.join(map(str, mapping.columns))}")
    for label, frame in (("styles_details", details), ("image_mapping", mapping)):
        if "id" not in frame.columns:
            raise SystemExit(f"[materialize] {label}.parquet has no id column")
        ids = frame["id"].astype(str)
        if ids.duplicated().any():
            examples = ", ".join(ids[ids.duplicated()].head(3))
            raise SystemExit(
                f"[materialize] {label}.parquet id is not unique; examples: {examples}")

    # Keep the original wide tables for existing agents.
    details.to_csv(outputs[0], index=False)
    styles.to_csv(outputs[1], index=False)
    mapping[["id", "filename"]].to_csv(outputs[2], index=False)

    # Flatten exactly the nested fields used by EComm's deterministic prefixes and
    # semantic predicates.  Joining by the physical id is equivalent to the query's
    # imageURL→link→filename chain, and lets DuckDB/pandas consume ordinary columns.
    flat = details.copy()
    flat["_join_id"] = flat["id"].astype(str)
    image_map = mapping[["id", "filename", "link"]].copy()
    image_map["_join_id"] = image_map["id"].astype(str)
    image_map = image_map.drop(columns=["id"])
    flat = flat.merge(image_map, on="_join_id", how="inner", validate="one_to_one")
    if len(flat) != len(details):
        detail_ids = set(details["id"].astype(str))
        mapped_ids = set(mapping["id"].astype(str))
        missing_ids = sorted(detail_ids - mapped_ids)[:5]
        raise SystemExit(
            f"[materialize] only {len(flat)}/{len(details)} product rows have images; "
            f"missing ids: {', '.join(missing_ids)}")
    flat["id"] = flat["_join_id"]
    flat["description"] = flat["productDescriptors"].map(
        lambda v: _nested(v, "description", "value"))
    flat["masterCategoryName"] = flat["masterCategory"].map(
        lambda v: _nested(v, "typeName"))
    flat["imageURL"] = flat["styleImages"].map(
        lambda v: _nested(v, "default", "imageURL"))
    mismatched_links = (
        flat["imageURL"].fillna("").astype(str)
        != flat["link"].fillna("").astype(str)
    )
    if mismatched_links.any():
        raise SystemExit(
            f"[materialize] {int(mismatched_links.sum())} style imageURL values do "
            "not match image_mapping.link")
    flat["semantic_text"] = (
        flat["productDisplayName"].fillna("").astype(str).str.strip()
        + " - "
        + flat["description"].fillna("").astype(str).str.strip()
    )
    fields = [
        "id", "filename", "link", "imageURL", "price", "baseColour",
        "colour1", "colour2", "brandName", "productDisplayName", "description",
        "masterCategoryName", "semantic_text",
    ]
    absent = [c for c in fields if c not in flat.columns]
    if absent:
        raise SystemExit(
            f"[materialize] styles_details.parquet lacks normalized field(s): "
            f"{', '.join(absent)}")
    flat[fields].to_csv(outputs[3], index=False)

    meta = {
        "version": 2,
        "sources": signatures,
        "rows": {
            "styles_details.csv": len(details),
            "styles.csv": len(styles),
            "IMAGES.csv": len(mapping),
            "ecomm_products.csv": len(flat),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")
    print(f"[materialize] styles_details.parquet -> {outputs[0]} ({len(details)} rows)")
    print(f"[materialize] styles.parquet -> {outputs[1]} ({len(styles)} rows)")
    print(f"[materialize] image_mapping.parquet -> {outputs[2]} ({len(mapping)} images)")
    print(f"[materialize] normalized product view -> {outputs[3]} ({len(flat)} rows)")
    return outputs


def main():
    if len(sys.argv) < 4:
        sys.stderr.write("usage: materialize.py <benchmark> <data_dir> <out_dir> [--force]\n")
        sys.exit(2)
    bench, data_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    force = "--force" in sys.argv[4:]
    try:
        import pandas  # noqa: F401
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"materialize.py needs pandas (run under the gendb/sembench conda env): {e}\n")
        sys.exit(2)
    if bench != "ecomm":
        sys.stderr.write(f"[materialize] no parquet materialization needed/known for '{bench}'\n")
        sys.exit(2)
    materialize_ecomm(data_dir, out_dir, force=force)
    print(f"[materialize] done -> {out_dir}")


if __name__ == "__main__":
    main()
