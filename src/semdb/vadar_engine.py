#!/usr/bin/env python3
"""
vadar_engine.py — the EXECUTION engine for VADAR-style generated extraction drivers.

A generated `extract_<corpus>.py` (the Program-agent output) supplies a `Driver` with:
  - map_columns(header) -> {"id":..., "image":..., "text":None, "context":[...]}
  - extract(patch) -> {field: value, ..., "conf": float}   (composes the ImagePatch API)
and calls `vadar_engine.run(Driver(), schema, table, out, model=..., image_dir=...)`.

This loads CLIP once, wraps each corpus image in an ImagePatch, execs `extract`, and writes
the SAME attribute-table contract semextract/semvision produce (so the compiled query +
evaluator are unchanged). Mirrors semvision.run — no VLM, no endpoint.
"""
import csv
import json
import os
import time


def run(driver, schema, table_path, out_path, *, model="openai/clip-vit-base-patch32",
        image_dir=None, limit=0):
    import semvision
    import semextract          # resolve_image_path + _none_record
    t0 = time.time()
    rows = list(csv.DictReader(open(table_path)))
    if limit:
        rows = rows[:limit]
    cols = driver.map_columns(list(rows[0].keys()) if rows else [])
    id_col = cols["id"]
    ctx = {"encoder": semvision.get_encoder(model), "palette": None}
    if hasattr(driver, "image_dir") and getattr(driver, "image_dir") is None:
        driver.image_dir = image_dir

    import imagepatch
    attrs, n_none, n_err = [], 0, 0
    primary = schema.get("attributes", [{}])[0].get("name")
    for i, r in enumerate(rows):
        uri = r.get(cols.get("image") or id_col, "")
        path = semextract.resolve_image_path(uri, image_dir)
        if not path or not os.path.exists(path):
            rec = semextract._none_record(schema, id_col, r[id_col]); n_none += 1
        else:
            try:
                rec = driver.extract(imagepatch.ImagePatch(path, ctx))
                rec.setdefault("conf", 1.0)
                rec[id_col] = r[id_col]
            except Exception as e:  # noqa: BLE001 — one bad row must not kill the batch
                rec = semextract._none_record(schema, id_col, r[id_col]); n_err += 1
                print(f"[vadar] row {i} extract error: {e}")
        attrs.append(rec)
        if rec.get(primary) in (None, "none", "", []):
            n_none += 1
        print(f"[vadar] {i+1}/{len(rows)} {str(r[id_col])[:32]!r} -> {rec.get(primary)}")

    json.dump(attrs, open(out_path, "w"), indent=2)
    elapsed = time.time() - t0
    meta = {"engine": "vadar", "clip_model": model, "rows": len(attrs),
            "none": n_none, "errors": n_err, "elapsed_sec": round(elapsed, 2),
            "rows_per_sec": round(len(attrs) / max(1e-9, elapsed), 2)}
    json.dump(meta, open(out_path + ".meta.json", "w"), indent=2)
    print(f"[vadar] wrote {len(attrs)} rows -> {out_path} ({elapsed:.1f}s)")
    return meta
