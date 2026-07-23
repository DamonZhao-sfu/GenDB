#!/usr/bin/env python3
"""
semtext.py — endpoint-backed TEXT inference backend for text DIRECT mode. The text
analog of semvision.py: predefined_text.py wraps these primitives, generated
solve_<q>.py programs compose them. All inference goes through an OpenAI-compatible
endpoint via semextract.gen_endpoint (guided-JSON), so there is ONE HTTP client.
"""
from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/semdb
import json
import semextract


class _Meter:
    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.calls = 0

    def incr(self) -> None:
        with self._lock:
            self.calls += 1


METER = _Meter()


@dataclass
class TextCtx:
    cfg: Any                       # semextract._Cfg
    cache: dict


def get_ctx(model: str, endpoint: str, api_key: str = "EMPTY", timeout: int = 120) -> TextCtx:
    cfg = semextract._Cfg(model=model, endpoint=endpoint, api_key=api_key,
                          max_new_tokens=64, timeout=timeout)
    return TextCtx(cfg=cfg, cache={})


def _ask(ctx: TextCtx, schema: dict, prompt: str, text: str) -> dict:
    """One guided-JSON call, memoized on (prompt, text)."""
    key = (prompt, text)
    if key in ctx.cache:
        return ctx.cache[key]
    METER.incr()
    raw = semextract.gen_endpoint(ctx.cfg, schema, prompt, "text", text=text)
    try:
        obj = json.loads(raw)
    except Exception:
        obj = {}
    ctx.cache[key] = obj
    return obj


class TextPatch:
    """A row's text, plus the shared endpoint ctx. Mirrors ImagePatch."""

    def __init__(self, text: str, ctx: TextCtx) -> None:
        self.text = text or ""
        self.ctx = ctx

    def judge(self, question: str) -> bool:
        schema = {"type": "object", "properties": {"answer": {"type": "boolean"}},
                  "required": ["answer"]}
        prompt = f"Answer the yes/no question about the INPUT text. Question: {question}"
        return bool(_ask(self.ctx, schema, prompt, self.text).get("answer", False))

    def classify(self, options: list[str]) -> str:
        schema = {"type": "object",
                  "properties": {"value": {"type": "string", "enum": list(options)}},
                  "required": ["value"]}
        prompt = ("Classify the INPUT text into exactly one of these options: "
                  + ", ".join(map(str, options)))
        v = _ask(self.ctx, schema, prompt, self.text).get("value", "")
        return v if v in options else (options[0] if options else "")

    def extract(self, field: str) -> str:
        schema = {"type": "object", "properties": {"value": {"type": "string"}},
                  "required": ["value"]}
        prompt = f"Extract the value of '{field}' from the INPUT text. If absent, return 'none'."
        return str(_ask(self.ctx, schema, prompt, self.text).get("value", "none"))

    def generate(self, instruction: str) -> str:
        schema = {"type": "object", "properties": {"value": {"type": "string"}},
                  "required": ["value"]}
        prompt = f"Follow this instruction over the INPUT text: {instruction}"
        return str(_ask(self.ctx, schema, prompt, self.text).get("value", ""))

    def score(self, query: str) -> float:
        schema = {"type": "object",
                  "properties": {"score": {"type": "number", "minimum": 0, "maximum": 1}},
                  "required": ["score"]}
        prompt = (f"Rate 0.0–1.0 how well the INPUT text matches: {query}. "
                  "Return only the number in JSON.")
        try:
            return float(_ask(self.ctx, schema, prompt, self.text).get("score", 0.0))
        except (TypeError, ValueError):
            return 0.0


import csv as _csv
import json as _json
import os as _os
import sys as _sys
import time as _time
from concurrent.futures import ThreadPoolExecutor


def _none_record(schema, id_col, id_val):
    rec = {a["name"]: ([] if "array" in a.get("type", "") or a.get("multi") else "none")
           for a in schema.get("attributes", [])}
    rec["conf"] = 0.0
    rec[id_col] = id_val
    return rec


def run_extraction(driver, schema, table_path, out_path, *, model, endpoint=None,
                   api_key="EMPTY", concurrency=8, theta=None, text_col=None,
                   limit=0, timeout=120):
    """Offline TEXT extraction engine (text analog of semextract.run / vadar_engine.run).
    For each corpus row it builds a TextPatch and calls driver.extract(patch), which
    composes predefined_text primitives. Writes attrs JSON + <out>.meta.json with the
    SAME contract as semextract.run. Returns the meta dict. Aborts (exit 3) without
    writing out_path if most rows error, so the orchestrator re-runs rather than caching
    a broken corpus."""
    ctx = get_ctx(model, endpoint or "", api_key, timeout)
    rows = list(_csv.DictReader(open(table_path)))
    if limit:
        rows = rows[:limit]
    cols = driver.map_columns(list(rows[0].keys()) if rows else [])
    id_col = cols["id"]
    tcol = text_col or cols.get("text")
    METER.reset()
    t_start = _time.time()
    attrs = [None] * len(rows)
    n_none = n_error = 0

    def process_row(i, r):
        id_val = r[id_col]
        try:
            patch = TextPatch(r.get(tcol, "") if tcol else "", ctx)
            fields = driver.extract(patch) or {}
            rec = dict(fields)
            primary = schema.get("attributes", [{}])[0].get("name")
            val = rec.get(primary)
            rec["conf"] = 0.0 if val in (None, "none", "", []) else 1.0
            rec[id_col] = id_val
            return {"i": i, "rec": rec, "error": None}
        except Exception as e:  # noqa: BLE001 — one bad row must not kill the batch
            return {"i": i, "rec": _none_record(schema, id_col, id_val), "error": str(e)}

    def tally(res):
        nonlocal n_none, n_error
        attrs[res["i"]] = res["rec"]
        if res["error"]:
            n_error += 1
            print(f"[semtext] {res['i']+1}/{len(rows)} ERROR: {res['error']}")
        elif res["rec"].get("conf", 0.0) == 0.0:
            n_none += 1

    workers = max(1, concurrency)
    if workers > 1 and endpoint:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(process_row, range(len(rows)), rows):
                tally(res)
    else:
        for i, r in enumerate(rows):
            tally(process_row(i, r))

    attrs = [a for a in attrs if a is not None]
    elapsed = _time.time() - t_start
    if len(rows) > 0 and n_error >= max(1, len(rows) // 2):
        _sys.stderr.write(f"[semtext] ABORT: {n_error}/{len(rows)} rows errored — likely a "
                          f"bad endpoint/model/creds. Not writing {out_path}.\n")
        _sys.exit(3)

    _json.dump(attrs, open(out_path, "w"), indent=2)
    meta = {
        "model": model, "endpoint": endpoint, "modality": "text",
        "rows": len(attrs), "extracted": len(attrs) - n_none, "none": n_none,
        "errors": n_error, "llm_calls": METER.calls, "theta": theta,
        "elapsed_sec": round(elapsed, 2),
        "sec_per_row": round(elapsed / max(1, len(attrs)), 3),
        "concurrency": workers,
    }
    _json.dump(meta, open(out_path + ".meta.json", "w"), indent=2)
    print(f"[semtext] wrote {len(attrs)} rows -> {out_path} ({elapsed:.1f}s, "
          f"{meta['llm_calls']} llm calls, {n_none} none)")
    return meta
