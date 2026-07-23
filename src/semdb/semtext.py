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
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/semdb
import json
import semextract


class _Meter:
    def __init__(self) -> None:
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0


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
    METER.calls += 1
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
