#!/usr/bin/env python3
"""
semvqa.py — OpImgVQA as a TYPED operator: (Answer, Score) over an image.

This is the residual/escalation backend, and the only image operator that spends a
generative VLM call. It differs from an ad-hoc VLM call in two ways that matter
downstream:

  * ANSWER is constrained. Pass `choices` and guided decoding restricts the answer to
    that value space (an enum), so the output is a real field value that joins/filters,
    not free-form prose.
  * SCORE is CALIBRATED. It comes from the model's token logprobs over the answer span,
    not from asking the model to rate itself. That distinction is the whole point: this
    score is the input a percentile router and a threshold test consume, and a
    self-reported confidence is too poorly calibrated to cut on. `score_source` in the
    detail dict records which path produced it, so a caller can tell when it degraded.

Fallback order for the score: logprobs → the model's self-reported `conf` → `default`.
"""

import json
import math
from concurrent.futures import ThreadPoolExecutor

import semextract


DEFAULT_SCORE = 0.5


def build_cfg(model, endpoint, api_key="EMPTY", max_new_tokens=64, timeout=120):
    return semextract._Cfg(model=model, endpoint=endpoint or "", api_key=api_key,
                           max_new_tokens=max_new_tokens, timeout=timeout)


def _schema(choices=None):
    ans = {"type": "string"}
    if choices:
        ans = {"type": "string", "enum": [str(c) for c in choices]}
    return {"type": "object",
            "properties": {"answer": ans, "conf": {"type": "number"}},
            "required": ["answer", "conf"],
            "additionalProperties": False}


def _prompt(question, choices=None):
    p = ("Answer the question about this image. Respond with STRICT JSON only, no prose:\n"
         '{"answer": <string>, "conf": <0.0-1.0>}\n'
         f"- question: {question}\n")
    if choices:
        p += f"- answer must be exactly one of: {', '.join(str(c) for c in choices)}\n"
    p += "- conf: how confident you are in the answer.\n"
    return p


def _answer_span(content, answer):
    """Character span of the answer VALUE inside the JSON response (not the whole blob),
    so the score reflects the decision rather than the surrounding punctuation."""
    if not answer:
        return None
    k = content.find('"answer"')
    start = content.find(answer, k if k >= 0 else 0)
    if start < 0:
        start = content.find(answer)
    return (start, start + len(answer)) if start >= 0 else None


def _score_from_logprobs(choice, content, answer):
    """Geometric mean of token probabilities over the answer span → [0,1].
    Returns None when the server did not return usable logprobs."""
    lp = (choice.get("logprobs") or {}).get("content")
    if not lp:
        return None
    span = _answer_span(content, answer)
    if span is None:
        return None
    lo, hi = span
    pos, picked = 0, []
    for tok in lp:
        s = tok.get("token", "")
        nxt = pos + len(s)
        if nxt > lo and pos < hi:          # this token overlaps the answer span
            picked.append(float(tok.get("logprob", 0.0)))
        pos = nxt
        if pos >= hi:
            break
    if not picked:
        return None
    return float(math.exp(sum(picked) / len(picked)))


def vqa_detail(question, choices=None, *, cfg, image_path=None, text=None,
               default=DEFAULT_SCORE):
    """One typed QA call over an image (`image_path`) or a text (`text`). Returns
    {"answer", "score", "score_source": "logprobs"|"self"|"default", "raw", "error"}."""
    modality = "image" if image_path is not None else "text"
    schema = _schema(choices)
    out = {"answer": "none", "score": 0.0, "score_source": "default", "raw": "", "error": None}
    try:
        choice = semextract.gen_endpoint_full(cfg, schema, _prompt(question, choices),
                                              modality, image_path=image_path, text=text,
                                              logprobs=True)
    except Exception as e:  # noqa: BLE001 — a residual miss must not kill the batch
        out["error"] = str(e)
        return out
    content = (choice.get("message") or {}).get("content", "") or ""
    out["raw"] = content
    try:
        obj = json.loads(content)
    except Exception:
        obj = semextract._find_json(content) or {}   # salvage JSON wrapped in prose
    if not isinstance(obj, dict):
        obj = {}
    answer = str(obj.get("answer", "none"))
    if choices and answer not in {str(c) for c in choices}:
        # guided decoding should prevent this; if a server ignored it, do not invent a value
        out["answer"], out["error"] = "none", f"off-vocabulary answer {answer!r}"
        return out
    out["answer"] = answer

    s = _score_from_logprobs(choice, content, answer)
    if s is not None:
        out["score"], out["score_source"] = max(0.0, min(1.0, s)), "logprobs"
    elif isinstance(obj.get("conf"), (int, float)):
        out["score"], out["score_source"] = max(0.0, min(1.0, float(obj["conf"]))), "self"
    else:
        out["score"], out["score_source"] = default, "default"
    return out


def img_vqa_detail(image_path, question, choices=None, *, cfg, default=DEFAULT_SCORE):
    """OpImgVQA over one image."""
    return vqa_detail(question, choices, cfg=cfg, image_path=image_path, default=default)


def txt_vqa_detail(text, question, choices=None, *, cfg, default=DEFAULT_SCORE):
    """The text-modality twin, so a text residual is a real call rather than a silent
    no-op. (A full OpTxtQA with its own backend family is a later phase.)"""
    return vqa_detail(question, choices, cfg=cfg, text=text, default=default)


def img_vqa(image_path, question, choices=None, *, cfg, default=DEFAULT_SCORE):
    """(answer, score) — the TxRA `OpImgVQA -> R_VQA(Answer, Score)` contract."""
    d = img_vqa_detail(image_path, question, choices, cfg=cfg, default=default)
    return d["answer"], d["score"]


def txt_vqa(text, question, choices=None, *, cfg, default=DEFAULT_SCORE):
    d = txt_vqa_detail(text, question, choices, cfg=cfg, default=default)
    return d["answer"], d["score"]


def img_vqa_batch(image_paths, question, choices=None, *, cfg, concurrency=8,
                  default=DEFAULT_SCORE):
    """Concurrent VQA over many images (a vLLM server batches concurrent requests).
    Returns a list of detail dicts, aligned with `image_paths`."""
    if not image_paths:
        return []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        return list(ex.map(
            lambda p: img_vqa_detail(p, question, choices, cfg=cfg, default=default),
            list(image_paths)))


def escalate(rows, question, choices=None, *, cfg, theta=0.5, path_key="image_path",
             value_key="value", score_key="conf", concurrency=8):
    """The residual pass: re-answer only the rows the cheap proxy was unsure about
    (`score < theta` or a missing value), leaving the rest untouched. Returns
    (new_rows, n_calls) — new_rows is a fresh list, inputs are not mutated."""
    idx = [i for i, r in enumerate(rows)
           if r.get(value_key) in (None, "", "none", []) or float(r.get(score_key, 0.0)) < theta]
    out = [dict(r) for r in rows]
    if not idx:
        return out, 0
    dets = img_vqa_batch([rows[i][path_key] for i in idx], question, choices,
                         cfg=cfg, concurrency=concurrency)
    for i, d in zip(idx, dets):
        if d["error"] is None and d["answer"] != "none":
            out[i][value_key] = d["answer"]
            out[i][score_key] = d["score"]
            out[i]["_escalated"] = True
    return out, len(idx)
