"""OpImgVQA — typed (Answer, Score), with the score derived from token logprobs.

The score must come from logprobs when available, because it is what a percentile router
and a threshold test cut on; a self-reported `conf` is the degraded fallback and must be
labelled as such.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semvqa  # noqa: E402


CFG = semvqa.build_cfg("m", "http://x/v1")


def _choice(content, token_logprobs=None):
    """Build an OpenAI-shaped choice; token_logprobs is [(token, logprob), ...]."""
    c = {"message": {"content": content}}
    if token_logprobs is not None:
        c["logprobs"] = {"content": [{"token": t, "logprob": lp} for t, lp in token_logprobs]}
    return c


def _tokens_for(content, answer, answer_lp):
    """Split `content` so the answer is its own token, at `answer_lp`; the rest ~certain."""
    head, _, tail = content.partition(answer)
    return [(head, -1e-9), (answer, answer_lp), (tail, -1e-9)]


def _patch(monkeypatch, choice, capture=None):
    def fake(cfg, schema, prompt, modality, image_path=None, text=None, logprobs=False):
        if capture is not None:
            capture.update(schema=schema, prompt=prompt, modality=modality,
                           image_path=image_path, logprobs=logprobs)
        return choice
    monkeypatch.setattr(semvqa.semextract, "gen_endpoint_full", fake)


def test_score_comes_from_logprobs(monkeypatch):
    content = json.dumps({"answer": "yes", "conf": 0.99})
    _patch(monkeypatch, _choice(content, _tokens_for(content, "yes", math.log(0.6))))
    d = semvqa.img_vqa_detail("i.png", "damaged?", ["yes", "no"], cfg=CFG)
    assert d["answer"] == "yes"
    assert d["score_source"] == "logprobs"
    # 0.6 from the token, NOT the 0.99 the model claimed about itself
    assert abs(d["score"] - 0.6) < 1e-6


def test_falls_back_to_self_reported_conf_without_logprobs(monkeypatch):
    _patch(monkeypatch, _choice(json.dumps({"answer": "no", "conf": 0.4})))
    d = semvqa.img_vqa_detail("i.png", "damaged?", ["yes", "no"], cfg=CFG)
    assert d["answer"] == "no" and d["score"] == 0.4 and d["score_source"] == "self"


def test_falls_back_to_default_without_any_signal(monkeypatch):
    _patch(monkeypatch, _choice(json.dumps({"answer": "no"})))
    d = semvqa.img_vqa_detail("i.png", "q?", cfg=CFG, default=0.25)
    assert d["score"] == 0.25 and d["score_source"] == "default"


def test_choices_become_a_guided_json_enum(monkeypatch):
    cap = {}
    _patch(monkeypatch, _choice(json.dumps({"answer": "yes", "conf": 0.5})), cap)
    semvqa.img_vqa_detail("i.png", "q?", ["yes", "no"], cfg=CFG)
    assert cap["schema"]["properties"]["answer"]["enum"] == ["yes", "no"]
    assert cap["logprobs"] is True and cap["modality"] == "image"


def test_off_vocabulary_answer_is_rejected_not_invented(monkeypatch):
    _patch(monkeypatch, _choice(json.dumps({"answer": "maybe", "conf": 0.9})))
    d = semvqa.img_vqa_detail("i.png", "q?", ["yes", "no"], cfg=CFG)
    assert d["answer"] == "none" and "off-vocabulary" in d["error"]


def test_json_wrapped_in_prose_is_salvaged(monkeypatch):
    _patch(monkeypatch, _choice('Sure! {"answer": "yes", "conf": 0.7} hope that helps'))
    d = semvqa.img_vqa_detail("i.png", "q?", ["yes", "no"], cfg=CFG)
    assert d["answer"] == "yes" and d["score"] == 0.7


def test_endpoint_failure_is_contained(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(semvqa.semextract, "gen_endpoint_full", boom)
    d = semvqa.img_vqa_detail("i.png", "q?", cfg=CFG)
    assert d["answer"] == "none" and d["score"] == 0.0 and "refused" in d["error"]


def test_img_vqa_returns_the_answer_score_tuple(monkeypatch):
    _patch(monkeypatch, _choice(json.dumps({"answer": "yes", "conf": 0.8})))
    assert semvqa.img_vqa("i.png", "q?", cfg=CFG) == ("yes", 0.8)


# --- the residual pass ------------------------------------------------------

def test_escalate_only_touches_low_confidence_rows(monkeypatch):
    _patch(monkeypatch, _choice(json.dumps({"answer": "delta", "conf": 0.9})))
    rows = [{"image_path": "a", "value": "united", "conf": 0.95},   # confident -> untouched
            {"image_path": "b", "value": "none", "conf": 1.0},      # empty -> escalate
            {"image_path": "c", "value": "spirit", "conf": 0.1}]    # unsure -> escalate
    out, n = semvqa.escalate(rows, "which airline?", cfg=CFG, theta=0.5)
    assert n == 2
    assert out[0]["value"] == "united" and "_escalated" not in out[0]
    assert out[1]["value"] == "delta" and out[2]["value"] == "delta"
    assert out[1]["conf"] == 0.9


def test_escalate_does_not_mutate_its_input(monkeypatch):
    _patch(monkeypatch, _choice(json.dumps({"answer": "delta", "conf": 0.9})))
    rows = [{"image_path": "b", "value": "none", "conf": 0.0}]
    semvqa.escalate(rows, "q?", cfg=CFG)
    assert rows[0]["value"] == "none"


def test_escalate_makes_no_calls_when_everything_is_confident(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not call the endpoint")
    monkeypatch.setattr(semvqa.semextract, "gen_endpoint_full", boom)
    rows = [{"image_path": "a", "value": "united", "conf": 0.9}]
    out, n = semvqa.escalate(rows, "q?", cfg=CFG, theta=0.5)
    assert n == 0 and out == rows
