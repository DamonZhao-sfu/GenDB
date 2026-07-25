"""The residual path: off without an endpoint (unchanged), typed and counted with one."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semruntime  # noqa: E402
import semvqa  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_meter(monkeypatch):
    semruntime.METER.__init__()
    monkeypatch.delenv("SEMDB_RESIDUAL", raising=False)


def _answer(content, patch):
    patch.setattr(semvqa.semextract, "gen_endpoint_full",
                  lambda *a, **k: {"message": {"content": content}})


# --- disabled paths must behave exactly as the old no-op version ------------

def test_no_endpoint_is_a_noop_that_counts_a_skip():
    assert semruntime.vlm_judge("is it a dog?", image_path="x.png") is False
    assert semruntime.METER.judge_calls == 0 and semruntime.METER.skipped == 1


def test_env_flag_forces_measurement_mode_even_with_an_endpoint(monkeypatch):
    monkeypatch.setenv("SEMDB_RESIDUAL", "0")
    monkeypatch.setattr(semvqa.semextract, "gen_endpoint_full",
                        lambda *a, **k: pytest.fail("must not call the endpoint"))
    assert semruntime.vlm_judge("q?", endpoint="http://x/v1", image_path="x.png") is False
    assert semruntime.METER.judge_calls == 0 and semruntime.METER.skipped == 1


# --- enabled path -----------------------------------------------------------

def test_judge_calls_the_model_when_an_endpoint_is_given(monkeypatch):
    _answer(json.dumps({"answer": "yes", "conf": 0.8}), monkeypatch)
    assert semruntime.vlm_judge("q?", endpoint="http://x/v1", model="m",
                                image_path="x.png") is True
    assert semruntime.METER.judge_calls == 1 and semruntime.METER.skipped == 0
    assert semruntime.METER.scores == [0.8]


def test_judge_is_false_on_a_no_answer(monkeypatch):
    _answer(json.dumps({"answer": "no", "conf": 0.9}), monkeypatch)
    assert semruntime.vlm_judge("q?", endpoint="http://x/v1", image_path="x.png") is False


def test_theta_gates_a_low_confidence_yes(monkeypatch):
    _answer(json.dumps({"answer": "yes", "conf": 0.3}), monkeypatch)
    assert semruntime.vlm_judge("q?", endpoint="http://x/v1", image_path="x.png",
                                theta=0.5) is False
    assert semruntime.vlm_judge("q?", endpoint="http://x/v1", image_path="x.png",
                                theta=0.2) is True


def test_vlm_answer_returns_a_real_field_value(monkeypatch):
    _answer(json.dumps({"answer": "Delta Air Lines", "conf": 0.77}), monkeypatch)
    ans, score = semruntime.vlm_answer("which airline?", ["Delta Air Lines", "Spirit"],
                                       endpoint="http://x/v1", image_path="x.png")
    assert ans == "Delta Air Lines" and score == 0.77


def test_endpoint_failure_counts_a_skip_not_a_match(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(semvqa.semextract, "gen_endpoint_full", boom)
    assert semruntime.vlm_judge("q?", endpoint="http://x/v1", image_path="x.png") is False
    assert semruntime.METER.skipped == 1


def test_llm_judge_uses_the_text_modality(monkeypatch):
    seen = {}

    def fake(cfg, schema, prompt, modality, image_path=None, text=None, logprobs=False):
        seen["modality"], seen["text"] = modality, text
        return {"message": {"content": json.dumps({"answer": "yes", "conf": 0.9})}}

    monkeypatch.setattr(semvqa.semextract, "gen_endpoint_full", fake)
    assert semruntime.llm_judge("comedy?", endpoint="http://x/v1", text="a funny film") is True
    assert seen == {"modality": "text", "text": "a funny film"}


def test_meter_summary_reports_the_residual(monkeypatch):
    _answer(json.dumps({"answer": "yes", "conf": 0.6}), monkeypatch)
    semruntime.vlm_judge("q?", endpoint="http://x/v1", image_path="x.png")
    semruntime.vlm_judge("q?", image_path="x.png")          # no endpoint -> skip
    assert semruntime.METER.summary() == {"judge_calls": 1, "skipped": 1, "mean_score": 0.6}
