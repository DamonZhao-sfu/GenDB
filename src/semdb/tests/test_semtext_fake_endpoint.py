import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import semtext


def _fake_gen(monkeypatch, reply):
    import semextract
    calls = {"n": 0}
    def fake(cfg, json_schema, prompt, modality, image_path=None, text=None):
        calls["n"] += 1
        return reply(prompt, text)
    monkeypatch.setattr(semextract, "gen_endpoint", fake)
    return calls


def test_judge_true_false(monkeypatch):
    _fake_gen(monkeypatch, lambda p, t: '{"answer": true}' if "good" in (t or "") else '{"answer": false}')
    semtext.METER.reset()
    ctx = semtext.get_ctx("m", "http://x/v1")
    assert semtext.TextPatch("a good movie", ctx).judge("Is it positive?") is True
    assert semtext.TextPatch("a bad movie", ctx).judge("Is it positive?") is False
    assert semtext.METER.calls == 2


def test_classify_returns_option(monkeypatch):
    _fake_gen(monkeypatch, lambda p, t: '{"value": "comedy"}')
    ctx = semtext.get_ctx("m", "http://x/v1")
    assert semtext.TextPatch("funny", ctx).classify(["comedy", "drama"]) == "comedy"


def test_cache_dedups_identical_calls(monkeypatch):
    calls = _fake_gen(monkeypatch, lambda p, t: '{"answer": true}')
    semtext.METER.reset()
    ctx = semtext.get_ctx("m", "http://x/v1")
    tp = semtext.TextPatch("same text", ctx)
    tp.judge("Q?"); tp.judge("Q?")
    assert calls["n"] == 1        # second call served from cache
