import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vadar import backend as semvision


def test_get_encoder_is_cached(monkeypatch):
    calls = {"n": 0}
    class Dummy:
        def __init__(self, model_id="m", device=None): calls["n"] += 1
    monkeypatch.setattr(semvision, "ClipEncoder", Dummy)
    semvision._ENCODER_CACHE.clear()
    a = semvision.get_encoder("m"); b = semvision.get_encoder("m")
    assert a is b and calls["n"] == 1
