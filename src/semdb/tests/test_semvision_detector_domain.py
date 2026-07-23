import os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semvision


class MockDetector:
    def __init__(self, dets): self._d = dets   # list[(name, conf)]
    def detect(self, img_path, min_conf=0.25): return [d for d in self._d if d[1] >= min_conf]


class MockDomain:
    def __init__(self, probs): self._p = probs
    def probs(self, img_path): return self._p


def _img(tmp): p = os.path.join(str(tmp), "x.png"); Image.new("RGB", (16, 16)).save(p); return p


def test_detect_presence_and_counts(tmp_path):
    det = MockDetector([("zebra", 0.9), ("zebra", 0.8), ("person", 0.5)])
    found, conf, counts = semvision.detect(_img(tmp_path), ["zebra", "impala"], det)
    assert found == ["zebra"] and conf == 0.9 and counts["zebra"] == 2


def test_dispatch_detector_single_class_presence(tmp_path):
    schema = {"attributes": [{"name": "has_zebra", "type": "boolean",
              "extractor": {"tier": "detector", "classes": ["zebra"]}}]}
    ctx = {"encoder": None, "palette": None, "detector": MockDetector([("zebra", 0.7)]), "domain": {}}
    rec = semvision.extract_record(_img(tmp_path), schema, ctx)
    assert rec["has_zebra"] == "yes" and rec["conf"] == 0.7


def test_dispatch_domain_xray(tmp_path):
    schema = {"attributes": [{"name": "xray_sick", "type": "boolean",
              "extractor": {"tier": "domain", "model": "torchxrayvision:densenet121-res224-all",
                            "labels": ["Pneumonia", "Effusion"], "params": {"threshold": 0.5}}}]}
    ctx = {"encoder": None, "palette": None, "detector": None,
           "domain": {"torchxrayvision:densenet121-res224-all": MockDomain({"Pneumonia": 0.8, "Effusion": 0.1})}}
    rec = semvision.extract_record(_img(tmp_path), schema, ctx)
    assert rec["xray_sick"] == "yes" and abs(rec["conf"] - 0.8) < 1e-6


def test_validate_detector_domain_specs():
    assert semvision.validate_extractor_spec(
        {"name": "z", "type": "boolean", "extractor": {"tier": "detector", "classes": ["zebra"]}}) == []
    assert any("classes" in e for e in semvision.validate_extractor_spec(
        {"name": "z", "type": "boolean", "extractor": {"tier": "detector"}}))
    assert any("model" in e for e in semvision.validate_extractor_spec(
        {"name": "x", "type": "boolean", "extractor": {"tier": "domain", "labels": ["Pneumonia"]}}))
