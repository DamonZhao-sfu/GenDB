"""Detector + domain-specialist backends in `vadar.backend`.

The schema-driven `extract_record` / `validate_extractor_spec` dispatch these used to
go through belonged to the compiled extract/compile pipeline and was removed with it —
generated programs call the primitives directly through `vadar.predefined`.
"""
import os, sys
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vadar import backend as semvision


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


def test_domain_classify_returns_yes_with_the_max_listed_probability(tmp_path):
    model = MockDomain({"Pneumonia": 0.8, "Effusion": 0.1})
    verdict, conf = semvision.domain_classify(_img(tmp_path), model,
                                              ["Pneumonia", "Effusion"], 0.5)
    assert verdict == "yes" and abs(conf - 0.8) < 1e-6


def test_domain_classify_fails_closed_below_threshold(tmp_path):
    model = MockDomain({"Pneumonia": 0.2, "Effusion": 0.1})
    verdict, conf = semvision.domain_classify(_img(tmp_path), model,
                                              ["Pneumonia", "Effusion"], 0.5)
    assert verdict == "no" and abs(conf - 0.2) < 1e-6
