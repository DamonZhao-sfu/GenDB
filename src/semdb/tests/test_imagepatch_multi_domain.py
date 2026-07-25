"""P1.2 — 多标签 CLIP 与 domain 专家模型的暴露。

两个后端在 semvision 里早就实现了（clip_multilabel / domain_classify + XrayClassifier），
但只有 extractor-spec 的 tier 路径够得着；VADAR 生成的程序完全用不上。
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imagepatch  # noqa: E402
import semvision  # noqa: E402


class MultiEncoder:
    def __init__(self, image_vec, label_vecs):
        self.iv = np.asarray(image_vec, np.float32)
        self.iv /= np.linalg.norm(self.iv)
        self.lv = {k: np.asarray(v, np.float32) / np.linalg.norm(v)
                   for k, v in label_vecs.items()}

    def encode_image(self, src, key=None):
        return self.iv

    def encode_text(self, labels, template="a photo of {}"):
        return np.stack([self.lv[str(l)] for l in labels])


class FakeDomainModel:
    def __init__(self, probs):
        self._p = probs
        self.calls = 0

    def probs(self, _src):
        self.calls += 1
        return self._p


def _img(tmp_path):
    p = os.path.join(str(tmp_path), "x.png")
    Image.new("RGB", (40, 40), (255, 255, 255)).save(p)
    return p


# --- multilabel -------------------------------------------------------------

def test_classify_multi_keeps_every_label_over_the_threshold(tmp_path):
    enc = MultiEncoder([1.0, 0.0], {"stripes": [1.0, 0.0], "spots": [0.99, 0.14],
                                    "plain": [-1.0, 0.0]})
    p = imagepatch.ImagePatch(_img(tmp_path), {"encoder": enc, "palette": None})
    labels, conf = p.classify_multi(["stripes", "spots", "plain"], thresh=0.5)
    assert set(labels) == {"stripes", "spots"}
    assert 0.0 <= conf <= 1.0


def test_classify_multi_can_return_nothing_but_still_scores(tmp_path):
    enc = MultiEncoder([1.0, 0.0], {"plain": [-1.0, 0.0]})
    p = imagepatch.ImagePatch(_img(tmp_path), {"encoder": enc, "palette": None})
    labels, conf = p.classify_multi(["plain"], thresh=0.9)
    assert labels == []
    assert 0.0 <= conf <= 1.0


# --- domain -----------------------------------------------------------------

def test_domain_classify_returns_yes_no_and_the_max_positive_prob(tmp_path):
    model = FakeDomainModel({"Pneumonia": 0.81, "Effusion": 0.10})
    ctx = {"encoder": None, "palette": None, "domain": {"fake:m": model}}
    p = imagepatch.ImagePatch(_img(tmp_path), ctx)
    assert p.domain_classify("fake:m", ["Pneumonia"], threshold=0.5) == ("yes", 0.81)
    assert p.domain_classify("fake:m", ["Effusion"], threshold=0.5) == ("no", 0.10)


def test_domain_model_is_cached_in_ctx_across_calls(tmp_path):
    model = FakeDomainModel({"Pneumonia": 0.9})
    ctx = {"encoder": None, "palette": None, "domain": {"fake:m": model}}
    p = imagepatch.ImagePatch(_img(tmp_path), ctx)
    p.domain_classify("fake:m", ["Pneumonia"])
    p.domain_classify("fake:m", ["Pneumonia"])
    assert model.calls == 2                     # 每次都推理
    assert ctx["domain"]["fake:m"] is model     # 但模型只加载一次


def test_domain_classify_creates_the_domain_cache_when_absent(tmp_path):
    """ctx 没有 'domain' 键时不能 KeyError —— 引擎并不总会预建它。"""
    ctx = {"encoder": None, "palette": None}
    p = imagepatch.ImagePatch(_img(tmp_path), ctx)
    loaded = {}

    def fake_get(model_id):
        loaded[model_id] = FakeDomainModel({"Pneumonia": 0.7})
        return loaded[model_id]

    real = semvision.get_domain_model
    semvision.get_domain_model = fake_get
    try:
        assert p.domain_classify("fake:m", ["Pneumonia"], threshold=0.5) == ("yes", 0.7)
    finally:
        semvision.get_domain_model = real
    assert "fake:m" in ctx["domain"]
