"""P0 — 每个 image 算子吐出论文 Table 1 的 (Value, Score) 输出 schema。

底座本来就算出了 score（clip_classify 返回 (label, prob)，detect_boxes 返回
(label, conf, box)），只是包装层扔掉了。没有 score 就没有 cascade / router /
阈值调优，所以这是整条线的前置。
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imagepatch  # noqa: E402


class FakeEncoder:
    """把每个 label 映射到一个固定向量，让 argmax 与 score 都是可预测的。"""

    def __init__(self, image_vec, label_vecs):
        self.iv = np.asarray(image_vec, np.float32)
        self.iv /= np.linalg.norm(self.iv)
        self.lv = {k: np.asarray(v, np.float32) / np.linalg.norm(v)
                   for k, v in label_vecs.items()}

    def encode_image(self, src, key=None):
        return self.iv

    def encode_text(self, labels, template="a photo of {}"):
        return np.stack([self.lv[str(l)] for l in labels])


class MockYolo:
    class _M:
        names = {0: "zebra", 1: "person"}

    def __init__(self, dets):
        self.model = self._M()
        self._d = dets

    def detect(self, img_path, min_conf=0.25):
        return [d for d in self._d if d[1] >= min_conf]


class MockOcr:
    def __init__(self, rows):
        self._rows = rows

    def readtext(self, _src, detail=1):
        return self._rows


def _img(tmp_path, size=(200, 100)):
    p = os.path.join(str(tmp_path), "x.png")
    Image.new("RGB", size, (255, 255, 255)).save(p)
    return p


def _ctx(**kw):
    base = {"encoder": None, "palette": None}
    base.update(kw)
    return base


# --- bbox (OpImgRegion 的 BBox 输出) ----------------------------------------

def test_bbox_is_the_full_frame_for_a_whole_image(tmp_path):
    p = imagepatch.ImagePatch(_img(tmp_path, (200, 100)), _ctx())
    assert p.bbox == (0, 0, 200, 100)


def test_bbox_of_a_crop_is_absolute(tmp_path):
    p = imagepatch.ImagePatch(_img(tmp_path, (200, 100)), _ctx())
    assert p.crop(0.5, 0.0, 1.0, 1.0).bbox == (100, 0, 200, 100)


# --- OpImgCls (Label, Score) ------------------------------------------------

def test_classify_detail_returns_label_and_score(tmp_path):
    enc = FakeEncoder([1, 0], {"cat": [1, 0], "dog": [0, 1]})
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(encoder=enc))
    label, score = p.classify_detail(["cat", "dog"])
    assert label == "cat"
    assert 0.0 <= score <= 1.0 and score > 0.5


def test_classify_still_returns_only_the_label(tmp_path):
    """向后兼容：已生成的程序调的是这个签名。"""
    enc = FakeEncoder([1, 0], {"cat": [1, 0], "dog": [0, 1]})
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(encoder=enc))
    assert p.classify(["cat", "dog"]) == "cat"


def test_verify_detail_returns_bool_and_score(tmp_path):
    enc = FakeEncoder([1, 0], {"a cat": [1, 0], "not a cat": [0, 1]})
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(encoder=enc))
    ok, score = p.verify_detail("a cat")
    assert ok is True and score > 0.5
    assert p.verify_property("a cat") is True


# --- OpImgObj (BBox, Label, Score) ------------------------------------------

def test_find_detail_returns_label_box_and_score(tmp_path):
    det = MockYolo([("zebra", 0.9, (10, 20, 60, 80))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(detector=det))
    d = p.find_detail("zebra")[0]
    assert d["label"] == "zebra" and d["score"] == 0.9
    assert d["box"] == (10, 20, 60, 80)
    assert isinstance(d["image"], imagepatch.ImagePatch)
    assert d["image"].box == (10, 20, 60, 80)


def test_find_detail_boxes_are_absolute_inside_a_crop(tmp_path):
    det = MockYolo([("zebra", 0.9, (0, 0, 20, 20))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(detector=det))
    d = p.crop(100, 50, 200, 100).find_detail("zebra")[0]
    assert d["box"] == (100, 50, 120, 70)


def test_find_detail_is_ordered_by_score(tmp_path):
    det = MockYolo([("zebra", 0.4, (0, 0, 10, 10)), ("zebra", 0.95, (20, 20, 30, 30))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(detector=det))
    assert [d["score"] for d in p.find_detail("zebra")] == [0.95, 0.4]


def test_find_is_unchanged_and_still_returns_patches(tmp_path):
    det = MockYolo([("zebra", 0.9, (10, 20, 60, 80))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(detector=det))
    hits = p.find("zebra")
    assert len(hits) == 1 and hits[0].box == (10, 20, 60, 80)


def test_find_detail_out_of_vocabulary_is_empty(tmp_path):
    det = MockYolo([("zebra", 0.9, (0, 0, 10, 10))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(detector=det))
    imagepatch.ImagePatch._OOV_WARNED.discard("impala")
    assert p.find_detail("impala") == []


# --- OpImgOCR (Text, Score) -------------------------------------------------

def test_best_ocr_match_detail_scores_a_fuzzy_hit(tmp_path):
    ocr = MockOcr([([[0, 0], [1, 0], [1, 1], [0, 1]], "DELTA AIRLINES", 0.9)])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(ocr=ocr))
    val, score = p.best_ocr_match_detail(["Delta Air Lines", "United"])
    assert val == "Delta Air Lines" and score > 0.9      # difflib ratio, near-exact
    assert p.best_ocr_match(["Delta Air Lines", "United"]) == "Delta Air Lines"


def test_best_ocr_match_detail_scores_the_distinctive_token_path(tmp_path):
    """长文本模糊匹配必然失败，只有 distinctive-token 命中能救回来 —— 它的 score
    是命中率，与模糊匹配的 ratio 不是一个量纲，但对同一算子跨行仍可比。"""
    text = "welcome aboard delta lines flight 42 departing from gate b7"
    ocr = MockOcr([([[0, 0], [1, 0], [1, 1], [0, 1]], text, 0.9)])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(ocr=ocr))
    val, score = p.best_ocr_match_detail(["Delta Air Lines", "United"])
    assert val == "Delta Air Lines" and score == 1.0     # 'delta' + 'lines' both read


def test_best_ocr_match_detail_scores_a_miss_as_zero(tmp_path):
    ocr = MockOcr([([[0, 0], [1, 0], [1, 1], [0, 1]], "xyzzy qwerty", 0.9)])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(ocr=ocr))
    assert p.best_ocr_match_detail(["Delta Air Lines", "United"]) == ("none", 0.0)


def test_best_ocr_match_detail_on_empty_text_is_none(tmp_path):
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(ocr=MockOcr([])))
    assert p.best_ocr_match_detail(["Delta Air Lines"]) == ("none", 0.0)
