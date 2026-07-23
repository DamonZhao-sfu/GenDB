import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semvision


class FakeEncoder:
    def __init__(self, img_vec, text_vecs):
        self._img = np.asarray(img_vec, np.float32); self._img /= np.linalg.norm(self._img)
        self._txt = {k: (np.asarray(v, np.float32) / np.linalg.norm(v)) for k, v in text_vecs.items()}
    def encode_image(self, _path): return self._img
    def encode_text(self, labels, template="a photo of {}"): return np.stack([self._txt[l] for l in labels])


def test_clip_classify_picks_nearest_label():
    enc = FakeEncoder([1, 0, 0], {"sports_shoes": [1, 0, 0], "sandal": [0, 1, 0], "boot": [0, 0, 1]})
    label, conf = semvision.clip_classify("x.jpg", ["sports_shoes", "sandal", "boot"], enc)
    assert label == "sports_shoes" and conf > 0.5


def test_clip_multilabel_thresholds():
    enc = FakeEncoder([1, 1, 0], {"yellow": [1, 0, 0], "silver": [0, 1, 0], "green": [0, 0, 1]})
    labels, _ = semvision.clip_multilabel("x.jpg", ["yellow", "silver", "green"], enc, thresh=0.5)
    assert set(labels) == {"yellow", "silver"}


def test_clip_match_similarity_range():
    enc = FakeEncoder([1, 0, 0], {"a red running shoe": [1, 0, 0]})
    s = semvision.clip_match("x.jpg", "a red running shoe", enc)
    assert 0.99 <= s <= 1.0
