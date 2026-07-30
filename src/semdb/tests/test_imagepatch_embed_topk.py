"""P1.1 — OpImgEmbed（向量）+ 向量化 top-k。

pair_score 是逐对 CLIP 调用；一个 image-image join / rank 需要的是「一次编码、
一次矩阵乘」。topk_text 则把 classify 的 argmax 放宽成带分数的候选集，
这正是 cascade 的候选来源。
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vadar import imagepatch  # noqa: E402
from vadar import backend as semvision  # noqa: E402


class VecEncoder:
    def __init__(self, vec, label_vecs=None):
        self.v = np.asarray(vec, np.float32) / np.linalg.norm(vec)
        self.lv = {k: np.asarray(x, np.float32) / np.linalg.norm(x)
                   for k, x in (label_vecs or {}).items()}

    def encode_image(self, src, key=None):
        return self.v

    def encode_text(self, labels, template="a photo of {}"):
        return np.stack([self.lv[str(l)] for l in labels])


def _img(tmp_path, name="x.png"):
    p = os.path.join(str(tmp_path), name)
    Image.new("RGB", (40, 40), (255, 255, 255)).save(p)
    return p


# --- semvision 层 -----------------------------------------------------------

def test_topk_similar_ranks_and_rescales_to_unit_interval():
    q = np.asarray([1.0, 0.0], np.float32)
    m = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], np.float32)
    rows = semvision.topk_similar(q, m, k=2)
    assert [i for i, _ in rows] == [0, 1]
    assert rows[0][1] == 1.0
    assert abs(rows[1][1] - 0.5) < 1e-6


def test_topk_similar_on_an_empty_corpus_is_empty():
    assert semvision.topk_similar(np.asarray([1.0, 0.0], np.float32),
                                  np.zeros((0, 2), np.float32)) == []


def test_topk_similar_clamps_k_to_the_corpus_size():
    q = np.asarray([1.0, 0.0], np.float32)
    m = np.asarray([[1.0, 0.0]], np.float32)
    assert len(semvision.topk_similar(q, m, k=10)) == 1


def test_embed_image_and_embed_text_are_unit_norm():
    enc = VecEncoder([3.0, 4.0], {"cat": [1.0, 0.0]})
    assert abs(np.linalg.norm(semvision.embed_image("a.png", enc)) - 1.0) < 1e-6
    assert abs(np.linalg.norm(semvision.embed_text("cat", enc)) - 1.0) < 1e-6


# --- ImagePatch 层 ----------------------------------------------------------

def test_embed_returns_a_plain_float_list(tmp_path):
    p = imagepatch.ImagePatch(_img(tmp_path), {"encoder": VecEncoder([1.0, 0.0])})
    v = p.embed()
    assert isinstance(v, list) and all(isinstance(x, float) for x in v)
    assert abs(sum(x * x for x in v) - 1.0) < 1e-6


def test_topk_similar_ranks_other_patches_by_index(tmp_path):
    enc = VecEncoder([1.0, 0.0])
    q = imagepatch.ImagePatch(_img(tmp_path, "q.png"), {"encoder": enc})
    others = [imagepatch.ImagePatch(_img(tmp_path, f"o{i}.png"), {"encoder": enc})
              for i in range(3)]
    rows = q.topk_similar(others, k=2)
    assert len(rows) == 2
    assert all(0 <= i < 3 for i, _ in rows)
    assert all(s == 1.0 for _, s in rows)      # 同一编码器 -> 全同向


def test_topk_similar_on_no_candidates_is_empty(tmp_path):
    q = imagepatch.ImagePatch(_img(tmp_path), {"encoder": VecEncoder([1.0, 0.0])})
    assert q.topk_similar([], k=3) == []


def test_topk_text_returns_texts_with_scores_best_first(tmp_path):
    enc = VecEncoder([1.0, 0.0], {"cat": [1.0, 0.0], "dog": [0.0, 1.0],
                                  "bird": [-1.0, 0.0]})
    p = imagepatch.ImagePatch(_img(tmp_path), {"encoder": enc})
    rows = p.topk_text(["dog", "cat", "bird"], k=2)
    assert [t for t, _ in rows] == ["cat", "dog"]
    assert rows[0][1] > rows[1][1]
