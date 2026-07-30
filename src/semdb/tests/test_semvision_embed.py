"""OpImgEmbed / OpImgPairScore — image-to-image similarity and corpus embedding."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vadar import backend as semvision  # noqa: E402


class FakeEncoder:
    """Cache-aware encoder: maps a path to a fixed unit vector."""

    def __init__(self, vecs):
        self._v = {k: (np.asarray(v, np.float32) / np.linalg.norm(v)) for k, v in vecs.items()}
        self.calls = 0

    def encode_image(self, src, key=None):
        self.calls += 1
        return self._v[src]

    def encode_images(self, srcs, batch=64, on_error="zero"):
        return np.stack([self._v[s] for s in srcs])


class LegacyEncoder:
    """Pre-cache signature — must keep working (no `key` parameter)."""

    def __init__(self, vec):
        self.vec = np.asarray(vec, np.float32) / np.linalg.norm(vec)

    def encode_image(self, src):
        return self.vec

    def encode_text(self, labels, template="a photo of {}"):
        return np.stack([self.vec] * len(labels))


def test_pair_score_is_1_for_identical_and_low_for_orthogonal():
    enc = FakeEncoder({"a.jpg": [1, 0, 0], "b.jpg": [1, 0, 0], "c.jpg": [0, 1, 0]})
    assert semvision.img_pair_score("a.jpg", "b.jpg", enc) == 1.0
    assert abs(semvision.img_pair_score("a.jpg", "c.jpg", enc) - 0.5) < 1e-6


def test_pair_score_is_bounded_for_opposite_vectors():
    enc = FakeEncoder({"a.jpg": [1, 0, 0], "z.jpg": [-1, 0, 0]})
    assert semvision.img_pair_score("a.jpg", "z.jpg", enc) == 0.0


def test_legacy_encoder_without_key_still_works():
    """A third-party encoder predating the region cache must not break."""
    enc = LegacyEncoder([1, 0, 0])
    assert semvision.clip_classify("x.jpg", ["a", "b"], enc, key=("x.jpg", (1, 2, 3, 4)))[0] == "a"
    assert semvision.img_pair_score("x.jpg", "y.jpg", enc, key_a=("x", None)) == 1.0


def test_embed_corpus_returns_row_aligned_matrix():
    enc = FakeEncoder({"a.jpg": [1, 0, 0], "b.jpg": [0, 1, 0]})
    m = semvision.embed_corpus(["a.jpg", "b.jpg"], enc)
    assert m.shape == (2, 3)
    assert np.allclose(m[0], [1, 0, 0])


def test_save_and_load_embeddings_roundtrip(tmp_path):
    mat = np.asarray([[1, 0], [0, 1]], np.float32)
    out = os.path.join(str(tmp_path), "emb.npy")
    semvision.save_embeddings(out, ["img1", "img2"], mat)
    loaded, index = semvision.load_embeddings(out)
    assert np.allclose(loaded, mat)
    assert index == {"img1": 0, "img2": 1}
