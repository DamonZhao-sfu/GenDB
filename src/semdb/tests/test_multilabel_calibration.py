"""`classify_multi`'s threshold must mean the same thing on every image.

Raw CLIP cosine carries a large per-image offset — measured at 0.089..0.306 across one
200-image corpus, a 0.216 swing — while the old sigmoid (centered at cos 0.2, width 0.05)
was effectively a step at 0.25. The offset alone therefore decided the outcome: 145 of
200 images returned either NOTHING or EVERY label regardless of what they showed, and
mmqa q2a's solver accepted all 6 racetrack names for every image it admitted (65 rows
out, 0 correct).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vadar import backend  # noqa: E402


class OffsetEncoder:
    """Emits a fixed label geometry plus a per-image constant offset.

    The offset is exactly the nuisance term real CLIP has: it shifts every label's
    cosine together and says nothing about which label matches.
    """

    def __init__(self, offset):
        self.offset = offset
        # relative shape: label 0 clearly best, label 1 middling, labels 2-3 poor
        self.rel = np.asarray([0.06, 0.01, -0.03, -0.04], dtype=np.float32)

    def encode_image(self, src, key=None):
        return np.asarray([1.0], dtype=np.float32)

    def encode_text(self, texts, template="{}"):
        return (self.rel[: len(texts)] + self.offset).reshape(-1, 1)


LABELS = ["a", "b", "c", "d"]


def _chosen(offset, thresh=0.5):
    enc = OffsetEncoder(offset)
    chosen, _ = backend.clip_multilabel("x.png", LABELS, enc, thresh=thresh)
    return chosen


def test_the_same_label_geometry_survives_any_per_image_offset():
    """Identical relative matches must produce an identical answer at any offset."""
    baseline = _chosen(0.20)
    for offset in (0.09, 0.15, 0.25, 0.31):
        assert _chosen(offset) == baseline, f"offset {offset} changed the answer"


def test_a_low_offset_image_is_not_forced_to_return_nothing():
    assert _chosen(0.09) != []


def test_a_high_offset_image_is_not_forced_to_return_everything():
    assert _chosen(0.31) != LABELS


def test_the_best_label_is_kept_and_the_worst_dropped():
    chosen = _chosen(0.20)
    assert "a" in chosen and "d" not in chosen


def test_raising_the_threshold_is_monotone():
    loose, tight = set(_chosen(0.20, thresh=0.4)), set(_chosen(0.20, thresh=0.9))
    assert tight <= loose


def test_an_empty_value_space_does_not_crash():
    chosen, conf = backend.clip_multilabel("x.png", [], OffsetEncoder(0.2))
    assert chosen == [] and conf == 0.0
