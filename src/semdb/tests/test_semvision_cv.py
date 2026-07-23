import os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semvision


def _write_half(tmp, top_rgb, bottom_rgb):
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:32, :] = top_rgb
    arr[32:, :] = bottom_rgb
    p = os.path.join(tmp, "img.png")
    Image.fromarray(arr).save(p)
    return p


def test_dominant_colors_detects_both_yellow_and_silver(tmp_path):
    p = _write_half(str(tmp_path), (255, 255, 0), (192, 192, 192))
    colors, conf = semvision.cv_dominant_colors(p)
    assert set(colors) == {"yellow", "silver"}
    assert conf == 1.0


def test_dominant_colors_drops_tiny_fraction(tmp_path):
    arr = np.full((64, 64, 3), (0, 0, 0), dtype=np.uint8)
    arr[:2, :2] = (255, 255, 0)
    p = os.path.join(str(tmp_path), "img.png")
    Image.fromarray(arr).save(p)
    colors, _ = semvision.cv_dominant_colors(p)
    assert colors == ["black"]
