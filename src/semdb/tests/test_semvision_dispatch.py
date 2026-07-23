import os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semvision
from test_semvision_clip import FakeEncoder


def test_dispatch_mixes_cv_and_clip(tmp_path):
    arr = np.zeros((64, 64, 3), np.uint8); arr[:32] = (240, 220, 40); arr[32:] = (192, 192, 192)
    p = os.path.join(str(tmp_path), "shoe.png"); Image.fromarray(arr).save(p)
    schema = {"attributes": [
        {"name": "product_type", "type": "enum",
         "extractor": {"tier": "clip", "method": "classify", "labels": ["sports_shoes", "sandal"]}},
        {"name": "colors", "type": "list[enum]",
         "extractor": {"tier": "cv", "method": "dominant_colors", "params": {"min_frac": 0.08}}},
    ]}
    enc = FakeEncoder([1, 0], {"sports_shoes": [1, 0], "sandal": [0, 1]})
    rec = semvision.extract_record(p, schema, {"encoder": enc, "palette": None})
    assert rec["product_type"] == "sports_shoes"
    assert set(rec["colors"]) == {"yellow", "silver"}
    assert 0.0 <= rec["conf"] <= 1.0


def test_dispatch_unknown_tier_is_none_not_crash(tmp_path):
    p = os.path.join(str(tmp_path), "x.png"); Image.new("RGB", (8, 8)).save(p)
    schema = {"attributes": [{"name": "vibe", "type": "string", "extractor": {"tier": "vlm"}}]}
    rec = semvision.extract_record(p, schema, {"encoder": None, "palette": None})
    assert rec["vibe"] == "none" and rec["conf"] == 0.0
