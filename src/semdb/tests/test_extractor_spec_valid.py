import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semvision


def test_valid_specs_pass():
    for attr in [
        {"name": "colors", "type": "list[enum]", "extractor": {"tier": "cv", "method": "dominant_colors"}},
        {"name": "cat", "type": "enum", "extractor": {"tier": "clip", "method": "classify", "labels": ["a", "b"]}},
        {"name": "vibe", "type": "string", "extractor": {"tier": "vlm"}},
    ]:
        assert semvision.validate_extractor_spec(attr) == []


def test_clip_classify_requires_labels():
    errs = semvision.validate_extractor_spec(
        {"name": "cat", "type": "enum", "extractor": {"tier": "clip", "method": "classify"}})
    assert any("labels" in e for e in errs)


def test_missing_extractor_flagged():
    errs = semvision.validate_extractor_spec({"name": "x", "type": "enum"})
    assert any("extractor" in e for e in errs)
