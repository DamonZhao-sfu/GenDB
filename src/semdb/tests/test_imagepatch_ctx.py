"""`ImagePatch` must accept a bare encoder as well as the ctx mapping.

mmqa q2a wrote `ImagePatch(path, get_encoder(model))` — the two calls sit on adjacent
lines of the prompt skeleton — and every primitive then raised
`TypeError: 'ClipEncoder' object is not subscriptable`. Wrapped in the mandated per-row
try/except that became a silent all-"none" result: 200/200 rows lost, exit code 0,
F1 0. The intent of a bare encoder is never ambiguous, so normalize it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vadar import ImagePatch  # noqa: E402
from vadar.imagepatch import _as_ctx  # noqa: E402


class FakeEncoder:
    """Shaped like backend.ClipEncoder: no __getitem__, so ctx["encoder"] would raise."""

    def encode_text(self, texts, template="{}"):
        return [[1.0, 0.0] for _ in texts]


def test_bare_encoder_is_normalized_into_a_ctx_mapping():
    enc = FakeEncoder()
    assert _as_ctx(enc) == {"encoder": enc, "palette": None}


def test_an_existing_mapping_is_passed_through_untouched():
    ctx = {"encoder": FakeEncoder(), "palette": {"red": (255, 0, 0)}}
    assert _as_ctx(ctx) is ctx


def test_both_call_forms_produce_the_same_patch_context(tmp_path):
    from PIL import Image
    path = os.path.join(str(tmp_path), "x.png")
    Image.new("RGB", (8, 8)).save(path)
    enc = FakeEncoder()
    bare = ImagePatch(path, enc)
    mapping = ImagePatch(path, {"encoder": enc, "palette": None})
    assert bare.ctx["encoder"] is mapping.ctx["encoder"] is enc
    assert bare.ctx.get("palette") is None


def test_sub_patches_inherit_the_normalized_context(tmp_path):
    """`_child` re-wraps ctx; a region must not re-trigger the original TypeError."""
    from PIL import Image
    path = os.path.join(str(tmp_path), "x.png")
    Image.new("RGB", (40, 40)).save(path)
    region = ImagePatch(path, FakeEncoder()).crop(0, 0, 0.5, 0.5)
    assert isinstance(region.ctx, dict) and "encoder" in region.ctx


def test_missing_ocr_warns_once_and_is_distinguishable_from_empty_text(capsys, tmp_path):
    """A missing optional dependency must be LOUD.

    OCR against the runtime value space is the recommended primitive for wordmark logos,
    so on mmqa q2a/q7 it is the plan's primary discriminative path. With easyocr absent
    it silently returned "" for every image and the query fell through to a CLIP path
    that cannot separate two logos — with nothing in any log saying so.
    """
    from PIL import Image
    from vadar.imagepatch import ImagePatch as IP
    path = os.path.join(str(tmp_path), "x.png")
    Image.new("RGB", (16, 16)).save(path)
    IP._OCR_WARNED.clear()

    patch = IP(path, FakeEncoder())
    patch.ctx.pop("ocr", None)
    assert patch.read_text() == ""
    first = capsys.readouterr().out

    if "easyocr" in first:                      # dependency genuinely absent here
        assert "OCR unavailable" in first
        assert patch.best_ocr_match(["Churchill Downs"]) == "none"
        # warned once, not once per image
        IP(path, FakeEncoder()).read_text()
        assert "OCR unavailable" not in capsys.readouterr().out
