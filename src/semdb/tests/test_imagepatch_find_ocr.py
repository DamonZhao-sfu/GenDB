"""`find` (OpImgObj -> sub-patches) and `read_text_boxes` (OpImgOCR -> text+box+score).

`find` used to be advertised to the Program agent via `predefined.detect` but did not
exist on ImagePatch at all — every call raised AttributeError, which the engine's
per-row guard swallowed into a silent 'none' column.
"""
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vadar import imagepatch  # noqa: E402
from vadar import backend as semvision  # noqa: E402


class MockYolo:
    """Shaped like semvision.YoloDetector: .model.names + detect() -> (name, conf, box)."""

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


# --- find -------------------------------------------------------------------

def test_find_returns_sub_patches_at_absolute_coords(tmp_path):
    det = MockYolo([("zebra", 0.9, (10, 20, 60, 80)), ("person", 0.8, (0, 0, 5, 5))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(detector=det))
    hits = p.find("zebra")
    assert len(hits) == 1
    assert isinstance(hits[0], imagepatch.ImagePatch)
    assert hits[0].box == (10, 20, 60, 80)


def test_find_supports_presence_and_count(tmp_path):
    det = MockYolo([("zebra", 0.9, (0, 0, 10, 10)), ("zebra", 0.7, (20, 20, 30, 30))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(detector=det))
    assert len(p.find("zebra")) == 2 and bool(p.find("zebra"))
    assert p.find("person") == []


def test_find_orders_by_confidence(tmp_path):
    det = MockYolo([("zebra", 0.4, (0, 0, 10, 10)), ("zebra", 0.95, (20, 20, 30, 30))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(detector=det))
    assert p.find("zebra")[0].box == (20, 20, 30, 30)


def test_find_out_of_vocabulary_returns_empty_not_garbage(tmp_path, capsys):
    det = MockYolo([("zebra", 0.9, (0, 0, 10, 10))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(detector=det))
    imagepatch.ImagePatch._OOV_WARNED.discard("impala")
    assert p.find("impala") == []
    assert "vocabulary" in capsys.readouterr().out


def test_find_inside_a_crop_offsets_to_absolute(tmp_path):
    det = MockYolo([("zebra", 0.9, (0, 0, 20, 20))])   # relative to the cropped region
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(detector=det))
    hit = p.crop(100, 50, 200, 100).find("zebra")[0]
    assert hit.box == (100, 50, 120, 70)


def test_detector_classes_reads_the_vocabulary():
    assert set(semvision.detector_classes(MockYolo([]))) == {"zebra", "person"}
    assert semvision.detector_classes(object()) == []


# --- detect_boxes / backward compatibility ----------------------------------

def test_detect_boxes_yields_one_row_per_instance(tmp_path):
    det = MockYolo([("zebra", 0.9, (0, 0, 10, 10)), ("zebra", 0.5, (20, 20, 30, 30))])
    rows = semvision.detect_boxes(_img(tmp_path), ["zebra"], det)
    assert [r[0] for r in rows] == ["zebra", "zebra"]
    assert rows[0][1] == 0.9 and rows[0][2] == (0.0, 0.0, 10.0, 10.0)


def test_detect_still_accepts_two_tuple_detectors(tmp_path):
    class Old:
        def detect(self, _p, min_conf=0.25):
            return [("zebra", 0.9), ("zebra", 0.6)]

    found, conf, counts = semvision.detect(_img(tmp_path), ["zebra"], Old())
    assert found == ["zebra"] and conf == 0.9 and counts["zebra"] == 2
    assert semvision.detect_boxes(_img(tmp_path), ["zebra"], Old()) == []   # no boxes to give


# --- OCR --------------------------------------------------------------------

def test_read_text_boxes_returns_text_box_and_score(tmp_path):
    ocr = MockOcr([([[10, 20], [90, 20], [90, 50], [10, 50]], "DELTA", 0.93)])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(ocr=ocr))
    d = p.read_text_boxes()[0]
    assert d["text"] == "DELTA" and d["score"] == 0.93
    assert d["box"] == (10.0, 20.0, 90.0, 50.0)


def test_read_text_joins_boxes_and_defaults_to_no_filtering(tmp_path):
    ocr = MockOcr([([[0, 0], [1, 0], [1, 1], [0, 1]], "FLY", 0.9),
                   ([[2, 0], [3, 0], [3, 1], [2, 1]], "DELTA", 0.05)])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(ocr=ocr))
    assert p.read_text() == "FLY DELTA"          # min_conf=0.0 keeps the weak box
    assert p.read_text(min_conf=0.5) == "FLY"


def test_read_text_boxes_offset_by_the_crop(tmp_path):
    ocr = MockOcr([([[0, 0], [10, 0], [10, 10], [0, 10]], "X", 0.9)])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(ocr=ocr))
    d = p.crop(100, 50, 200, 100).read_text_boxes()[0]
    assert d["box"] == (100.0, 50.0, 110.0, 60.0)
