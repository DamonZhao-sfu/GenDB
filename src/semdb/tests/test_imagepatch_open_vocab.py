"""P1.4 — OpImgObj 的开放词表实现。

YOLOv8n 的词表是闭的（COCO-80）：find("impala") / find("bumper") 永远返回 []。
Wildlife 的物种和 CarDamage 的部件都在词表外，所以需要一个 prompt 即词表的检测器。
真实 OWLv2 权重不一定在环境里，所以这里全部对着 mock 测接口契约。
"""
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imagepatch  # noqa: E402
import semvision  # noqa: E402


class MockOwl:
    """形状同 semvision.OwlDetector：detect_prompts(src, prompts, min_conf)。"""

    def __init__(self, rows):
        self._rows = rows
        self.seen = []

    def detect_prompts(self, src, prompts, min_conf=0.1):
        self.seen.append((tuple(prompts), min_conf))
        return sorted([r for r in self._rows if r[0] in prompts and r[1] >= min_conf],
                      key=lambda r: -r[1])


def _img(tmp_path, size=(200, 100)):
    p = os.path.join(str(tmp_path), "x.png")
    Image.new("RGB", size, (255, 255, 255)).save(p)
    return p


def _ctx(**kw):
    base = {"encoder": None, "palette": None}
    base.update(kw)
    return base


def test_detect_open_boxes_sorts_by_confidence(tmp_path):
    owl = MockOwl([("impala", 0.4, (0, 0, 10, 10)), ("impala", 0.8, (20, 20, 40, 40))])
    rows = semvision.detect_open_boxes(_img(tmp_path), ["impala"], owl)
    assert [r[1] for r in rows] == [0.8, 0.4]


def test_find_open_detects_a_name_yolo_can_never_see(tmp_path):
    owl = MockOwl([("impala", 0.7, (10, 20, 60, 80))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(open_detector=owl))
    hits = p.find_open("impala")
    assert len(hits) == 1 and hits[0].box == (10, 20, 60, 80)
    assert owl.seen[0][0] == ("impala",)


def test_find_open_detail_carries_label_box_and_score(tmp_path):
    owl = MockOwl([("bumper", 0.66, (10, 20, 60, 80))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(open_detector=owl))
    d = p.find_open_detail("bumper")[0]
    assert d["label"] == "bumper" and d["score"] == 0.66 and d["box"] == (10, 20, 60, 80)


def test_find_open_inside_a_crop_offsets_to_absolute(tmp_path):
    owl = MockOwl([("impala", 0.7, (0, 0, 20, 20))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(open_detector=owl))
    hit = p.crop(100, 50, 200, 100).find_open("impala")[0]
    assert hit.box == (100, 50, 120, 70)


def test_find_open_honors_min_conf(tmp_path):
    owl = MockOwl([("impala", 0.2, (0, 0, 10, 10))])
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(open_detector=owl))
    assert p.find_open("impala", min_conf=0.5) == []


def test_find_open_degrades_to_empty_when_no_backend_is_installed(tmp_path, capsys):
    """OWLv2 是可选依赖：装不上时告警并返回 []，不能让整条查询崩掉。"""
    p = imagepatch.ImagePatch(_img(tmp_path), _ctx())
    imagepatch.ImagePatch._OPEN_WARNED.clear()
    real = semvision.get_open_detector

    def boom(*a, **kw):
        raise ImportError("no transformers owlv2")

    semvision.get_open_detector = boom
    try:
        assert p.find_open("impala") == []
    finally:
        semvision.get_open_detector = real
    assert "open-vocabulary" in capsys.readouterr().out


def test_closed_find_still_warns_and_points_at_find_open(tmp_path, capsys):
    class MockYolo:
        class _M:
            names = {0: "zebra"}

        model = _M()

        def detect(self, _p, min_conf=0.25):
            return []

    p = imagepatch.ImagePatch(_img(tmp_path), _ctx(detector=MockYolo()))
    imagepatch.ImagePatch._OOV_WARNED.discard("impala")
    assert p.find("impala") == []
    assert "find_open" in capsys.readouterr().out
