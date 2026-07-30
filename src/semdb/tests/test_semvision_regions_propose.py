"""P1.3 — OpImgRegion：前景连通域区域提案（确定性、model-free）。

论文的 OpImgRegion 输出 (RegIdx, BBox, Mask)；我们此前只有 regions_grid 这种
盲切。E-Commerce 一张图多件衣服、CarDamage 局部损伤，都需要按内容切而不是按网格切。
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vadar import imagepatch  # noqa: E402
from vadar import backend as semvision  # noqa: E402


def _two_blobs(tmp_path, size=(200, 100)):
    """白底 + 左右两个分离的黑块。"""
    im = Image.new("RGB", size, (255, 255, 255))
    px = im.load()
    for x in range(10, 60):
        for y in range(20, 80):
            px[x, y] = (0, 0, 0)
    for x in range(130, 190):
        for y in range(20, 80):
            px[x, y] = (0, 0, 0)
    p = os.path.join(str(tmp_path), "blobs.png")
    im.save(p)
    return p


def _blank(tmp_path):
    p = os.path.join(str(tmp_path), "blank.png")
    Image.new("RGB", (100, 100), (255, 255, 255)).save(p)
    return p


# --- connected components ---------------------------------------------------

def test_label_4c_separates_disconnected_blobs():
    m = np.zeros((5, 7), bool)
    m[1:3, 1:3] = True
    m[1:3, 5:7] = True
    lab = semvision._label_4c(m)
    assert lab[0, 0] == 0                       # 背景是 0
    assert lab[1, 1] != lab[1, 5]               # 两个块不同标签
    assert len(set(lab[m].tolist())) == 2


def test_label_4c_joins_an_l_shape_into_one_component():
    m = np.zeros((4, 4), bool)
    m[1, 1:4] = True
    m[1:4, 1] = True
    lab = semvision._label_4c(m)
    assert len(set(lab[m].tolist())) == 1


def test_label_4c_on_an_empty_mask_is_all_background():
    assert semvision._label_4c(np.zeros((3, 3), bool)).max() == 0


# --- region proposal --------------------------------------------------------

def test_propose_region_boxes_finds_both_blobs(tmp_path):
    boxes = semvision.propose_region_boxes(_two_blobs(tmp_path), max_regions=8)
    assert len(boxes) == 2
    for l, t, r, b in boxes:
        assert 0.0 <= l < r <= 1.0 and 0.0 <= t < b <= 1.0
    lefts = sorted(l for l, _, _, _ in boxes)
    assert lefts[0] < 0.4 and lefts[1] > 0.5    # 一左一右


def test_propose_region_boxes_respects_max_regions(tmp_path):
    assert len(semvision.propose_region_boxes(_two_blobs(tmp_path), max_regions=1)) == 1


def test_propose_region_boxes_drops_specks_below_min_area(tmp_path):
    assert semvision.propose_region_boxes(_two_blobs(tmp_path), min_area_frac=0.9) == []


def test_propose_region_boxes_on_a_uniform_image_is_empty(tmp_path):
    assert semvision.propose_region_boxes(_blank(tmp_path)) == []


def test_propose_region_boxes_is_deterministic(tmp_path):
    p = _two_blobs(tmp_path)
    assert semvision.propose_region_boxes(p) == semvision.propose_region_boxes(p)


# --- ImagePatch 层 ----------------------------------------------------------

def test_propose_regions_returns_sub_patches_in_absolute_pixels(tmp_path):
    p = imagepatch.ImagePatch(_two_blobs(tmp_path), {"encoder": None, "palette": None})
    regions = p.propose_regions()
    assert len(regions) == 2
    for r in regions:
        assert isinstance(r, imagepatch.ImagePatch)
        l, t, rr, b = r.bbox
        assert 0 <= l < rr <= 200 and 0 <= t < b <= 100


def test_propose_regions_inside_a_crop_offsets_to_absolute(tmp_path):
    p = imagepatch.ImagePatch(_two_blobs(tmp_path), {"encoder": None, "palette": None})
    sub = p.crop(100, 0, 200, 100)              # 只含右边那块
    regions = sub.propose_regions()
    assert len(regions) == 1
    assert regions[0].bbox[0] >= 100            # 已偏移回整图坐标


def test_propose_regions_warns_and_falls_back_for_an_unavailable_method(tmp_path, capsys):
    p = imagepatch.ImagePatch(_two_blobs(tmp_path), {"encoder": None, "palette": None})
    regions = p.propose_regions(method="sam")
    assert len(regions) == 2                    # 降级到 contour，而不是假装支持
    assert "contour" in capsys.readouterr().out
