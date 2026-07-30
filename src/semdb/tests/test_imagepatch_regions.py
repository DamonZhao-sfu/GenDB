"""Regions actually restrict the proxy's input.

Before the box fix, `patch.crop(...)` returned a patch whose every primitive still ran on
the FULL image — silently, so composed code looked right and scored wrong. Each test here
fails against that behavior.
"""
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vadar import imagepatch  # noqa: E402


def _half_and_half(tmp_path):
    """100x60: left half red, right half blue."""
    p = os.path.join(str(tmp_path), "halves.png")
    im = Image.new("RGB", (100, 60), (200, 30, 30))
    im.paste(Image.new("RGB", (50, 60), (40, 70, 190)), (50, 0))
    im.save(p)
    return p


def _patch(tmp_path):
    return imagepatch.ImagePatch(_half_and_half(tmp_path), {"encoder": None, "palette": None})


def test_crop_restricts_the_region_pixels(tmp_path):
    full = _patch(tmp_path)
    assert set(full.dominant_colors(0.1)) == {"red", "blue"}
    assert full.crop(0, 0, 50, 60).dominant_colors(0.1) == ["red"]
    assert full.crop(50, 0, 100, 60).dominant_colors(0.1) == ["blue"]


def test_crop_accepts_fractions(tmp_path):
    full = _patch(tmp_path)
    assert full.crop(0.0, 0.0, 0.5, 1.0).dominant_colors(0.1) == ["red"]
    assert full.crop(0.5, 0.0, 1.0, 1.0).dominant_colors(0.1) == ["blue"]


def test_crop_composes_into_absolute_coordinates(tmp_path):
    full = _patch(tmp_path)
    # right half, then the right half OF THAT -> the rightmost quarter
    assert full.crop(0.5, 0, 1, 1).crop(0.5, 0, 1, 1).box == (75, 0, 100, 60)


def test_crop_is_clamped_and_order_insensitive(tmp_path):
    full = _patch(tmp_path)
    assert full.crop(-40, -10, 500, 500).box == (0, 0, 100, 60)
    assert full.crop(80, 50, 20, 10).box == full.crop(20, 10, 80, 50).box   # swapped edges


def test_size_reports_the_patch_not_the_image(tmp_path):
    full = _patch(tmp_path)
    assert full.size == (100, 60)
    assert full.crop(0, 0, 50, 60).size == (50, 60)


def test_regions_grid_tiles_the_image(tmp_path):
    full = _patch(tmp_path)
    cells = full.regions_grid(1, 2)
    assert [c.box for c in cells] == [(0, 0, 50, 60), (50, 0, 100, 60)]
    assert [c.dominant_colors(0.1) for c in cells] == [["red"], ["blue"]]
    assert len(full.regions_grid(3, 4)) == 12


def test_regions_grid_overlap_widens_cells(tmp_path):
    full = _patch(tmp_path)
    plain = full.regions_grid(1, 2)[0].box
    over = full.regions_grid(1, 2, overlap=0.2)[0].box
    assert over[2] > plain[2] and over[0] == 0        # extends right, clamped at the left


def test_regions_center_drops_the_margin(tmp_path):
    full = _patch(tmp_path)
    assert full.regions_center(0.5).box == (25, 15, 75, 45)


def test_region_key_distinguishes_patches(tmp_path):
    """The encoder cache is keyed by region — two different crops must not collide."""
    full = _patch(tmp_path)
    assert full._key != full.crop(0, 0, 50, 60)._key
    assert full.crop(0, 0, 50, 60)._key != full.crop(50, 0, 100, 60)._key
