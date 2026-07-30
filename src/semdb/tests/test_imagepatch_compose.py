"""Validates the ImagePatch base API + per-query composition on real SemBench data.
Integration (downloads CLIP + easyocr). Skips if data/models absent."""
import os, sys, csv, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
MM = "/localhome/hza214/SemBench/files/mmqa"
EC = "/localhome/hza214/SemBench/files/ecomm"
MAN = "/local-scratch/localhome/hza214/GenDB/src/semdb/runs/_materialized/ecomm_sf250/IMAGES.csv"


def _ctx():
    from vadar import backend as semvision
    return {"encoder": semvision.get_encoder("openai/clip-vit-base-patch32"), "palette": None}


def _skip_if_absent(*paths):
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f"absent: {p}")


def test_q2a_classify_small_valuespace():
    _skip_if_absent(f"{MM}/data/sf_200/ap_warrior.csv")
    from vadar import imagepatch
    ctx = _ctx()
    tracks = sorted({r["Track"] for r in csv.DictReader(open(f"{MM}/data/sf_200/ap_warrior.csv"))})
    id2t = {r["ID"]: r["Track"] for r in csv.DictReader(open(f"{MM}/data/sf_200/ap_warrior.csv"))}
    gt = json.load(open(f"{MM}/raw_results/ground_truth/Q2a.json"))["ground_truth"]
    ok = tot = 0
    for im in sorted({x for _i, x in gt}):
        p = f"{MM}/data/sf_200/images/{im}"
        if not os.path.exists(p):
            continue
        field = imagepatch.ImagePatch(p, ctx).classify(tracks, "the logo of {}")
        tot += 1; ok += field in {id2t[str(i)] for i, x in gt if x == im}
    assert tot and ok == tot                      # CLIP classify nails the small value space


def test_q7_ocr_match_large_valuespace():
    _skip_if_absent(f"{MM}/data/sf_200/tampa_international_airport.csv")
    from vadar import imagepatch
    ctx = _ctx()
    airl = sorted({r["Airlines"] for r in csv.DictReader(open(f"{MM}/data/sf_200/tampa_international_airport.csv")) if r.get("Airlines")})
    gt = json.load(open(f"{MM}/raw_results/ground_truth/Q7.json"))["ground_truth"]
    ok = tot = 0
    for air, im in gt:
        p = f"{MM}/data/sf_200/images/{im}"
        if not os.path.exists(p):
            continue
        tot += 1; ok += imagepatch.ImagePatch(p, ctx).best_ocr_match(airl) == air
    assert tot and ok >= tot - 1                  # OCR+fuzzy recovers ≥4/5 airline logos


def test_ecomm_q2_classify_plus_colors():
    _skip_if_absent(MAN)
    from vadar import imagepatch
    ctx = _ctx()
    gt = set(r["id"] for r in csv.DictReader(open(f"{EC}/raw_results/ground_truth/Q2.csv")))
    labels = ["sports_shoes", "sandal", "boot", "other_footwear", "not_footwear"]
    pred = set()
    for r in csv.DictReader(open(MAN)):
        p = f"{EC}/data/sf_250/images/{r['filename']}"
        if os.path.exists(p) and imagepatch.ImagePatch(p, ctx).classify(labels) == "sports_shoes" \
           and {"yellow", "silver"}.issubset(set(imagepatch.ImagePatch(p, ctx).dominant_colors(0.03))):
            pred.add(r["id"])
    assert len(pred & gt) >= 4                     # recovers ≥4/5 yellow+silver sports shoes
