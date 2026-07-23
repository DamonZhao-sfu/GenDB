#!/usr/bin/env python3
"""
vadar_run.py — END-TO-END example of the VADAR-style 3-stage generated-code pipeline
for three SemBench image queries (mmqa q2a, mmqa q7, ecomm q2), run over the FULL corpus
and scored against ground truth.

The three stages (this script embodies what the converted agents produce):
  1. SIGNATURE (Schema Designer): the per-query field spec — fields + method + value space.
  2. API+PROGRAM (Extractor): `extract(patch)` composing the imagepatch base primitives.
  3. ENGINE + COMPILED (semvision.run + Code Generator): run extract over the corpus,
     then the relational join/filter, then score.

The extract() bodies below are exactly the compositions validated per query:
  q2a  -> gate "racetrack logo?" then classify over Track values          (CLIP)
  q7   -> best_ocr_match over Airlines values (OCR reads the wordmark)     (OCR+fuzzy)
  q2   -> classify(sports_shoe) AND dominant_colors ⊇ {yellow, silver}     (CLIP+CV)

Run:  <sembench python> src/semdb/vadar_run.py
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imagepatch  # noqa: E402
import semvision   # noqa: E402

MM = "/localhome/hza214/SemBench/files/mmqa"
EC = "/localhome/hza214/SemBench/files/ecomm"
ECMAN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "runs/_materialized/ecomm_sf250/IMAGES.csv")


def _distinct(csv_path, col):
    return sorted({r[col] for r in csv.DictReader(open(csv_path)) if r.get(col)})


def _norm(s):
    return str(s).strip().lower()


def prf(pred, gold):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(pred=len(pred), gt=len(gold), tp=tp, P=round(p, 3), R=round(r, 3), F1=round(f1, 3))


# ---------------------------------------------------------------------------
# STAGE 2+3: extract() compositions — one per query (the Program-agent output)
# ---------------------------------------------------------------------------

def extract_q2a(patch, tracks):
    kind = patch.classify(["a horse racetrack logo", "an airline company logo",
                           "a product photo", "a scenic landscape photo"])
    if kind == "a horse racetrack logo":
        return {"racetrack": patch.classify(tracks, "the logo of {}")}
    return {"racetrack": "none"}


def extract_q7(patch, airlines):
    return {"airline": patch.best_ocr_match(airlines)}   # self-gates: non-airline -> 'none'


def extract_q2(patch):
    return {"product_type": patch.classify(["sports_shoes", "sandal", "boot",
                                            "other_footwear", "not_footwear"]),
            "colors": patch.dominant_colors(0.03)}


def main():
    ctx = {"encoder": semvision.get_encoder("openai/clip-vit-base-patch32"), "palette": None}

    def P(path):
        return imagepatch.ImagePatch(path, ctx)

    imgs = list(csv.DictReader(open(f"{MM}/data/sf_200/images.csv")))          # 200 mmqa images

    # ---------- mmqa q2a: racetrack logo join ----------
    tracks = _distinct(f"{MM}/data/sf_200/ap_warrior.csv", "Track")
    ap = list(csv.DictReader(open(f"{MM}/data/sf_200/ap_warrior.csv")))
    gt2a = {(_norm(i), _norm(x)) for i, x in
            json.load(open(f"{MM}/raw_results/ground_truth/Q2a.json"))["ground_truth"]}
    pred2a = set()
    for row in imgs:
        rec = extract_q2a(P(row["image_filepath"]), tracks)
        if rec["racetrack"] == "none":
            continue
        for t in ap:                                    # relational join on Track
            if _norm(t["Track"]) == _norm(rec["racetrack"]):
                pred2a.add((_norm(t["ID"]), _norm(row["image_filename"])))
    print("mmqa q2a (classify+gate):", prf(pred2a, gt2a))

    # ---------- mmqa q7: airline logo join ----------
    airlines = _distinct(f"{MM}/data/sf_200/tampa_international_airport.csv", "Airlines")
    gt7 = {(_norm(a), _norm(x)) for a, x in
           json.load(open(f"{MM}/raw_results/ground_truth/Q7.json"))["ground_truth"]}
    pred7 = set()
    for row in imgs:
        air = extract_q7(P(row["image_filepath"]), airlines)["airline"]
        if air != "none":                               # emit (Airlines, uri)
            pred7.add((_norm(air), _norm(row["image_filename"])))
    print("mmqa q7  (ocr_match)     :", prf(pred7, gt7))

    # ---------- ecomm q2: sports shoe ∧ yellow ∧ silver ----------
    gt2 = {_norm(r["id"]) for r in csv.DictReader(open(f"{EC}/raw_results/ground_truth/Q2.csv"))}
    pred2 = set()
    for r in csv.DictReader(open(ECMAN)):
        p = f"{EC}/data/sf_250/images/{r['filename']}"
        if not os.path.exists(p):
            continue
        rec = extract_q2(P(p))
        if rec["product_type"] == "sports_shoes" and {"yellow", "silver"}.issubset(set(rec["colors"])):
            pred2.add(_norm(r["id"]))
    print("ecomm q2 (classify+cv)  :", prf(pred2, gt2))


if __name__ == "__main__":
    main()
