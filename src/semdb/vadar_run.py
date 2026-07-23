#!/usr/bin/env python3
"""
vadar_run.py — END-TO-END example of the VADAR-style generated-code pipeline over the
image-only SemBench queries, run on the FULL corpora and scored vs ground truth.

Covers: mmqa q2a, mmqa q7, ecomm q2 (id-set F1), ecomm q4 & q6 (ARI). Each query's
`extract()` composes the ImagePatch base API (CLIP / CV / OCR) — the Program-agent output.

Run:  <sembench python> src/semdb/vadar_run.py [q2a q7 q2 q4 q6]
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
MAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs/_materialized/ecomm_sf250")


def _distinct(path, col):
    return sorted({r[col] for r in csv.DictReader(open(path)) if r.get(col)})


def _n(s):
    return str(s).strip().lower()


def prf(pred, gold):
    tp = len(pred & gold); p = tp / len(pred) if pred else 0.0; r = tp / len(gold) if gold else 0.0
    return dict(pred=len(pred), gt=len(gold), tp=tp, P=round(p, 3), R=round(r, 3),
                F1=round(2 * p * r / (p + r), 3) if p + r else 0.0)


def ari(pred_pairs, gt_pairs):
    from sklearn.metrics import adjusted_rand_score
    gt = {i: c for i, c in gt_pairs}
    common = [i for i, _ in pred_pairs if i in gt]
    if not common:
        return {"n": 0, "ARI": 0.0}
    pd = {i: c for i, c in pred_pairs}
    return {"n": len(common), "ARI": round(adjusted_rand_score([gt[i] for i in common],
                                                               [pd[i] for i in common]), 3)}


# ---------------- extract() compositions (Program-agent output) ----------------

def extract_q2a(patch, tracks):
    # negative-option gate: include "not a racetrack logo" in the value space
    r = patch.classify(tracks + ["not a racetrack logo"], "the logo of {}")
    return {"racetrack": r if r != "not a racetrack logo" else "none"}


def extract_q7(patch, airlines):
    return {"airline": patch.best_ocr_match(airlines)}          # OCR self-gates non-logos


def extract_q2(patch):
    return {"product_type": patch.classify(["sports_shoes", "sandal", "boot",
                                            "other_footwear", "not_footwear"]),
            "colors": patch.dominant_colors(0.03)}


def extract_q4(patch):
    # center-crop past the white product background; take the top color in the DB space
    cols = patch.dominant_colors(0.08, center_frac=0.55)
    space = ["black", "blue", "red", "white", "orange", "green"]
    top = next((c for c in cols if c in space), cols[0] if cols else "none")
    return {"color": top}


def extract_q6(patch):
    return {"category": patch.classify(["Dress", "Bottomwear", "Socks", "Topwear", "Innerwear"],
                                       "a product photo of {} clothing")}


def run(which):
    ctx = {"encoder": semvision.get_encoder("openai/clip-vit-base-patch32"), "palette": None}
    P = lambda p: imagepatch.ImagePatch(p, ctx)
    imgs = list(csv.DictReader(open(f"{MM}/data/sf_200/images.csv")))
    ec_img = {r["id"]: r["filename"] for r in csv.DictReader(open(f"{MAT}/IMAGES.csv"))}
    styles = {r["id"]: r for r in csv.DictReader(open(f"{MAT}/styles.csv"))}

    if "q2a" in which:
        tracks = _distinct(f"{MM}/data/sf_200/ap_warrior.csv", "Track")
        ap = list(csv.DictReader(open(f"{MM}/data/sf_200/ap_warrior.csv")))
        gt = {(_n(i), _n(x)) for i, x in json.load(open(f"{MM}/raw_results/ground_truth/Q2a.json"))["ground_truth"]}
        pred = set()
        for row in imgs:
            rc = extract_q2a(P(row["image_filepath"]), tracks)["racetrack"]
            if rc == "none":
                continue
            for t in ap:
                if _n(t["Track"]) == _n(rc):
                    pred.add((_n(t["ID"]), _n(row["image_filename"])))
        print("mmqa q2a (classify+neg-gate):", prf(pred, gt))

    if "q7" in which:
        airlines = _distinct(f"{MM}/data/sf_200/tampa_international_airport.csv", "Airlines")
        gt = {(_n(a), _n(x)) for a, x in json.load(open(f"{MM}/raw_results/ground_truth/Q7.json"))["ground_truth"]}
        pred = set()
        for row in imgs:
            air = extract_q7(P(row["image_filepath"]), airlines)["airline"]
            if air != "none":
                pred.add((_n(air), _n(row["image_filename"])))
        print("mmqa q7  (ocr_match)        :", prf(pred, gt))

    if "q2" in which:
        gt = {_n(r["id"]) for r in csv.DictReader(open(f"{EC}/raw_results/ground_truth/Q2.csv"))}
        pred = set()
        for r in csv.DictReader(open(f"{MAT}/IMAGES.csv")):
            p = f"{EC}/data/sf_250/images/{r['filename']}"
            if not os.path.exists(p):
                continue
            rc = extract_q2(P(p))
            if rc["product_type"] == "sports_shoes" and {"yellow", "silver"}.issubset(set(rc["colors"])):
                pred.add(_n(r["id"]))
        print("ecomm q2 (classify+cv)      :", prf(pred, gt))

    if "q4" in which:
        gt = [(r["id"], r["category"]) for r in csv.DictReader(open(f"{EC}/raw_results/ground_truth/Q4.csv"))]
        space = {"Black", "Blue", "Red", "White", "Orange", "Green"}
        ids = [i for i, s in styles.items() if s.get("baseColour") in space]      # DB filter
        pred = []
        for i in ids:
            p = f"{EC}/data/sf_250/images/{ec_img.get(i,'')}"
            if os.path.exists(p):
                pred.append((i, extract_q4(P(p))["color"].capitalize()))
        print("ecomm q4 (cv primary-color) :", ari(pred, gt))

    if "q6" in which:
        gt = [(r["id"], r["category"]) for r in csv.DictReader(open(f"{EC}/raw_results/ground_truth/Q6.csv"))]
        bad = {"Saree", "Apparel Set", "Loungewear and Nightwear"}
        ids = [i for i, s in styles.items() if s.get("masterCategory") == "Apparel" and s.get("subCategory") not in bad]
        pred = []
        for i in ids:
            p = f"{EC}/data/sf_250/images/{ec_img.get(i,'')}"
            if os.path.exists(p):
                pred.append((i, extract_q6(P(p))["category"]))
        print("ecomm q6 (classify category):", ari(pred, gt))


if __name__ == "__main__":
    run(sys.argv[1:] or ["q2a", "q7", "q2", "q4", "q6"])
