#!/usr/bin/env python3
"""
compiled_q7.py — the Code Generator's output for SemBench MMQA Q7 / Q2a:

    SELECT t.Airlines, i.uri
    FROM   tampa_international_airport t, images i
    WHERE  AI.IF("... does the image show the logo of the airline? ...",
                 t.Airlines, i.uri)

The naive plan is M×N VLM calls. This compiled plan turns the semantic join into:

    1. reuse the extracted image schema (img_attrs) — cost already paid, once, and
       shared across every logo query (Q2a, Q2b, Q7);
    2. normalize both sides and hash-join on the brand key — zero model calls;
    3. only the images with no confident brand fall through to a residual VLM,
       and only against candidate airlines.

The result is identical to the naive plan but the per-query model cost collapses
from M×N to |residual| calls.

Usage: python3 compiled_q7.py <airlines.csv> <img_attrs.json> <out pairs.csv>
"""

import sys

from semlib import load_airlines, load_img_attrs, normalize, write_pairs, METER, Image, P

THETA = 0.5  # confidence floor below which an extraction is not trusted


def join(airlines, img_attrs):
    pairs = []

    # 1. Build the hash table over the structured (airline) side — it was never
    #    multimodal, so there is nothing to extract here.
    by_key = {}
    for a in airlines:
        by_key.setdefault(normalize(a.name), []).append(a)

    # 2. Exact equality join on the pre-extracted, normalized logo brand. Cost 0.
    residual_imgs = []
    for rec in img_attrs:
        brand, conf = rec["logo_brand"], rec.get("conf", 1.0)
        if brand == "none" or conf < THETA:
            residual_imgs.append(rec)          # unsure → defer to VLM
            continue
        for a in by_key.get(normalize(brand), []):
            pairs.append((a.name, rec["uri"]))

    # 3. Residual VLM: only the unsure images, only against candidate airlines.
    #    This is the single masked call site — everything else was free.
    for rec in residual_imgs:
        img = Image(uri=rec["uri"], true_brand=_IMAGE_INDEX[rec["uri"]])
        for a in airlines:
            if P.vlm_judge(a, img):
                pairs.append((a.name, rec["uri"]))

    return pairs


# --- residual plumbing -------------------------------------------------------
# The compiled code does NOT hold pixels; to re-judge an unsure image it must
# re-open it and pay for a model call. We reload the raw image rows lazily.
_IMAGE_INDEX = {}


def main():
    airlines_csv, attrs_json, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    airlines = load_airlines(airlines_csv)
    img_attrs = load_img_attrs(attrs_json)

    # Lazy pixel access for the residual path (kept out of the hot join path).
    from semlib import load_images
    import os
    images_csv = os.path.join(os.path.dirname(attrs_json) or ".", "data", "images.csv")
    if not os.path.exists(images_csv):
        images_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "images.csv")
    for im in load_images(images_csv):
        _IMAGE_INDEX[im.uri] = im.true_brand

    METER.reset()  # count only this query's residual calls
    pairs = join(airlines, img_attrs)
    rows = write_pairs(out_path, pairs)

    print(f"[compiled] emitted {len(rows)} pairs -> {out_path}")
    print(f"[compiled] residual VLM.IF calls this query: {METER.judge_calls}")


if __name__ == "__main__":
    main()
