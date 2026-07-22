#!/usr/bin/env python3
"""
mock_extractor.py — deterministic stand-in for a small VLM (Qwen3-VL-2B /
SmolVLM-256M) running the Extractor phase.

It walks every image ONCE and emits the schema the Schema Designer asked for:

    {"uri": ..., "logo_brand": ..., "conf": ..., "ocr_text": ...}

To make the PoC realistic it models two failure modes a 256M–2B model actually
exhibits, without needing a GPU:

  * abbreviation — it reads "Delta" off a "Delta Air Lines" logo, so the join key
    only lines up after normalization (the compile-time synonym map).
  * low-confidence miss — for faint/rotated logos it returns logo_brand="none"
    with low conf, which is exactly what should fall through to the residual VLM.

Swap this file for vlm_extractor.py to run the identical contract on real pixels.

Usage: python3 mock_extractor.py <images.csv> <out img_attrs.json>
"""

import json
import sys

from semlib import load_images, METER

# Images a small model struggles to read (faint / low-res / rotated logo).
# Documented explicitly so the PoC's residual path is intentional, not accidental.
HARD_IMAGES = {"img/008.jpg"}

CONF_CONFIDENT = 0.92
CONF_NO_LOGO = 0.97
CONF_UNSURE = 0.30


def small_vlm_read(image):
    """Simulate one forward pass of a small VLM over an image → (brand, conf)."""
    METER.extractions += 1
    if image.true_brand.lower() == "none":
        return "none", CONF_NO_LOGO
    if image.uri in HARD_IMAGES:
        return "none", CONF_UNSURE  # couldn't resolve the brand from pixels
    # A small model tends to read only the salient word of a multi-word logo.
    brand = image.true_brand.split()[0] if " " in image.true_brand else image.true_brand
    return brand, CONF_CONFIDENT


def main():
    images_csv, out_path = sys.argv[1], sys.argv[2]
    images = load_images(images_csv)
    METER.reset()

    attrs = []
    for img in images:
        brand, conf = small_vlm_read(img)
        attrs.append({
            "uri": img.uri,
            "logo_brand": brand,
            "conf": round(conf, 2),
            "ocr_text": "" if brand == "none" else brand,
        })

    with open(out_path, "w") as f:
        json.dump(attrs, f, indent=2)

    print(f"[extractor] extracted {METER.extractions} images -> {out_path}")
    n_none = sum(1 for a in attrs if a["logo_brand"] == "none")
    print(f"[extractor] {n_none} image(s) returned no confident brand "
          f"(will fall through to residual VLM)")


if __name__ == "__main__":
    main()
