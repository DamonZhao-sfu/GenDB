#!/usr/bin/env python3
"""
baseline_q7.py — the naive semantic execution of Q7 / Q2a: one VLM.IF call for
every (airline, image) pair. This is the M×N plan the compiler replaces, and it
also serves as the correctness oracle the compiled plan must match exactly.

Usage: python3 baseline_q7.py <airlines.csv> <images.csv> <out pairs.csv>
"""

import sys

from semlib import load_airlines, load_images, write_pairs, METER, P


def main():
    airlines_csv, images_csv, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    airlines = load_airlines(airlines_csv)
    images = load_images(images_csv)

    METER.reset()
    pairs = []
    for a in airlines:
        for im in images:
            if P.vlm_judge(a, im):     # M×N model calls
                pairs.append((a.name, im.uri))

    rows = write_pairs(out_path, pairs)
    print(f"[baseline] emitted {len(rows)} pairs -> {out_path}")
    print(f"[baseline] VLM.IF calls: {METER.judge_calls} "
          f"({len(airlines)} airlines x {len(images)} images)")


if __name__ == "__main__":
    main()
