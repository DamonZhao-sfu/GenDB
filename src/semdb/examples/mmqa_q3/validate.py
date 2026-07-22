#!/usr/bin/env python3
"""
validate.py — is the small model's reusable schema good enough to answer the SQL?

Oracle: the naive plan runs AI.IF per (movie, genre) with a perfect judge, i.e. it
returns exactly the hand-labeled genres in data/genres_key.csv. We compare the
compiled answers (which read the SMALL MODEL's extracted genres in
movie_attrs.json) against that oracle, per genre query, and report precision /
recall / F1 — plus the model-call reduction.

This is the crux the user asked to verify: can a small model turn each item into a
structured row that downstream relational code reuses correctly?

Usage: python3 validate.py [movie_attrs.json] [data/genres_key.csv]
"""

import csv
import sys

from compiled_q3 import GENRES, norm, q3_filter, load_attrs


def load_key(path):
    key = {}
    for r in csv.DictReader(open(path)):
        key[r["title"]] = {norm(g) for g in r["genres"].split("|") if g.strip()}
    return key


def main():
    attrs_path = sys.argv[1] if len(sys.argv) > 1 else "movie_attrs.json"
    key_path = sys.argv[2] if len(sys.argv) > 2 else "data/genres_key.csv"

    attrs = load_attrs(attrs_path)
    key = load_key(key_path)
    all_titles = [r["title"] for r in attrs]

    print(f"{'genre':<10} {'TP':>3} {'FP':>3} {'FN':>3} {'prec':>6} {'recall':>7} {'F1':>6}   disagreements")
    print("-" * 78)
    TP = FP = FN = 0
    for genre in GENRES:
        pred = set(q3_filter(attrs, genre))                       # small-model schema
        gold = {t for t in all_titles if norm(genre) in key[t]}   # oracle (hand labels)
        tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        diffs = []
        for t in sorted(pred - gold):
            diffs.append(f"+{t}")
        for t in sorted(gold - pred):
            diffs.append(f"-{t}")
        print(f"{genre:<10} {tp:>3} {fp:>3} {fn:>3} {p:>6.2f} {r:>7.2f} {f1:>6.2f}   {', '.join(diffs)}")

    P = TP / (TP + FP) if TP + FP else 1.0
    R = TP / (TP + FN) if TP + FN else 1.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    print("-" * 78)
    print(f"{'MICRO':<10} {TP:>3} {FP:>3} {FN:>3} {P:>6.2f} {R:>7.2f} {F1:>6.2f}")

    n, q = len(attrs), len(GENRES)
    print()
    print("=== model-call reduction ===")
    print(f"  naive (AI.IF per movie per query) : {n} x {q} = {n * q} model calls")
    print(f"  compiled (extract once, reuse)    : {n} extractions + 0 per query")
    print(f"  reduction                         : {n * q} -> {n}")
    print(f"\n  (+ q4 aggregation is free: it reads the same {n} extracted rows)")


if __name__ == "__main__":
    main()
