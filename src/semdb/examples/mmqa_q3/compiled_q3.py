#!/usr/bin/env python3
"""
compiled_q3.py — Code Generator output for the SemBench MMQA q3 genre family + q4.

The SemBench queries are semantic scans, one model call per (movie, query):

    q3a: SELECT title FROM movies t
         WHERE AI.IF(t.title || " is a comedy movie given description: " || t.text)
    q3b: ... "sci-fi" ...   q3c: horror   q3d: thriller ...   (7 genres)
    q4:  SELECT AI.GENERATE(..., output_schema=>"genres ARRAY<STRING>").genres
         FROM movies t  -- then UNNEST + GROUP BY genre

The Schema Designer decided all 8 queries factor through ONE slot — each movie's
`genres` set — and the Extractor (a small model) materialized it once into
movie_attrs.json. So every query below compiles to pure relational code over that
set, with ZERO further model calls:

    q3<genre>  ->  SELECT title WHERE '<genre>' IN genres        (set membership)
    q4         ->  UNNEST(genres) GROUP BY genre                 (inverted index)

This is the Python analog of GenDB's generated C++ query — the semantic operator
is gone; what remains is a scan over the pre-extracted schema.

Usage: python3 compiled_q3.py [movie_attrs.json]
"""

import json
import sys

# Controlled vocabulary (from schema.json). q3 asks one filter per genre.
GENRES = ["comedy", "sci-fi", "horror", "thriller", "drama", "action", "romance"]

# Compile-time normalization (the schema's synonym map) — free-form model wording
# is canonicalized to the vocabulary so filters are exact membership, not fuzzy.
SYNONYMS = {
    "science fiction": "sci-fi", "scifi": "sci-fi", "sf": "sci-fi",
    "romantic": "romance", "rom-com": "romance",
    "suspense": "thriller", "psychological thriller": "thriller",
    "biopic": "drama", "biographical": "drama", "biography": "drama",
}


def norm(g):
    g = g.strip().lower()
    return SYNONYMS.get(g, g)


def load_attrs(path):
    rows = json.load(open(path))
    for r in rows:
        r["_gset"] = {norm(g) for g in r.get("genres", [])}
    return rows


# --- compiled operators ------------------------------------------------------

def q3_filter(attrs, genre):
    """SELECT title WHERE '<genre>' IN genres — zero model calls."""
    g = norm(genre)
    return [r["title"] for r in attrs if g in r["_gset"]]


def q4_group_by_genre(attrs):
    """UNNEST(genres) GROUP BY genre — zero model calls."""
    out = {}
    for r in attrs:
        for g in sorted(r["_gset"]):
            out.setdefault(g, []).append(r["title"])
    return dict(sorted(out.items()))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "movie_attrs.json"
    attrs = load_attrs(path)

    print(f"# q3 family — one extraction over {len(attrs)} movies, reused by all queries\n")
    for genre in GENRES:
        titles = q3_filter(attrs, genre)
        print(f"q3[{genre}]  SELECT title WHERE '{genre}' IN genres")
        for t in titles:
            print(f"    {t}")
        print()

    print("q4  SELECT genre, STRING_AGG(title) GROUP BY genre")
    for genre, titles in q4_group_by_genre(attrs).items():
        print(f"    {genre:<9} -> {', '.join(titles)}")

    print(f"\n# model calls this run: 0 (all answers served from the extracted schema)")


if __name__ == "__main__":
    main()
