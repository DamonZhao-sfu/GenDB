"""
semlib.py — shared primitives for the SemDB compilation PoC.

This module is the "runtime" that the compiled program links against. It provides:

  1. CSV loaders for the two SemBench-style tables (airlines, images).
  2. A brand-name normalizer (the artifact the Schema Designer emits at compile time).
  3. `P.vlm_judge(...)` — a *simulated* vision-language model call. In the real
     system this is one call to Qwen3-VL-2B / SmolVLM. Here it is deterministic so
     the PoC is reproducible and needs no GPU.

Simulation contract
-------------------
Each image row carries a hidden `true_brand` column. This column stands in for
"what a perfect model would read off the pixels". It is the ONLY channel through
which any model (the extractor or the residual judge) is allowed to see the image.
Compiled query code must never read `true_brand` directly — it may only see the
extracted schema (img_attrs) and may only re-touch a pixel by paying for a
`P.vlm_judge` call. A global counter records every simulated model call so the PoC
can prove the M×N → N cost collapse.
"""

import csv
import json
import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Cost accounting — every simulated model touch is counted here.
# ---------------------------------------------------------------------------

class ModelMeter:
    def __init__(self):
        self.extractions = 0      # one per image, paid once, shared by all queries
        self.judge_calls = 0      # residual per-pair VLM.IF calls

    def reset(self):
        self.extractions = 0
        self.judge_calls = 0

    def as_dict(self):
        return {"extractions": self.extractions, "judge_calls": self.judge_calls}


METER = ModelMeter()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class Airline:
    name: str
    destinations: str


@dataclass
class Image:
    uri: str
    true_brand: str  # hidden pixels — only reachable via a model call (see below)


def load_airlines(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(Airline(name=r["Airlines"].strip(),
                                 destinations=r.get("Destinations", "").strip()))
    return rows


def load_images(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(Image(uri=r["uri"].strip(), true_brand=r["true_brand"].strip()))
    return rows


# ---------------------------------------------------------------------------
# Normalization — emitted by the Schema Designer at COMPILE time, runs once.
#
# Airlines have aliases, abbreviations, and marketing suffixes. A hash join only
# works if both sides collapse to the same key. This map + suffix stripping is
# exactly the "hidden difficulty" the Q2a write-up calls out.
# ---------------------------------------------------------------------------

# Designer-generated synonym map: alias -> canonical token.
SYNONYMS = {
    "swa": "southwest",
    "southwest airlines": "southwest",
    "delta air lines": "delta",
    "spirit airlines": "spirit",
    "silver airways": "silver",
    "british airways": "british",
}

_SUFFIXES = ["air lines", "airlines", "airways", "airline", "air"]


def normalize(name):
    """Collapse an airline / logo brand string to a canonical join key."""
    if name is None:
        return ""
    s = re.sub(r"[^a-z0-9 ]", " ", name.lower()).strip()
    s = re.sub(r"\s+", " ", s)
    if s in SYNONYMS:
        return SYNONYMS[s]
    for suf in _SUFFIXES:
        if s.endswith(" " + suf):
            s = s[: -(len(suf) + 1)].strip()
            break
    return SYNONYMS.get(s, s)


# ---------------------------------------------------------------------------
# P — the "physical operator" library exposed to compiled code.
#
# P.vlm_judge is the expensive primitive. It answers the ORIGINAL semantic
# predicate ("does this image show the logo of {airline}?") by consulting the
# hidden pixels. Both the naive baseline and the compiled residual path go
# through here, so they share one oracle and must agree.
# ---------------------------------------------------------------------------

class P:
    @staticmethod
    def vlm_judge(airline: Airline, image: Image) -> bool:
        """Simulated VLM.IF — one model call, counted."""
        METER.judge_calls += 1
        return normalize(image.true_brand) == normalize(airline.name)


# ---------------------------------------------------------------------------
# Schema I/O
# ---------------------------------------------------------------------------

def load_img_attrs(path):
    with open(path) as f:
        return json.load(f)


def write_pairs(path, pairs):
    """pairs: list of (airline_name, uri). Written as sorted CSV for stable diffs."""
    rows = sorted(set(pairs))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Airlines", "uri"])
        for a, u in rows:
            w.writerow([a, u])
    return rows
