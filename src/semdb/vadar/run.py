#!/usr/bin/env python3
"""
vadar/run.py — end-to-end runner for the strict VADAR pipeline on SemBench image queries.

For a query: SignatureAgent -> APIAgent -> ProgramAgent synthesize helpers + `extract(image)`
over the predefined API; the engine assembles + execs that generated code over the corpus,
then the relational compiled step + scoring.

Modes:
  --llm      : synthesize the program via the code-LLM (needs VADAR_BASE_URL/VADAR_MODEL).
  (default)  : use the REFERENCE program (validated composition) to exercise the engine
               deterministically without an LLM.

Usage:
  <sembench python> -m vadar.run q2a q7 q2   [--llm]
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEMDB = os.path.dirname(HERE)                        # src/semdb (has imagepatch, semvision, vadar/)
sys.path.insert(0, SEMDB)

import imagepatch    # noqa: E402
import semvision     # noqa: E402
from vadar import predefined  # noqa: E402

MM = "/localhome/hza214/SemBench/files/mmqa"
EC = "/localhome/hza214/SemBench/files/ecomm"
MAT = os.path.join(SEMDB, "runs/_materialized/ecomm_sf250")


def _distinct(path, col):
    return sorted({r[col] for r in csv.DictReader(open(path)) if r.get(col)})


def _n(s):
    return str(s).strip().lower()


def prf(pred, gold):
    tp = len(pred & gold); p = tp / len(pred) if pred else 0.0; r = tp / len(gold) if gold else 0.0
    return dict(pred=len(pred), gt=len(gold), tp=tp, P=round(p, 3), R=round(r, 3),
                F1=round(2 * p * r / (p + r), 3) if p + r else 0.0)


# --- value spaces + reference programs (what the Program agent should synthesize) ---
def specs():
    tracks = _distinct(f"{MM}/data/sf_200/ap_warrior.csv", "Track")
    airlines = _distinct(f"{MM}/data/sf_200/tampa_international_airport.csv", "Airlines")
    return {
        "q2a": dict(
            value_spaces={"TRACKS": tracks},
            question="Field `racetrack` = which racetrack's logo the image shows, from TRACKS.",
            reference='def extract(image):\n    return {"racetrack": classify(image, TRACKS, "the logo of {}")}'),
        "q7": dict(
            value_spaces={"AIRLINES": airlines},
            question="Field `airline` = which airline's logo the image shows, from AIRLINES.",
            reference='def extract(image):\n    return {"airline": best_ocr_match(image, AIRLINES)}'),
        "q2": dict(
            value_spaces={"SHOES": ["sports_shoes", "sandal", "boot", "other_footwear", "not_footwear"]},
            question="Fields: `product_type` from SHOES; `colors` = colors present.",
            reference=('def extract(image):\n'
                       '    return {"product_type": classify(image, SHOES),\n'
                       '            "colors": dominant_colors(image, 0.03)}')),
    }


def _build_extract(program_src, value_spaces, helpers_src=""):
    """Assemble + exec the generated module; return the `extract` function."""
    ns = {name: getattr(predefined, name) for name in
          ("classify", "best_ocr_match", "dominant_colors", "verify_property", "score", "read_text", "detect")}
    ns.update(value_spaces)
    exec(compile(helpers_src + "\n\n" + program_src, "<generated>", "exec"), ns)
    return ns["extract"]


def _score(qid, extract, ctx):
    P = lambda p: imagepatch.ImagePatch(p, ctx)
    if qid in ("q2a", "q7"):
        imgs = list(csv.DictReader(open(f"{MM}/data/sf_200/images.csv")))
        if qid == "q2a":
            ap = list(csv.DictReader(open(f"{MM}/data/sf_200/ap_warrior.csv")))
            gt = {(_n(i), _n(x)) for i, x in json.load(open(f"{MM}/raw_results/ground_truth/Q2a.json"))["ground_truth"]}
            pred = set()
            for row in imgs:
                rc = extract(P(row["image_filepath"])).get("racetrack", "none")
                for t in ap:
                    if _n(t["Track"]) == _n(rc):
                        pred.add((_n(t["ID"]), _n(row["image_filename"])))
            return prf(pred, gt)
        gt = {(_n(a), _n(x)) for a, x in json.load(open(f"{MM}/raw_results/ground_truth/Q7.json"))["ground_truth"]}
        pred = {(_n(extract(P(r["image_filepath"]))["airline"]), _n(r["image_filename"]))
                for r in imgs if extract(P(r["image_filepath"]))["airline"] != "none"}
        return prf(pred, gt)
    # ecomm q2
    gt = {_n(r["id"]) for r in csv.DictReader(open(f"{EC}/raw_results/ground_truth/Q2.csv"))}
    pred = set()
    for r in csv.DictReader(open(f"{MAT}/IMAGES.csv")):
        p = f"{EC}/data/sf_250/images/{r['filename']}"
        if not os.path.exists(p):
            continue
        rec = extract(P(p))
        if rec.get("product_type") == "sports_shoes" and {"yellow", "silver"}.issubset(set(rec.get("colors", []))):
            pred.add(_n(r["id"]))
    return prf(pred, gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="*", default=["q2a", "q7", "q2"])
    ap.add_argument("--llm", action="store_true", help="synthesize via the code-LLM (VADAR_BASE_URL/VADAR_MODEL)")
    args = ap.parse_args()
    ctx = {"encoder": semvision.get_encoder("openai/clip-vit-base-patch32"), "palette": None}
    S = specs()
    gen = None
    if args.llm:
        from vadar.generator import Generator
        from vadar import agents
        gen = Generator()
    for qid in (args.queries or ["q2a", "q7", "q2"]):
        spec = S[qid]
        helpers_src = ""
        if args.llm:
            vs_doc = "\n".join(f"{k} = {v}" for k, v in spec["value_spaces"].items())
            helpers_src, program_src, _ = agents.synthesize(gen, [spec["question"]], vs_doc)
            print(f"[{qid}] synthesized program:\n{program_src}\n")
        else:
            program_src = spec["reference"]
        extract = _build_extract(program_src, spec["value_spaces"], helpers_src)
        print(f"{qid}: {_score(qid, extract, ctx)}")


if __name__ == "__main__":
    main()
