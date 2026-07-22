#!/usr/bin/env python3
"""
extract.py — SemDB Extractor: run a SMALL model once over a SemBench table and
materialize the schema the Schema Designer asked for.

One CLI for both modalities:

  text   : a small text LLM reads a text column (e.g. lizzy_caplan_text_data.text)
  image  : a small VLM reads each image (SmolVLM-256M / Qwen3-VL-2B)

Output contract (same for both, same as the PoC):
  [{"<id_col>": ..., <schema attribute columns>, "conf": <float>}, ...]

The attribute columns and the per-attribute instruction come from schema.json, so
this driver is query-agnostic — it serves every query over the corpus.

Dependency-gated: needs torch + transformers (+ Pillow for images). If missing it
prints install guidance and exits 2, so the rest of the repo stays runnable
without a GPU. HuggingFace access is required to fetch the model weights.

Examples
--------
Text (movie genres over lizzy_caplan_text_data.csv):
  python3 extract.py --schema runs/mmqa-q3a/schema.json \
      --table /localhome/hza214/SemBench/files/mmqa/data/sf_200/lizzy_caplan_text_data.csv \
      --modality text --id-col title --text-col text \
      --model Qwen/Qwen2.5-0.5B-Instruct --out movie_attrs.json

Image (airline logos over the images table + images/ dir):
  python3 extract.py --schema runs/mmqa-q7/schema.json \
      --table /localhome/hza214/SemBench/files/mmqa/data/sf_200/thalamusdb_images.csv \
      --modality image --id-col uri --image-col uri \
      --image-dir /localhome/hza214/SemBench/files/mmqa/data/sf_200/images \
      --model HuggingFaceTB/SmolVLM-256M-Instruct --out img_attrs.json
"""

import argparse
import csv
import json
import os
import re
import sys


# ---------------------------------------------------------------------------
# Prompt built from the schema so the driver is query-agnostic.
# ---------------------------------------------------------------------------

def build_prompt(schema):
    attrs = schema.get("attributes", [])
    fields, instrs = [], []
    for a in attrs:
        t = a.get("type", "string")
        fields.append(f'"{a["name"]}": <{t}>')
        instrs.append(f'- {a["name"]}: {a.get("extract_instruction", a.get("description",""))}'
                      + (" (use 'none'/[] if not present)" if a.get("allow_none") else ""))
    vocab = ""
    for a in attrs:
        if a.get("vocabulary"):
            vocab += f'\nAllowed values for {a["name"]}: {", ".join(a["vocabulary"])}.'
    schema_line = "{" + ", ".join(fields) + ', "conf": <0.0-1.0>}'
    return (
        "Extract the following fields. Respond with STRICT JSON only, no prose:\n"
        + schema_line + "\n" + "\n".join(instrs) + vocab
    )


def parse_json_object(text, schema):
    """Returns (record, ok). ok=False means no valid JSON was found in the model
    output — i.e. the 'none' is a PARSE FAILURE, not a genuine empty extraction."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    default = {a["name"]: ([] if "array" in a.get("type", "") or a.get("multi")
                           else "none") for a in schema.get("attributes", [])}
    default["conf"] = 0.0
    if not m:
        return dict(default), False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return dict(default), False
    out = {}
    for a in schema.get("attributes", []):
        out[a["name"]] = obj.get(a["name"], default[a["name"]])
    out["conf"] = float(obj.get("conf", 0.0))
    return out, True


# ---------------------------------------------------------------------------
# Model backends (loaded lazily so --help works without deps).
# ---------------------------------------------------------------------------

def load_text_model(model_id):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        _dep_exit("torch transformers", e)
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    return ("text", tok, mdl)


def _load_vlm_class():
    """transformers renamed AutoModelForVision2Seq -> AutoModelForImageTextToText
    (>=4.45); the old name is removed in recent releases. Try new, fall back to old."""
    try:
        from transformers import AutoModelForImageTextToText as VLM
        return VLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as VLM  # older transformers
        return VLM


def load_image_model(model_id):
    try:
        import torch  # noqa
        from transformers import AutoProcessor  # noqa
        from PIL import Image  # noqa
    except Exception as e:
        _dep_exit("torch transformers pillow accelerate", e)
    from transformers import AutoProcessor
    VLM = _load_vlm_class()
    proc = AutoProcessor.from_pretrained(model_id)
    mdl = VLM.from_pretrained(model_id, torch_dtype="auto")
    return ("image", proc, mdl)


def _dep_exit(pkgs, e):
    sys.stderr.write(
        f"extract.py needs: {pkgs}\n  pip install {pkgs}\n"
        "and network access to huggingface.co for the model weights.\n"
        f"(import error: {e})\n")
    sys.exit(2)


def gen_text(backend, prompt, max_new_tokens):
    import torch
    _, tok, mdl = backend
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def gen_image(backend, image_path, prompt, max_new_tokens):
    import torch
    from PIL import Image
    _, proc, mdl = backend
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    chat = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=chat, images=[image], return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.batch_decode(out, skip_special_tokens=True)[0].split("Assistant:")[-1]


def resolve_image_path(uri, image_dir):
    if image_dir:
        cand = os.path.join(image_dir, os.path.basename(uri))
        if os.path.exists(cand):
            return cand
    return uri  # assume the column already holds a local path


def main():
    ap = argparse.ArgumentParser(description="SemDB small-model Extractor")
    ap.add_argument("--schema", required=True, help="schema.json from the Schema Designer")
    ap.add_argument("--table", required=True, help="SemBench table CSV")
    ap.add_argument("--modality", choices=["text", "image"], required=True)
    ap.add_argument("--id-col", required=True, help="row key column (e.g. title / uri)")
    ap.add_argument("--text-col", help="text column to read (modality=text)")
    ap.add_argument("--image-col", help="column holding the image uri/path (modality=image)")
    ap.add_argument("--image-dir", help="directory of image files (modality=image)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="only process first N rows (0=all)")
    ap.add_argument("--debug", type=int, default=0,
                    help="print the raw model output for the first N rows (diagnose 'none')")
    args = ap.parse_args()

    schema = json.load(open(args.schema))
    prompt = build_prompt(schema)
    rows = list(csv.DictReader(open(args.table)))
    if args.limit:
        rows = rows[: args.limit]

    backend = (load_text_model(args.model) if args.modality == "text"
               else load_image_model(args.model))

    primary = schema.get("attributes", [{}])[0].get("name")
    theta = schema.get("residual", {}).get("theta", 0.5)
    attrs = []
    n_parse_fail = n_none = n_lowconf = 0
    for i, r in enumerate(rows):
        if args.modality == "text":
            col = args.text_col or "text"
            raw = gen_text(backend, prompt + "\n\nINPUT:\n" + r.get(col, ""), args.max_new_tokens)
        else:
            uri = r[args.image_col or args.id_col]
            path = resolve_image_path(uri, args.image_dir)
            if not os.path.exists(path):
                # image missing → not a model 'none'; record and warn loudly
                print(f"[extract] {i+1}/{len(rows)} MISSING IMAGE: {path}")
                rec = {a["name"]: ([] if "array" in a.get("type","") else "none")
                       for a in schema.get("attributes", [])}
                rec["conf"] = 0.0
                rec[args.id_col] = r[args.id_col]
                attrs.append(rec); n_none += 1
                continue
            raw = gen_image(backend, path, prompt, args.max_new_tokens)

        rec, ok = parse_json_object(raw, schema)
        rec[args.id_col] = r[args.id_col]
        if i < args.debug:
            print(f"[debug row {i}] raw output:\n{raw}\n[debug row {i}] parsed ok={ok} -> {rec}\n")

        val = rec.get(primary)
        if not ok:
            n_parse_fail += 1
        elif val in (None, "none", "", []):
            n_none += 1
        elif rec["conf"] < theta:
            n_lowconf += 1
        attrs.append(rec)
        print(f"[extract] {i+1}/{len(rows)} {str(r[args.id_col])[:40]!r} -> {rec.get(primary)}"
              + ("" if ok else "  [PARSE-FAIL]"))

    json.dump(attrs, open(args.out, "w"), indent=2)
    n_ok = len(attrs) - n_parse_fail - n_none
    print(f"\n[extract] wrote {len(attrs)} rows -> {args.out}")
    print(f"[extract]   extracted a value : {n_ok}")
    print(f"[extract]   genuine 'none'    : {n_none}   (no such attribute in the item)")
    print(f"[extract]   low confidence    : {n_lowconf}   (< theta {theta}; residual)")
    print(f"[extract]   JSON PARSE FAILS  : {n_parse_fail}   (model didn't emit valid JSON — see below)")
    if n_parse_fail > len(attrs) * 0.3:
        print("[extract] >30% parse failures: the model is too weak for strict JSON. "
              "Try --model Qwen/Qwen3-VL-2B-Instruct, raise --max-new-tokens, or rerun with --debug 3.")


if __name__ == "__main__":
    main()
