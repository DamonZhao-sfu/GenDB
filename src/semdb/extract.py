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
import ast
import os
import re
import sys
import time


# ---------------------------------------------------------------------------
# Prompt built from the schema so the driver is query-agnostic.
# ---------------------------------------------------------------------------

def entity_attr(schema):
    """The single attribute a weak model should extract in --prompt-style simple:
    the join key (brand/entity/name), not the boolean/type/alias/conf helpers."""
    attrs = schema.get("attributes", [])
    for a in attrs:
        n = a["name"].lower()
        if re.search(r"entity|brand|name", n) and not re.search(r"type|alias|conf", n):
            return a["name"]
    return attrs[0]["name"] if attrs else "value"


def build_prompt(schema, style="json"):
    attrs = schema.get("attributes", [])
    if style == "simple":
        ent = entity_attr(schema)
        a = next((x for x in attrs if x["name"] == ent), {})
        instr = a.get("extract_instruction", a.get("description", f"the {ent}"))
        return ("Look at the image. " + instr +
                "\nAnswer with ONLY the name (at most a few words). "
                "If there is no logo, answer exactly: none. Do not describe the image.")
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


def parse_simple(text, schema):
    """Salvage the entity name from a weak model's prose/dict output.
    Handles: "The logo is 'edelweiss'." / bare name / "none" / rambling → none."""
    ent = entity_attr(schema)
    t = (text or "").strip()
    # Prefer a quoted phrase, but skip schema keys / literals if the model emitted a dict.
    skip = {a["name"].lower() for a in schema.get("attributes", [])} | {"true", "false", "none"}
    quotes = [q.strip() for q in re.findall(r"['\"]([^'\"]{1,60})['\"]", t)
              if q.strip().lower() not in skip]
    val = quotes[0] if quotes else (t.splitlines()[0].strip() if t else "")
    val = re.sub(r"^(the logo is( a| an)?|this is( a| an)?|it is( a| an)?|logo:|answer:)\s*",
                 "", val, flags=re.I).strip().strip(".").strip()
    if not val or re.match(r"^(none|no logo|no|n/?a|unknown|not a logo)$", val, re.I) \
            or len(val.split()) > 8:                         # long prose ⇒ no clean logo
        val = "none"
    rec = {a["name"]: ([] if "array" in a.get("type", "") else "none")
           for a in schema.get("attributes", [])}
    rec[ent] = val
    rec["conf"] = 0.0 if val == "none" else 0.7
    return rec, True   # simple mode always "parses"; genuine none is signalled by the value


def _find_json(text):
    """Return the LAST parseable JSON object in the text, or None.

    Models often echo the prompt (which contains the schema TEMPLATE, e.g.
    {"is_logo": <boolean>, ...}) before the real answer. A greedy {.*} match would
    fuse the two. We scan for balanced {...} objects and prefer the last one that
    actually parses — the template has placeholders like <boolean> and won't."""
    blobs, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                blobs.append(text[start:i + 1])
                start = None
    for blob in reversed(blobs):
        for candidate in (blob, _repair_json(blob)):
            try:
                o = json.loads(candidate)
                if isinstance(o, dict):
                    return o
            except json.JSONDecodeError:
                pass
        try:
            o = ast.literal_eval(blob)          # tolerate single-quoted dicts
            if isinstance(o, dict):
                return o
        except (ValueError, SyntaxError):
            pass
    return None


def _repair_json(blob):
    """Best-effort repair of the malformations small models emit and that json
    rejects: trailing commas, `0.`/`.5` numbers, and Python True/False/None."""
    b = re.sub(r",(\s*[}\]])", r"\1", blob)             # trailing comma before } or ]
    b = re.sub(r"(?<=[:\s\[,])\.(\d)", r"0.\1", b)      # .5 -> 0.5
    b = re.sub(r"(\d)\.(?=\s*[,}\]])", r"\1.0", b)      # 0.  -> 0.0
    b = re.sub(r"\bTrue\b", "true", b)
    b = re.sub(r"\bFalse\b", "false", b)
    b = re.sub(r"\bNone\b", "null", b)
    return b


def _norm_key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def parse_json_object(text, schema):
    """Returns (record, ok). ok=False means no valid JSON was found in the model
    output — i.e. the 'none' is a PARSE FAILURE, not a genuine empty extraction."""
    default = {a["name"]: ([] if "array" in a.get("type", "") or a.get("multi")
                           else "none") for a in schema.get("attributes", [])}
    default["conf"] = 0.0
    obj = _find_json(text)
    if not isinstance(obj, dict):
        return dict(default), False
    # Match keys tolerantly: strip case/spaces/underscores so "is_logo " or
    # "logoentitytype" still map to is_logo / logo_entity_type.
    norm = {_norm_key(k): v for k, v in obj.items()}
    out = {}
    for a in schema.get("attributes", []):
        v = obj.get(a["name"], norm.get(_norm_key(a["name"]), default[a["name"]]))
        out[a["name"]] = "none" if v is None else v   # JSON null → 'none' sentinel
    out["conf"] = float(obj.get("conf") or norm.get("conf") or norm.get("confidence")
                        or obj.get("logo_conf") or 0.0)
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


# Anti-degeneration decoding — stops the "santa anita park" ×50 repetition
# collapse that small models fall into with plain greedy decoding.
_GEN = dict(do_sample=False, repetition_penalty=1.3, no_repeat_ngram_size=3)


def gen_text(backend, prompt, max_new_tokens):
    import torch
    _, tok, mdl = backend
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_new_tokens, **_GEN)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def gen_image(backend, image_path, prompt, max_new_tokens):
    import torch
    from PIL import Image
    _, proc, mdl = backend
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    chat = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=chat, images=[image], return_tensors="pt").to(mdl.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_new_tokens, **_GEN)
    # Slice off the echoed prompt — decode only the newly generated tokens.
    return proc.batch_decode(out[:, input_len:], skip_special_tokens=True)[0]


def resolve_image_path(uri, image_dir):
    if image_dir:
        cand = os.path.join(image_dir, os.path.basename(uri))
        if os.path.exists(cand):
            return cand
    return uri  # assume the column already holds a local path


# ---------------------------------------------------------------------------
# vLLM / OpenAI-compatible endpoint backend, with GUIDED JSON decoding.
#
# This is the recommended production path: the server constrains generation to a
# JSON Schema, so the output is ALWAYS schema-valid (no parse failures at all),
# and requests are batched/fast. Serve the small VLM with, e.g.:
#   vllm serve Qwen/Qwen3-VL-2B-Instruct --port 8000
# then pass  --endpoint http://localhost:8000/v1  to this script.
# ---------------------------------------------------------------------------

def build_json_schema(schema):
    """Turn schema.json attributes into a JSON Schema for guided decoding."""
    props, required = {}, []
    for a in schema.get("attributes", []):
        t = a.get("type", "string").lower()
        if a.get("vocabulary"):
            js = {"type": "string", "enum": list(a["vocabulary"])}
        elif "bool" in t:
            js = {"type": "boolean"}
        elif any(k in t for k in ("float", "double", "number", "real", "int")):
            js = {"type": "number"}
        elif "array" in t or "list" in t:
            js = {"type": "array", "items": {"type": "string"}}
        else:
            js = {"type": "string"}
        if a.get("allow_none") and js.get("type") in ("string", "number", "boolean"):
            js = {"type": [js["type"], "null"], **({"enum": js["enum"]} if "enum" in js else {})}
        props[a["name"]] = js
        required.append(a["name"])
    props["conf"] = {"type": "number"}
    required.append("conf")
    return {"type": "object", "properties": props, "required": required,
            "additionalProperties": False}


def _data_url(path):
    import base64
    ext = (os.path.splitext(path)[1].lstrip(".") or "png").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/{mime};base64,{b64}"


def gen_endpoint(args, json_schema, prompt, modality, image_path=None, text=None):
    """POST one chat completion to an OpenAI-compatible server with guided_json.
    Returns the assistant message content (guaranteed schema-valid JSON)."""
    import urllib.request
    content = [{"type": "text", "text": prompt}]
    if modality == "image":
        content.append({"type": "image_url", "image_url": {"url": _data_url(image_path)}})
    else:
        content[0]["text"] = prompt + "\n\nINPUT:\n" + (text or "")
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_new_tokens,
        "temperature": 0,
        # vLLM guided decoding — also mirror as response_format for other servers.
        "guided_json": json_schema,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "extract", "schema": json_schema}},
    }
    req = urllib.request.Request(
        args.endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {args.api_key}"},
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        out = json.loads(resp.read())
    return out["choices"][0]["message"]["content"]


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
    ap.add_argument("--prompt-style", choices=["json", "simple"], default="json",
                    help="'json' = strict multi-field JSON (capable models, e.g. Qwen3-VL-2B); "
                         "'simple' = ask only for the entity name + lenient parse (small models "
                         "like SmolVLM-256M that ramble/emit prose)")
    ap.add_argument("--endpoint",
                    help="OpenAI-compatible base URL (e.g. http://localhost:8000/v1). When set, "
                         "calls a vLLM server with GUIDED JSON decoding instead of loading a "
                         "local model — output is always schema-valid, no parse failures.")
    ap.add_argument("--api-key", default="EMPTY", help="bearer token for --endpoint (vLLM: EMPTY)")
    ap.add_argument("--timeout", type=int, default=120, help="--endpoint request timeout (s)")
    args = ap.parse_args()

    schema = json.load(open(args.schema))
    use_endpoint = bool(args.endpoint)
    # Guided decoding enforces the schema, so always use the json prompt + parser there.
    style = "json" if use_endpoint else args.prompt_style
    prompt = build_prompt(schema, style)
    json_schema = build_json_schema(schema) if use_endpoint else None
    parse = (lambda raw: parse_simple(raw, schema)) if style == "simple" \
        else (lambda raw: parse_json_object(raw, schema))
    t_start = time.time()
    rows = list(csv.DictReader(open(args.table)))
    if args.limit:
        rows = rows[: args.limit]

    backend = None
    if not use_endpoint:
        backend = (load_text_model(args.model) if args.modality == "text"
                   else load_image_model(args.model))
    else:
        print(f"[extract] using guided-JSON endpoint {args.endpoint} (model {args.model})")

    # In simple mode the reporting/join key is the entity attribute, not attr[0].
    primary = entity_attr(schema) if style == "simple" \
        else schema.get("attributes", [{}])[0].get("name")
    theta = schema.get("residual", {}).get("theta", 0.5)
    attrs = []
    n_parse_fail = n_none = n_lowconf = 0
    for i, r in enumerate(rows):
        if args.modality == "text":
            col = args.text_col or "text"
            if use_endpoint:
                raw = gen_endpoint(args, json_schema, prompt, "text", text=r.get(col, ""))
            else:
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
            if use_endpoint:
                raw = gen_endpoint(args, json_schema, prompt, "image", image_path=path)
            else:
                raw = gen_image(backend, path, prompt, args.max_new_tokens)

        rec, ok = parse(raw)
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

    elapsed = time.time() - t_start
    json.dump(attrs, open(args.out, "w"), indent=2)
    n_ok = len(attrs) - n_parse_fail - n_none
    # Sidecar consumed by orchestrator telemetry (code-execution time + counts).
    meta = {
        "model": args.model, "endpoint": args.endpoint, "modality": args.modality,
        "rows": len(attrs), "extracted": n_ok, "none": n_none,
        "low_conf": n_lowconf, "parse_fail": n_parse_fail,
        "elapsed_sec": round(elapsed, 2),
        "sec_per_row": round(elapsed / max(1, len(attrs)), 3),
    }
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=2)
    print(f"\n[extract] wrote {len(attrs)} rows -> {args.out}  ({elapsed:.1f}s, {meta['sec_per_row']}s/row)")
    print(f"[extract]   extracted a value : {n_ok}")
    print(f"[extract]   genuine 'none'    : {n_none}   (no such attribute in the item)")
    print(f"[extract]   low confidence    : {n_lowconf}   (< theta {theta}; residual)")
    print(f"[extract]   JSON PARSE FAILS  : {n_parse_fail}   (model didn't emit valid JSON — see below)")
    print(f"[extract]   meta -> {args.out}.meta.json")
    if n_parse_fail > len(attrs) * 0.3:
        print("[extract] >30% parse failures: the model is too weak for strict JSON. "
              "Try --model Qwen/Qwen3-VL-2B-Instruct, raise --max-new-tokens, or rerun with --debug 3.")


if __name__ == "__main__":
    main()
