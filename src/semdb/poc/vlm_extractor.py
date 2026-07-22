#!/usr/bin/env python3
"""
vlm_extractor.py — the REAL Extractor phase: run a small vision-language model
over each image once and emit the same schema mock_extractor.py produces:

    {"uri": ..., "logo_brand": ..., "conf": ..., "ocr_text": ...}

This is the drop-in replacement that proves the PoC's core hypothesis on real
pixels: *a 256M–2B VLM can turn an image into a reusable structured row*, after
which every logo query is plain relational code (see compiled_q7.py).

It is intentionally dependency-gated: if torch/transformers/Pillow are missing it
prints install guidance and exits 2, so `run_poc.sh` stays GPU-free by default.

Supported models (--model):
    HuggingFaceTB/SmolVLM-256M-Instruct   (tiny, CPU-friendly)
    Qwen/Qwen3-VL-2B-Instruct             (stronger, needs ~6GB)

Manifest CSV: columns `uri,path` where `path` is a local image file.

Usage:
    python3 vlm_extractor.py --manifest images_manifest.csv --out img_attrs.json \
            [--model HuggingFaceTB/SmolVLM-256M-Instruct]
"""

import argparse
import csv
import json
import re
import sys

# The single instruction that defines the schema slot the Designer chose.
# Constrained to strict JSON so the output is machine-parseable.
EXTRACT_PROMPT = (
    "You are a logo reader. Look at the image and identify the brand or company "
    "whose logo it shows. Respond with STRICT JSON only, no prose:\n"
    '{"logo_brand": "<brand name, or none if no logo>", '
    '"conf": <0.0-1.0 how sure you are>, '
    '"ocr_text": "<any text visible in the logo>"}'
)


def _require_deps():
    try:
        import torch  # noqa: F401
        from transformers import AutoProcessor  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        sys.stderr.write(
            "vlm_extractor needs torch + transformers + Pillow.\n"
            "  pip install 'torch' 'transformers>=4.49' 'pillow' 'accelerate'\n"
            f"(import error: {e})\n"
        )
        sys.exit(2)


def _load_vlm_class():
    """transformers renamed AutoModelForVision2Seq -> AutoModelForImageTextToText
    (>=4.45); the old name is removed in recent releases."""
    try:
        from transformers import AutoModelForImageTextToText as VLM
        return VLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as VLM  # older transformers
        return VLM


def parse_json_object(text):
    """Best-effort recovery of the first JSON object in the model output."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"logo_brand": "none", "conf": 0.0, "ocr_text": ""}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"logo_brand": "none", "conf": 0.0, "ocr_text": ""}
    return {
        "logo_brand": str(obj.get("logo_brand", "none")).strip() or "none",
        "conf": float(obj.get("conf", 0.0)),
        "ocr_text": str(obj.get("ocr_text", "")).strip(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with columns uri,path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()

    _require_deps()
    import torch
    from transformers import AutoProcessor
    from PIL import Image
    VLM = _load_vlm_class()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = VLM.from_pretrained(args.model, torch_dtype=dtype).to(device)

    rows = list(csv.DictReader(open(args.manifest)))
    attrs = []
    for r in rows:
        image = Image.open(r["path"]).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": EXTRACT_PROMPT},
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = processor.batch_decode(out, skip_special_tokens=True)[0]
        rec = parse_json_object(text.split("Assistant:")[-1])
        rec["uri"] = r["uri"]
        attrs.append(rec)
        print(f"[vlm] {r['uri']} -> {rec['logo_brand']} (conf {rec['conf']:.2f})")

    json.dump(attrs, open(args.out, "w"), indent=2)
    print(f"[vlm] wrote {len(attrs)} rows -> {args.out}")


if __name__ == "__main__":
    main()
