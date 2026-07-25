#!/usr/bin/env python3
"""
semcaption.py — OpImgCap: one VLM caption per corpus image, materialized as a column.

WHY A SEPARATE PASS. A caption is the cross-modal proxy that lets the CHEAP TEXT side
answer an image predicate (CADENZA's "Cross-Modal Proxying" template: Apply_t_img ⇝
Apply_t_txt ∘ Apply_c). It cannot live inside a VADAR-generated program — the
orchestrator's offline guard forbids network calls there — so it runs here, once per
CORPUS, and every query in the family reads the resulting column. Same amortization the
attribute table already gets: extract once, join many.

Output contract mirrors the attribute table: [{<id_col>: ..., "caption": str, "conf": float}].

  python3 semcaption.py <table.csv> <out.json> --image-col uri --id-col uri \
      --image-dir DIR --model Qwen/Qwen3-VL-2B-Instruct --endpoint http://localhost:8000/v1
"""

import argparse
import json
import os
import sys

import semextract


# One string attribute — `semextract.build_json_schema` turns this into the guided-JSON
# contract, so the VLM cannot answer with prose around the JSON.
CAPTION_SCHEMA = {
    "attributes": [{
        "name": "caption",
        "type": "string",
        "description": "a literal description of what is visible in the image",
        "extract_instruction": (
            "One or two sentences describing what is literally visible: the main "
            "subject, its type/category, colors, any legible text or logo, and the "
            "setting. Name things concretely. Do not speculate about what is not "
            "shown, and do not editorialize."
        ),
    }]
}

CAPTION_PROMPT = (
    "Describe this image for a search index. Respond with STRICT JSON only, no prose:\n"
    '{"caption": <string>, "conf": <0.0-1.0>}\n'
    "- caption: one or two sentences covering the main subject, its category, colors, "
    "any legible text or brand name, and the setting. Be concrete and literal; mention "
    "only what is visible.\n"
    "- conf: how confident you are that the caption is accurate."
)


class CaptionDriver(semextract.Default):
    """The Default driver with a caption-specific prompt (the schema is fixed, so the
    generic schema-rendered prompt would be needlessly vague)."""

    def build_prompt(self, schema, row, cols):
        return CAPTION_PROMPT


def run(table_path, out_path, *, id_col, image_col=None, image_dir=None, model,
        endpoint=None, api_key="EMPTY", concurrency=8, limit=0, max_new_tokens=160,
        timeout=120):
    """Caption the corpus. Returns semextract's meta dict."""
    driver = CaptionDriver(id_col=id_col, image_col=image_col or id_col, image_dir=image_dir)
    return semextract.run(driver, CAPTION_SCHEMA, table_path, out_path,
                          modality="image", model=model, endpoint=endpoint,
                          api_key=api_key, concurrency=concurrency, image_dir=image_dir,
                          limit=limit, max_new_tokens=max_new_tokens, timeout=timeout)


def load(path):
    """{id -> caption} from a captions table, for joining into a compiled query."""
    rows = json.load(open(path))
    if not rows:
        return {}
    id_col = next(k for k in rows[0] if k not in ("caption", "conf"))
    return {str(r[id_col]): r.get("caption", "") for r in rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description="OpImgCap — caption a corpus of images")
    ap.add_argument("table")
    ap.add_argument("out")
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--image-col")
    ap.add_argument("--image-dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    a = ap.parse_args(argv)
    run(a.table, a.out, id_col=a.id_col, image_col=a.image_col, image_dir=a.image_dir,
        model=a.model, endpoint=a.endpoint, api_key=a.api_key,
        concurrency=a.concurrency, limit=a.limit, max_new_tokens=a.max_new_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
