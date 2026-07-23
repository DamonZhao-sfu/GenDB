#!/usr/bin/env python3
"""
extract.py — thin CLI over the shared `semextract` engine, using the generic
`Default` driver. Kept for backward-compat and manual runs.

Per-corpus extraction is now driven by agent-GENERATED drivers (extract_<corpus>.py)
that call `semextract.run(<their Driver>, ...)` — this script is the query-agnostic
fallback that reproduces the previous behavior.

Examples
--------
Text (movie genres over lizzy_caplan_text_data.csv):
  python3 extract.py --schema runs/mmqa-q3a/schema.json \
      --table /.../sf_200/lizzy_caplan_text_data.csv \
      --modality text --id-col title --text-col text \
      --model Qwen/Qwen2.5-0.5B-Instruct --out movie_attrs.json

Image over a vLLM endpoint (guided JSON, concurrent):
  python3 extract.py --schema runs/mmqa-q7/schema.json \
      --table /.../sf_200/images.csv \
      --modality image --id-col row_id --image-col image_filename \
      --image-dir /.../sf_200/images \
      --model HuggingFaceTB/SmolVLM-256M-Instruct --out img_attrs.json \
      --endpoint http://localhost:8000/v1 --concurrency 16
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import semextract  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="SemDB Extractor (thin CLI over semextract.Default)")
    ap.add_argument("--schema", required=True, help="schema.json from the Schema Designer")
    ap.add_argument("--table", required=True, help="SemBench table CSV")
    ap.add_argument("--modality", choices=["text", "image"], required=True)
    ap.add_argument("--id-col", required=True, help="row key column (e.g. title / uri / row_id)")
    ap.add_argument("--text-col", help="text column to read (modality=text)")
    ap.add_argument("--image-col", help="column holding the image uri/path (modality=image)")
    ap.add_argument("--image-dir", help="directory of image files (modality=image)")
    ap.add_argument("--context-cols", help="comma-separated structured columns (unused by Default)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="only process first N rows (0=all)")
    ap.add_argument("--debug", type=int, default=0, help="print raw model output for the first N rows")
    ap.add_argument("--prompt-style", choices=["json", "simple"], default="json",
                    help="'json' = strict multi-field JSON; 'simple' = entity-only lenient parse")
    ap.add_argument("--endpoint", help="OpenAI-compatible base URL (e.g. http://localhost:8000/v1); "
                                       "enables guided-JSON decoding + concurrency")
    ap.add_argument("--api-key", default="EMPTY", help="bearer token for --endpoint (vLLM: EMPTY)")
    ap.add_argument("--timeout", type=int, default=120, help="--endpoint request timeout (s)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="in-flight --endpoint requests (ignored for a local model)")
    ap.add_argument("--theta", type=float, default=None, help="residual confidence floor")
    args = ap.parse_args()

    schema = json.load(open(args.schema))
    context = [c for c in (args.context_cols or "").split(",") if c] or None
    driver = semextract.Default(
        id_col=args.id_col, text_col=args.text_col, image_col=args.image_col,
        image_dir=args.image_dir, context_cols=context,
        style=("json" if args.endpoint else args.prompt_style))
    semextract.run(
        driver, schema, args.table, args.out,
        modality=args.modality, model=args.model,
        endpoint=args.endpoint, api_key=args.api_key, concurrency=args.concurrency,
        theta=args.theta, limit=args.limit, max_new_tokens=args.max_new_tokens,
        timeout=args.timeout, prompt_style=args.prompt_style, debug=args.debug)


if __name__ == "__main__":
    main()
