#!/usr/bin/env python3
"""
vadar — the complete operator library that generated semantic programs run on.

A generated `solve_<query>.py` needs exactly one import root:

    import sys
    sys.path.insert(0, "<semdb_dir>")

    from vadar import ImagePatch, get_encoder, resolve_image_path
    from vadar.predefined import classify, detect, dominant_colors, contains_any, ...

Layout:

    predefined.py   the operator library the generated program composes —
                    VISION functions over an ImagePatch, TEXT functions over strings.
                    `MODULES_SIGNATURES` is what the agents are shown.
    imagepatch.py   `ImagePatch`: an image (or a region of one) with the primitives
                    bound to it.
    backend.py      model loading and the raw proxies (CLIP / OCR / CV / YOLO / OWL).
    paths.py        `resolve_image_path` — corpus manifest reference -> real file.
    models/         bundled detector weights.
    API.md          the ImagePatch API spec shown to the agents.

Everything here is OFFLINE: no VLM, no LLM, no HTTP client, no API key. The
orchestrator enforces that on every generated file before running it. The
endpoint-backed tools (semvqa, semcaption, semextract) live OUTSIDE this package and
import from it, never the other way round.
"""
from .backend import DEFAULT_PALETTE, get_detector, get_encoder, get_open_detector
from .imagepatch import ImagePatch
from .paths import resolve_image_path
from .predefined import MODULES_SIGNATURES, PREDEFINED_API

__all__ = [
    "ImagePatch",
    "DEFAULT_PALETTE",
    "get_detector",
    "get_encoder",
    "get_open_detector",
    "resolve_image_path",
    "MODULES_SIGNATURES",
    "PREDEFINED_API",
]
