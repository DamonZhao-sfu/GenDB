#!/usr/bin/env python3
"""
vadar/predefined.py — the PREDEFINED API (VADAR's `predefined_modules.py` analog) that the
agent-synthesized helper functions and the final program compose. VADAR's base is
loc/vqa/depth (GroundingDINO/SAM2+VLM/UniDepth); ours is the non-VLM SemBench-image set,
backed by ImagePatch/semvision (CLIP / OCR / CV / YOLO). Free functions taking an
`ImagePatch` `image`, mirroring VADAR's `loc(image, ...)` call style.

`MODULES_SIGNATURES` is the docstring+signature block shown to the Signature/API/Program
agents (VADAR's `prompts/modules.py` analog).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/semdb


# --- the predefined primitives (backed by ImagePatch, which wraps semvision) ---

def classify(image, options, template="a photo of {}"):
    """CLIP zero-shot over a CLOSED value space; returns the best-matching VALUE."""
    return image.classify(options, template)


def best_ocr_match(image, options):
    """OCR the image and fuzzy-match its text to the value space (best for legible
    wordmark logos). Returns the matched VALUE, or 'none'."""
    return image.best_ocr_match(options)


def dominant_colors(image, min_frac=0.06, center_frac=1.0):
    """The colors present in the image (HSV CV). center_frac<1 crops past a product's
    background. Returns a list of color names."""
    return image.dominant_colors(min_frac, center_frac)


def verify_property(image, prop):
    """True iff the image matches `prop` (CLIP) better than its negation."""
    return image.verify_property(prop)


def score(image, text):
    """CLIP image-text similarity in [0, 1]. NOTE: CLIP's text side sees only the first
    ~77 tokens — pass a SHORT phrase (e.g. "a black handbag"), NEVER a full product
    description. For long text predicates, extract visual attributes with classify /
    dominant_colors / detect and match those instead."""
    return image.score(text)


def read_text(image):
    """Raw OCR text of the image."""
    return image.read_text()


def detect(image, object_prompt):
    """Detected instances of `object_prompt` (YOLO). Returns a list (presence/count)."""
    return image.find(object_prompt)


MODULES_SIGNATURES = '''
"""
Classifies the image into the single best-matching option from a closed value space, and
returns that VALUE (a real field). Use for enum categories or a DB column's values; the
`template` frames the CLIP prompt (e.g. "the logo of {}" for logos).
Args:
    image (image): the image.
    options (list): the value space (candidate string values).
    template (string): prompt template with one "{}", default "a photo of {}".
Returns:
    string: the best-matching option value.
"""
def classify(image, options, template="a photo of {}"):

"""
Reads the text in the image (OCR) and fuzzy-matches it to the value space; returns the
matched VALUE or "none". PREFER over classify for legible WORDMARK logos (airline/brand
names) over a large value space, where the printed name is readable.
Args:
    image (image): the image.
    options (list): the value space.
Returns:
    string: the matched value, or "none".
"""
def best_ocr_match(image, options):

"""
Returns the colors present in the image (HSV, includes pale accents). center_frac<1 crops
to the central region first (excludes a product photo's white background).
Args:
    image (image): the image.
    min_frac (float): min pixel fraction for a color to count (default 0.06).
    center_frac (float): central fraction to analyze (default 1.0 = whole image).
Returns:
    list: color-name strings present.
"""
def dominant_colors(image, min_frac=0.06, center_frac=1.0):

"""
Checks whether the image matches a property (CLIP), True/False.
Args:
    image (image): the image.
    prop (string): the property, e.g. "a damaged car".
Returns:
    bool: True if it matches `prop` better than "not `prop`".
"""
def verify_property(image, prop):

"""
Detects instances of an object/species in the image (YOLO). Returns a list (empty if none).
Args:
    image (image): the image.
    object_prompt (string): simple object/species name, e.g. "zebra".
Returns:
    list: detected instances (use len(...) for count, truthiness for presence).
"""
def detect(image, object_prompt):
'''
