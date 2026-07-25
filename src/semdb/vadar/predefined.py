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
    """Detected instances of `object_prompt` (YOLO) as SUB-IMAGES. Returns a list —
    len() counts, truthiness tests presence, and each element can be classified/read."""
    return image.find(object_prompt)


def crop(image, left, top, right, bottom):
    """A sub-region of the image (PIL order, TOP-LEFT origin). Pixels, or fractions of
    the image when all four are in [0,1]."""
    return image.crop(left, top, right, bottom)


def regions_grid(image, rows=2, cols=2, overlap=0.0):
    """Split the image into a rows x cols grid of sub-images (partitioned extraction)."""
    return image.regions_grid(rows, cols, overlap)


def regions_center(image, frac=0.6):
    """The central `frac` of the image — drops a product photo's background margin."""
    return image.regions_center(frac)


def pair_score(image, other):
    """IMAGE-IMAGE similarity in [0,1] (CLIP cosine) — for image-to-image joins."""
    return image.pair_score(other)


# --- P0: the (Value, Score) views — the paper's Table 1 output schemas ----------

def classify_detail(image, options, template="a photo of {}"):
    """OpImgCls: (best-matching VALUE, confidence)."""
    return image.classify_detail(options, template)


def verify_detail(image, prop):
    """(True iff the image matches `prop`, confidence of the winning side)."""
    return image.verify_detail(prop)


def detect_detail(image, object_prompt, min_conf=0.25):
    """OpImgObj: one dict per instance — {"image", "label", "box", "score"}."""
    return image.find_detail(object_prompt, min_conf)


def ocr_detail(image, min_conf=0.0):
    """OpImgOCR: one dict per text box — {"text", "box", "score"}."""
    return image.read_text_boxes(min_conf)


def best_ocr_match_detail(image, options):
    """(the OCR-matched VALUE or "none", match strength)."""
    return image.best_ocr_match_detail(options)


def bbox(image):
    """This image's own box in ABSOLUTE pixels — (0, 0, w, h) for a whole image."""
    return image.bbox


# --- P1.1: Latent — OpImgEmbed and vectorized ranking --------------------------

def embed(image):
    """OpImgEmbed: the image's dense vector."""
    return image.embed()


def topk_similar(image, others, k=5):
    """Vectorized OpImgPairScore: rank `others` against `image`."""
    return image.topk_similar(others, k)


def topk_text(image, texts, k=5):
    """Rank candidate texts against the image; the top-k with scores."""
    return image.topk_text(texts, k)


# --- P1.2: more OpImgCls backends — multilabel and domain specialists ----------

def classify_multi(image, options, thresh=0.5):
    """OpImgCls, multilabel: every option over the threshold, with a confidence."""
    return image.classify_multi(options, thresh)


def domain_classify(image, model_id, labels, threshold=0.5):
    """A domain-specialist classifier (e.g. chest X-ray pathologies)."""
    return image.domain_classify(model_id, labels, threshold)


# --- P1.3: Structural — OpImgRegion by content, not by grid --------------------

def regions_propose(image, max_regions=8, min_area_frac=0.01):
    """OpImgRegion: foreground regions as sub-images, largest first."""
    return image.propose_regions(max_regions, min_area_frac)


# --- P1.4: OpImgObj over an open vocabulary -----------------------------------

def detect_open(image, object_prompt, min_conf=0.1):
    """OpImgObj, open vocabulary: instances of ANY object name, as sub-images."""
    return image.find_open(object_prompt, min_conf)


MODULES_SIGNATURES = '''
SCORES. Every `*_detail` variant returns the operator's confidence alongside its value.
A score is comparable ACROSS ROWS for the SAME primitive (so a threshold on it is
meaningful, and a cheap pass can hand only its low-score rows to a heavier one), but
NOT across different primitives — CLIP probabilities, detector confidences and OCR
match strengths are on different scales. Prefer the plain variant when you only need
the value; reach for `_detail` when you need to gate, rank or cascade.

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
CLIP image-to-TEXT similarity in [0,1] — a raw score, for thresholding when no closed
value space exists. CLIP's text side sees only the first ~77 tokens, so pass a SHORT
phrase ("a black handbag"), NEVER a full product description; for a long predicate,
extract visual attributes with classify / dominant_colors / detect and match those.
Args:
    image (image): the image.
    text (string): a short phrase.
Returns:
    float: similarity in [0,1].
"""
def score(image, text):

"""
The raw OCR text of the image, reading order preserved. Use `best_ocr_match` instead when
you want to land on a value from a known value space; use `ocr_detail` when WHERE the text
sits matters.
Args:
    image (image): the image.
Returns:
    string: the text ("" if none was read).
"""
def read_text(image):

"""
Detects instances of an object in the image (YOLO) and returns them as SUB-IMAGES, most
confident first. Use len(...) for a count, truthiness for presence, and pass an element
back into classify/read_text/dominant_colors to inspect just that object. The detector
has a CLOSED vocabulary (COCO-80: person, car, dog, zebra, bird, ...); a name outside it
returns [] and warns — use classify/verify_property for those.
Args:
    image (image): the image.
    object_prompt (string): simple object name, e.g. "zebra".
Returns:
    list: detected instances as images (empty if none).
"""
def detect(image, object_prompt):

"""
A sub-region of the image. PIL order with the TOP-LEFT as origin: (left, top, right,
bottom). Pass pixels, or fractions of the image when all four values are in [0,1].
Every primitive applied to the result reads ONLY that region.
Args:
    image (image): the image.
    left, top, right, bottom (float): the box.
Returns:
    image: the sub-image.
"""
def crop(image, left, top, right, bottom):

"""
Splits the image into a rows x cols grid of sub-images (row-major) — partitioned
extraction. Run a primitive on each cell and combine, when one whole-image call is too
coarse (a small target in a big frame, several products in one photo). `overlap` (a
fraction of a cell) keeps a target that straddles a cut line whole in some cell.
Args:
    image (image): the image.
    rows (int), cols (int): grid shape (default 2x2).
    overlap (float): per-side overlap fraction (default 0.0).
Returns:
    list: sub-images.
"""
def regions_grid(image, rows=2, cols=2, overlap=0.0):

"""
The central `frac` of the image — excludes a product photo's white background margin.
Args:
    image (image): the image.
    frac (float): central fraction to keep (default 0.6).
Returns:
    image: the central sub-image.
"""
def regions_center(image, frac=0.6):

"""
IMAGE-to-IMAGE similarity in [0,1] (CLIP cosine). Use to join/dedup/rank one image
against another. NOTE `score` compares an image against TEXT; this compares two images.
Args:
    image (image): the first image.
    other (image): the second image.
Returns:
    float: similarity in [0,1].
"""
def pair_score(image, other):

"""
OpImgCls with a score: same as `classify`, but returns (VALUE, confidence in [0,1]).
Args:
    image (image): the image.
    options (list): the value space.
    template (string): prompt template with one "{}", default "a photo of {}".
Returns:
    tuple: (best-matching option value, confidence).
"""
def classify_detail(image, options, template="a photo of {}"):

"""
Same as `verify_property`, but also returns the confidence of the winning side.
Args:
    image (image): the image.
    prop (string): the property, e.g. "a damaged car".
Returns:
    tuple: (bool, confidence in [0,1]).
"""
def verify_detail(image, prop):

"""
OpImgObj proper: one entry PER DETECTED INSTANCE, most confident first, each a dict
{"image": the sub-image, "label": the class name, "box": (left, top, right, bottom) in
ABSOLUTE image pixels, "score": the detector confidence}. Use over `detect` when you need
the box or the score (to rank instances, or to keep only confident ones).
Args:
    image (image): the image.
    object_prompt (string): simple object name from the detector's closed vocabulary.
    min_conf (float): drop detections below this confidence (default 0.25).
Returns:
    list: dicts as above (empty if none).
"""
def detect_detail(image, object_prompt, min_conf=0.25):

"""
OpImgOCR proper: one entry per detected text box — {"text", "box": (left, top, right,
bottom) in ABSOLUTE image pixels, "score"}. Use over `read_text` when WHERE the text sits
matters (a label in a corner, a caption strip) or to drop unreliable reads.
Args:
    image (image): the image.
    min_conf (float): drop boxes below this OCR confidence (default 0.0 = keep all).
Returns:
    list: dicts as above.
"""
def ocr_detail(image, min_conf=0.0):

"""
Same as `best_ocr_match`, but also returns how strongly the OCR text matched. A miss is
("none", 0.0), so the score doubles as a gate for non-logo images.
Args:
    image (image): the image.
    options (list): the value space.
Returns:
    tuple: (matched value or "none", match strength in [0,1]).
"""
def best_ocr_match_detail(image, options):

"""
This image's own box in ABSOLUTE pixels, (0, 0, width, height) for a whole image and the
region's box for a sub-image from crop / regions_grid / detect / regions_propose. Pair it
with the region's index in the list it came from to identify a region.
Args:
    image (image): the image.
Returns:
    tuple: (left, top, right, bottom).
"""
def bbox(image):

"""
OpImgEmbed: encodes the image into a dense vector (a plain list of floats, unit-norm).
Two vectors are compared by `pair_score`-style cosine; use this when you want to encode
once and compare many times, or to materialize a vector column.
Args:
    image (image): the image.
Returns:
    list: the vector.
"""
def embed(image):

"""
Ranks OTHER images against this one by image-to-image similarity — the vectorized form of
`pair_score` (one encode per candidate, then one matmul, instead of a pairwise call per
comparison). Use for an image-to-image join, dedup or top-k.
Args:
    image (image): the query image.
    others (list): candidate images.
    k (int): how many to keep (default 5).
Returns:
    list: (index into `others`, score in [0,1]) tuples, best first.
"""
def topk_similar(image, others, k=5):

"""
Ranks candidate TEXTS against the image and keeps the top-k WITH scores. `classify` keeps
only the single best option; use this when you want a short candidate list out of a large
value space (e.g. narrow 135 airline names down to 5, then verify those 5 more carefully).
Args:
    image (image): the image.
    texts (list): candidate strings.
    k (int): how many to keep (default 5).
Returns:
    list: (text, score in [0,1]) tuples, best first.
"""
def topk_text(image, texts, k=5):

"""
OpImgCls, multilabel: keeps EVERY option whose match clears `thresh`, not just the best
one. Use when the field is a SET (several attributes hold at once, e.g. damage types on
one car) rather than a single enum value — `classify` would force one winner.
Args:
    image (image): the image.
    options (list): the value space.
    thresh (float): keep options scoring at or above this (default 0.5).
Returns:
    tuple: (list of matching values, confidence in [0,1]).
"""
def classify_multi(image, options, thresh=0.5):

"""
Runs a DOMAIN-SPECIALIST classifier — a model trained for this exact domain, far more
accurate there than zero-shot CLIP. Currently available: chest X-ray pathologies via
model_id "torchxrayvision:densenet121-res224-all" (labels are pathology names such as
"Pneumonia", "Effusion", "Cardiomegaly"). Returns "yes" when the strongest listed label
clears `threshold`.
Args:
    image (image): the image.
    model_id (string): the specialist model, e.g. "torchxrayvision:densenet121-res224-all".
    labels (list): the positive labels to test for.
    threshold (float): decision threshold (default 0.5).
Returns:
    tuple: ("yes" or "no", the max probability over `labels`).
"""
def domain_classify(image, model_id, labels, threshold=0.5):

"""
Splits the image into regions along its CONTENT — it estimates the background from the
border, then returns each foreground blob as a sub-image, largest first. Prefer over
`regions_grid` when the image holds several distinct objects (a product photo with two
garments, a localized defect on a car): a grid cuts blindly and can slice one object
across cells, while this cuts around each object. Deterministic and model-free.
A region's index in the returned list identifies it; its box is `bbox(region)`.
Args:
    image (image): the image.
    max_regions (int): keep at most this many, largest first (default 8).
    min_area_frac (float): ignore blobs smaller than this fraction of the image
        (default 0.01).
Returns:
    list: sub-images (empty for a uniform image).
"""
def regions_propose(image, max_regions=8, min_area_frac=0.01):

"""
Detects instances of ANY object name and returns them as SUB-IMAGES, most confident first
— the prompt IS the vocabulary. Use when `detect` reports the name is outside its closed
COCO-80 list (a species like "impala", a part like "bumper"). Slower and less precise than
`detect`, so prefer `detect` whenever the name is in its vocabulary. Returns [] with a
warning when the open-vocabulary backend is not installed.
Args:
    image (image): the image.
    object_prompt (string): any object name.
    min_conf (float): drop detections below this confidence (default 0.1).
Returns:
    list: detected instances as images (empty if none).
"""
def detect_open(image, object_prompt, min_conf=0.1):
'''


def _public_api():
    """The agent-visible API surface, derived from this module rather than hand-listed —
    a new predefined function is exposed to the generated program the moment it is defined
    here, and `tests/test_predefined_api_surface.py` fails if it is not also documented in
    `MODULES_SIGNATURES`."""
    import inspect
    import sys as _sys
    mod = _sys.modules[__name__]
    return {n: o for n, o in vars(mod).items()
            if inspect.isfunction(o) and not n.startswith("_") and o.__module__ == __name__}


PREDEFINED_API = _public_api()
