#!/usr/bin/env python3
"""
vadar/predefined.py — the PREDEFINED operator library (VADAR's `predefined_modules.py`
analog) that the agent-synthesized helper functions and the final program compose.

Two families live here, because one generated program routinely needs both (an image
predicate joined against a text column):

  VISION — free functions taking an `ImagePatch` `image` first, mirroring VADAR's
           `loc(image, ...)` call style. Backed by `vadar.imagepatch`, which wraps the
           CLIP / OCR / CV / YOLO / OWL backends in `vadar.backend`. VADAR's base is
           loc/vqa/depth (GroundingDINO/SAM2+VLM/UniDepth); ours is the non-VLM set.

  TEXT   — deterministic offline functions over ordinary strings. They never create a
           model client, make a network request, or reach an endpoint. Query-specific
           helpers may compose them with Python's standard library.

Both families are OFFLINE: no VLM, no LLM, no HTTP. The orchestrator enforces that on
every generated file before running it.

Where a name would collide across the two families the text one carries a `text_`
prefix (`text_classify_detail`), because the first argument is a different KIND of
thing — an ImagePatch versus a str — and a single dispatching name would make the
docstring the agent reads ambiguous.

`MODULES_SIGNATURES` is the docstring+signature block shown to the Signature / API /
Program / Solver agents (VADAR's `prompts/modules.py` analog). `PREDEFINED_API` is
derived from this module, so a new function becomes visible to the generated program
the moment it is defined here — and `tests/test_predefined_api_surface.py` fails if it
is not also documented in `MODULES_SIGNATURES`.

NO HARDCODED VALUE SPACES. Every operator here takes its value space as an argument.
A closed set of labels (genres, regions, brands, track names) belongs to the QUERY and
must be read from the structured column at runtime — never baked into this library.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping


# =====================================================================================
# VISION primitives — backed by ImagePatch, which wraps vadar.backend
# =====================================================================================

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


# --- the (Value, Score) views — the paper's Table 1 output schemas ------------------

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


# --- Latent — OpImgEmbed and vectorized ranking -------------------------------------

def embed(image):
    """OpImgEmbed: the image's dense vector."""
    return image.embed()


def topk_similar(image, others, k=5):
    """Vectorized OpImgPairScore: rank `others` against `image`."""
    return image.topk_similar(others, k)


def topk_text(image, texts, k=5):
    """Rank candidate texts against the image; the top-k with scores."""
    return image.topk_text(texts, k)


# --- more OpImgCls backends — multilabel and domain specialists ---------------------

def classify_multi(image, options, thresh=0.5):
    """OpImgCls, multilabel: every option over the threshold, with a confidence."""
    return image.classify_multi(options, thresh)


def domain_classify(image, model_id, labels, threshold=0.5):
    """A domain-specialist classifier (e.g. chest X-ray pathologies)."""
    return image.domain_classify(model_id, labels, threshold)


# --- Structural — OpImgRegion by content, not by grid -------------------------------

def regions_propose(image, max_regions=8, min_area_frac=0.01):
    """OpImgRegion: foreground regions as sub-images, largest first."""
    return image.propose_regions(max_regions, min_area_frac)


# --- OpImgObj over an open vocabulary ----------------------------------------------

def detect_open(image, object_prompt, min_conf=0.1):
    """OpImgObj, open vocabulary: instances of ANY object name, as sub-images."""
    return image.find_open(object_prompt, min_conf)


def classify_or_none(image, options, min_conf=0.5, template="a photo of {}"):
    """`classify` that CAN ABSTAIN: the best option, or "none" below `min_conf`.

    `classify`/`classify_detail` are argmax — they always return one of `options`, so an
    image that shows none of them still gets a label. That is the right behaviour when
    every row is known to hold one of the values, and the wrong one for a FILTER, where
    most rows hold none of them. Use this for a filter and pick `min_conf` deliberately.
    """
    label, score = image.classify_detail(options, template)
    return label if score >= min_conf else "none"


# =====================================================================================
# TEXT primitives — deterministic, offline, over ordinary strings
# =====================================================================================

def _plain_text(value) -> str:
    """Accept a string or an object exposing a `.text` string."""
    return str(getattr(value, "text", value) or "")


def normalize(text) -> str:
    """Case-fold text and collapse punctuation/whitespace for stable matching."""
    value = unicodedata.normalize("NFKD", _plain_text(text)).casefold()
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


def tokens(text) -> set[str]:
    """The normalized word-token set."""
    return set(normalize(text).split())


def contains_phrase(text, phrase) -> bool:
    """True when a normalized phrase occurs on token boundaries."""
    haystack = f" {normalize(text)} "
    needle = normalize(phrase)
    return bool(needle) and f" {needle} " in haystack


def contains_any(text, phrases: Iterable[str]) -> bool:
    """True when at least one normalized phrase occurs in the text."""
    return any(contains_phrase(text, phrase) for phrase in phrases)


def contains_all(text, phrases: Iterable[str]) -> bool:
    """True when every normalized phrase occurs in the text."""
    return all(contains_phrase(text, phrase) for phrase in phrases)


def lexical_score(text, query) -> float:
    """Token-overlap score in [0, 1], with an exact-phrase match scoring 1."""
    if contains_phrase(text, query):
        return 1.0
    left, right = tokens(text), tokens(query)
    if not left or not right:
        return 0.0
    return len(left & right) / len(right)


def best_lexical_match(text, options, aliases: Mapping[str, Iterable[str]] | None = None,
                       default="none"):
    """The best locally matched option, or `default` when none overlaps."""
    best, best_score = default, 0.0
    for option in options:
        candidates = [str(option)]
        if aliases:
            candidates.extend(str(v) for v in aliases.get(option, ()))
        score = max((lexical_score(text, candidate) for candidate in candidates), default=0.0)
        if score > best_score:
            best, best_score = option, score
    return best


def text_classify_detail(text, options,
                         descriptions: Mapping[str, str] | None = None,
                         aliases: Mapping[str, Iterable[str]] | None = None,
                         default="none") -> tuple[str, float]:
    """Bounded offline text classification over an explicit value space.

    Scores label names, caller-supplied descriptions, and aliases by token overlap.
    This is deliberately a bounded approximation — not a claim to reproduce a remote
    model's behavior — and returns its confidence so a plan can expose that boundary.

    `options`, `descriptions` and `aliases` come from the QUERY (a database column's
    distinct values, or the SQL's literals), never from a taxonomy baked in here.
    """
    best, best_score = default, 0.0
    for option in options:
        candidates = [str(option)]
        if descriptions and option in descriptions:
            candidates.append(str(descriptions[option]))
        if aliases:
            candidates.extend(str(value) for value in aliases.get(option, ()))
        score = max((lexical_score(text, candidate) for candidate in candidates),
                    default=0.0)
        if score > best_score:
            best, best_score = option, score
    return best, float(best_score)


def text_classify_multi_detail(text, options,
                               descriptions: Mapping[str, str] | None = None,
                               aliases: Mapping[str, Iterable[str]] | None = None,
                               threshold: float = 0.34) -> tuple[list[str], dict[str, float]]:
    """Every supported label from an explicit multi-label value space, with scores.

    The multi-label counterpart of `text_classify_detail`: use it when the field holds
    a SET of values at once (several genres on one film) rather than a single label.
    Pass `aliases` to name the cue PHRASES that stand in for a label — matching on the
    bare label name alone is what makes token overlap over-fire.
    """
    selected: list[str] = []
    scores: dict[str, float] = {}
    for option in options:
        candidates = [str(option)]
        if descriptions and option in descriptions:
            candidates.append(str(descriptions[option]))
        if aliases:
            candidates.extend(str(value) for value in aliases.get(option, ()))
        score = max((lexical_score(text, candidate) for candidate in candidates),
                    default=0.0)
        scores[str(option)] = float(score)
        if score >= float(threshold):
            selected.append(option)
    return selected, scores


def any_value_in_set(values, members, strip_parentheticals=True,
                     separators=r"[,;/|]") -> bool:
    """True when any value of a DELIMITED field falls in `members`.

    The general form of a "does this multi-valued column hit this set?" predicate —
    a destination list against a region's cities, a cast list against an award roster.
    `members` is the value space, supplied by the query at runtime.
    `strip_parentheticals` drops trailing qualifiers such as an airport code in
    "London (LHR)" before matching.
    """
    space = [normalize(m) for m in members]
    space = [m for m in space if m]
    if not space:
        return False
    for value in split_values(values, separators=separators):
        cleaned = re.sub(r"\([^)]*\)", " ", value) if strip_parentheticals else value
        if any(contains_phrase(cleaned, member) for member in space):
            return True
    return False


def extract_from_candidates(text, candidates, default="none") -> tuple[str, float]:
    """The longest candidate occurring in text, else the best lexical match.

    Candidate values must come from an ordinary database column or SQL literal value
    space, never validation labels. This supports bounded entity extraction such as a
    brand name without network access.
    """
    values = [str(value).strip() for value in candidates if str(value).strip()]
    contained = [value for value in values if contains_phrase(text, value)]
    if contained:
        value = max(contained, key=lambda item: (len(normalize(item)), item))
        return value, 1.0
    value = best_lexical_match(text, values, default=default)
    return value, lexical_score(text, value) if value != default else 0.0


def regex_extract(text, pattern, group=1, flags=re.IGNORECASE, default="none"):
    """A regex capture from text, or `default` when it is absent."""
    match = re.search(pattern, _plain_text(text), flags)
    if not match:
        return default
    try:
        value = match.group(group)
    except (IndexError, KeyError):
        return default
    value = str(value).strip()
    return value if value else default


def extract_person_names(text) -> list[str]:
    """Explicit multi-token proper names, without a remote NER service.

    This intentionally returns candidates, not a claim that every candidate is a
    person. Relational evidence such as occurrence across all target documents can
    safely disambiguate the candidates downstream.
    """
    pattern = re.compile(
        r"\b(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+"
        r"(?:\s+(?:de|del|van|von|da|dos|la|le))?\s+)"
        r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+"
        r"(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+){0,2}\b")
    seen: set[str] = set()
    result: list[str] = []
    sentence_leaders = {
        "a", "after", "an", "as", "at", "before", "by", "during", "for",
        "from", "in", "later", "meanwhile", "on", "the", "then", "when",
    }
    for match in pattern.finditer(_plain_text(text)):
        value = " ".join(match.group(0).split()).strip(" ,.;:()[]")
        pieces = value.split()
        if len(pieces) >= 3 and pieces[0].casefold() in sentence_leaders:
            value = " ".join(pieces[1:])
        key = normalize(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def split_values(text, separators=r"[,;/|]", allowed=None):
    """Split a delimited field, optionally retaining only allowed normalized values."""
    values = [part.strip() for part in re.split(separators, _plain_text(text)) if part.strip()]
    if allowed is None:
        return values
    lookup = {normalize(value): value for value in allowed}
    return [lookup[normalize(value)] for value in values if normalize(value) in lookup]


MODULES_SIGNATURES = '''
The predefined API has TWO families. VISION functions take an `image` (an ImagePatch)
first; TEXT functions take an ordinary string first. Both are OFFLINE — no VLM, no LLM,
no HTTP, no API key. Query-specific helpers may also use Python's standard-library
string, regex, numeric and date operations.

NO HARDCODED VALUE SPACES. Every closed set of labels (genres, regions, brands, track
names) is an ARGUMENT, read from the structured column at runtime. There is no built-in
taxonomy to call.

ARGMAX NEVER ABSTAINS. `classify`, `classify_detail` and `best_lexical_match` always
return one of `options` — they can NEVER return "none", so `if classify(...) != "none"`
is a guard that is always true and silently labels every row. For a FILTER, where most
rows match none of the options, you must either threshold the confidence from the
`_detail` variant or call `classify_or_none`. A filter bound to a bare argmax labels every
row, including the ones showing none of the options.

SCORES. Every `*_detail` variant returns the operator's confidence alongside its value.
A score is comparable ACROSS ROWS for the SAME primitive (so a threshold on it is
meaningful, and a cheap pass can hand only its low-score rows to a heavier one), but
NOT across different primitives — CLIP probabilities, detector confidences and OCR
match strengths are on different scales. Prefer the plain variant when you only need
the value; reach for `_detail` when you need to gate, rank or cascade.


========================= VISION PRIMITIVES (take an image) =========================

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
matched VALUE or "none". PREFER over classify for legible WORDMARK logos over a large
value space, where the printed name is readable.
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
value space (e.g. narrow a few hundred names down to 5, then verify those 5 carefully).
SCALE WARNING: these are RAW CLIP cosines, not the softmaxed confidences `classify_detail`
returns. They sit in a narrow band — six competing prompts on one logo spanned 0.6184 to
0.6325, a total spread of 0.014 — so ORDER here is near-noise and a margin like 0.01 is
meaningless. Use it to SHORTLIST, never as the gate that decides a row.
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
NOT A MEMBERSHIP TEST. Each option is scored INDEPENDENTLY and relative to this image's
own average match, so over a mutually-exclusive value space (a handful of names, exactly
one of which applies) it returns roughly HALF the space for every input. Deciding a filter with
it makes each accepted row match nearly every candidate value. For "which one of these",
use `classify_detail` with a high threshold, or `classify_or_none`.
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

"""
`classify` that CAN ABSTAIN: returns the best-matching option, or "none" when its
confidence is below `min_conf`. Use this for a FILTER — a predicate deciding row
membership, where most rows match none of the options. `classify`/`classify_detail` are
argmax and label every row no matter what it shows, which produces recall 1.0 at
precision near the base rate. Choose `min_conf` deliberately and record it as the site's
`confidence_signal` so the optimizer has a threshold to tune.
Args:
    image (image): the image.
    options (list): the value space.
    min_conf (float): abstain below this confidence (default 0.5).
    template (string): prompt template with one "{}", default "a photo of {}".
Returns:
    string: the best-matching option value, or "none".
"""
def classify_or_none(image, options, min_conf=0.5, template="a photo of {}"):


========================== TEXT PRIMITIVES (take a string) ==========================

"""
Case-folds text and collapses punctuation/whitespace/accents into space-separated word
tokens. Every other text primitive matches on this form, so two spellings that normalize
alike compare equal. Use it to key a dict or to compare a field against a literal.
Args:
    text (string): the text.
Returns:
    string: the normalized text.
"""
def normalize(text):

"""
The normalized word-token SET of the text — for set algebra (overlap, subset) when
phrase order does not matter.
Args:
    text (string): the text.
Returns:
    set: the tokens.
"""
def tokens(text):

"""
True when `phrase` occurs in `text` on TOKEN BOUNDARIES, after normalizing both. Unlike
`in`, "comed" does not match "comedy" and "sci-fi" matches "SCI FI". This is the right
test for "does this field mention X".
Args:
    text (string): the text.
    phrase (string): the phrase to look for.
Returns:
    bool: True if the phrase occurs.
"""
def contains_phrase(text, phrase):

"""
True when AT LEAST ONE of `phrases` occurs in the text (`contains_phrase` over a list) —
an OR over a value space or a synonym list.
Args:
    text (string): the text.
    phrases (list): the phrases.
Returns:
    bool: True if any occurs.
"""
def contains_any(text, phrases):

"""
True when EVERY phrase occurs in the text — an AND over a conjunctive predicate.
Args:
    text (string): the text.
    phrases (list): the phrases.
Returns:
    bool: True if all occur.
"""
def contains_all(text, phrases):

"""
Token-overlap score in [0,1]: the fraction of `query`'s tokens present in `text`, with an
exact phrase match scoring 1.0. Use to RANK or threshold when no closed value space
exists — the text analog of vision's `score`.
Args:
    text (string): the text.
    query (string): the phrase to score against.
Returns:
    float: overlap in [0,1].
"""
def lexical_score(text, query):

"""
The option from `options` that best overlaps the text, or `default` when none does. Use
when you only want the winning VALUE; use `text_classify_detail` when you also need the
confidence.
Args:
    text (string): the text.
    options (list): the value space, read from the query's column at runtime.
    aliases (dict): optional {option: [other spellings]}.
    default (string): returned when nothing overlaps (default "none").
Returns:
    string: the best option, or `default`.
"""
def best_lexical_match(text, options, aliases=None, default="none"):

"""
The TEXT counterpart of vision's `classify_detail`: classifies text into the single best
option of an explicit value space and returns (VALUE, confidence in [0,1]). Scores the
label name, an optional description, and optional aliases by token overlap. `options`
comes from the query's column or SQL literals — there is no built-in taxonomy.
Named `text_`* because vision's `classify_detail` takes an image, not a string.
Args:
    text (string): the text.
    options (list): the value space.
    descriptions (dict): optional {option: a sentence describing it}.
    aliases (dict): optional {option: [cue phrases]}.
    default (string): returned when nothing overlaps (default "none").
Returns:
    tuple: (best option value, confidence in [0,1]).
"""
def text_classify_detail(text, options, descriptions=None, aliases=None, default="none"):

"""
Multi-label form of `text_classify_detail`: EVERY option clearing `threshold`, plus the
per-option scores. Use when the field holds a SET of values at once (several genres on
one synopsis) rather than one label. Pass `aliases` with the cue PHRASES that stand in
for each label — matching the bare label name alone is what makes token overlap over-fire
(a synopsis containing only "film" would otherwise match "action film").
CAUTION: scoring is TOKEN OVERLAP, so a multi-word cue can half-match — "love story"
scores 0.5 against "...horror story...". When you want WHOLE-PHRASE semantics, build the
predicate from `contains_any` instead:
    labels = [k for k, cues in cue_map.items() if contains_any(text, cues)]
Reach for this function when you need the SCORES (to gate, rank or cascade).
Args:
    text (string): the text.
    options (list): the value space.
    descriptions (dict): optional {option: a sentence describing it}.
    aliases (dict): optional {option: [cue phrases]}.
    threshold (float): keep options scoring at or above this (default 0.34).
Returns:
    tuple: (list of matching values, {option: score}).
"""
def text_classify_multi_detail(text, options, descriptions=None, aliases=None, threshold=0.34):

"""
True when any value of a DELIMITED field falls in `members` — the general "does this
multi-valued column hit this set?" test (a destination list against a region's cities, a
cast list against a roster). `members` is the value space and MUST be supplied by the
query at runtime; there is no built-in geography or taxonomy.
`strip_parentheticals` drops trailing qualifiers such as the airport code in
"London (LHR)" before matching.
Args:
    values (string): the delimited field.
    members (list): the value space to test against.
    strip_parentheticals (bool): drop "(...)" qualifiers first (default True).
    separators (string): regex character class of delimiters (default "[,;/|]").
Returns:
    bool: True if any value is in the set.
"""
def any_value_in_set(values, members, strip_parentheticals=True, separators=r"[,;/|]"):

"""
Bounded entity extraction: the LONGEST candidate that occurs verbatim in the text (score
1.0), else the best lexical match with its score. Use to pull a brand / person / place
out of free text when the candidate list comes from a database column.
Args:
    text (string): the text.
    candidates (list): the candidate value space.
    default (string): returned when nothing matches (default "none").
Returns:
    tuple: (the extracted value or `default`, confidence in [0,1]).
"""
def extract_from_candidates(text, candidates, default="none"):

"""
A regex capture group from the text, or `default` when the pattern does not match. Use
for structurally marked fields ("Director: Jane Doe"), not for semantic judgement.
Args:
    text (string): the text.
    pattern (string): the regex.
    group (int|string): capture group (default 1).
    flags (int): regex flags (default re.IGNORECASE).
    default (string): returned when absent (default "none").
Returns:
    string: the capture, or `default`.
"""
def regex_extract(text, pattern, group=1, flags=re.IGNORECASE, default="none"):

"""
Candidate multi-token proper names in the text, de-duplicated, original spelling kept.
These are CANDIDATES, not a claim that each is a person — disambiguate downstream with
relational evidence (e.g. the name occurring across all target documents).
Args:
    text (string): the text.
Returns:
    list: candidate name strings, first occurrence order.
"""
def extract_person_names(text):

"""
Splits a delimited field into its values. With `allowed`, keeps only the values whose
normalized form is in that value space and returns them in the value space's own
spelling — the way to reconcile a free-text list against a DB column.
Args:
    text (string): the delimited field.
    separators (string): regex character class of delimiters (default "[,;/|]").
    allowed (list): optional value space to filter and re-spell against.
Returns:
    list: the values.
"""
def split_values(text, separators=r"[,;/|]", allowed=None):
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
