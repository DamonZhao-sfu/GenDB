#!/usr/bin/env python3
"""
vadar/backend.py — the non-VLM vision BACKENDS the operator library runs on.

This is the bottom layer of `vadar/`: model loading and the raw per-image proxies,
with no notion of a query, a corpus or a schema. `vadar/imagepatch.py` wraps these
into the `ImagePatch` object, and `vadar/predefined.py` exposes them to generated
code as free functions.

  pure CV      — dominant colors, region proposal (numpy/PIL, zero model)
  CLIP         — classify / multilabel / match / embed / pair score
  OCR          — easyocr (lazy)
  detector     — YOLO closed vocabulary, OWLv2 open vocabulary
  domain       — specialist classifiers (e.g. torchxrayvision)

Every proxy takes a PATH or an already-loaded PIL image, and every model is loaded
once through a cached `get_*` accessor.
"""

import json
import os

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Source resolution — every proxy accepts a PATH or an already-loaded PIL image
# ---------------------------------------------------------------------------

def _open(src, mode="RGB"):
    """Resolve a proxy input to a PIL image in `mode`.

    `src` is either a path (the whole-image fast path) or an already-cropped
    PIL image — that second form is what lets `ImagePatch.crop`/`find`/`regions_*`
    run a proxy on a REGION instead of silently re-running it on the full frame.
    Passing a path reproduces the previous behavior byte-for-byte.
    """
    im = Image.open(src) if isinstance(src, (str, os.PathLike)) else src
    return im.convert(mode)


# ---------------------------------------------------------------------------
# ① Pure CV — dominant colors
# ---------------------------------------------------------------------------

# name -> RGB anchor. 'silver' is a light gray; 'gray' a mid gray (kept distinct).
DEFAULT_PALETTE = {
    "black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128),
    "silver": (192, 192, 192), "red": (200, 30, 30), "green": (30, 160, 60),
    "blue": (40, 70, 190), "yellow": (240, 220, 40), "orange": (230, 130, 30),
    "brown": (120, 70, 40), "purple": (120, 50, 160), "pink": (230, 130, 170),
    "gold": (210, 170, 60), "beige": (220, 200, 160),
}


# Chromatic hue bands (HSV hue in [0,1)) → color name. Hue-based detection catches
# PALE/ACCENT colors (e.g. a light-yellow shoe accent) that nearest-RGB would collapse
# into white/beige. Achromatic (low-saturation) pixels bucket by lightness below.
_HUE_BANDS = [
    ("red", [(0.0, 0.04), (0.96, 1.0)]), ("orange", [(0.04, 0.10)]),
    ("yellow", [(0.10, 0.20)]), ("green", [(0.20, 0.45)]),
    ("blue", [(0.45, 0.70)]), ("purple", [(0.70, 0.86)]), ("pink", [(0.86, 0.96)]),
]


def _rgb_to_hsv(arr):
    """Vectorized RGB[0..1] -> (h,s,v), each [H,W]."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx, mn = arr.max(-1), arr.min(-1)
    diff = mx - mn
    v = mx
    s = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0)
    h = np.zeros_like(v)
    m = diff > 1e-6
    ir = m & (mx == r)
    ig = m & (mx == g) & ~ir
    ib = m & (mx == b) & ~ir & ~ig
    h[ir] = (((g - b) / np.maximum(diff, 1e-6))[ir]) % 6.0
    h[ig] = (((b - r) / np.maximum(diff, 1e-6))[ig]) + 2.0
    h[ib] = (((r - g) / np.maximum(diff, 1e-6))[ib]) + 4.0
    return (h / 6.0) % 1.0, s, v


def cv_dominant_colors(img_path, palette=None, min_frac=0.04, size=96,
                       sat_thresh=0.20, val_lo=0.12, center_frac=1.0):
    """Return (colors ≥ min_frac of pixels, confidence). Chromatic pixels (saturation
    ≥ sat_thresh) map to a color by HSV HUE (so pale accents keep their hue); the rest
    bucket to black/gray/silver/white by lightness. `center_frac`<1 crops to the central
    region first (excludes a product photo's white background). Deterministic → conf 1.0."""
    im = _open(img_path)
    if center_frac < 1.0:
        w, h = im.size
        cw, ch = int(w * center_frac), int(h * center_frac)
        l, t = (w - cw) // 2, (h - ch) // 2
        im = im.crop((l, t, l + cw, t + ch))
    im = im.resize((size, size))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    h, s, v = _rgb_to_hsv(arr)
    n = h.size
    names = np.empty(h.shape, dtype=object)
    chroma = (s >= sat_thresh) & (v >= val_lo)
    ach = ~chroma
    names[ach & (v < 0.2)] = "black"
    names[ach & (v >= 0.2) & (v < 0.5)] = "gray"
    names[ach & (v >= 0.5) & (v < 0.8)] = "silver"
    names[ach & (v >= 0.8)] = "white"
    for cname, bands in _HUE_BANDS:
        mask = np.zeros(h.shape, bool)
        for lo, hi in bands:
            mask |= (h >= lo) & (h < hi)
        names[chroma & mask] = cname
    names[chroma & (names == "orange") & (v < 0.5)] = "brown"  # dark orange → brown
    flat = names.reshape(-1)
    counts = {}
    for nm in flat:
        counts[nm] = counts.get(nm, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    colors = [nm for nm, c in ordered if nm is not None and c / n >= min_frac]
    return colors, 1.0


# ---------------------------------------------------------------------------
# ① Structural — OpImgRegion by foreground connected components (model-free)
# ---------------------------------------------------------------------------

def _label_4c(mask):
    """4-connected component labels for a boolean mask (two-pass union-find). Returns an
    int32 array of the same shape where 0 is background and each component has its own id.
    Deterministic and dependency-free — deliberately not scipy.ndimage.label, so region
    proposal works in any environment that can already run the CLIP path."""
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    parent = [0]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nxt = 1
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            up = int(lab[y - 1, x]) if y else 0
            left = int(lab[y, x - 1]) if x else 0
            if up and left:
                lab[y, x] = min(up, left)
                union(up, left)
            elif up or left:
                lab[y, x] = up or left
            else:
                lab[y, x] = nxt
                parent.append(nxt)
                nxt += 1
    if nxt > 1:
        root = np.array([find(i) for i in range(nxt)], np.int32)
        lab = root[lab]
    return lab


def propose_region_boxes(src, max_regions=8, min_area_frac=0.01, size=128, tol=0.12):
    """OpImgRegion (BBox only, no Mask): content-driven region proposals as boxes.

    Estimates the background color from the border pixels, marks every pixel further than
    `tol` from it as foreground, labels the 4-connected components, and returns their
    bounding boxes as FRACTIONS of the image — (left, top, right, bottom) each in [0,1],
    largest component first. Unlike `regions_grid` this cuts along content, which is what
    a product photo with several items or a localized defect needs. Model-free, so it is
    the cheap first backend for OpImgRegion; a learned proposer (SAM/DINO) would be a
    second one under the same signature.
    """
    im = _open(src).resize((size, size))
    arr = np.asarray(im, np.float32) / 255.0
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    bg = np.median(border, axis=0)
    fg = np.abs(arr - bg).max(-1) > tol
    if not fg.any():
        return []
    lab = _label_4c(fg)
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    keep = [(int(i), int(c)) for i, c in zip(ids, counts)
            if c / float(size * size) >= min_area_frac]
    keep.sort(key=lambda ic: (-ic[1], ic[0]))          # largest first, id breaks ties
    out = []
    for cid, _c in keep[:max_regions]:
        ys, xs = np.nonzero(lab == cid)
        out.append((float(xs.min()) / size, float(ys.min()) / size,
                    float(xs.max() + 1) / size, float(ys.max() + 1) / size))
    return out


# ---------------------------------------------------------------------------
# ② CLIP zero-shot — classify / multilabel / match (injectable encoder)
# ---------------------------------------------------------------------------

_ENCODER_TAKES_KEY = {}


def _encode_image(encoder, src, key=None):
    """Encode via `encoder`, passing the region cache key only if it accepts one.
    An encoder is any object with `encode_image` — the pre-cache signature `(src)` and
    the cache-aware `(src, key=...)` must both keep working."""
    if key is None:
        return encoder.encode_image(src)
    cls = type(encoder)
    if cls not in _ENCODER_TAKES_KEY:
        import inspect
        try:
            _ENCODER_TAKES_KEY[cls] = "key" in inspect.signature(encoder.encode_image).parameters
        except (TypeError, ValueError):
            _ENCODER_TAKES_KEY[cls] = False
    return (encoder.encode_image(src, key=key) if _ENCODER_TAKES_KEY[cls]
            else encoder.encode_image(src))


def _softmax(x, temp=0.01):
    x = np.asarray(x, np.float32) / temp
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def clip_classify(img_path, labels, encoder, template="a photo of {}", key=None):
    """Zero-shot classify into `labels` (the field's VALUE SPACE — an enum or a
    structured column's values). Returns (the winning VALUE, prob). `template`
    frames the text prompt (e.g. 'the logo of {}' for logos) but the returned value
    is the raw label, so this yields a real structured FIELD value, not a score.
    `key` identifies the image region for the encoder's cache (see ClipEncoder)."""
    iv = _encode_image(encoder, img_path, key)        # [D], normalized
    tv = encoder.encode_text(list(labels), template)  # [L,D], normalized
    sims = tv @ iv                               # [L] cosine
    probs = _softmax(sims)
    j = int(np.argmax(probs))
    return labels[j], float(probs[j])


def clip_multilabel(img_path, labels, encoder, thresh=0.5, template="a photo of {}", key=None):
    """Keep every label whose match clears `thresh`, scored RELATIVE to this image.

    The score is centered on the image's own mean similarity across `labels` before the
    sigmoid. Without that centering `thresh` is not comparable across images at all: raw
    CLIP cosine carries a large per-image offset (0.089..0.306 over one 200-image corpus,
    a 0.216 swing) while the sigmoid's transition width is 0.05, so the offset alone
    decided the outcome — 72% of images returned either NOTHING or EVERY label, whatever
    they actually showed.

    Centering makes `thresh` mean "better matched than this image's average candidate".
    That is the most an independent-label rule can promise, and it is still the WRONG
    primitive for a mutually-exclusive value space: asking which of six racetrack names an
    image shows has exactly one answer, so use `clip_classify` (argmax + softmax
    confidence) there and keep this for genuinely multi-valued attributes.
    """
    iv = _encode_image(encoder, img_path, key)
    tv = encoder.encode_text(list(labels), template)
    sims = np.asarray(tv @ iv, dtype=np.float32)  # cosine in [-1,1]
    centered = sims - sims.mean() if sims.size else sims
    probs = 1.0 / (1.0 + np.exp(-centered / 0.05))
    chosen = [l for l, p in zip(labels, probs) if p >= thresh]
    if chosen:
        conf = float(np.mean([p for l, p in zip(labels, probs) if p >= thresh]))
    else:
        conf = float(np.max(probs)) if len(probs) else 0.0
    return chosen, conf


def clip_match(img_path, text, encoder, key=None):
    iv = _encode_image(encoder, img_path, key)
    tv = encoder.encode_text([text])[0]
    return float(np.clip((tv @ iv + 1.0) / 2.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# ② Latent — OpImgEmbed / OpImgPairScore
# ---------------------------------------------------------------------------

def img_pair_score(a, b, encoder, key_a=None, key_b=None):
    """IMAGE–IMAGE similarity in [0,1] (cosine, rescaled like `clip_match` so the two
    are on one scale). This is the primitive an image-to-image join/dedup/top-k needs;
    `clip_match` only compares an image to TEXT."""
    va = _encode_image(encoder, a, key_a)
    vb = _encode_image(encoder, b, key_b)
    return float(np.clip((float(va @ vb) + 1.0) / 2.0, 0.0, 1.0))


def embed_image(src, encoder, key=None):
    """OpImgEmbed: the unit-norm image vector for one image or region. `pair_score` and
    `topk_similar` both consume this space, so a corpus can be encoded once and reused."""
    return np.asarray(_encode_image(encoder, src, key), np.float32)


def embed_text(text, encoder, template="{}"):
    """OpTxtEmbed on the image side: the unit-norm text vector in the SAME space as
    `embed_image`. This is how the paper grounds OpTxtImgSim into OpImgEmbed->OpTxtEmbed."""
    return np.asarray(encoder.encode_text([str(text)], template)[0], np.float32)


def topk_similar(query_vec, matrix, k=5):
    """Top-k rows of an [N, D] unit-norm `matrix` by cosine against `query_vec`, rescaled
    to [0,1] on the same scale as `img_pair_score`/`clip_match`. Returns [(row index,
    score)], best first — one matmul instead of N pairwise encoder calls."""
    q = np.asarray(query_vec, np.float32)
    m = np.asarray(matrix, np.float32)
    if m.size == 0:
        return []
    sims = np.clip((m @ q + 1.0) / 2.0, 0.0, 1.0)
    k = max(0, min(int(k), int(sims.shape[0])))
    order = np.argsort(-sims, kind="stable")[:k]
    return [(int(i), float(sims[i])) for i in order]


def embed_corpus(paths, encoder, batch=64, on_error="zero"):
    """Encode a whole corpus into an [N, D] float32 matrix (batched — an order of
    magnitude faster than per-row `encode_image`). Unreadable images become zero rows
    when `on_error='zero'` so the matrix stays aligned with `paths`."""
    return encoder.encode_images(list(paths), batch=batch, on_error=on_error)


def save_embeddings(out_path, ids, matrix):
    """Persist a corpus embedding table as <out_path> (.npy) + <out_path>.ids.json,
    so it amortizes across a query family exactly like the attribute table does."""
    np.save(out_path, np.asarray(matrix, np.float32))
    json.dump({"ids": [str(i) for i in ids], "dim": int(np.shape(matrix)[1]) if len(matrix) else 0},
              open(str(out_path) + ".ids.json", "w"))
    return out_path


def load_embeddings(path):
    """Returns (matrix, {id -> row index})."""
    mat = np.load(str(path))
    ids = json.load(open(str(path) + ".ids.json"))["ids"]
    return mat, {v: i for i, v in enumerate(ids)}


# ---------------------------------------------------------------------------
# Real CLIP encoder (transformers), lazy + process-cached
# ---------------------------------------------------------------------------

_ENCODER_CACHE = {}


def get_encoder(model_id="openai/clip-vit-base-patch32"):
    if model_id not in _ENCODER_CACHE:
        _ENCODER_CACHE[model_id] = ClipEncoder(model_id)
    return _ENCODER_CACHE[model_id]


class ClipEncoder:
    # Bounded caches. The image cache pays off WITHIN a row (a patch that is cropped
    # into regions and scored several times); the text cache pays off ACROSS rows —
    # a value space (e.g. 135 airline names) was previously re-encoded once per image.
    _IMG_CACHE_CAP = 256
    _TXT_CACHE_CAP = 64

    def __init__(self, model_id="openai/clip-vit-base-patch32", device=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)
        self._icache, self._tcache = {}, {}

    def _norm(self, t):
        return (t / t.norm(dim=-1, keepdim=True)).detach().cpu().numpy()

    @staticmethod
    def _put(cache, cap, key, val):
        if len(cache) >= cap:
            cache.clear()            # cheap bounded eviction; these are hit-locality caches
        cache[key] = val
        return val

    def _encode_pixels(self, ims):
        # Compute via the vision tower + projection (version-robust: some transformers
        # builds make get_image_features return a model-output object, not a tensor).
        inp = self.proc(images=ims, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.vision_model(pixel_values=inp["pixel_values"])
            v = self.model.visual_projection(out.pooler_output)
        return self._norm(v)

    def encode_image(self, src, key=None):
        """`src` is a path or a PIL region. `key` (a hashable region identity, e.g.
        (path, box)) enables caching — omit it and behavior is exactly as before."""
        if key is not None and key in self._icache:
            return self._icache[key]
        v = self._encode_pixels([_open(src)])[0]
        return v if key is None else self._put(self._icache, self._IMG_CACHE_CAP, key, v)

    def encode_images(self, srcs, batch=64, on_error="zero"):
        """Batched corpus encoding → [N, D]. Unreadable images become zero rows so the
        matrix stays row-aligned with `srcs`."""
        dim = int(getattr(self.model.config, "projection_dim", 512))
        out = np.zeros((len(srcs), dim), np.float32)
        for i in range(0, len(srcs), batch):
            chunk = srcs[i:i + batch]
            ims, keep = [], []
            for j, s in enumerate(chunk):
                try:
                    ims.append(_open(s)); keep.append(j)
                except Exception as e:  # noqa: BLE001 — one bad image must not kill the batch
                    if on_error != "zero":
                        raise
                    print(f"[semvision] embed skip {s!r}: {e}")
            if ims:
                out[[i + j for j in keep]] = self._encode_pixels(ims)
        return out

    def encode_text(self, labels, template="a photo of {}"):
        key = (tuple(str(l) for l in labels), template)
        if key in self._tcache:
            return self._tcache[key]
        prompts = [template.format(str(l).replace('_', ' ')) for l in labels]
        # CLIP's text context length is 77 tokens; truncate so long inputs (e.g. a full
        # product description) don't blow past max_position_embeddings and crash.
        inp = self.proc(text=prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=77).to(self.device)
        with self.torch.no_grad():
            out = self.model.text_model(input_ids=inp["input_ids"],
                                        attention_mask=inp["attention_mask"])
            v = self.model.text_projection(out.pooler_output)
        return self._put(self._tcache, self._TXT_CACHE_CAP, key, self._norm(v))


# ---------------------------------------------------------------------------
# Real TEXT encoder (sentence-transformers), lazy + process-cached
#
# The text dual of ClipEncoder. It exists because the previous text classifier scored
# by TOKEN OVERLAP, which is not a semantic signal at all: over 1,000 real movie
# reviews, scoring against the bare labels ["positive","negative"] put 998 of them at
# exactly 0.0 — only the two reviews that literally contain the word "positive" or
# "negative" scored anything. A ranking metric over a score with two distinct values is
# noise, which is why every Spearman and ARI query lagged.
# ---------------------------------------------------------------------------

DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_TEXT_ENCODER_CACHE = {}


def get_text_encoder(model_id=DEFAULT_TEXT_MODEL):
    if model_id not in _TEXT_ENCODER_CACHE:
        _TEXT_ENCODER_CACHE[model_id] = TextEncoder(model_id)
    return _TEXT_ENCODER_CACHE[model_id]


class TextEncoder:
    """Sentence embeddings with mean pooling + L2 normalization.

    Mirrors ClipEncoder deliberately: same lazy construction, same bounded caches, and
    a batched `encode_texts` that is the exact dual of `encode_images`. Corpus-scale
    work MUST go through the batched call — encoding row by row is the mistake that
    made the image path slow (638 single-image forward passes at ~24 ms each).
    """

    _CACHE_CAP = 4096          # label/description/alias side: small set, reused per row

    def __init__(self, model_id=DEFAULT_TEXT_MODEL, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.dim = int(self.model.config.hidden_size)
        self._cache = {}

    def _norm(self, t):
        return (t / t.norm(dim=-1, keepdim=True)).detach().cpu().numpy()

    def _forward(self, texts):
        """Mean-pool the token states under the attention mask, then normalize.

        Mean pooling (not the CLS/pooler output) is what all-MiniLM-L6-v2 was trained
        to be read with; using pooler_output silently degrades the embedding.
        """
        inp = self.tok(list(texts), return_tensors="pt", padding=True,
                       truncation=True, max_length=512).to(self.device)
        with self.torch.no_grad():
            out = self.model(**inp).last_hidden_state
        mask = inp["attention_mask"].unsqueeze(-1).to(out.dtype)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self._norm(pooled)

    def encode_texts(self, texts, batch=256):
        """Batched corpus encoding → [N, D], row-aligned with `texts`.

        The dual of `encode_images`. Empty/missing strings become zero rows so the
        matrix keeps its alignment rather than shifting every later row.
        """
        texts = [("" if t is None else str(t)) for t in texts]
        out = np.zeros((len(texts), self.dim), np.float32)
        idx = [i for i, t in enumerate(texts) if t.strip()]
        for i in range(0, len(idx), batch):
            chunk = idx[i:i + batch]
            out[chunk] = self._forward([texts[j] for j in chunk])
        return out

    def encode_cached(self, texts):
        """Encode with a bounded cache — for the LABEL side, which repeats every row."""
        texts = [str(t) for t in texts]
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vectors = self._forward(missing)
            if len(self._cache) + len(missing) > self._CACHE_CAP:
                self._cache.clear()          # same cheap bounded eviction as ClipEncoder
            for text, vector in zip(missing, vectors):
                self._cache[text] = vector
        return np.stack([self._cache[t] for t in texts])


# ---------------------------------------------------------------------------
# ③ Detectors (YOLO) — object/species presence & counting
# ---------------------------------------------------------------------------

def _unpack_det(d):
    """A detection is (name, conf) or (name, conf, box) — tolerate both so that
    third-party/mock detectors written against the older 2-tuple shape keep working."""
    return (d[0], d[1], d[2] if len(d) >= 3 else None)


def detect(img_path, classes, detector, min_conf=0.25):
    """`detector.detect(img_path, min_conf) -> list[(class_name, conf[, box])]`.
    Returns (present classes ⊆ `classes`, max confidence, per-class counts)."""
    conf_by, count_by = {}, {}
    for d in detector.detect(img_path, min_conf):
        name, c, _box = _unpack_det(d)
        if name in classes:
            conf_by[name] = max(conf_by.get(name, 0.0), c)
            count_by[name] = count_by.get(name, 0) + 1
    found = [c for c in classes if c in conf_by]
    return found, (max(conf_by.values()) if conf_by else 0.0), count_by


def detect_boxes(img_path, classes, detector, min_conf=0.25):
    """OpImgObj proper: one row PER INSTANCE — [(label, conf, (x1,y1,x2,y2))], sorted by
    confidence. `classes=None` keeps every detected class. Detections whose backend gave
    no box are dropped (a box is the whole point of this entry point)."""
    rows = []
    for d in detector.detect(img_path, min_conf):
        name, c, box = _unpack_det(d)
        if box is not None and (classes is None or name in classes):
            rows.append((name, float(c), tuple(float(v) for v in box)))
    return sorted(rows, key=lambda r: -r[1])


def detector_classes(detector):
    """The detector's CLOSED vocabulary (YOLOv8n = COCO-80). Anything outside it can
    never be detected, so callers surface that instead of silently returning nothing."""
    names = getattr(getattr(detector, "model", None), "names", None)
    if isinstance(names, dict):
        return [str(v) for v in names.values()]
    if isinstance(names, (list, tuple)):
        return [str(v) for v in names]
    return []


_DETECTOR_CACHE = {}

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _resolve_weights(model_id):
    """Prefer the weights bundled in `vadar/models/` over ultralytics' cwd-relative
    lookup, so a generated program finds them from whatever directory it runs in."""
    bundled = os.path.join(MODELS_DIR, os.path.basename(str(model_id)))
    return bundled if os.path.exists(bundled) else model_id


def get_detector(model_id="yolov8n.pt"):
    if model_id not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[model_id] = YoloDetector(model_id)
    return _DETECTOR_CACHE[model_id]


class YoloDetector:
    def __init__(self, model_id="yolov8n.pt"):
        from ultralytics import YOLO
        self.model = YOLO(_resolve_weights(model_id))

    def detect(self, img_path, min_conf=0.25):
        res = self.model.predict(img_path, verbose=False, conf=min_conf)[0]
        names = res.names
        return [(names[int(b.cls)], float(b.conf), tuple(b.xyxy[0].tolist()))
                for b in res.boxes]


# ---------------------------------------------------------------------------
# ③ Open-vocabulary detector (OWLv2) — the prompt IS the vocabulary
# ---------------------------------------------------------------------------

def detect_open_boxes(src, prompts, detector, min_conf=0.1):
    """OpImgObj with an OPEN vocabulary: [(label, conf, (x1,y1,x2,y2))], most confident
    first. Where `detect_boxes` can only answer for the detector's fixed class list,
    here the prompts ARE the class list, so a species or a car part is detectable."""
    return detector.detect_prompts(src, list(prompts), min_conf)


_OPEN_DETECTOR_CACHE = {}


def get_open_detector(model_id="google/owlv2-base-patch16-ensemble"):
    if model_id not in _OPEN_DETECTOR_CACHE:
        _OPEN_DETECTOR_CACHE[model_id] = OwlDetector(model_id)
    return _OPEN_DETECTOR_CACHE[model_id]


class OwlDetector:
    """OWLv2 open-vocabulary detection. Optional: `transformers` must be able to load the
    weights, and callers degrade to [] with a warning when it cannot."""

    def __init__(self, model_id="google/owlv2-base-patch16-ensemble"):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.proc = Owlv2Processor.from_pretrained(model_id)

    def detect_prompts(self, src, prompts, min_conf=0.1):
        im = _open(src)
        queries = [[f"a photo of a {p}" for p in prompts]]
        inp = self.proc(text=queries, images=im, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inp)
        sizes = self.torch.tensor([[im.size[1], im.size[0]]]).to(self.device)
        # The post-process helper was renamed across transformers releases.
        post = (getattr(self.proc, "post_process_grounded_object_detection", None)
                or self.proc.post_process_object_detection)
        res = post(outputs=out, target_sizes=sizes, threshold=min_conf)[0]
        rows = [(prompts[int(l)], float(s), tuple(float(v) for v in b))
                for s, l, b in zip(res["scores"], res["labels"], res["boxes"])]
        return sorted(rows, key=lambda r: -r[1])


# ---------------------------------------------------------------------------
# ③ Domain classifiers (chest X-ray) — pathology probabilities
# ---------------------------------------------------------------------------

def domain_classify(img_path, model, positive_labels, threshold=0.5):
    """`model.probs(img_path) -> dict[label->prob]`. Returns (yes/no, score) where
    score = max prob over `positive_labels` (the 'sick/abnormal' pathologies)."""
    p = model.probs(img_path)
    score = max((p.get(l, 0.0) for l in positive_labels), default=0.0)
    return ("yes" if score >= threshold else "no"), float(score)


_DOMAIN_CACHE = {}


def get_domain_model(model_id):
    """model_id like 'torchxrayvision:densenet121-res224-all'."""
    if model_id not in _DOMAIN_CACHE:
        kind, _, name = model_id.partition(":")
        if kind == "torchxrayvision":
            _DOMAIN_CACHE[model_id] = XrayClassifier(name or "densenet121-res224-all")
        else:
            raise ValueError(f"unknown domain model {model_id!r}")
    return _DOMAIN_CACHE[model_id]


class XrayClassifier:
    def __init__(self, weights="densenet121-res224-all"):
        import torch
        import torchxrayvision as xrv  # pip install torchxrayvision
        self.torch, self.xrv = torch, xrv
        self.model = xrv.models.DenseNet(weights=weights).eval()

    def probs(self, img_path):
        img = np.asarray(_open(img_path, "L"), dtype=np.float32)
        img = self.xrv.datasets.normalize(img, 255)          # → [-1024, 1024]
        img = self.xrv.datasets.XRayCenterCrop()(img[None, ...])
        t = self.torch.from_numpy(img)[None, ...]            # [1,1,H,W]
        with self.torch.no_grad():
            out = self.model(t)[0]
        return {p: float(v) for p, v in zip(self.model.pathologies, out) if p}


# ---------------------------------------------------------------------------
# ④ OCR (easyocr) — process-wide, like every other heavy model here
# ---------------------------------------------------------------------------
#
# This cache is module-level for the same reason _ENCODER_CACHE / _DETECTOR_CACHE /
# _DOMAIN_CACHE are. OCR used to be memoized on the per-patch `ctx` instead, and a
# generated solver that builds one ImagePatch per image (mmqa q2a: 200 of them) got a
# FRESH ctx per patch, so easyocr.Reader was constructed once per image and never
# freed: 275s of a 288s run was model loading, and once the accumulated readers filled
# the GPU the last 68 of 200 images failed OCR outright and were silently dropped from
# the result. Standalone, the same 200 images OCR in 9.2s with zero failures.

_OCR_CACHE = {}
_OCR_WARNED = set()


def get_ocr_reader(langs=("en",), gpu=None):
    """The shared easyocr reader, built at most once per (langs, device).

    Deliberately NOT `cudnn_benchmark=True`, despite an OCR-only microbenchmark showing
    1.14x (8.59s -> 7.56s, text byte-identical). Inside the real solver it was 5.9x
    SLOWER: 18.5s -> 108.5s on mmqa q2a. easyocr sets the flag PROCESS-WIDE, so it also
    governs the interleaved CLIP calls, and cuDNN re-runs its exhaustive algorithm
    search on every new input shape — a 200-image corpus has ~200 distinct shapes, and
    the autotune workspaces have to fit alongside a co-resident vLLM holding ~83 GB per
    device. The microbenchmark missed all of that by running OCR alone.

    Other easyocr knobs measured on the same corpus and rejected: `batch_size` is ~1.45x
    but changed the recognized text on 39 of 200 images, `workers>0` is 16x slower (a
    DataLoader spawn per call), `readtext_batched` needs 10-26 GB that vLLM leaves
    unavailable.

    Returns None — warning ONCE — when easyocr is not installed. That distinction
    matters: OCR is often a plan's primary discriminative path, and a missing
    dependency that reads as "this image has no text" produces a plausible, wrong,
    exit-code-0 result.
    """
    key = (tuple(langs), gpu)
    if key in _OCR_CACHE:
        return _OCR_CACHE[key]
    try:
        import easyocr
    except Exception as e:               # noqa: BLE001 — optional heavy dependency
        if "load" not in _OCR_WARNED:
            _OCR_WARNED.add("load")
            print(f"[vadar] OCR unavailable ({e}) — read_text returns \"\" and "
                  f"best_ocr_match returns \"none\" for EVERY image. Install easyocr, "
                  f"or bind classify/classify_or_none instead of an OCR path.")
        _OCR_CACHE[key] = None
        return None
    if gpu is None:
        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
    _OCR_CACHE[key] = easyocr.Reader(list(langs), gpu=gpu, verbose=False)
    return _OCR_CACHE[key]
