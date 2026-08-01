#!/usr/bin/env python3
"""
vadar/imagepatch.py — the base vision-primitive API that agent-SYNTHESIZED extraction
code composes (ViperGPT `ImagePatch` shape; VADAR predefined-module role). Each method
is a thin adapter over the `vadar.backend` proxies — NO new models:

  classify / best_text_match / verify_property / score  -> CLIP  (backend.clip_*)
  pair_score                                            -> CLIP  image-image cosine
  dominant_colors                                        -> HSV CV (backend.cv_*)
  read_text / read_text_boxes                            -> OCR    (easyocr, lazy)
  find / regions_grid / regions_center / crop            -> region decomposition

The synthesized `extract(patch) -> {field: value, "conf": float}` calls ONLY these,
so the value space stays bounded (real field values, no free-form VLM). `ctx` carries
the loaded CLIP encoder + palette, built once per run by the engine.

REGIONS. A patch carries an optional `box` in ABSOLUTE image pixels, PIL order
`(left, top, right, bottom)` with the origin at the TOP-LEFT. Every primitive runs on
that region — `crop(...).dominant_colors()` reads the crop, not the whole frame. Region
patches compose: `patch.crop(...).crop(...)` and `patch.find(...)[0].read_text()` work,
and `find`/`read_text_boxes` report boxes back in absolute image coordinates.
"""
import weakref
from collections.abc import Mapping

import numpy as np

from . import backend

#: encoder -> its one normalized ctx. Weak so an encoder that goes out of scope does
#: not pin its caches (image embeddings) for the life of the process.
_CTX_BY_ENCODER: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _as_ctx(ctx):
    """Accept either the ctx MAPPING or a bare encoder.

    `ctx` is documented as `{"encoder": ..., "palette": ...}`, but generated solvers
    regularly write `ImagePatch(path, get_encoder(model))` — the two calls sit on
    adjacent lines of the prompt skeleton. That used to raise
    `TypeError: 'ClipEncoder' object is not subscriptable` on EVERY row, deep inside
    the first primitive call; wrapped in the mandated per-row try/except it became a
    silent all-"none" result with exit code 0 (mmqa q2a: 200/200 rows lost, F1 0).

    A bare encoder is unambiguous, so normalize instead of failing: the caller's intent
    is never in doubt, and the alternative is a whole query's iteration budget spent
    rediscovering a two-character mistake.

    The normalized ctx is MEMOIZED per encoder identity. Returning a fresh dict each
    time silently un-shared every cache the ctx exists to hold — `_size_cache`,
    `palette`, `domain` and (before it moved to `backend`) the OCR reader — because a
    solver that writes `ImagePatch(path, encoder)` inside its row loop builds one ctx
    per IMAGE. The accommodation above is right; making it allocate was the bug.
    """
    if isinstance(ctx, Mapping):
        return ctx
    try:
        shared = _CTX_BY_ENCODER.get(ctx)
        if shared is None:
            shared = _CTX_BY_ENCODER[ctx] = {"encoder": ctx, "palette": None}
        return shared
    except TypeError:
        # Not weak-referenceable or not hashable — correctness first, share nothing.
        return {"encoder": ctx, "palette": None}


def _norm_box(box, size):
    """Resolve a user box to absolute int pixels, clamped to `size`. Values that are all
    within [0,1] are read as FRACTIONS of the patch (an agent rarely knows the pixel
    dimensions); anything larger is read as pixels."""
    w, h = size
    l, t, r, b = (float(v) for v in box)
    if max(abs(l), abs(t), abs(r), abs(b)) <= 1.0:
        l, t, r, b = l * w, t * h, r * w, b * h
    l, r = sorted((int(round(l)), int(round(r))))
    t, b = sorted((int(round(t)), int(round(b))))
    l, t = max(0, min(l, w - 1)), max(0, min(t, h - 1))
    r, b = max(l + 1, min(r, w)), max(t + 1, min(b, h))
    return (l, t, r, b)


class ImagePatch:
    def __init__(self, image_path, ctx, box=None):
        self.image_path = image_path
        # {"encoder": CLIP, "palette": dict|None} — a bare encoder is also accepted and
        # normalized here, see `_as_ctx`.
        self.ctx = _as_ctx(ctx)
        self.box = tuple(box) if box else None   # absolute (left, top, right, bottom)

    # --- geometry -----------------------------------------------------------
    @property
    def _key(self):
        """Region identity — lets the CLIP encoder cache repeated scoring of one patch."""
        return (self.image_path, self.box)

    def _image_size(self):
        cache = self.ctx.setdefault("_size_cache", {})
        if self.image_path not in cache:
            from PIL import Image
            with Image.open(self.image_path) as im:   # lazy: reads the header, not pixels
                cache[self.image_path] = im.size
        return cache[self.image_path]

    @property
    def size(self):
        """(width, height) of THIS patch."""
        if self.box is None:
            return self._image_size()
        l, t, r, b = self.box
        return (r - l, b - t)

    @property
    def bbox(self):
        """This patch's box in ABSOLUTE image pixels — `(0, 0, w, h)` for a whole image.
        `OpImgRegion`'s BBox output; a region's RegIdx is its index in the producing list."""
        if self.box is not None:
            return self.box
        w, h = self._image_size()
        return (0, 0, w, h)

    def _src(self):
        """What to hand a proxy: the path (whole image, zero-copy) or the cropped region."""
        if self.box is None:
            return self.image_path
        from PIL import Image
        return Image.open(self.image_path).convert("RGB").crop(self.box)

    def _child(self, box_in_self):
        """Build a sub-patch from a box expressed in THIS patch's coordinate frame."""
        l, t, r, b = _norm_box(box_in_self, self.size)
        ox, oy = (self.box[0], self.box[1]) if self.box else (0, 0)
        return ImagePatch(self.image_path, self.ctx, box=(l + ox, t + oy, r + ox, b + oy))

    def crop(self, left, top, right, bottom):
        """A sub-region of this patch. PIL order and TOP-LEFT origin: (left, top, right,
        bottom). Accepts pixels, or fractions of this patch when all four are in [0,1]."""
        return self._child((left, top, right, bottom))

    # --- CLIP: (VALUE, Score) — the paper's R_Cls(Label, Score) output schema ---
    def classify_detail(self, options, template="a photo of {}"):
        """OpImgCls: (best-matching VALUE, confidence in [0,1]). The score is comparable
        ACROSS ROWS for this operator — not against another operator's score."""
        label, s = backend.clip_classify(self._src(), list(options),
                                           self.ctx["encoder"], template, key=self._key)
        return label, float(s)

    def classify(self, options, template="a photo of {}"):
        """Best-matching option from a closed value space (enum or DB column values)."""
        return self.classify_detail(options, template)[0]

    def best_text_match(self, options, template="{}"):
        return self.classify(options, template)

    def verify_detail(self, prop, template="a photo of {}"):
        """(True iff the patch matches `prop` better than its negation, confidence of the
        WINNING side)."""
        label, s = backend.clip_classify(self._src(), [prop, f"not {prop}"],
                                           self.ctx["encoder"], template, key=self._key)
        return label == prop, float(s)

    def verify_property(self, prop, template="a photo of {}"):
        """True iff the patch matches `prop` better than its negation."""
        return self.verify_detail(prop, template)[0]

    def classify_multi(self, options, thresh=0.5, template="a photo of {}"):
        """OpImgCls, multilabel: (every option scoring over `thresh`, confidence). Use when
        the field is a SET (several attributes true at once), not a single enum value."""
        return backend.clip_multilabel(self._src(), list(options), self.ctx["encoder"],
                                         thresh, template, key=self._key)

    def score(self, text, template="a photo of {}"):
        """RELATIVE CLIP image-text similarity. Use it to RANK or COMPARE, never to
        threshold on an absolute value.

        The nominal range is [0,1] but the EFFECTIVE range is roughly 0.54..0.67:
        `clip_match` rescales cosine from [-1,1], and real CLIP image-text cosines sit
        around 0.1..0.35. So `score(...) >= 0.5` is TRUE FOR EVERY PAIR, including
        gibberish — measured over 60 (image, text) pairs, all 60 cleared 0.5, and the
        nonsense string "xyzzy plugh frobnicate" outscored a coherent but unrelated
        description. A query that thresholded this at 0.5 returned all 1,000 candidate
        pairs against a 4-row ground truth.

        FOR A YES/NO DECISION USE `verify_detail(prop)`, which contrasts `prop` against
        "not prop" and is therefore calibrated: on the same 20 images it accepted 5 for
        a plausible description and 0 for an absurd one.
        """
        return backend.clip_match(self._src(), template.format(text),
                                    self.ctx["encoder"], key=self._key)

    def pair_score(self, other):
        """IMAGE-IMAGE similarity in [0,1] against another patch — the primitive for an
        image-to-image join, dedup, or top-k. (`score` compares against TEXT.)"""
        return backend.img_pair_score(self._src(), other._src(), self.ctx["encoder"],
                                        key_a=self._key, key_b=other._key)

    # --- Latent: OpImgEmbed + vectorized top-k ------------------------------
    def embed(self):
        """OpImgEmbed: this patch's unit-norm vector as a plain float list (JSON-safe, so
        it can be materialized into a column)."""
        return [float(v) for v in
                backend.embed_image(self._src(), self.ctx["encoder"], self._key)]

    def topk_similar(self, others, k=5):
        """Rank other patches against this one by IMAGE-IMAGE similarity, returning
        [(index into `others`, score)] best first. The vectorized form of `pair_score` —
        encodes each candidate once, then a single matmul. Use for an image-to-image
        join, dedup, or top-k."""
        others = list(others)
        if not others:
            return []
        q = backend.embed_image(self._src(), self.ctx["encoder"], self._key)
        mat = np.stack([backend.embed_image(o._src(), self.ctx["encoder"], o._key)
                        for o in others])
        return backend.topk_similar(q, mat, k)

    def topk_text(self, texts, k=5, template="a photo of {}"):
        """Rank candidate TEXTS against this patch, returning [(text, score)] best first.
        `classify` keeps only the argmax; this keeps the top-k WITH scores — the candidate
        set a cascade needs before a heavier verifier runs on a few options."""
        texts = [str(t) for t in texts]
        if not texts:
            return []
        enc = self.ctx["encoder"]
        q = backend.embed_image(self._src(), enc, self._key)
        rows = backend.topk_similar(q, enc.encode_text(texts, template), k)
        return [(texts[i], s) for i, s in rows]

    # --- CV: colors present (incl. pale accents) ----------------------------
    def dominant_colors(self, min_frac=0.06, center_frac=1.0):
        colors, _ = backend.cv_dominant_colors(self._src(), self.ctx.get("palette"),
                                                 min_frac=min_frac, center_frac=center_frac)
        return colors

    # --- domain specialists (a task-tuned model, not a zero-shot one) --------
    def domain_classify(self, model_id, labels, threshold=0.5):
        """A domain-specialist classifier — e.g.
        `domain_classify("torchxrayvision:densenet121-res224-all", ["Pneumonia"])`.
        Returns ("yes"|"no", the max probability over `labels`). The model is loaded once
        and cached in `ctx`, so a corpus scan pays for it a single time."""
        cache = self.ctx.setdefault("domain", {})
        if model_id not in cache:
            cache[model_id] = backend.get_domain_model(model_id)
        return backend.domain_classify(self._src(), cache[model_id], list(labels), threshold)

    # --- OCR (printed text; stylized logos are unreliable — prefer classify) -
    #: Same set object as `backend._OCR_WARNED`, so clearing it here (tests do) also
    #: re-arms the loader's warn-once.
    _OCR_WARNED = backend._OCR_WARNED

    def _ocr(self):
        """The OCR reader — shared process-wide, or an explicitly injected one.

        `ctx["ocr"]` still wins when a caller sets it, which is how tests inject a fake
        reader and how a run can pin a non-default language set. Everything else goes to
        `backend.get_ocr_reader()`.

        The reader deliberately does NOT get memoized back into `self.ctx`. It used to
        be, and a generated solver that builds one ImagePatch per image gets a fresh
        ctx per patch (see `_as_ctx`) — so the "load once per run" cache degraded to
        "load once per IMAGE": on mmqa q2a that was 200 easyocr.Reader constructions,
        275s of a 288s run, and a GPU filled with leaked readers until the last 68 of
        200 images failed OCR and were silently dropped.

        Distinguishing "OCR is unavailable" from "this image has no text" matters more
        here than it looks: OCR against the runtime value space is the primitive the
        planning rules recommend for wordmark logos, so it is often a plan's PRIMARY
        discriminative path."""
        if "ocr" in self.ctx:
            return self.ctx["ocr"]
        return backend.get_ocr_reader()

    def read_text_boxes(self, min_conf=0.0):
        """OpImgOCR proper: [{"text", "box": (l,t,r,b) in ABSOLUTE image pixels, "score"}].
        `min_conf=0.0` keeps every detection, so `read_text()` is unchanged by default;
        raise it to trade recall for precision."""
        reader = self._ocr()
        if reader is None:
            return []
        try:
            import numpy as np
            src = self._src()
            raw = reader.readtext(src if isinstance(src, str) else np.asarray(src), detail=1)
        except Exception as e:               # noqa: BLE001 — one unreadable image
            if "read" not in ImagePatch._OCR_WARNED:
                ImagePatch._OCR_WARNED.add("read")
                print(f"[imagepatch] OCR failed on {self.image_path!r} ({e})")
            return []
        ox, oy = (self.box[0], self.box[1]) if self.box else (0, 0)
        out = []
        for quad, text, conf in raw:
            if float(conf) < min_conf:
                continue
            xs = [float(p[0]) for p in quad]
            ys = [float(p[1]) for p in quad]
            out.append({"text": text,
                        "box": (min(xs) + ox, min(ys) + oy, max(xs) + ox, max(ys) + oy),
                        "score": float(conf)})
        return out

    def read_text(self, min_conf=0.0):
        """Raw OCR text of the patch, reading order preserved."""
        return " ".join(d["text"] for d in self.read_text_boxes(min_conf))

    _GENERIC_TOKENS = {"air", "airlines", "airways", "airline", "aviation", "co", "ltd",
                       "the", "of", "and", "group", "international"}

    def best_ocr_match_detail(self, options, cutoff=0.6, min_len=3):
        """Read the image text (OCR) and match it to the closest VALUE in a value space —
        returns (the field value, match strength in [0,1]); ("none", 0.0) when nothing
        matches. Wins over `classify` for wordmark logos (airline names). Strict, to avoid
        false positives on non-logo images: requires a strong fuzzy match OR that the
        option's DISTINCTIVE (non-generic) tokens actually appear in the OCR text."""
        import difflib
        text = self.read_text().lower().strip()
        if len(text) < min_len:
            return "none", 0.0
        by_lower = {o.lower(): o for o in options}
        m = difflib.get_close_matches(text, list(by_lower), n=1, cutoff=cutoff)
        if m:
            return by_lower[m[0]], float(difflib.SequenceMatcher(None, text, m[0]).ratio())
        toks = set(text.split())
        best, best_s = "none", 0.0
        for o in options:
            distinctive = [w for w in o.lower().split() if w not in self._GENERIC_TOKENS]
            if not distinctive:
                continue
            hit = sum(1 for w in distinctive if w in toks) / len(distinctive)
            if hit >= 0.6 and hit > best_s:            # its distinctive name must be read
                best, best_s = o, hit
        return best, float(best_s)

    def best_ocr_match(self, options, cutoff=0.6, min_len=3):
        """The matched VALUE only — see `best_ocr_match_detail` for the match strength."""
        return self.best_ocr_match_detail(options, cutoff, min_len)[0]

    # --- region decomposition ----------------------------------------------
    _OOV_WARNED = set()

    def _detect_rows(self, object_prompt, min_conf=0.25):
        """[(label, conf, box in THIS patch's frame)], best first — [] for a class outside
        the detector's closed vocabulary (it warns rather than pretending)."""
        det = self.ctx.get("detector")
        if det is None:
            det = self.ctx["detector"] = backend.get_detector()
        want = str(object_prompt).strip().lower()
        vocab = {c.lower(): c for c in backend.detector_classes(det)}
        if vocab and want not in vocab:
            if want not in ImagePatch._OOV_WARNED:
                ImagePatch._OOV_WARNED.add(want)
                print(f"[imagepatch] find({object_prompt!r}): outside the detector's "
                      f"vocabulary ({len(vocab)} classes) — use classify/verify_property, "
                      f"or find_open for an open-vocabulary detector")
            return []
        cls = [vocab[want]] if vocab else [object_prompt]
        return backend.detect_boxes(self._src(), cls, det, min_conf=min_conf)

    def find_detail(self, object_prompt, min_conf=0.25):
        """OpImgObj proper: [{"image": patch, "label", "box" (ABSOLUTE px), "score"}],
        most confident first — the paper's R_Obj(BBox, Label, Score)."""
        out = []
        for label, conf, box in self._detect_rows(object_prompt, min_conf):
            child = self._child(box)
            out.append({"image": child, "label": label, "box": child.box, "score": float(conf)})
        return out

    def find(self, object_prompt, min_conf=0.25):
        """Detected instances of `object_prompt` as SUB-PATCHES (YOLO), highest confidence
        first. `len(...)` counts, truthiness tests presence, and each element can be
        further classified / read. Returns [] for a class outside the detector's closed
        vocabulary (see `backend.detector_classes`) — it warns rather than pretending."""
        return [d["image"] for d in self.find_detail(object_prompt, min_conf)]

    # --- open vocabulary: the prompt IS the class ---------------------------
    _OPEN_WARNED = set()

    def _open_detector(self):
        """The open-vocabulary backend, loaded once per run. Returns None (warning once)
        when the optional weights are unavailable — a missing extra must degrade the
        query, not crash it."""
        det = self.ctx.get("open_detector")
        if det is not None:
            return det
        try:
            det = self.ctx["open_detector"] = backend.get_open_detector()
            return det
        except Exception as e:               # noqa: BLE001 — optional heavy dependency
            if "load" not in ImagePatch._OPEN_WARNED:
                ImagePatch._OPEN_WARNED.add("load")
                print(f"[imagepatch] no open-vocabulary detector available ({e}) — "
                      f"find_open returns []; use classify/verify_property instead")
            return None

    def find_open_detail(self, object_prompt, min_conf=0.1):
        """OpImgObj over an OPEN vocabulary: [{"image", "label", "box" (ABSOLUTE px),
        "score"}]. The prompt IS the class, so names outside the closed detector's
        COCO-80 (a species, a car part) are detectable here."""
        det = self._open_detector()
        if det is None:
            return []
        out = []
        for label, conf, box in backend.detect_open_boxes(
                self._src(), [str(object_prompt)], det, min_conf):
            child = self._child(box)
            out.append({"image": child, "label": label, "box": child.box, "score": float(conf)})
        return out

    def find_open(self, object_prompt, min_conf=0.1):
        """Instances of ANY object name as SUB-PATCHES, most confident first. Slower than
        `find` but not limited to a closed vocabulary."""
        return [d["image"] for d in self.find_open_detail(object_prompt, min_conf)]

    def regions_grid(self, rows=2, cols=2, overlap=0.0):
        """Split the patch into a rows×cols grid of sub-patches (row-major). `overlap` is
        the fraction of a cell each side extends by, so a target straddling a cut line is
        still whole in some cell. Deterministic and model-free — the cheapest way to run
        partitioned extraction over a large image."""
        out = []
        for i in range(int(rows)):
            for j in range(int(cols)):
                t, b = i / rows, (i + 1) / rows
                l, r = j / cols, (j + 1) / cols
                if overlap:
                    dy, dx = overlap / rows, overlap / cols
                    t, b, l, r = t - dy, b + dy, l - dx, r + dx
                out.append(self._child((max(0.0, l), max(0.0, t),
                                        min(1.0, r), min(1.0, b))))
        return out

    def regions_center(self, frac=0.6):
        """The central `frac` of the patch — excludes a product photo's white margin."""
        m = (1.0 - float(frac)) / 2.0
        return self._child((m, m, 1.0 - m, 1.0 - m))

    def propose_regions(self, max_regions=8, min_area_frac=0.01, method="contour"):
        """OpImgRegion: content-driven regions as SUB-PATCHES, largest first (a region's
        RegIdx is its index here, its BBox is `.bbox`). `method="contour"` splits on
        foreground connected components — deterministic and model-free, unlike
        `regions_grid`'s blind cut. Any other method warns and falls back to contour
        rather than pretending a learned proposer is installed."""
        if method != "contour":
            print(f"[imagepatch] propose_regions: method {method!r} is not available — "
                  f"using 'contour'")
        boxes = backend.propose_region_boxes(self._src(), max_regions, min_area_frac)
        return [self._child(b) for b in boxes]
