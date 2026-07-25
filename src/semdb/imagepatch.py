#!/usr/bin/env python3
"""
imagepatch.py — the base vision-primitive API that agent-SYNTHESIZED extraction code
composes (ViperGPT `ImagePatch` shape; VADAR predefined-module role). Each method is a
thin adapter over the existing `semvision` proxies — NO new models:

  classify / best_text_match / verify_property / score  -> CLIP  (semvision.clip_*)
  pair_score                                            -> CLIP  image-image cosine
  dominant_colors                                        -> HSV CV (semvision.cv_*)
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
import semvision


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
        self.ctx = ctx                      # {"encoder": CLIP, "palette": dict|None}
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

    # --- CLIP: return the winning VALUE (a real field), not a score ---------
    def classify(self, options, template="a photo of {}"):
        """Best-matching option from a closed value space (enum or DB column values)."""
        label, _ = semvision.clip_classify(self._src(), list(options),
                                           self.ctx["encoder"], template, key=self._key)
        return label

    def best_text_match(self, options, template="{}"):
        return self.classify(options, template)

    def verify_property(self, prop, template="a photo of {}"):
        """True iff the patch matches `prop` better than its negation."""
        label, _ = semvision.clip_classify(self._src(), [prop, f"not {prop}"],
                                           self.ctx["encoder"], template, key=self._key)
        return label == prop

    def score(self, text, template="a photo of {}"):
        """Raw CLIP image-text similarity in [0,1] (for yes/no thresholds)."""
        return semvision.clip_match(self._src(), template.format(text),
                                    self.ctx["encoder"], key=self._key)

    def pair_score(self, other):
        """IMAGE-IMAGE similarity in [0,1] against another patch — the primitive for an
        image-to-image join, dedup, or top-k. (`score` compares against TEXT.)"""
        return semvision.img_pair_score(self._src(), other._src(), self.ctx["encoder"],
                                        key_a=self._key, key_b=other._key)

    # --- CV: colors present (incl. pale accents) ----------------------------
    def dominant_colors(self, min_frac=0.06, center_frac=1.0):
        colors, _ = semvision.cv_dominant_colors(self._src(), self.ctx.get("palette"),
                                                 min_frac=min_frac, center_frac=center_frac)
        return colors

    # --- OCR (printed text; stylized logos are unreliable — prefer classify) -
    def _ocr(self):
        if "ocr" not in self.ctx:
            import easyocr
            try:
                import torch
                gpu = torch.cuda.is_available()
            except Exception:
                gpu = False
            self.ctx["ocr"] = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        return self.ctx["ocr"]

    def read_text_boxes(self, min_conf=0.0):
        """OpImgOCR proper: [{"text", "box": (l,t,r,b) in ABSOLUTE image pixels, "score"}].
        `min_conf=0.0` keeps every detection, so `read_text()` is unchanged by default;
        raise it to trade recall for precision."""
        try:
            import numpy as np
            src = self._src()
            reader = self._ocr()
            raw = reader.readtext(src if isinstance(src, str) else np.asarray(src), detail=1)
        except Exception:
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

    def best_ocr_match(self, options, cutoff=0.6, min_len=3):
        """Read the image text (OCR) and match it to the closest VALUE in a value space
        — the field. Wins over `classify` for wordmark logos (airline names). Strict, to
        avoid false positives on non-logo images: requires a strong fuzzy match OR that
        the option's DISTINCTIVE (non-generic) tokens actually appear in the OCR text."""
        import difflib
        text = self.read_text().lower().strip()
        if len(text) < min_len:
            return "none"
        by_lower = {o.lower(): o for o in options}
        m = difflib.get_close_matches(text, list(by_lower), n=1, cutoff=cutoff)
        if m:
            return by_lower[m[0]]
        toks = set(text.split())
        best, best_s = "none", 0.0
        for o in options:
            distinctive = [w for w in o.lower().split() if w not in self._GENERIC_TOKENS]
            if not distinctive:
                continue
            hit = sum(1 for w in distinctive if w in toks) / len(distinctive)
            if hit >= 0.6 and hit > best_s:            # its distinctive name must be read
                best, best_s = o, hit
        return best

    # --- region decomposition ----------------------------------------------
    _OOV_WARNED = set()

    def find(self, object_prompt, min_conf=0.25):
        """Detected instances of `object_prompt` as SUB-PATCHES (YOLO), highest confidence
        first. `len(...)` counts, truthiness tests presence, and each element can be
        further classified / read. Returns [] for a class outside the detector's closed
        vocabulary (see `semvision.detector_classes`) — it warns rather than pretending."""
        det = self.ctx.get("detector")
        if det is None:
            det = self.ctx["detector"] = semvision.get_detector()
        want = str(object_prompt).strip().lower()
        vocab = {c.lower(): c for c in semvision.detector_classes(det)}
        if vocab and want not in vocab:
            if want not in ImagePatch._OOV_WARNED:
                ImagePatch._OOV_WARNED.add(want)
                print(f"[imagepatch] find({object_prompt!r}): outside the detector's "
                      f"vocabulary ({len(vocab)} classes) — use classify/verify_property instead")
            return []
        cls = [vocab[want]] if vocab else [object_prompt]
        rows = semvision.detect_boxes(self._src(), cls, det, min_conf=min_conf)
        return [self._child(box) for _label, _conf, box in rows]

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
