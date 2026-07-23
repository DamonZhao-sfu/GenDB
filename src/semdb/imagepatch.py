#!/usr/bin/env python3
"""
imagepatch.py — the base vision-primitive API that agent-SYNTHESIZED extraction code
composes (ViperGPT `ImagePatch` shape; VADAR predefined-module role). Each method is a
thin adapter over the existing `semvision` proxies — NO new models:

  classify / best_text_match / verify_property / score  -> CLIP  (semvision.clip_*)
  dominant_colors                                        -> HSV CV (semvision.cv_*)
  read_text                                              -> OCR    (easyocr, lazy)
  crop                                                   -> PIL geometry

The synthesized `extract(patch) -> {field: value, "conf": float}` calls ONLY these,
so the value space stays bounded (real field values, no free-form VLM). `ctx` carries
the loaded CLIP encoder + palette, built once per run by the engine.
"""
import os
import semvision


class ImagePatch:
    def __init__(self, image_path, ctx, box=None):
        self.image_path = image_path
        self.ctx = ctx                      # {"encoder": CLIP, "palette": dict|None}
        self.box = box                      # (l, low, r, up) or None (full image)

    # --- CLIP: return the winning VALUE (a real field), not a score ---------
    def classify(self, options, template="a photo of {}"):
        """Best-matching option from a closed value space (enum or DB column values)."""
        label, _ = semvision.clip_classify(self.image_path, list(options),
                                           self.ctx["encoder"], template)
        return label

    def best_text_match(self, options, template="{}"):
        return self.classify(options, template)

    def verify_property(self, prop, template="a photo of {}"):
        """True iff the patch matches `prop` better than its negation."""
        label, _ = semvision.clip_classify(self.image_path, [prop, f"not {prop}"],
                                           self.ctx["encoder"], template)
        return label == prop

    def score(self, text, template="a photo of {}"):
        """Raw CLIP image-text similarity in [0,1] (for yes/no thresholds)."""
        return semvision.clip_match(self.image_path, template.format(text), self.ctx["encoder"])

    # --- CV: colors present (incl. pale accents) ----------------------------
    def dominant_colors(self, min_frac=0.06):
        colors, _ = semvision.cv_dominant_colors(self.image_path, self.ctx.get("palette"),
                                                 min_frac=min_frac)
        return colors

    # --- OCR (printed text; stylized logos are unreliable — prefer classify) -
    def read_text(self):
        try:
            import easyocr
            if "ocr" not in self.ctx:
                try:
                    import torch
                    gpu = torch.cuda.is_available()
                except Exception:
                    gpu = False
                self.ctx["ocr"] = easyocr.Reader(["en"], gpu=gpu, verbose=False)
            return " ".join(self.ctx["ocr"].readtext(self.image_path, detail=0))
        except Exception:
            return ""

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

    def crop(self, left, lower, right, upper):
        return ImagePatch(self.image_path, self.ctx, box=(left, lower, right, upper))
