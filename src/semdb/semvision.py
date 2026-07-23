#!/usr/bin/env python3
"""
semvision.py — tiered, per-attribute IMAGE extraction proxies (non-VLM).

Each schema attribute carries an `extractor` spec ({tier, method, params, labels});
this engine runs the LIGHTEST proxy that answers it instead of one generative VLM:
  ① pure CV (dominant colors — numpy/PIL, zero model)
  ② CLIP zero-shot (transformers CLIP — classify / multilabel / match)
Later tiers (③ detector/domain, ② DINOv2 probe, ④ distilled) plug into `_run_attr`.

Mirrors semextract's driver hooks (map_columns/preprocess) and reuses
`semextract.resolve_image_path` + `_none_record`. Text extraction stays in semextract.
"""

import csv
import json
import os
import time

import numpy as np
from PIL import Image


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
                       sat_thresh=0.20, val_lo=0.12):
    """Return (colors ≥ min_frac of pixels, confidence). Chromatic pixels (saturation
    ≥ sat_thresh) map to a color by HSV HUE (so pale accents keep their hue); the rest
    bucket to black/gray/silver/white by lightness. Deterministic → conf 1.0."""
    im = Image.open(img_path).convert("RGB").resize((size, size))
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
# ② CLIP zero-shot — classify / multilabel / match (injectable encoder)
# ---------------------------------------------------------------------------

def _softmax(x, temp=0.01):
    x = np.asarray(x, np.float32) / temp
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def clip_classify(img_path, labels, encoder):
    iv = encoder.encode_image(img_path)          # [D], normalized
    tv = encoder.encode_text(list(labels))       # [L,D], normalized
    sims = tv @ iv                               # [L] cosine
    probs = _softmax(sims)
    j = int(np.argmax(probs))
    return labels[j], float(probs[j])


def clip_multilabel(img_path, labels, encoder, thresh=0.5):
    iv = encoder.encode_image(img_path)
    tv = encoder.encode_text(list(labels))
    sims = tv @ iv                               # cosine in [-1,1]
    probs = 1.0 / (1.0 + np.exp(-(sims - 0.2) / 0.05))   # sigmoid centered ~0.2 cos
    chosen = [l for l, p in zip(labels, probs) if p >= thresh]
    if chosen:
        conf = float(np.mean([p for l, p in zip(labels, probs) if p >= thresh]))
    else:
        conf = float(np.max(probs)) if len(probs) else 0.0
    return chosen, conf


def clip_match(img_path, text, encoder):
    iv = encoder.encode_image(img_path)
    tv = encoder.encode_text([text])[0]
    return float(np.clip((tv @ iv + 1.0) / 2.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Real CLIP encoder (transformers), lazy + process-cached
# ---------------------------------------------------------------------------

_ENCODER_CACHE = {}


def get_encoder(model_id="openai/clip-vit-base-patch32"):
    if model_id not in _ENCODER_CACHE:
        _ENCODER_CACHE[model_id] = ClipEncoder(model_id)
    return _ENCODER_CACHE[model_id]


class ClipEncoder:
    def __init__(self, model_id="openai/clip-vit-base-patch32", device=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)

    def _norm(self, t):
        return (t / t.norm(dim=-1, keepdim=True)).detach().cpu().numpy()

    def encode_image(self, img_path):
        # Compute via the vision tower + projection (version-robust: some transformers
        # builds make get_image_features return a model-output object, not a tensor).
        im = Image.open(img_path).convert("RGB")
        inp = self.proc(images=[im], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.vision_model(pixel_values=inp["pixel_values"])
            v = self.model.visual_projection(out.pooler_output)
        return self._norm(v)[0]

    def encode_text(self, labels):
        prompts = [f"a photo of {l.replace('_', ' ')}" for l in labels]
        inp = self.proc(text=prompts, return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            out = self.model.text_model(input_ids=inp["input_ids"],
                                        attention_mask=inp["attention_mask"])
            v = self.model.text_projection(out.pooler_output)
        return self._norm(v)


# ---------------------------------------------------------------------------
# ③ Detectors (YOLO) — object/species presence & counting
# ---------------------------------------------------------------------------

def detect(img_path, classes, detector, min_conf=0.25):
    """`detector.detect(img_path, min_conf) -> list[(class_name, conf)]`.
    Returns (present classes ⊆ `classes`, max confidence, per-class counts)."""
    conf_by, count_by = {}, {}
    for name, c in detector.detect(img_path, min_conf):
        if name in classes:
            conf_by[name] = max(conf_by.get(name, 0.0), c)
            count_by[name] = count_by.get(name, 0) + 1
    found = [c for c in classes if c in conf_by]
    return found, (max(conf_by.values()) if conf_by else 0.0), count_by


_DETECTOR_CACHE = {}


def get_detector(model_id="yolov8n.pt"):
    if model_id not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[model_id] = YoloDetector(model_id)
    return _DETECTOR_CACHE[model_id]


class YoloDetector:
    def __init__(self, model_id="yolov8n.pt"):
        from ultralytics import YOLO
        self.model = YOLO(model_id)

    def detect(self, img_path, min_conf=0.25):
        res = self.model.predict(img_path, verbose=False, conf=min_conf)[0]
        names = res.names
        return [(names[int(b.cls)], float(b.conf)) for b in res.boxes]


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
        img = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
        img = self.xrv.datasets.normalize(img, 255)          # → [-1024, 1024]
        img = self.xrv.datasets.XRayCenterCrop()(img[None, ...])
        t = self.torch.from_numpy(img)[None, ...]            # [1,1,H,W]
        with self.torch.no_grad():
            out = self.model(t)[0]
        return {p: float(v) for p, v in zip(self.model.pathologies, out) if p}


# ---------------------------------------------------------------------------
# Per-attribute dispatch
# ---------------------------------------------------------------------------

def _run_attr(image_path, attr, ctx):
    """Return (value, score) for one attribute per its extractor spec."""
    spec = attr.get("extractor") or {"tier": "vlm"}
    tier, method = spec.get("tier"), spec.get("method")
    params = spec.get("params") or {}
    is_list = "array" in attr.get("type", "") or "list" in attr.get("type", "")
    if tier == "cv" and method == "dominant_colors":
        colors, s = cv_dominant_colors(image_path, ctx.get("palette"),
                                       min_frac=params.get("min_frac", 0.08))
        return (colors if is_list else (colors[0] if colors else "none")), s
    if tier == "clip":
        enc = ctx["encoder"]
        if method == "classify":
            return clip_classify(image_path, spec["labels"], enc)
        if method == "multilabel":
            return clip_multilabel(image_path, spec["labels"], enc, params.get("thresh", 0.5))
        if method == "match":
            score = clip_match(image_path, params["text"], enc)
            thr = params.get("threshold")
            return (("yes" if score >= thr else "no") if thr is not None else score), score
    if tier == "detector":
        found, s, _counts = detect(image_path, spec["classes"], ctx["detector"],
                                   min_conf=params.get("min_conf", 0.25))
        if is_list:
            return found, s
        return ("yes" if found else "no"), s     # single-class presence
    if tier == "domain":
        model = ctx["domain"][spec["model"]]
        return domain_classify(image_path, model,
                               spec.get("labels") or spec.get("positive") or [],
                               params.get("threshold", 0.5))
    # unknown / vlm / dino / distilled (not implemented) → residual miss
    return ([] if is_list else "none"), 0.0


def extract_record(image_path, schema, ctx):
    rec, scores = {}, []
    for attr in schema.get("attributes", []):
        val, s = _run_attr(image_path, attr, ctx)
        rec[attr["name"]] = val
        scores.append(s)
    rec["conf"] = float(min(scores)) if scores else 0.0
    return rec


# ---------------------------------------------------------------------------
# extractor-spec validator (locks the Schema-Designer contract)
# ---------------------------------------------------------------------------

def validate_extractor_spec(attr):
    errs = []
    spec = attr.get("extractor")
    if not spec:
        return [f"attribute '{attr.get('name')}' missing 'extractor' spec"]
    tier = spec.get("tier")
    if tier not in ("cv", "clip", "dino", "detector", "domain", "distilled", "vlm"):
        errs.append(f"'{attr.get('name')}' unknown tier {tier!r}")
    if tier == "clip" and spec.get("method") in ("classify", "multilabel") and not spec.get("labels"):
        errs.append(f"'{attr.get('name')}' clip {spec.get('method')} needs 'labels'")
    if tier == "clip" and spec.get("method") == "match" and not (spec.get("params") or {}).get("text"):
        errs.append(f"'{attr.get('name')}' clip match needs params.text")
    if tier == "cv" and spec.get("method") != "dominant_colors":
        errs.append(f"'{attr.get('name')}' cv only supports method 'dominant_colors'")
    if tier == "detector" and not spec.get("classes"):
        errs.append(f"'{attr.get('name')}' detector needs 'classes'")
    if tier == "domain" and not spec.get("model"):
        errs.append(f"'{attr.get('name')}' domain needs 'model' (e.g. torchxrayvision:densenet121-res224-all)")
    if tier == "domain" and not (spec.get("labels") or spec.get("positive")):
        errs.append(f"'{attr.get('name')}' domain needs 'labels'/'positive' pathologies")
    return errs


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _tiers_used(schema):
    return {(a.get("extractor") or {}).get("tier") for a in schema.get("attributes", [])}


def run(driver, schema, table_path, out_path, *, image_dir=None,
        clip_model="openai/clip-vit-base-patch32", detector_model="yolov8n.pt",
        limit=0, palette=None):
    import semextract  # reuse resolve_image_path + _none_record
    t0 = time.time()
    rows = list(csv.DictReader(open(table_path)))
    if limit:
        rows = rows[:limit]
    cols = driver.map_columns(list(rows[0].keys()) if rows else [])
    id_col = cols["id"]
    tiers = _tiers_used(schema)
    ctx = {"encoder": get_encoder(clip_model) if "clip" in tiers else None,
           "palette": palette,
           "detector": get_detector(detector_model) if "detector" in tiers else None,
           "domain": {}}
    for a in schema.get("attributes", []):
        sp = a.get("extractor") or {}
        if sp.get("tier") == "domain" and sp.get("model") and sp["model"] not in ctx["domain"]:
            ctx["domain"][sp["model"]] = get_domain_model(sp["model"])
    if hasattr(driver, "image_dir") and driver.image_dir is None:
        driver.image_dir = image_dir

    attrs, n_none, n_missing = [], 0, 0
    primary = schema.get("attributes", [{}])[0].get("name")
    for i, r in enumerate(rows):
        pre = driver.preprocess(r, cols)
        path = pre.get("image_path")
        if not path or not os.path.exists(path):
            rec = semextract._none_record(schema, id_col, r[id_col]); n_missing += 1
        else:
            rec = extract_record(path, schema, ctx)
            rec[id_col] = r[id_col]
        attrs.append(rec)
        if rec.get(primary) in (None, "none", "", []):
            n_none += 1
        print(f"[semvision] {i+1}/{len(rows)} {str(r[id_col])[:32]!r} -> {rec.get(primary)}")

    json.dump(attrs, open(out_path, "w"), indent=2)
    elapsed = time.time() - t0
    meta = {"engine": "semvision", "clip_model": clip_model if ctx["encoder"] else None,
            "rows": len(attrs), "extracted": len(attrs) - n_none - n_missing,
            "none": n_none, "missing": n_missing,
            "elapsed_sec": round(elapsed, 2),
            "rows_per_sec": round(len(attrs) / max(1e-9, elapsed), 2)}
    json.dump(meta, open(out_path + ".meta.json", "w"), indent=2)
    print(f"[semvision] wrote {len(attrs)} rows -> {out_path} ({elapsed:.1f}s)")
    return meta
