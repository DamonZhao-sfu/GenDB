# Tiered Image Extractors (non-VLM proxies) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-VLM-per-row image extraction with a **per-attribute tiered proxy** pipeline — the Schema Designer assigns each attribute the lightest extractor (pure-CV / CLIP-zero-shot / … / residual-VLM), and a new `semvision.py` runtime runs the assigned proxy per attribute instead of one generative VLM call.

**Architecture:** A new `src/semdb/semvision.py` image engine mirrors `semextract` (same `map_columns`/`preprocess` driver hooks, same checkpoint/meta/attrs contract) but **dispatches per attribute** to a tier proxy read from each attribute's new `extractor` spec in `schema.json`. This first plan implements the framework + two tiers that need **no new dependencies and no vLLM**: **① pure CV** (dominant colors, numpy/PIL) and **② CLIP zero-shot** (`transformers.CLIPModel`), covering closed-enum categories, brand/logo matching, and image-text scoring. Text extraction stays on `semextract` unchanged. Later tiers (domain classifiers, detectors, DINOv2 probes) are a follow-on plan.

**Tech Stack:** Python 3.13 (gendb conda env); `torch`, `numpy`, `PIL`, `transformers` (CLIP) — all already installed; reuse `semextract`'s `.partial` checkpoint / systemic-guard / `<out>.meta.json` machinery; `pytest` for tests. Node orchestrator: only model-plumbing changes.

## Global Constraints

- Run everything under `/localhome/hza214/miniconda3/envs/gendb/bin/python` (has torch/transformers/sklearn).
- NO new pip dependencies for this plan: CLIP via `transformers.CLIPModel` (`openai/clip-vit-base-patch32`), CV via numpy/PIL. `open_clip` and `cv2` are NOT available — do not import them.
- The image path resolver is `semextract.resolve_image_path(uri, image_dir)` — reuse it, do not reinvent.
- The produced attribute record contract is unchanged: `{"<id_col>": ..., <schema attribute columns>, "conf": <float>}` — one JSON object per row, so the compiled query and `evaluate.py` keep working.
- Confidence: each proxy returns a per-attribute score; the record `conf` = the **min** of the row's per-attribute scores (a row is only as confident as its weakest feature).
- Tests must not require network downloads in unit scope — CLIP-dependent logic takes an **injectable encoder**; the real-CLIP path is exercised only in the integration task (Task 7), which may download `clip-vit-base-patch32` once.

---

### Task 1: CV dominant-colors proxy (pure, no model)

**Files:**
- Create: `src/semdb/semvision.py`
- Test: `src/semdb/tests/test_semvision_cv.py`

**Interfaces:**
- Produces: `cv_dominant_colors(img_path: str, palette: dict[str,tuple[int,int,int]] | None = None, min_frac: float = 0.08, size: int = 64) -> tuple[list[str], float]` — returns (colors present at ≥ min_frac of pixels, confidence). `DEFAULT_PALETTE: dict[str, tuple[int,int,int]]` maps color name → RGB anchor.

- [ ] **Step 1: Write the failing test**

```python
# src/semdb/tests/test_semvision_cv.py
import os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semvision


def _write_half(tmp, top_rgb, bottom_rgb):
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:32, :] = top_rgb
    arr[32:, :] = bottom_rgb
    p = os.path.join(tmp, "img.png")
    Image.fromarray(arr).save(p)
    return p


def test_dominant_colors_detects_both_yellow_and_silver(tmp_path):
    p = _write_half(str(tmp_path), (255, 255, 0), (192, 192, 192))  # yellow / silver
    colors, conf = semvision.cv_dominant_colors(p)
    assert set(colors) == {"yellow", "silver"}
    assert conf == 1.0


def test_dominant_colors_drops_tiny_fraction(tmp_path):
    arr = np.full((64, 64, 3), (0, 0, 0), dtype=np.uint8)   # ~all black
    arr[:2, :2] = (255, 255, 0)                              # <0.8% yellow
    p = os.path.join(str(tmp_path), "img.png")
    Image.fromarray(arr).save(p)
    colors, _ = semvision.cv_dominant_colors(p)
    assert colors == ["black"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/localhome/hza214/miniconda3/envs/gendb/bin/python -m pytest src/semdb/tests/test_semvision_cv.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'semvision'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/semdb/semvision.py  (top of new file)
"""semvision.py — tiered, per-attribute IMAGE extraction proxies (non-VLM).

Each schema attribute carries an `extractor` spec ({tier, method, params, labels});
this module runs the lightest proxy that answers it: ① pure CV, ② CLIP zero-shot.
Text extraction stays in semextract; this is the image path.
"""
import os
import numpy as np
from PIL import Image

# name -> RGB anchor. 'silver' is a light gray; 'gray' a mid gray (kept distinct).
DEFAULT_PALETTE = {
    "black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128),
    "silver": (192, 192, 192), "red": (200, 30, 30), "green": (30, 160, 60),
    "blue": (40, 70, 190), "yellow": (240, 220, 40), "orange": (230, 130, 30),
    "brown": (120, 70, 40), "purple": (120, 50, 160), "pink": (230, 130, 170),
    "gold": (210, 170, 60), "beige": (220, 200, 160),
}


def cv_dominant_colors(img_path, palette=None, min_frac=0.08, size=64):
    """Return (colors ≥ min_frac of pixels, confidence). Deterministic: each pixel
    maps to its nearest palette color by RGB distance; colors covering ≥ min_frac
    are returned (sorted by coverage, descending)."""
    palette = palette or DEFAULT_PALETTE
    names = list(palette)
    anchors = np.array([palette[n] for n in names], dtype=np.float32)   # [C,3]
    im = Image.open(img_path).convert("RGB").resize((size, size))
    px = np.asarray(im, dtype=np.float32).reshape(-1, 3)                # [N,3]
    d = ((px[:, None, :] - anchors[None, :, :]) ** 2).sum(-1)           # [N,C]
    nearest = d.argmin(1)
    frac = np.bincount(nearest, minlength=len(names)).astype(np.float32) / len(px)
    order = np.argsort(-frac)
    colors = [names[i] for i in order if frac[i] >= min_frac]
    return colors, 1.0   # CV readout is deterministic → full confidence
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/localhome/hza214/miniconda3/envs/gendb/bin/python -m pytest src/semdb/tests/test_semvision_cv.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/semdb/semvision.py src/semdb/tests/test_semvision_cv.py
git commit -m "feat(semvision): pure-CV dominant-colors proxy (tier ①)"
```

---

### Task 2: CLIP zero-shot proxies (classify / multilabel / match) with injectable encoder

**Files:**
- Modify: `src/semdb/semvision.py`
- Test: `src/semdb/tests/test_semvision_clip.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `clip_classify(img_path, labels, encoder) -> tuple[str, float]` — argmax label + softmax prob.
  - `clip_multilabel(img_path, labels, encoder, thresh=0.5) -> tuple[list[str], float]` — labels whose sigmoid(sim) ≥ thresh; conf = mean selected prob (or top prob if none selected).
  - `clip_match(img_path, text, encoder) -> float` — cosine similarity in [0,1].
  - An `Encoder` protocol: `encode_image(img_path) -> np.ndarray[D]` (L2-normalized), `encode_text(list[str]) -> np.ndarray[L,D]` (L2-normalized).

- [ ] **Step 1: Write the failing test**

```python
# src/semdb/tests/test_semvision_clip.py
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semvision


class FakeEncoder:
    """Deterministic encoder: image vec fixed; each label vec crafted so a chosen
    label is nearest. No model, no files."""
    def __init__(self, img_vec, text_vecs):
        self._img = np.asarray(img_vec, np.float32)
        self._img /= np.linalg.norm(self._img)
        self._txt = {k: (np.asarray(v, np.float32) / np.linalg.norm(v)) for k, v in text_vecs.items()}

    def encode_image(self, _path):
        return self._img

    def encode_text(self, labels):
        return np.stack([self._txt[l] for l in labels])


def test_clip_classify_picks_nearest_label():
    enc = FakeEncoder([1, 0, 0], {"sports_shoes": [1, 0, 0], "sandal": [0, 1, 0], "boot": [0, 0, 1]})
    label, conf = semvision.clip_classify("x.jpg", ["sports_shoes", "sandal", "boot"], enc)
    assert label == "sports_shoes"
    assert conf > 0.5


def test_clip_multilabel_thresholds():
    enc = FakeEncoder([1, 1, 0], {"yellow": [1, 0, 0], "silver": [0, 1, 0], "green": [0, 0, 1]})
    labels, _ = semvision.clip_multilabel("x.jpg", ["yellow", "silver", "green"], enc, thresh=0.5)
    assert set(labels) == {"yellow", "silver"}


def test_clip_match_similarity_range():
    enc = FakeEncoder([1, 0, 0], {"a red running shoe": [1, 0, 0]})
    s = semvision.clip_match("x.jpg", "a red running shoe", enc)
    assert 0.99 <= s <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/localhome/hza214/miniconda3/envs/gendb/bin/python -m pytest src/semdb/tests/test_semvision_clip.py -v`
Expected: FAIL with `AttributeError: module 'semvision' has no attribute 'clip_classify'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/semdb/semvision.py  (append)
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
    probs = 1.0 / (1.0 + np.exp(-(sims - 0.2) / 0.05))   # sigmoid, centered ~0.2 cos
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/localhome/hza214/miniconda3/envs/gendb/bin/python -m pytest src/semdb/tests/test_semvision_clip.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/semdb/semvision.py src/semdb/tests/test_semvision_clip.py
git commit -m "feat(semvision): CLIP zero-shot classify/multilabel/match proxies (tier ②) with injectable encoder"
```

---

### Task 3: Real CLIP encoder (transformers) — lazy, cached, batched-per-call

**Files:**
- Modify: `src/semdb/semvision.py`
- Test: `src/semdb/tests/test_semvision_encoder.py`

**Interfaces:**
- Produces: `class ClipEncoder` implementing `encode_image(img_path)->np.ndarray[D]` and `encode_text(list[str])->np.ndarray[L,D]`, both L2-normalized; ctor `ClipEncoder(model_id="openai/clip-vit-base-patch32", device=None)`. `get_encoder(model_id)` returns a process-cached instance.

- [ ] **Step 1: Write the failing test** (structure only — no model download in unit scope)

```python
# src/semdb/tests/test_semvision_encoder.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semvision


def test_get_encoder_is_cached(monkeypatch):
    calls = {"n": 0}
    class Dummy:
        def __init__(self, model_id="m", device=None): calls["n"] += 1
    monkeypatch.setattr(semvision, "ClipEncoder", Dummy)
    semvision._ENCODER_CACHE.clear()
    a = semvision.get_encoder("m")
    b = semvision.get_encoder("m")
    assert a is b and calls["n"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/localhome/hza214/miniconda3/envs/gendb/bin/python -m pytest src/semdb/tests/test_semvision_encoder.py -v`
Expected: FAIL with `AttributeError: module 'semvision' has no attribute '_ENCODER_CACHE'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/semdb/semvision.py  (append)
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
        from PIL import Image
        im = Image.open(img_path).convert("RGB")
        inp = self.proc(images=[im], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            v = self.model.get_image_features(**inp)
        return self._norm(v)[0]

    def encode_text(self, labels):
        # CLIP works best with a prompt template.
        prompts = [f"a photo of {l.replace('_', ' ')}" for l in labels]
        inp = self.proc(text=prompts, return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            v = self.model.get_text_features(**inp)
        return self._norm(v)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/localhome/hza214/miniconda3/envs/gendb/bin/python -m pytest src/semdb/tests/test_semvision_encoder.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/semdb/semvision.py src/semdb/tests/test_semvision_encoder.py
git commit -m "feat(semvision): cached transformers CLIP encoder"
```

---

### Task 4: Per-attribute dispatch — `extract_record(row, cols, schema, ctx)`

**Files:**
- Modify: `src/semdb/semvision.py`
- Test: `src/semdb/tests/test_semvision_dispatch.py`

**Interfaces:**
- Consumes: `cv_dominant_colors`, `clip_classify`, `clip_multilabel`, `clip_match` (Tasks 1–2); `resolve_image_path` from `semextract`.
- Produces: `extract_record(image_path, schema, ctx) -> dict` — for each attribute reads `attr["extractor"]` and runs its tier; returns `{attr_name: value, ..., "conf": min_score}`. `ctx` is a dict `{"encoder": <Encoder or None>, "palette": dict|None}`. Supported specs:
  - `{"tier":"cv","method":"dominant_colors","params":{...}}` → `cv_dominant_colors`.
  - `{"tier":"clip","method":"classify","labels":[...]}` → `clip_classify`.
  - `{"tier":"clip","method":"multilabel","labels":[...],"params":{"thresh":0.5}}` → `clip_multilabel`.
  - `{"tier":"clip","method":"match","params":{"text":"..."}}` → `clip_match` (value = score, boolean via `params.threshold`).
  - Unknown/`vlm` tier → value `"none"`, score `0.0` (residual is disabled; recorded so the row is a low-conf miss, never a crash).

- [ ] **Step 1: Write the failing test**

```python
# src/semdb/tests/test_semvision_dispatch.py
import os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semvision
from test_semvision_clip import FakeEncoder


def test_dispatch_mixes_cv_and_clip(tmp_path):
    arr = np.zeros((64, 64, 3), np.uint8); arr[:32] = (240, 220, 40); arr[32:] = (192, 192, 192)
    p = os.path.join(str(tmp_path), "shoe.png"); Image.fromarray(arr).save(p)
    schema = {"attributes": [
        {"name": "product_type", "type": "enum",
         "extractor": {"tier": "clip", "method": "classify", "labels": ["sports_shoes", "sandal"]}},
        {"name": "colors", "type": "list[enum]",
         "extractor": {"tier": "cv", "method": "dominant_colors", "params": {"min_frac": 0.08}}},
    ]}
    enc = FakeEncoder([1, 0], {"sports_shoes": [1, 0], "sandal": [0, 1]})
    rec = semvision.extract_record(p, schema, {"encoder": enc, "palette": None})
    assert rec["product_type"] == "sports_shoes"
    assert set(rec["colors"]) == {"yellow", "silver"}
    assert 0.0 <= rec["conf"] <= 1.0


def test_dispatch_unknown_tier_is_none_not_crash(tmp_path):
    p = os.path.join(str(tmp_path), "x.png"); Image.new("RGB", (8, 8)).save(p)
    schema = {"attributes": [{"name": "vibe", "type": "string", "extractor": {"tier": "vlm"}}]}
    rec = semvision.extract_record(p, schema, {"encoder": None, "palette": None})
    assert rec["vibe"] == "none" and rec["conf"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/localhome/hza214/miniconda3/envs/gendb/bin/python -m pytest src/semdb/tests/test_semvision_dispatch.py -v`
Expected: FAIL with `AttributeError: module 'semvision' has no attribute 'extract_record'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/semdb/semvision.py  (append)
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
    # unknown / vlm / dino / detector / domain (not yet implemented) → residual miss
    return ([] if is_list else "none"), 0.0


def extract_record(image_path, schema, ctx):
    rec, scores = {}, []
    for attr in schema.get("attributes", []):
        val, s = _run_attr(image_path, attr, ctx)
        rec[attr["name"]] = val
        scores.append(s)
    rec["conf"] = float(min(scores)) if scores else 0.0
    return rec
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/localhome/hza214/miniconda3/envs/gendb/bin/python -m pytest src/semdb/tests/test_semvision_dispatch.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/semdb/semvision.py src/semdb/tests/test_semvision_dispatch.py
git commit -m "feat(semvision): per-attribute tier dispatch (extract_record)"
```

---

### Task 5: `semvision.run(driver, schema, table, out, ...)` engine

**Files:**
- Modify: `src/semdb/semvision.py`
- Test: `src/semdb/tests/test_semvision_run.py`

**Interfaces:**
- Consumes: `extract_record` (Task 4); `get_encoder` (Task 3); `resolve_image_path`, `_none_record` from `semextract`; the driver's `map_columns(header)` / `preprocess(row, cols)` hooks (same protocol as `semextract.ExtractDriver`, minus `build_prompt`).
- Produces: `run(driver, schema, table_path, out_path, *, image_dir=None, clip_model="openai/clip-vit-base-patch32", limit=0, palette=None) -> dict` — reads the table CSV, resolves each row's image, calls `extract_record`, writes the attrs JSON (+`<out>.meta.json` with `rows/extracted/none/elapsed_sec/rows_per_sec`), returns meta. Loads the CLIP encoder ONCE iff any attribute uses tier `clip`.

- [ ] **Step 1: Write the failing test**

```python
# src/semdb/tests/test_semvision_run.py
import os, sys, csv, json
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semvision


class CVDriver:  # CV-only → no encoder needed
    def __init__(self, image_dir): self.image_dir = image_dir
    def map_columns(self, header): return {"id": "id", "image": "filename", "text": None, "context": []}
    def preprocess(self, row, cols):
        import semextract
        return {"image_path": semextract.resolve_image_path(row["filename"], self.image_dir)}


def test_run_writes_attrs_and_meta(tmp_path):
    imgdir = tmp_path / "images"; imgdir.mkdir()
    for i, rgb in [(1, (240, 220, 40)), (2, (0, 0, 0))]:
        Image.fromarray(np.full((64, 64, 3), rgb, np.uint8)).save(imgdir / f"{i}.png")
    table = tmp_path / "IMAGES.csv"
    with open(table, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "filename"]); w.writerow([1, "1.png"]); w.writerow([2, "2.png"])
    schema = {"attributes": [{"name": "colors", "type": "list[enum]",
              "extractor": {"tier": "cv", "method": "dominant_colors"}}]}
    out = tmp_path / "attrs.json"
    meta = semvision.run(CVDriver(str(imgdir)), schema, str(table), str(out), image_dir=str(imgdir))
    recs = json.load(open(out))
    assert meta["rows"] == 2 and len(recs) == 2
    by = {r["id"]: r for r in recs}
    assert "yellow" in by["1"]["colors"] and "black" in by["2"]["colors"]
    assert os.path.exists(str(out) + ".meta.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/localhome/hza214/miniconda3/envs/gendb/bin/python -m pytest src/semdb/tests/test_semvision_run.py -v`
Expected: FAIL with `AttributeError: module 'semvision' has no attribute 'run'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/semdb/semvision.py  (append)
import csv, json, time, sys


def _uses_clip(schema):
    return any((a.get("extractor") or {}).get("tier") == "clip" for a in schema.get("attributes", []))


def run(driver, schema, table_path, out_path, *, image_dir=None,
        clip_model="openai/clip-vit-base-patch32", limit=0, palette=None):
    import semextract  # reuse resolve_image_path
    t0 = time.time()
    rows = list(csv.DictReader(open(table_path)))
    if limit:
        rows = rows[:limit]
    cols = driver.map_columns(list(rows[0].keys()) if rows else [])
    id_col = cols["id"]
    ctx = {"encoder": get_encoder(clip_model) if _uses_clip(schema) else None, "palette": palette}
    if hasattr(driver, "image_dir") and driver.image_dir is None:
        driver.image_dir = image_dir

    attrs, n_none, n_missing = [], 0, 0
    for i, r in enumerate(rows):
        pre = driver.preprocess(r, cols)
        path = pre.get("image_path")
        if not path or not os.path.exists(path):
            rec = semextract._none_record(schema, id_col, r[id_col]); n_missing += 1
        else:
            rec = extract_record(path, schema, ctx)
            rec[id_col] = r[id_col]
        attrs.append(rec)
        primary = schema.get("attributes", [{}])[0].get("name")
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/localhome/hza214/miniconda3/envs/gendb/bin/python -m pytest src/semdb/tests/test_semvision_run.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/semdb/semvision.py src/semdb/tests/test_semvision_run.py
git commit -m "feat(semvision): run() engine (attrs + meta, CLIP loaded once)"
```

---

### Task 6: Schema Designer emits a per-attribute `extractor` spec

**Files:**
- Modify: `src/semdb/agents/schema-designer/prompt.md`
- Modify: `src/semdb/agents/schema-designer/user-prompt.md`
- Create: `src/semdb/tests/test_extractor_spec_valid.py`

**Interfaces:**
- Produces: schema.json attributes each carry `extractor: {tier, method?, labels?, params?, model?}`. A helper `validate_extractor_spec(attr) -> list[str]` (errors) is added to `semvision.py` and unit-tested; the prompt change itself is validated by running the designer, but the *contract* is locked by this validator.

- [ ] **Step 1: Write the failing test**

```python
# src/semdb/tests/test_extractor_spec_valid.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semvision


def test_valid_specs_pass():
    for attr in [
        {"name": "colors", "type": "list[enum]", "extractor": {"tier": "cv", "method": "dominant_colors"}},
        {"name": "cat", "type": "enum", "extractor": {"tier": "clip", "method": "classify", "labels": ["a", "b"]}},
        {"name": "vibe", "type": "string", "extractor": {"tier": "vlm"}},
    ]:
        assert semvision.validate_extractor_spec(attr) == []


def test_clip_classify_requires_labels():
    errs = semvision.validate_extractor_spec(
        {"name": "cat", "type": "enum", "extractor": {"tier": "clip", "method": "classify"}})
    assert any("labels" in e for e in errs)


def test_missing_extractor_flagged():
    errs = semvision.validate_extractor_spec({"name": "x", "type": "enum"})
    assert any("extractor" in e for e in errs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/localhome/hza214/miniconda3/envs/gendb/bin/python -m pytest src/semdb/tests/test_extractor_spec_valid.py -v`
Expected: FAIL with `AttributeError: module 'semvision' has no attribute 'validate_extractor_spec'`.

- [ ] **Step 3: Write minimal implementation** (validator + prompt edits)

Add to `src/semdb/semvision.py`:

```python
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
    return errs
```

Append to `src/semdb/agents/schema-designer/prompt.md` a new section (verbatim):

```markdown
## Pick an EXTRACTOR per attribute (image corpora)
For every attribute, add an `extractor` object choosing the LIGHTEST proxy that
answers the predicate — do NOT default to a generative VLM:

- color / brightness / texture → `{"tier":"cv","method":"dominant_colors","params":{"min_frac":0.08}}`
- closed-enum semantic category (product_type, species, garment class, damaged yes/no)
  → `{"tier":"clip","method":"classify","labels":[<enum values>]}`
- a MULTI-label closed set (all colors present, multiple attributes)
  → `{"tier":"clip","method":"multilabel","labels":[...],"params":{"thresh":0.5}}`
- logo→brand identity / open-ish nameable → `{"tier":"clip","method":"match","params":{"text":"<brand or concept>"}}`
  (or, when a small trained model exists later: tier "domain"/"detector"/"distilled")
- ONLY holistic/compositional predicates that cannot be factored → `{"tier":"vlm"}`

Decompose conjunctions: "sports shoe that is yellow and silver" → product_type
attribute (clip classify) + colors attribute (cv dominant_colors, so BOTH colors are
detected). Every attribute MUST have an `extractor`.
```

Add to `src/semdb/agents/schema-designer/user-prompt.md` after the schema-output bullet:

```markdown
- Each attribute MUST include an `extractor` spec (tier cv|clip|…|vlm) chosen per the
  decision tree in the system prompt — prefer non-VLM tiers.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/localhome/hza214/miniconda3/envs/gendb/bin/python -m pytest src/semdb/tests/test_extractor_spec_valid.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/semdb/semvision.py src/semdb/tests/test_extractor_spec_valid.py src/semdb/agents/schema-designer/prompt.md src/semdb/agents/schema-designer/user-prompt.md
git commit -m "feat(schema-designer): assign a non-VLM extractor tier per attribute + validator"
```

---

### Task 7: Extractor generates a semvision image driver; end-to-end on ecomm q2

**Files:**
- Modify: `src/semdb/agents/extractor/prompt.md`
- Modify: `src/semdb/agents/extractor/user-prompt.md`
- Create: `src/semdb/tests/test_ecomm_q2_semvision.py` (integration; downloads CLIP once)

**Interfaces:**
- Consumes: `semvision.run` (Task 5); the materialized `IMAGES.csv` manifest (`runs/_materialized/ecomm_sf250/IMAGES.csv`) and `<dataDir>/images` from the existing ecomm wiring.
- Produces: for IMAGE corpora the Extractor agent writes `extract_<corpus>.py` that imports `semvision`, provides `map_columns`/`preprocess`, and calls `semvision.run(...)` instead of `semextract.run(...)`. Text corpora still use `semextract`.

- [ ] **Step 1: Write the failing integration test**

```python
# src/semdb/tests/test_ecomm_q2_semvision.py
import os, sys, csv, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semvision, semextract

DATA = "/localhome/hza214/SemBench/files/ecomm/data/sf_250"
MANIFEST = "/local-scratch/localhome/hza214/GenDB/src/semdb/runs/_materialized/ecomm_sf250/IMAGES.csv"
GT = "/localhome/hza214/SemBench/files/ecomm/raw_results/ground_truth/Q2.csv"


class ImgDriver:
    def __init__(self, image_dir): self.image_dir = image_dir
    def map_columns(self, header): return {"id": "id", "image": "filename", "text": None, "context": []}
    def preprocess(self, row, cols):
        return {"image_path": semextract.resolve_image_path(row["filename"], self.image_dir)}


def test_ecomm_q2_hits_yellow_silver_sports_shoes(tmp_path):
    if not os.path.exists(MANIFEST):
        import pytest; pytest.skip("materialize ecomm first")
    schema = {"attributes": [
        {"name": "product_type", "type": "enum",
         "extractor": {"tier": "clip", "method": "classify",
                       "labels": ["sports_shoes", "sandal", "boot", "other_footwear", "not_footwear"]}},
        {"name": "colors", "type": "list[enum]",
         "extractor": {"tier": "cv", "method": "dominant_colors", "params": {"min_frac": 0.06}}},
    ]}
    out = tmp_path / "attrs.json"
    semvision.run(ImgDriver(os.path.join(DATA, "images")), schema, MANIFEST, str(out),
                  image_dir=os.path.join(DATA, "images"))
    recs = {r["id"]: r for r in json.load(open(out))}
    gt = {r["id"] for r in csv.DictReader(open(GT))}
    # a yellow+silver sports shoe must now have BOTH colors detected (CV), not one
    hit = [i for i, r in recs.items()
           if r.get("product_type") == "sports_shoes"
           and {"yellow", "silver"}.issubset(set(r.get("colors", [])))]
    # at least one GT shoe recovered — proves CV multi-color fixes the empty result
    assert set(hit) & gt, f"no GT shoe recovered; hits={hit[:5]} gt={list(gt)[:5]}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/localhome/hza214/miniconda3/envs/gendb/bin/python -m pytest src/semdb/tests/test_ecomm_q2_semvision.py -v`
Expected: FAIL initially if the CV `min_frac`/palette can't catch the accent colors — tune `min_frac`/palette (silver vs gray) until at least one GT shoe is recovered, OR document that these specific shoes need a finer color proxy. (This is the real acceptance check; a genuine data-driven fail here is informative, not a plan defect.)

- [ ] **Step 3: Write minimal implementation** (prompt edits so the agent emits a semvision driver)

Edit `src/semdb/agents/extractor/prompt.md` — change the "What you write" section so that **for image corpora** the driver targets `semvision`:

```markdown
## Image corpora use semvision (tiered proxies), NOT a VLM
When the corpus modality is image, generate a driver over `semvision` (not semextract):
its `run()` reads each attribute's `extractor` spec (from schema.json) and runs the
assigned proxy — pure-CV colors, CLIP zero-shot category/brand — so NO generative VLM
is called. Provide `map_columns(header)` and `preprocess(row, cols)` (resolve the image
with `semextract.resolve_image_path(row[cols['image']], self.image_dir)`), then call:

    import semvision
    semvision.run(driver, schema, args.table, args.out,
                  image_dir=args.image_dir, clip_model=args.model)

Text corpora keep using semextract.run as before.
```

Edit `src/semdb/agents/extractor/user-prompt.md` — add:

```markdown
- If Modality is `image`, target `semvision.run` (tiered non-VLM proxies driven by each
  attribute's `extractor` spec). `--model` is the CLIP model id for tier `clip`.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/localhome/hza214/miniconda3/envs/gendb/bin/python -m pytest src/semdb/tests/test_ecomm_q2_semvision.py -v`
Expected: PASS — at least one GT yellow+silver sports shoe recovered (CV now returns both colors; CLIP tags it a sports shoe).

- [ ] **Step 5: Commit**

```bash
git add src/semdb/agents/extractor/prompt.md src/semdb/agents/extractor/user-prompt.md src/semdb/tests/test_ecomm_q2_semvision.py
git commit -m "feat(extractor): image corpora extract via semvision tiered proxies; ecomm q2 recovers GT shoes"
```

---

### Task 8: Orchestrator plumbing — image corpora don't need a vLLM endpoint

**Files:**
- Modify: `src/semdb/orchestrator.mjs` (Phase B execute block for image corpora)
- Modify: `src/semdb/semdb.config.mjs` (add `extraction.clipModel`)

**Interfaces:**
- Consumes: the generated `extract_<corpus>.py` now calling `semvision.run` for image corpora (Task 7).
- Produces: for image corpora, the orchestrator passes `--model <clipModel>` and `--image-dir`, and does NOT require `--endpoint` (CV/CLIP run locally). Text corpora still pass `--endpoint`/`--concurrency`.

- [ ] **Step 1: Write the failing test**

```python
# src/semdb/tests/test_orch_image_no_endpoint.py  (node-driven smoke via subprocess)
import subprocess, sys
def test_config_has_clip_model():
    out = subprocess.run(
        ["node", "-e", "import('./src/semdb/semdb.config.mjs').then(m=>console.log(!!m.defaults.extraction.clipModel))"],
        cwd="/local-scratch/localhome/hza214/GenDB", capture_output=True, text=True)
    assert out.stdout.strip() == "true", out.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/localhome/hza214/miniconda3/envs/gendb/bin/python -m pytest src/semdb/tests/test_orch_image_no_endpoint.py -v`
Expected: FAIL (prints `false`).

- [ ] **Step 3: Write minimal implementation**

In `semdb.config.mjs` `extraction` block add:
```javascript
    clipModel: "openai/clip-vit-base-patch32",   // tier-② CLIP for image extraction
```

In `orchestrator.mjs` Phase B execute exArgs, when `isImage`, pass the CLIP model and drop the endpoint requirement:
```javascript
    const exArgs = [driverPath, corpus.path, attrsPath, "--schema", schemaPath,
      "--model", (isImage ? (args.extractModel || defaults.extraction.clipModel) : extractModel),
      ...(isImage ? ["--image-dir", args.imageDir] : []),
      ...(!isImage && args.endpoint ? ["--endpoint", args.endpoint, "--api-key", args.apiKey,
                           "--concurrency", String(args.concurrency)] : []),
      ...(args.theta != null ? ["--theta", String(args.theta)] : [])];
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/localhome/hza214/miniconda3/envs/gendb/bin/python -m pytest src/semdb/tests/test_orch_image_no_endpoint.py -v` and `node --check src/semdb/orchestrator.mjs`
Expected: PASS + `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/semdb/orchestrator.mjs src/semdb/semdb.config.mjs src/semdb/tests/test_orch_image_no_endpoint.py
git commit -m "feat(orchestrator): image corpora extract locally via CLIP (no vLLM endpoint)"
```

---

## Phase 2 — additional tiers (separate follow-on plan, not in this plan)

These are independent add-ons; each is its own spec→plan→build cycle because each pulls a distinct dependency and validation set. Listed here so coverage is explicit, NOT implemented by this plan:

- **③ Domain classifiers** (medical Q3/Q5–Q9/Q11 chest-X-ray + skin; ~16 queries): add `semvision.domain_classify(img, model_id)` backed by `torchxrayvision` (X-ray) and a HAM10000/ISIC `timm` model (skin). New deps: `torchxrayvision`, `timm`. Designer emits `{"tier":"domain","model":"torchxrayvision:densenet121-res224-all"}`.
- **③ Detectors** (animals species presence, cars fine-grained defects; ~14 queries): add `semvision.detect(img, classes)` backed by MegaDetector/YOLOv8 (`ultralytics`). Designer emits `{"tier":"detector","model":"yolov8","classes":[...]}`.
- **② DINOv2 linear probe** (when CLIP zero-shot underperforms and labels exist): add `semvision.dino_probe(img, head_path)` with `dinov2_vits14` frozen + a trained linear head; includes a tiny probe-training utility.
- **④ Distilled student** (mmqa logos, ecomm same-brand): VLM labels K exemplars → train a small student head over CLIP embeddings.

Each Phase-2 tier reuses Tasks 4–5's dispatch/engine (add a branch in `_run_attr` + a proxy fn + a validator case) and follows the same TDD shape.

---

## Notes for the implementer
- Put tests under `src/semdb/tests/`; run with the gendb python: `/localhome/hza214/miniconda3/envs/gendb/bin/python -m pytest src/semdb/tests/ -v`.
- `semvision.py` deliberately mirrors `semextract.py`'s driver hooks and reuses `resolve_image_path` + `_none_record` — do NOT duplicate those.
- Residual is disabled globally (`semruntime.vlm_judge` is a no-op); tier `vlm` therefore yields a low-conf `none`, never a live call — that is intended for this plan.
- Confidence is `min` over per-attribute scores so a weak sub-feature marks the whole row low-conf (surfaces in the attrs `conf`, usable later if residual is re-enabled).
