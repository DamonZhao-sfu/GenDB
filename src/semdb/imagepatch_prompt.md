# ImagePatch API — for the Program agent (generated `extract`)

You write a Python function `extract(patch)` that returns one field per schema attribute
plus `"conf"`, composing ONLY the `ImagePatch` methods below (backed by CLIP / CV / OCR /
YOLO / domain — no generative VLM). Pick the LIGHTEST composition that answers each field;
the returned label IS the field value (joins/filters downstream).

## API (methods on `patch`)
```python
patch.classify(options, template="a photo of {}") -> str
    # best-matching option from a CLOSED value space (enum, or a DB column's values).
    # returns the winning VALUE. Best when the value space is SMALL/visual.
    # template shapes the prompt: "the logo of {}" for logos, "a {} shoe", etc.
patch.best_ocr_match(options) -> str
    # OCR the image, fuzzy-match the text to the value space; "none" if no strong match.
    # PREFER over classify for WORDMARK logos / large value spaces (airline names) where
    # the printed name is legible and CLIP over many options is unreliable.
patch.dominant_colors(min_frac=0.06) -> list[str]
    # colors present (incl. pale accents). Use for color fields / color predicates.
patch.verify_property(prop) -> bool          # CLIP: does the patch match `prop`?
patch.score(text) -> float                   # CLIP image-TEXT similarity in [0,1]
patch.pair_score(other) -> float             # CLIP image-IMAGE similarity in [0,1]
patch.read_text() -> str                     # raw OCR text
patch.read_text_boxes() -> list[dict]        # [{"text", "box": (l,t,r,b), "score"}]
patch.size -> (w, h)                         # of THIS patch
patch.bbox -> (l, t, r, b)                   # THIS patch's box in ABSOLUTE image pixels
```

### With a confidence — gate, rank, or hand the unsure rows to a heavier pass
A score is comparable ACROSS ROWS for the same primitive, but NOT across different
primitives (CLIP probabilities, detector confidences and OCR match strengths are on
different scales). Use the plain variant when you only need the value.
```python
patch.classify_detail(options, template="a photo of {}") -> (str, float)
patch.verify_detail(prop) -> (bool, float)
patch.best_ocr_match_detail(options) -> (str, float)   # a miss is ("none", 0.0)
patch.find_detail(name) -> list[dict]        # [{"image", "label", "box", "score"}]
patch.classify_multi(options, thresh=0.5) -> (list[str], float)
    # set-valued fields: keeps EVERY option over the threshold, not one winner
patch.domain_classify(model_id, labels, threshold=0.5) -> (str, float)
    # a model trained for this domain, e.g.
    # domain_classify("torchxrayvision:densenet121-res224-all", ["Pneumonia"])
```

### Vectors — encode once, compare many times
```python
patch.embed() -> list[float]                 # unit-norm image vector
patch.topk_similar(others, k=5) -> list[(int, float)]
    # ranks OTHER patches by image-image similarity; vectorized `pair_score`
patch.topk_text(texts, k=5) -> list[(str, float)]
    # ranks candidate TEXTS; narrows a large value space to a short scored list
```

### Regions — run a primitive on PART of the image
Every primitive below returns sub-patches; calling any method on one reads ONLY that
region. Boxes are `(left, top, right, bottom)` with the origin at the **TOP-LEFT**
(PIL order); pass pixels, or fractions of the patch when all four are in `[0, 1]`.
```python
patch.crop(left, top, right, bottom) -> ImagePatch   # e.g. crop(0.5, 0, 1, 1) = right half
patch.regions_grid(rows, cols, overlap=0.0) -> list[ImagePatch]   # partitioned extraction
patch.regions_center(frac=0.6) -> ImagePatch         # drops a product photo's margin
patch.find(object_name) -> list[ImagePatch]          # YOLO instances, most confident first
patch.find_open(object_name) -> list[ImagePatch]     # OPEN vocabulary — any name
patch.propose_regions(max_regions=8) -> list[ImagePatch]   # cuts along CONTENT
```
`find` has a CLOSED vocabulary (COCO-80: person, car, dog, zebra, bird, bottle, ...).
A name outside it returns `[]` and warns — use `find_open` (the prompt IS the class, so a
species or a car part works), or `classify`/`verify_property`.
`propose_regions` splits on foreground blobs instead of a blind grid — prefer it over
`regions_grid` when the image holds several distinct objects.
Reach for regions when ONE whole-image call is too coarse: a small target in a big
frame, several products in one photo, or text that only appears in one corner.

## Worked examples (instruction -> extract) — generalize, do not copy blindly
```python
# mmqa q2a: which racetrack's logo? value space = ap_warrior.Track (SMALL) -> classify
def extract(patch):
    return {"racetrack": patch.classify(TRACK_VALUES, "the logo of {}")}

# mmqa q7: which airline's logo? value space = Airlines (135, LARGE) -> OCR reads the wordmark
def extract(patch):
    return {"airline": patch.best_ocr_match(AIRLINE_VALUES)}   # 'none' self-gates non-logos

# ecomm q2: sports shoe featuring yellow AND silver -> category (CLIP) + colors (CV)
def extract(patch):
    return {"product_type": patch.classify(["sports_shoes","sandal","boot","other_footwear","not_footwear"]),
            "colors": patch.dominant_colors(0.03)}

# PARTITIONED: the category is decided by the product, not the white background;
# and a small logo can be lost when CLIP sees the whole frame -> decide per region
# and combine. Cheap: still zero VLM calls, just a few more CLIP passes.
def extract(patch):
    product = patch.regions_center(0.6)                       # drop the margin
    cat = product.classify(CATEGORIES)
    logo = "none"
    for cell in patch.regions_grid(2, 2, overlap=0.1):        # scan quadrants for a wordmark
        hit = cell.best_ocr_match(BRAND_VALUES)
        if hit != "none":
            logo = hit
            break
    return {"category": cat, "brand": logo, "colors": product.dominant_colors(0.03)}
```

## Rules
- SMALL/visual value space or a category enum -> `classify`. LARGE value space of legible
  WORDMARK names (airlines, brands with text) -> `best_ocr_match`. Colors -> `dominant_colors`.
- Whole image first. Reach for `regions_center` / `regions_grid` / `find` only when the
  whole-image call is genuinely too coarse — each region costs another pass.
- Return the field VALUE (or list), never a bare score. Value-space lists (e.g. TRACK_VALUES)
  are provided to you (from the schema's `labels`/`labels_from` — the DB value space).
- Compose when a single primitive is insufficient (gate then classify; crop then read).
- `conf` optional; if omitted the engine sets it. Do not import models — only use `patch`.
