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
patch.score(text) -> float                   # CLIP image-text similarity in [0,1]
patch.read_text() -> str                     # raw OCR text
patch.crop(l, low, r, up) -> ImagePatch      # region
```

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
```

## Rules
- SMALL/visual value space or a category enum -> `classify`. LARGE value space of legible
  WORDMARK names (airlines, brands with text) -> `best_ocr_match`. Colors -> `dominant_colors`.
- Return the field VALUE (or list), never a bare score. Value-space lists (e.g. TRACK_VALUES)
  are provided to you (from the schema's `labels`/`labels_from` — the DB value space).
- Compose when a single primitive is insufficient (gate then classify; crop then read).
- `conf` optional; if omitted the engine sets it. Do not import models — only use `patch`.
