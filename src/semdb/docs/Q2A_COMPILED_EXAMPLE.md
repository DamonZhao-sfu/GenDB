# Worked example: compiling MMQA Q2a / Q7 (the airline-logo join)

SemBench query (`files/mmqa/query/bigquery/q7.sql`; q2a is the same shape over a
racetrack table):

```sql
SELECT t.Airlines, i.uri
FROM   mmqa.tampa_international_airport t, mmqa.images i
WHERE  AI.IF(
         STRUCT(
           "You will be provided with an airline name and an image. ",
           "Determine if the image shows the logo of the airline. ",
           "Airline: ", t.Airlines, ", Image: ", i.uri
         ),
         connection_id => '<<connection>>',
         model_params  => JSON '{...thinking_budget...}'
       );
```

The naive plan calls the VLM once per `(airline, image)` pair — **M×N** calls.
SemDB compiles it in three agent phases. Each phase's real artifact is in
`../poc/`.

---

## Phase A — Schema Designer  →  `poc/expected/schema.json`

The Designer sees the join predicate and applies the decomposition test:

> The image side has a nameable, extractable slot — the brand the logo belongs
> to. The airline side (`t.Airlines`) is **already structured**. The join is
> therefore **asymmetric**: extract one side only.

It emits the schema (abridged):

```json
{
  "decomposable": true,
  "extract_side": { "table": "images", "column": "uri", "modality": "image" },
  "attributes": [
    { "name": "logo_brand", "type": "string", "allow_none": true,
      "extract_instruction": "Identify the brand whose logo appears; else 'none'." },
    { "name": "ocr_text", "type": "string", "allow_none": true }
  ],
  "join_key":  { "op": "equality_after_normalize" },
  "normalization": { "synonym_map": { "southwest airlines": "southwest", "...": "..." } },
  "residual":  { "condition": "logo_brand == 'none' OR conf < theta", "theta": 0.5 },
  "reused_by": ["q2a", "q2b", "q7"]
}
```

Key decision: **only the image side is extracted.** Normalization (aliases,
"Southwest" vs "Southwest Airlines") is a compile-time artifact built once.

## Phase B — Extractor  →  `poc/mock_extractor.py` / `poc/vlm_extractor.py`

Run a **small VLM once per image** (SmolVLM-256M / Qwen3-VL-2B), constrained to
strict JSON:

```json
{"uri": "img/001.jpg", "logo_brand": "Southwest", "conf": 0.92, "ocr_text": "Southwest"}
```

N calls total, once. The resulting `img_attrs` table is **shared by q2a, q2b, q7**
and any future logo query. Under-confident images record `logo_brand: "none"` —
they are not guessed; they are deferred.

## Phase C — Code Generator  →  `poc/compiled_q7.py`

The join compiles to normalize + hash-join + masked residual:

```python
def join(airlines, img_attrs):
    by_key = {}
    for a in airlines:                      # structured side: nothing to extract
        by_key.setdefault(normalize(a.name), []).append(a)

    residual = []
    for rec in img_attrs:                    # extracted side: cost 0
        if rec["logo_brand"] == "none" or rec["conf"] < THETA:
            residual.append(rec); continue   # unsure → defer to VLM
        for a in by_key.get(normalize(rec["logo_brand"]), []):
            pairs.append((a.name, rec["uri"]))   # exact hash join, no model call

    for rec in residual:                     # masked residual VLM: only unsure rows
        img = reopen(rec["uri"])
        for a in airlines:
            if P.vlm_judge(a, img): pairs.append((a.name, rec["uri"]))
    return pairs
```

---

## What the PoC measures (run `poc/run_poc.sh`)

On the 6-airline × 10-image sample:

```
naive VLM.IF calls         : 60   (M x N)
extractions (amortized)    : 10   (paid once, shared by every logo query)
residual VLM.IF calls      : 12
per-query model calls      : 60 -> 12
amortized over K logo queries: K*60 -> 10 + K*12
PASS: compiled plan is result-equivalent to the naive semantic plan.
```

- **Correct:** the compiled result CSV is byte-identical to the naive oracle,
  because the residual path re-runs the *exact* original predicate on every row
  the cheap path was unsure about.
- **Cheap:** the confident majority is answered by a hash join at zero model cost.
- **Auditable:** a wrong pair is traceable to a mis-extraction (wrong brand) or a
  normalization miss ("Southwest" vs "Southwest Airlines") — not an opaque call.
- **Amortized:** the 10 extractions are shared across q2a, q2b, q7; the marginal
  cost of each additional logo query is just its residual.
