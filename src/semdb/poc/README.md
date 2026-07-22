# SemDB PoC — verifiable, GPU-free

Proves the core claim: a small VLM can turn each image into a **reusable schema
row**, after which a multimodal semantic join is plain relational code plus a
tiny residual — and the compiled result is **identical** to the naive M×N plan.

## Run it

```bash
bash run_poc.sh
```

Expected tail:

```
PASS: compiled plan is result-equivalent to the naive semantic plan.
  naive VLM.IF calls         : 60  (M x N)
  extractions (amortized)    : 10  (paid once, shared by every logo query)
  residual VLM.IF calls      : 12
  per-query model calls      : 60 -> 12
  amortized over K logo queries: K*60 -> 10 + K*12
```

## Files

| File | Role | Maps to |
|------|------|---------|
| `data/airlines.csv` | structured side (`mmqa.tampa_international_airport`) | — |
| `data/images.csv` | image corpus; `true_brand` = hidden pixels (only a model may read it) | `mmqa.images` |
| `semlib.py` | runtime: loaders, `normalize` (compile-time synonym map), `P.vlm_judge` (counted model call), pair writer | — |
| `mock_extractor.py` | deterministic small-VLM stand-in → `img_attrs.json` | **Extractor (Phase B)** |
| `vlm_extractor.py` | real SmolVLM-256M / Qwen3-VL-2B extractor, same output contract (dependency-gated) | **Extractor (Phase B)** |
| `compiled_q7.py` | hash-join + masked residual VLM | **Code Generator (Phase C)** |
| `baseline_q7.py` | naive M×N `AI.IF` — correctness oracle | naive plan |
| `run_poc.sh` | extract → compile → oracle → assert equal → report savings | Verify |
| `expected/schema.json` | what the Schema Designer emits for this query | **Schema Designer (Phase A)** |
| `expected/img_attrs.json`, `expected/pairs.csv` | committed reference outputs | — |

## Simulation honesty
`true_brand` stands in for "what a perfect model reads off the pixels". It is the
**only** channel through which the extractor or the residual judge may see an
image; compiled query code never touches it directly. Every simulated model touch
is counted by `semlib.METER`, so the M×N → N collapse is measured, not asserted.
The mock models a *well-behaved* small VLM (abbreviations + low-confidence misses,
never confidently wrong); `docs/PLAN.md` lists confidently-wrong extraction as the
real-world risk and its mitigations.

## Real pixels
```bash
# needs: pip install torch transformers pillow accelerate
python3 vlm_extractor.py --manifest images_manifest.csv --out img_attrs.json \
        --model HuggingFaceTB/SmolVLM-256M-Instruct
python3 compiled_q7.py data/airlines.csv img_attrs.json out/compiled.csv
```
`images_manifest.csv` needs columns `uri,path` pointing at local image files.
