# MMQA q3 — the full agent loop, run for real

This example runs the actual SemDB pipeline end-to-end: an **Agent** designs the
schema, a **small model** generates the schema records, and **Python** simulates
the SemBench SQL over those records. Unlike `../../poc/` (which uses a
deterministic mock), the schema and the extraction here were produced by live
models.

Target: the SemBench MMQA **q3 genre-filter family** (`q3a`–`q3g`, seven
`AI.IF("... is a <genre> movie ...")` scans) plus **q4** (`AI.GENERATE` genres +
`GROUP BY`). All eight queries run over one movie-description corpus — the best
amortization case in MMQA.

## What produced each artifact

| Artifact | Produced by | How |
|----------|-------------|-----|
| `data/movies.csv` | — | 12 movies from q4's title list, with short descriptions |
| `data/genres_key.csv` | — | hand-labeled genres = the oracle for validation |
| `schema.json` | **Schema Designer agent** | ran `agents/schema-designer/prompt.md` on the q3/q4 queries → decomposable, slot = `genres` set, controlled vocab, normalization map, residual θ |
| `movie_attrs.json` | **Extractor = small model (Claude Haiku)** | one pass over the descriptions → `{title, genres[], conf}` per movie |
| `compiled_q3.py` | **Code Generator** (this file) | compiles each `AI.IF` to `'<genre>' IN genres` and q4 to `UNNEST + GROUP BY` — 0 model calls |
| `validate.py` | — | compiled answers vs oracle labels: precision/recall/F1 + call reduction |

## Run it

```bash
bash run_q3.sh
```

Measured result (this run):

```
MICRO   TP=22 FP=3 FN=1   prec 0.88  recall 0.96  F1 0.92
naive (AI.IF per movie per query) : 12 x 7 = 84 model calls
compiled (extract once, reuse)    : 12 extractions + 0 per query
reduction                         : 84 -> 12
```

So a small model turns each description into a reusable `genres` row; the seven
genre filters and the q4 aggregation are then plain relational code that answers
correctly on 22/23 memberships (F1 0.92) at **1/7th** the model calls — and every
extra genre query added later is free.

## Honesty notes
- **Why Haiku and not Qwen3-VL-2B / SmolVLM-256M?** HuggingFace is blocked by this
  environment's network policy (403 on CONNECT), so those weights can't be
  downloaded here. Claude Haiku is the smallest model reachable through the
  harness, used as the small-model stand-in. `../../poc/vlm_extractor.py` is the
  drop-in for the real tiny models wherever HF is reachable — the output contract
  is identical.
- **The F1 is < 1.0 on purpose.** The small model over-tags "drama" and mislabels
  a thriller/horror edge case — exactly the residual/normalization tradeoff the
  design calls out. Nothing here is mocked to look perfect.
- The oracle is the hand-labeled `genres_key.csv`; it stands in for a perfect
  per-row `AI.IF`. The disagreements column in `validate.py` shows every case
  where the small model's schema differs from that oracle.
