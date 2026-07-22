# SemDB — Semantic Operator Compilation & Execution

SemDB compiles **semantic operators over multimodal data** (`AI.IF`,
`AI.GENERATE`, semantic join / map / filter / rank / classify) into **vectorized
relational programs** that call a live vision/language model only on a residual.

It is a sibling of [GenDB](../../README.md): the same "agents generate execution
code" idea, but the thing being designed is an **extracted schema over
unstructured data**, and the expensive primitive is a model call, not a disk
scan. It reuses GenDB's Claude Agent SDK plumbing (`src/gendb/providers`,
`src/gendb/shared.mjs`) verbatim.

Target workload: [SemBench](https://github.com/SemBench/SemBench) (MMQA first).
Theory: semantic-operator decomposition / compilation (arXiv 2607.13407).

## The idea in one query

`AI.IF("does this image show the logo of {airline}?")` joining airlines × images
is **M×N** VLM calls. SemDB decomposes it:

1. **Schema Designer** notices the image side has a nameable slot (`logo_brand`)
   while the airline side is already structured → extract one side only.
2. **Extractor** runs a *small* VLM once per image → a reusable `img_attrs` table.
3. **Code Generator** compiles the join to normalize + hash-join, with a **masked
   residual** VLM call only for images the small model was unsure about.

Result: identical answers, `M×N → N extractions + hash join + k residual`, and
the extraction is shared across every logo query (q2a, q2b, q7).

## Three agents (mirroring GenDB's layout)

```
agents/
  schema-designer/   {index.mjs, prompt.md, user-prompt.md}   # decomposability + schema
  extractor/         {index.mjs, prompt.md, user-prompt.md}   # small VLM → attribute table
  code-generator/    {index.mjs, prompt.md, user-prompt.md}   # relational program + residual
orchestrator.mjs     # wires A→B→C, imports ../gendb/shared.mjs
semdb.config.mjs     # models / thresholds
```

## Runnable PoC (no GPU, no API key)

```bash
bash src/semdb/poc/run_poc.sh
```

Extracts a synthetic image corpus with a deterministic small-VLM stand-in,
compiles the airline-logo join, runs the naive M×N oracle, and **asserts the two
produce identical results** while reporting the model-call reduction (60 → 12 on
the sample). Swap `mock_extractor.py` for `vlm_extractor.py` (SmolVLM-256M /
Qwen3-VL-2B) to run the same contract on real pixels.

Dry-run the live agent pipeline (renders the three prompts, no credentials):

```bash
node src/semdb/orchestrator.mjs --query q7 --dry-run
```

## Use it on your own SemBench data
Point it at your query folder + data folder and a query id — see
[`USAGE.md`](USAGE.md):

```bash
node src/semdb/orchestrator.mjs --query q3a \
  --query-dir /localhome/hza214/SemBench/files/mmqa/query/bigquery \
  --data-dir  /localhome/hza214/SemBench/files/mmqa/data/sf_200
```

## Docs
- [`USAGE.md`](USAGE.md) — how to run it on your SemBench paths (both autonomous and step-by-step).
- [`docs/PLAN.md`](docs/PLAN.md) — implementation roadmap (PoC → real VLM → full SemBench → optimizer).
- [`docs/SEMBENCH_ANALYSIS.md`](docs/SEMBENCH_ANALYSIS.md) — which SemBench queries compile, and why.
- [`docs/Q2A_COMPILED_EXAMPLE.md`](docs/Q2A_COMPILED_EXAMPLE.md) — the airline-logo join, phase by phase.

## Status
Phase 0 (PoC) complete and verifiable. Phase 1 (real small-VLM extraction on an
MMQA image subset) is the next step — the extractor and gating are already in
place. See `docs/PLAN.md`.
