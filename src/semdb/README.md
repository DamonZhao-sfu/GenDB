# SemDB — Semantic Operator Compilation & Execution

SemDB answers **semantic operators over multimodal data** (`AI.IF`, `AI.GENERATE`,
semantic join / map / filter / rank / classify) by having agents synthesize **one
offline Python program per query** that composes a local, non-generative operator
library — CLIP / OCR / CV / detectors for images, deterministic string primitives for
text — instead of issuing a VLM call per row.

It is a sibling of [GenDB](../../README.md): the same "agents generate execution code"
idea, but the expensive primitive is a model call, not a disk scan. It reuses GenDB's
Claude Agent SDK plumbing (`src/gendb/providers`, `src/gendb/shared.mjs`) verbatim.

Target workload: [SemBench](https://github.com/SemBench/SemBench).
Theory: semantic-operator decomposition / compilation (arXiv 2607.13407), VADAR
dynamic-API synthesis (arXiv 2502.06787).

## The idea in one query

`AI.IF("does this image show the logo of {airline}?")` joining airlines × images is
**M×N** VLM calls. SemDB replaces it with **zero**: the planner binds the predicate to
a local primitive (`best_ocr_match` over the airline value space read from the
structured column at runtime), the generator writes `solve_q7.py` around it, and the
join/filter/projection happen in plain Python over the primitive's output.

## Layout

```
vadar/                 # THE OPERATOR LIBRARY — everything generated code runs on
  __init__.py          #   ImagePatch / get_encoder / resolve_image_path
  predefined.py        #   the operators: VISION (over an ImagePatch) + TEXT (over str)
  imagepatch.py        #   an image, or a region of one, with the primitives bound to it
  backend.py           #   model loading + raw proxies (CLIP / OCR / CV / YOLO / OWL)
  paths.py             #   corpus reference -> real file
  models/              #   bundled detector weights
  API.md               #   the ImagePatch API spec shown to the agents

agents/
  query-planner/            # SQL -> a typed semantic plan (which primitive, which value space)
  semantic-code-generator/  # plan -> solve_<q>.py + helpers
  semantic-optimizer/       # scored run -> the next action (patch / replan / reprompt)
  vadar-signature/          # propose new helper signatures over the operator library
  vadar-api/                # implement one proposed signature
  vadar-solver/             # write the end-to-end solver
skills/                # the shared instruction bodies those agents load

orchestrator.mjs       # wires the agents, runs the solver, scores, iterates
semdb.config.mjs       # models / effort / thresholds
```

Everything under `vadar/` is **offline**: no VLM, no LLM, no HTTP client, no API key.
`orchestrator.mjs` enforces that on every generated file before executing it
(`offlineVadarViolations`). The endpoint-backed tools — `semvqa.py` (oracle labelling),
`semcaption.py` (captions), `semextract.py` (the guided-JSON client they share) — live
OUTSIDE the package and import from it, never the other way round.

## What a generated program looks like

One import root, and every value space read at runtime:

```python
import sys
sys.path.insert(0, "<semdb_dir>")
from vadar import ImagePatch, get_encoder, resolve_image_path
from vadar.predefined import classify, best_ocr_match, dominant_colors, contains_any
```

## Run it

```bash
./run_image_queries.sh                 # all image scenarios
QUERIES=q2a,q7 ./run_image_queries.sh mmqa
ITERS=0 ./run_image_queries.sh         # single-shot, no refinement loop
AGENT_EXECUTION=agent ./run_image_queries.sh mmqa
                                        # legacy full tool agents for Planner/Optimizer
```

Or drive the orchestrator directly — see [`USAGE.md`](USAGE.md):

```bash
node src/semdb/orchestrator.mjs --benchmark mmqa \
  --sembench-dir /localhome/hza214/SemBench --sf 200 \
  --run --agent-provider vllm \
  --agent-execution structured \
  --base-url http://localhost:8000/v1 \
  --endpoint http://localhost:8000/v1 --api-key EMPTY \
  --oracle-model Qwen/Qwen3.8-27B-FP8 \
  --val-rate 0.05 --max-iterations 5 \
  --out src/semdb/runs/mmqa
```

movie

```
node src/semdb/orchestrator.mjs     --benchmark movie     --sembench-dir /localhome/hza214/SemBench     --sf 1000     --direct     --run     --agent-provider vllm     --base-url http://localhost:8000/v1     --endpoint http://localhost:8000/v1     --api-key EMPTY     --oracle-model Qwen/Qwen3.8-27B-FP8   --val-rate 0.05   --max-iterations 3   --out src/semdb/runs/movie_new
```


The `--endpoint` / `--oracle-model` here are for **validation labelling only** (building
the val set the refinement loop scores against). The generated solver itself never sees
them.

## Docs
- [`USAGE.md`](USAGE.md) — running it on your SemBench paths, autonomous and step-by-step.
- [`vadar/API.md`](vadar/API.md) — the ImagePatch API surface.
- [`docs/SEMBENCH_ANALYSIS.md`](docs/SEMBENCH_ANALYSIS.md) — which SemBench queries compile, and why.

### Historical
`poc/`, `examples/mmqa_q3/`, `docs/Q2A_COMPILED_EXAMPLE.md` and `docs/PLAN.md` document
the earlier **Schema-Designer → Extractor → Code-Generator** pipeline (extract an
attribute table once per corpus, then compile the SQL against it). That pipeline was
removed; those artifacts are self-contained and still run, but they no longer describe
how the system works.

### Deploy the agent model

```
export MODEL_PATH=Qwen/Qwen3.8-27B-FP8

vllm serve $MODEL_PATH \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --served-model-name qwen3.8 \
  --max-num-seqs 32 \
  --max-model-len 262144 \
  --max-num-batched-tokens 16384 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --mm-encoder-tp-mode data \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --default-chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20}'
```

```
RUN_NAME="mmqa-qwen38-structured"
  ITERS=3
  ALL=1 \
  AGENT_EXECUTION=structured \
  PROVIDER=vllm \
  VLLM_BASE_URL=http://localhost:8000/v1 \
  VAL_RATE=0.05 \
  ENDPOINT=http://localhost:8000/v1 \
  ORACLE=Qwen/Qwen3.8-27B-FP8 \
  ITERS="$ITERS" \
  OUT="$PWD/src/semdb/runs/$RUN_NAME" \
  ./src/semdb/run_image_queries.sh mmqa
```

```
RUN_NAME="ecomm-qwen38-structured"
  ITERS=3
  ALL=1 \
  AGENT_EXECUTION=structured \
  PROVIDER=vllm \
  VLLM_BASE_URL=http://localhost:8000/v1 \
  VAL_RATE=0.05 \
  ENDPOINT=http://localhost:8000/v1 \
  ORACLE=Qwen/Qwen3.8-27B-FP8 \
  ITERS="$ITERS" \
  OUT="$PWD/src/semdb/runs/$RUN_NAME" \
  ./src/semdb/run_image_queries.sh ecomm
```