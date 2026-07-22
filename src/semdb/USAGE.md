# Using SemDB with your SemBench data

You provide two paths — the query folder and the data folder — plus a query id.

```
query folder :  /localhome/hza214/SemBench/files/mmqa/query/bigquery      (*.sql)
data folder  :  /localhome/hza214/SemBench/files/mmqa/data/sf_200         (*.csv + images/)
```

The pipeline has three agent phases and then execution:

```
[A] Schema Designer  →  schema.json           (which slot to extract, normalization, residual θ)
[B] Extractor        →  <table>_attrs.json     (a SMALL model reads each row/image, ONCE)
[C] Code Generator   →  compiled_<q>.py        (the semantic operator becomes relational code)
    Execute          →  run compiled_<q>.py on your real CSVs; verify vs raw_results
```

## 0. Install

```bash
# repo root — the agent side (Claude Agent SDK)
npm install
# the extraction side (only where you run the small model; needs GPU + HuggingFace)
pip install torch transformers pillow accelerate
```

The agent phases need Claude credentials (`ANTHROPIC_API_KEY`, same as GenDB).
The extraction phase needs `huggingface.co` reachable to fetch the small model.

### Choosing the agent provider (Claude or Codex)
The three compiler agents run on either provider — set it in `semdb.config.mjs`
(`defaults.agentProvider`) or per run with `--agent-provider`:

```bash
# use OpenAI Codex (gpt-5.6-codex) instead of Claude
node src/semdb/orchestrator.mjs --query q3a --query-dir <...> --data-dir <...> \
     --agent-provider codex
```

Codex needs `@openai/codex-sdk` (already in `package.json`) and Codex auth. The
model per agent lives in `semdb.config.mjs → defaults.providers.codex` — change
the single `model:` line to re-point every agent, or edit `agentModels` per agent.
`--model <id>` forces one model for all agents (testing). The *extraction* small
models (`defaults.extraction.*`) are independent of this choice.

---

## Mode 1 — autonomous (agents do A→C for you)

```bash
node src/semdb/orchestrator.mjs \
  --query q3a \
  --query-dir /localhome/hza214/SemBench/files/mmqa/query/bigquery \
  --data-dir  /localhome/hza214/SemBench/files/mmqa/data/sf_200
```

The orchestrator reads the real `q3a.sql` and the real table headers, then runs
the three agents into `src/semdb/runs/mmqa-q3a/` (`schema.json`,
`<table>_attrs.json`, `compiled_q3a.py`). It prints the exact `extract.py` and
run commands to finish.

Preview everything without credentials (renders the three prompts):

```bash
node src/semdb/orchestrator.mjs --query q3a --query-dir <...> --data-dir <...> --dry-run
```

---

## Mode 2 — step by step (explicit control of the small model)

### A. Design the schema (once per query family)
Run Phase A of Mode 1, or hand-write `schema.json` (see
`examples/mmqa_q3/schema.json` and `poc/expected/schema.json` for the shape).

### B. Extract with a small model (once per corpus, reused by every query)

**Text queries** — q1, q3a–g, q4, q5, q6 (a small text LLM reads a text column):

```bash
python3 src/semdb/extract.py \
  --schema   src/semdb/runs/mmqa-q3a/schema.json \
  --table    /localhome/hza214/SemBench/files/mmqa/data/sf_200/lizzy_caplan_text_data.csv \
  --modality text --id-col title --text-col text \
  --model    Qwen/Qwen2.5-0.5B-Instruct \
  --out      movie_attrs.json
```

**Image queries** — q2a, q2b, q7 (a small VLM reads each image):

```bash
python3 src/semdb/extract.py \
  --schema    src/semdb/runs/mmqa-q7/schema.json \
  --table     /localhome/hza214/SemBench/files/mmqa/data/sf_200/thalamusdb_images.csv \
  --modality  image --id-col uri --image-col uri \
  --image-dir /localhome/hza214/SemBench/files/mmqa/data/sf_200/images \
  --model     HuggingFaceTB/SmolVLM-256M-Instruct \
  --out       img_attrs.json
```

`--image-dir` lets the driver resolve `uri` → `images/<basename>`; if the column
already holds a local path, omit it. Use `--limit N` to smoke-test on a few rows.

**Recommended: serve the model with vLLM + guided JSON decoding.** This constrains
generation to the schema, so output is *always* valid JSON (zero parse failures)
and requests are batched/fast:

```bash
# 1) serve the small VLM (separate terminal)
vllm serve Qwen/Qwen3-VL-2B-Instruct --port 8000

# 2) point the extractor at it — same flags, plus --endpoint
python3 src/semdb/extract.py \
  --schema    src/semdb/runs/mmqa-q2a/schema.json \
  --table     .../sf_200/thalamusdb_images.csv \
  --modality  image --id-col image_filepath --image-col image_filepath \
  --image-dir .../sf_200/images \
  --model     Qwen/Qwen3-VL-2B-Instruct \
  --endpoint  http://localhost:8000/v1 \
  --out       img_attrs.json
```

`--endpoint` builds a JSON Schema from `schema.json` and sends it as `guided_json`
(also `response_format: json_schema`), so the server can't emit malformed output.
`--api-key` defaults to `EMPTY` (vLLM). Local (no-endpoint) runs still work; for a
weak local model without guided decoding, add `--prompt-style simple`.

### C. Run the compiled query on your real data

The Code Generator writes `compiled_<q>.py` referencing the attribute table.
Run it, then compare against SemBench's ground truth in
`files/mmqa/raw_results/`. See the two worked, committed examples:

- `examples/mmqa_q3/` — text, genre family, run for real (`bash run_q3.sh`).
- `poc/` — the image logo-join compiled plan vs the naive M×N oracle (`bash run_poc.sh`).

---

## Which file backs each SemBench table

| SQL identifier | CSV in `data/sf_200/` | queries |
|----------------|-----------------------|---------|
| `mmqa.tampa_international_airport` | `tampa_international_airport.csv` (Airlines, Destinations, …) | q5, q6, q7 |
| `mmqa.ap_warrior` | `ap_warrior.csv` (ID, Track, …) | q2a, q2b |
| `mmqa.images` | `thalamusdb_images.csv` (uri, ref) + `images/` dir | q2a, q2b, q7 |
| `mmqa.lizzy_caplan_text_data` | `lizzy_caplan_text_data.csv` (title, text) | q3a–g, q4 |
| `mmqa.ben_piazza_text_data` / `ben_piazza` | `ben_piazza_text_data.csv`, `ben_piazza.csv` | q1 |

> `mmqa.images` in the SQL maps to `thalamusdb_images.csv` on disk — pass that
> path to `--table`.

## What runs where

| Step | Needs |
|------|-------|
| `--dry-run` (see prompts) | nothing |
| Phase A/B/C agents (Mode 1) | Claude credentials + `npm install` |
| `extract.py` real small model | GPU-friendly host + `pip install` + HuggingFace access |
| compiled `.py` execution + validate | plain Python 3 |
| offline demos (`poc/`, `examples/mmqa_q3/`) | plain Python 3 (no GPU, no keys) |

If HuggingFace is blocked on your host (as in this sandbox), the compiled-query
logic and validation still run; only Phase B needs the model. `examples/mmqa_q3/`
shows the full loop already executed with a small model.
