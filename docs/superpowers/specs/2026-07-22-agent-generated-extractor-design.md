# Design — Agent-generated thin extractor over a shared `semextract` engine

- **Date:** 2026-07-22
- **Branch:** `claude/semantic-operator-compilation-hiplmc`
- **Status:** IMPLEMENTED (2026-07-22) — engine + thin driver contract verified against fake vLLM; orchestrator dry-run renders the extractor template; agent-generated full run pending a real Extractor-agent invocation.

## Context & motivation

SemDB has three pipeline agents, each scaffolded under `src/semdb/agents/`:
`schema-designer`, `extractor`, `code-generator`. Each has a system `prompt.md` and a
`user-prompt.md` template rendered with variables.

Two of the three follow the intended pattern — an agent renders its template and
**generates code**:
- **Schema Designer** (Phase A, once per corpus, **sees ALL queries over the corpus**)
  produces `schema.json`.
- **Code Generator** (Phase C, per query) writes `compiled_<q>.py`, validates it against a
  baseline oracle, and imports the shared `semruntime` library for the risky residual model
  calls.

The **Extractor** breaks the pattern. Its agent is scaffolded (`prompt.md` literally says
*"Emit a small, deterministic extraction driver…"*) but **not wired**: `orchestrator.mjs`
Phase B (`ensureCorpus`, ~line 377) bypasses the agent and hardcodes
`spawnSync("python3", ["extract.py", …])`, a hand-written **static** driver. The agent config
is imported only for a model-name display line.

This design finishes the original intent: the Extractor agent should **generate** the
extraction driver — the way the Code Generator generates compiled queries — while a shared,
tested engine holds the mechanical plumbing (including the concurrency/error-handling work
already landed in `extract.py`).

## Goals

1. Phase B generates a per-corpus extraction driver via the Extractor agent from a template +
   `schema.json` (+ corpus headers + modality).
2. Schema — hence extraction — is driven by **all** of the workload's queries (already true
   via Schema Designer; this design keeps it).
3. All mechanical plumbing lives in one shared, tested engine (`semextract.py`), reused by
   every generated driver — the recently-added concurrency, guided-JSON, per-row error
   tolerance, systemic-failure guard, `.partial` checkpoint, and `<out>.meta.json`.
4. The generated driver owns exactly the **semantic/corpus-specific** decisions: prompt
   construction (incl. injecting `structured_context_columns`), column/key mapping, and
   modality preprocessing.

## Non-goals

- Small→strong **escalation** policy (deferred; keep it simple).
- A correctness **oracle** for extraction quality (none exists; see Risks).
- The other 6 SemBench scenarios (still mmqa-first).

## Architecture

```
Schema Designer  ──sees ALL queries──▶  schema.json            (WHAT to extract; unchanged)
Extractor agent  ──schema + headers──▶  extract_<corpus>.py    (HOW: thin, generated per corpus)
                                              │ imports
                                              ▼
                     semextract.py  ← shared engine (today's extract.py plumbing)
```

Responsibility split:

| Concern | Owner |
|---|---|
| endpoint HTTP, `--concurrency` ThreadPool, local-HF serial fallback | `semextract.py` |
| guided-JSON (`build_json_schema`), generic `parse_json_object` | `semextract.py` |
| `.partial` checkpoint, systemic-failure guard, `<out>.meta.json` | `semextract.py` |
| prompt construction + `structured_context_columns` injection | generated driver |
| column/key mapping (corpus cols → id/text/image/context) | generated driver |
| modality preprocessing (image resolve/crop, long-text chunk/truncate) | generated driver |

## Components

### 1. `semextract.py` (shared engine, refactored from `extract.py`)

Moves the current mechanical functions into an importable library:
`gen_endpoint`, `gen_text`, `gen_image`, `load_text_model`, `load_image_model`,
`build_json_schema`, `parse_json_object` (+ `_find_json`/`_repair_json`/`_norm_key`),
`resolve_image_path`, `_data_url`, and the run loop (`process_row`/`record`/`checkpoint`/
systemic guard/meta). Public entry point:

```python
def run(driver, schema, table_path, out_path, *, modality, model, endpoint=None,
        api_key="EMPTY", concurrency=8, theta=None, image_dir=None, limit=0,
        max_new_tokens=128) -> dict:  # returns the meta dict, also written to out+".meta.json"
```

Driver interface (duck-typed / `typing.Protocol`):

```python
class ExtractDriver(Protocol):
    def map_columns(self, header: list[str]) -> dict: ...
        # -> {"id": str, "text": str|None, "image": str|None, "context": list[str]}
    def preprocess(self, row: dict, cols: dict) -> dict: ...
        # -> {"text": str}  OR  {"image_path": str}   (engine feeds this to the model)
    def build_prompt(self, schema: dict, row: dict, cols: dict) -> str: ...
        # rendered prompt incl. any context columns
```

`run()` reads the table, calls `driver.map_columns` once, then per row (concurrent when
`endpoint` set, serial for local HF): `driver.preprocess` → `driver.build_prompt` → model call
→ parse → tally → checkpoint; finally the systemic-failure guard + meta write. **A `Default`
driver ships in the library** reproducing today's behavior (generic `build_prompt` from schema,
heuristic column mapping, image-resolve/text-read preprocessing) so a generic corpus works
without agent generation and serves as the reference the agent starts from.

### 2. `extract_<corpus>.py` (generated per corpus by the Extractor agent)

The 3 hooks specialized for the corpus + an entry point:

```python
import semextract
class Driver:                     # implements map_columns / preprocess / build_prompt
    ...
if __name__ == "__main__":
    # argparse: <table.csv> <attrs.json> [--schema S --endpoint URL --api-key K
    #                                     --model M --concurrency N --theta T --image-dir D]
    semextract.run(Driver(), schema, args.table, args.out, modality="image", ...)
```

Fixed CLI signature so the orchestrator invokes it exactly like `compiled_<q>.py`. Thin by
design → little for the agent to get wrong.

### 3. `orchestrator.mjs` Phase B rewrite (`ensureCorpus`)

Replace the hardcoded `extract.py` spawn with the two-step agent pattern already used for
Phase C:
1. **Generate** (agent): render `extractor/user-prompt.md` → `runPhase(extractorConfig, vars)`
   → agent writes `extract_<corpus>.py` (Write/Bash tools). The agent MAY self-test on a
   1–2 row sample, but the authoritative full-corpus run is the orchestrator's (step 2) — the
   heavy concurrent run against vLLM stays orchestrator-controlled, not inside the agent turn.
   Cache: skip if the driver file exists unless `--force`.
2. **Execute** (orchestrator): `spawnSync("python3", ["extract_<corpus>.py", table, attrs,
   "--schema", …, "--endpoint", …, "--api-key", …, "--model", extractModel, "--concurrency",
   …, …])`. Cache: skip if attrs exist unless `--force`.

Both driver and attrs cache **per corpus**, amortized across all its queries (like schema
design). Extractor telemetry flows into `corpusTelemetry` next to `schema_design`.
`corpusCols()` shrinks to model selection (column-guessing moves into the agent, which sees
headers + schema). `extractorConfig` (index.mjs) gains `allowedTools:
[Read,Write,Edit,Glob,Grep,Bash]` and a **strong codegen model** (opus/codex) — note this is
the model that *writes the driver*, distinct from `--extract-model` (the small model the driver
calls at runtime).

### 4. Agent prompts

- `extractor/prompt.md`: update from "emit a driver" to "generate a thin driver implementing
  the `semextract` `ExtractDriver` hooks; import `semextract`; do NOT re-implement HTTP/
  concurrency/parsing — the engine owns those." Emphasize context-column injection and column
  mapping.
- `extractor/user-prompt.md`: add vars for `driver_path`, corpus `header`/`columns`, and the
  `semextract` API contract (mirroring how code-generator's template documents `semruntime`).

## Data flow

1. Phase A → `schema.json` (all queries).
2. Phase B generate → `extract_<corpus>.py` (agent).
3. Phase B execute → orchestrator runs the driver → `<corpus>_attrs.json` + `.meta.json`.
4. Phase C per query → `compiled_<q>.py` reuses attrs (unchanged).

## Validation & error handling

- Structural validation only (no quality oracle): `rows == corpus_size`, every schema column
  present, JSON parses — already asserted by the extractor prompt + engine.
- Runtime robustness from the engine (unchanged from Feature 2): per-row error tolerance,
  systemic-failure guard (exit non-zero + don't write `attrs` so the orchestrator re-runs),
  `.partial` checkpoint with atomic promote.
- Acceptance regression: the generated driver for the `images` corpus must reproduce the
  current `extract.py` attrs on the same corpus + model (byte-comparable), proving the
  refactor is behavior-preserving.

## Migration / backward-compat

- `extract.py` becomes a **thin CLI wrapper** over `semextract.run(Default(), …)` so existing
  manual invocations and any scripts keep working; all logic lives in `semextract.py`.
- Feature-2 behavior (concurrency, guards, checkpoint, meta) is **moved intact** into the
  engine — nothing discarded.

## Risks & mitigations

1. **No extraction correctness oracle.** A subtly-wrong generated driver silently degrades all
   downstream queries. *Mitigations:* keep the driver thin (3 small hooks); structural
   validation; systemic guard; cached, human-inspectable generated file; the library `Default`
   driver as a correct starting reference; the byte-compat regression test above.
2. **Extra agent call per corpus.** One codegen call per corpus, amortized across its queries
   (same economics as schema design). Acceptable.
3. **Agent re-generates on `--force`.** Non-determinism between runs. *Mitigation:* cache the
   driver; only regenerate on `--force`; the regression test gates behavior.

## Testing

- **Engine unit tests** (migrate the Feature-2 fake-server tests to `semextract`): serial vs
  concurrent byte-identical + ordered; sporadic + parse-exception tolerance; systemic-failure
  exit/no-write.
- **Driver-hook unit tests:** `Default` driver reproduces current prompt/column/preprocess
  behavior.
- **Integration:** Phase B generates a driver for a corpus, orchestrator runs it, attrs match
  schema columns and row count.
- **Regression:** generated `images` driver reproduces current `extract.py` attrs.

## Open questions

None blocking. (Escalation, quality oracle, other scenarios: explicitly deferred.)
