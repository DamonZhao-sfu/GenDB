# Integration Plan: Experience Graphs → GenDB Memory System

**Paper:** *Experience Graphs: The Data Foundation for Self-Improving Agents* (arXiv:2606.29823)
**Target:** GenDB Memory v3 (6-layer HAG + Memory Manager differential learning)
**Design constraints:** file-backed JSON (no DB engine), large artifacts by reference, non-breaking at every phase, memory stays opt-in (`--memory-dir`, disabled by default).

---

## 0. Thesis in one paragraph (and why GenDB is a perfect fit)

GenDB already persists search *outcomes* — the best `.cpp`, distilled Claude skills, consolidated `L0` nodes — but throws away the search *process*: the branching tree of attempts, their parents, their sibling alternatives, and the reward deltas along each edge. The paper's claim is that this process is the valuable substrate and must be **first-class queryable state**, not ephemeral logs, because "search over the experience graph is a database access pattern."

GenDB's optimize loop is literally a tree search that we currently flatten. Each iteration warm-starts from **best-so-far** — `orchestrator.mjs` copies the best passing code as the starting point (`orchestrator.mjs:1523`, `:1612`, *"Copy best code as starting point"*). So every `iter_N` has a real parent, but `optimization_history.json` records only a **linear list** and drops the edge. The plan makes that tree explicit as the raw layer beneath `L0`, captures it live, and consumes it for correct credit assignment.

**Worked example — `output/deprecated/sec-edgar/2026-03-08T22-08-07/queries/Q24`:**

| iter | timing_ms | improved | true parent (best-so-far at start) |
|---|---|---|---|
| 0 | 447.97 | ✓ | — (root) |
| 1 | 2112.34 | ✗ | 0 |
| 2 | **157.13** | ✓ | 0 |
| 3 | 560.45 | ✗ | 2 |
| 4 | 287.84 | ✗ | 2 |
| 5 | **127.38** | ✓ (best) | 2 |

The linear Memory Manager diffs `iter_5` against `iter_4` (its list-predecessor, a **287ms dead-branch regression**). The true parent of `iter_5` is `iter_2` (157ms). **Correct credit assignment requires the edge**, and `iter_3`/`iter_4` are the *negative* evidence (siblings that failed from the same base) the paper wants surfaced — today they collapse into `gotchas.md` prose.

---

## 1. Four-table schema → GenDB artifact mapping

The paper's `Tasks / Sessions / Nodes / Prompts` map cleanly onto artifacts GenDB already writes:

| Paper table | GenDB meaning | Backed by (existing artifacts) | Stable key |
|---|---|---|---|
| **Tasks** | An optimization problem: (benchmark, query template, scale factor) | `L1` template node + `run.json.workload/scaleFactor` + `queries/<Q>/template.sql` | `task_id = <benchmark>__<template‑or‑queryId>__sf<N>` |
| **Sessions** | One run's attempt at one task = the per-query optimize loop inside a run | `run.json` (`runId, model, agentProvider, optimizationTarget`) + hardware fingerprint + `queries/<Q>/` | `session_id = <runId>__<queryId>` |
| **Nodes** | One iteration = one search-tree node (attempt + artifact + reward) | `queries/<Q>/iter_<N>/{q.cpp, plan.json, execution_results.json, results/}` + `optimization_history.iterations[N]` | `node_id = <session_id>__iter_<N>` |
| **Prompts** | The agent I/O that produced a node | `strategyNote` (already in history `.strategy`), agent telemetry, per-iteration planner/optimizer/codegen prompt+response | referenced from node (`artifacts.prompts`) |

**Fields we already have per Node** (from real `optimization_history.iterations[]`):
`iteration, improved, categories[], timing_ms, cold_timing_ms, hot_timing_ms, validation, strategy, operation_timings{}`.

**Fields we must *add* per Node** (the discarded search evidence): `parent_node_id`, reward deltas (`delta_vs_parent_pct`, `delta_vs_root_pct`), artifact **references** (paths, not blobs), and backprop rollups (`visit_count`, `best_descendant_ms`).

**Prompts gap:** the optimizer/codegen prompt+response are currently transient (only `strategy` text survives into history). Phase (a) persists a small `iter_<N>/agent_io.json` (planner+optimizer+codegen request/response refs) so the Prompts table is real, not synthesized.

---

## 2. Where the experience graph sits relative to the HAG

**It is the raw substrate beneath `L0`.** Introduce it as a new bottom layer — call it **`L-1` (trace layer)** — physically separate from `graph/nodes/` so nothing about existing L0–L5 changes:

```
L5 Performance Principles      ┐
L4 Optimization Strategies     │  distillations (skills / HAG)
L3 Operator Techniques         │  — unchanged
L2 Sub-Structure Patterns      │
L1 Query Templates             │
L0 Query Instances (best/run)  ┘
        ▲  SUMMARIZES (roll-up of the best path)
────────┴──────────────────────────────────────
L-1 Experience Graph (Tasks/Sessions/Nodes/Prompts)   ← NEW raw substrate
    every attempt, reward, parent edge, sibling set, dead branch
```

- `L0` becomes a **`SUMMARIZES` roll-up** over an `L-1` session tree (its `optimization_trajectory` is the *best path*, not the whole tree). L0's `best_cpp_path` already points at `iter_N/…` — we just add `experience_ref = best_node_id`.
- The Memory Manager's distillation turns from a **linear scan** into a **graph reduction** over `L-1` (diff along true edges; contrast winning node vs failed siblings).

**Search evidence currently discarded (and where it will now live):**

| Evidence | Today | With `L-1` |
|---|---|---|
| Parent lineage (which attempt each derived from) | lost (flat list) | `node.parent_node_id` + `derives_from` edge |
| Sibling attempts from same base (Q24 `iter_3/4/5` ← `iter_2`) | lost | `getSiblings(node)` |
| Regressions *with the reason they failed* | folded into prose `gotchas.md` | node w/ `improved:false` + `strategy` (structured negative evidence) |
| Abandoned/dead branches | lost | subtrees with no improving descendant (`best_descendant_ms` unbeaten) |
| Reward delta along an edge | approximated linearly | `delta_vs_parent_pct` on the true edge |

---

## 3. Schema, storage layout, retrieval API

### 3.1 Storage layout (mirrors HAG conventions; artifacts by reference)

```
gendb-memory/experience/
  tasks/<task_id>.json
  sessions/<session_id>.json
  nodes/<session_id>/<node_id>.json        # one file per attempt (concurrency-safe)
  edges.jsonl                              # append-only: derives_from, warm_started_from
  index.json                               # task→sessions, session→nodes, stats
```

Design choices dictated by the codebase:
- **By reference, never inline** (same discipline as `storage-pool.mjs` symlinks and `L0.best_cpp_path`): node JSON stores `artifacts.{cpp,plan,exec_results,prompts}` as paths into `output/**`, plus enough scalar reward to be useful even if the referenced dir is later pruned/gitignored.
- **Append-only `edges.jsonl`, per-session node files.** The existing `graph.mjs addEdge` rewrites `edges.json` wholesale — unsafe under Phase-2 parallelism (`maxConcurrent` queries writing at once). One node file per `(session,node)` + append-only edges = no write races; `index.json` is rebuilt idempotently (same pattern as `graph.mjs updateIndex`).

### 3.2 Node schema (`L-1` trace node)

```json
{
  "node_id": "2026-03-08T22-08-07__Q24__iter_5",
  "session_id": "2026-03-08T22-08-07__Q24",
  "task_id": "sec-edgar__Q24__sf3",
  "iteration": 5,
  "parent_node_id": "2026-03-08T22-08-07__Q24__iter_2",
  "reward": {
    "timing_ms": 127.38, "hot_ms": 127.38, "cold_ms": null,
    "validation": "pass", "improved": true, "is_best_in_session": true,
    "delta_vs_parent_pct": -18.9, "delta_vs_root_pct": -71.6
  },
  "rollup": { "visit_count": 1, "best_descendant_ms": 127.38 },
  "categories": ["storage_extension"],
  "strategy": "Root cause: ... Strategy category E/A: ...",
  "operation_timings": { "data_loading": 0.11, "main_scan": 56.89, "total": 125.36 },
  "artifacts": {
    "cpp":          "output/.../queries/Q24/iter_5/q24.cpp",
    "plan":         "output/.../queries/Q24/iter_5/plan.json",
    "exec_results": "output/.../queries/Q24/iter_5/execution_results.json",
    "prompts":      "output/.../queries/Q24/iter_5/agent_io.json",
    "available":    true
  },
  "created_at": "2026-03-08T22:20:00.000Z"
}
```

### 3.3 Session & Task schemas

```json
// sessions/<session_id>.json
{
  "session_id": "2026-03-08T22-08-07__Q24", "task_id": "sec-edgar__Q24__sf3",
  "run_id": "2026-03-08T22-08-07", "query_id": "Q24",
  "model": "...", "agent_provider": "claude", "optimization_target": "hot",
  "hardware_fingerprint": "64c-376g-...", "root_node_id": "...__iter_0",
  "best_node_id": "...__iter_5", "node_ids": ["...iter_0", "...iter_5"],
  "status": "completed", "started_at": "...", "completed_at": "..."
}
// tasks/<task_id>.json
{
  "task_id": "sec-edgar__Q24__sf3", "benchmark": "sec-edgar", "query_id": "Q24",
  "scale_factor": 3, "template_signature": { /* reuse retrieval.mjs features */ },
  "session_ids": ["2026-03-08T22-08-07__Q24", "..."],
  "global_best": { "session_id": "...", "node_id": "...", "timing_ms": 127.38 }
}
```

Edge record (`edges.jsonl`, one JSON object per line):
```json
{"type":"derives_from","from":"...iter_5","to":"...iter_2","reward_delta_pct":-18.9,"created_at":"..."}
{"type":"warm_started_from","from":"<new session root>","to":"<prior best node>","created_at":"..."}   // Phase (c)
```

### 3.4 Retrieval API — new module `src/gendb/memory/experience.mjs`

```js
// ---- init & identity (pure) ----
export async function initExperienceDirs(memoryDir)
export function taskId(benchmark, queryId, scaleFactor)            // "bench__Q__sfN"
export function sessionId(runId, queryId)                          // "runId__Q"
export function nodeId(sessionId, iteration)                       // "session__iter_N"

// ---- capture (write side; called by orchestrator, guarded by --memory-dir) ----
export async function openSession(memoryDir, meta)                 // meta: {runId,queryId,benchmark,scaleFactor,model,agentProvider,optimizationTarget,hardwareFingerprint,rootNodeId}
export async function recordNode(memoryDir, node)                  // writes node file, appends derives_from edge, updates index (idempotent)
export async function closeSession(memoryDir, sessionId, {bestNodeId, status})
export async function backpropReward(memoryDir, leafNodeId)        // MCTS-style: walk ancestors, update visit_count & best_descendant_ms

// ---- retrieval ("database access patterns" from the paper) ----
export async function getNode(memoryDir, nodeId)
export async function getAncestors(memoryDir, nodeId)              // node → root path (array, root last)
export async function getChildren(memoryDir, nodeId)
export async function getSiblings(memoryDir, nodeId)               // same parent_node_id, excl. self
export async function getBestNode(memoryDir, sessionId)            // argmin reward.timing_ms where validation=pass
export async function getSessionTree(memoryDir, sessionId)         // {nodes[], edges[]} for one query's search
export async function findSimilarTasks(memoryDir, {benchmark, queryId, sql, scaleFactor, k}) // cross-session, reuses retrieval.classifyQuery structural features
export async function bestSiblingAcrossTasks(memoryDir, taskId)    // global best node for warm-start (Phase c)
```

`backpropReward` writes `rollup.{visit_count,best_descendant_ms}` onto ancestors, so a query like *"which early-iteration strategy most reliably leads to the global best"* becomes a scan over root children ordered by `best_descendant_ms` — the paper's backpropagation-as-query.

Expose the read side through `src/gendb/memory/index.mjs` (re-export) and add an `experience` subcommand to `search-tool.mjs` so agents/CLI can query trees the same way they call `load_memory`.

---

## 4. Minimal, non-breaking wiring

### 4.1 `orchestrator.mjs` — capture during the optimize loop (additive only)

All new code is `if (args.memoryDir) { try { … } catch (e) { /* non-fatal */ } }` — identical to the existing memory-is-optional discipline, so a capture failure never breaks a run and behavior is byte-identical when memory is off (the default).

One import + one thin helper:

```js
import * as xg from "./memory/experience.mjs";

// reward + parent are already in scope at each capture site
async function captureIteration(args, sessionId, iteration, parentIteration, iterDir, histEntry) {
  if (!args.memoryDir) return;
  try {
    await xg.recordNode(args.memoryDir, {
      node_id: xg.nodeId(sessionId, iteration),
      session_id: sessionId,
      task_id: xg.taskId(args.targetBenchmark, /*queryId*/, args.scaleFactor),
      iteration,
      parent_node_id: parentIteration == null ? null : xg.nodeId(sessionId, parentIteration),
      reward: { timing_ms: histEntry.timing_ms, hot_ms: histEntry.hot_timing_ms,
                cold_ms: histEntry.cold_timing_ms, validation: histEntry.validation,
                improved: histEntry.improved },
      categories: histEntry.categories, strategy: histEntry.strategy,
      operation_timings: histEntry.operation_timings,
      artifacts: { cpp: resolve(iterDir, `${queryId.toLowerCase()}.cpp`),
                   plan: resolve(iterDir, "plan.json"),
                   exec_results: resolve(iterDir, "execution_results.json"),
                   prompts: resolve(iterDir, "agent_io.json") },
    });
  } catch (e) { console.warn(`[xg] capture non-fatal: ${e.message}`); }
}
```

Insertion points (exact, all guarded):
1. **iter_0 baseline** (`orchestrator.mjs:~1447`): `openSession(...)` then `captureIteration(..., iteration=0, parentIteration=null, ...)`.
2. **Track the true parent.** Where best code is copied as the base (`:1523` / `:1612`), record `const baseIterationForAttempt = bestResult?.iterations ?? 0;` for this attempt.
3. **Optimize loop** (right after `writeFile(historyPath, …)` at `:1983`): `await captureIteration(args, sessionId, iteration, baseIterationForAttempt, optIterDir, lastHistEntry);`.
4. **Escalation branch** (after `:1594`): same call with the escalation base.
5. **Query completion** (after the loop): `await xg.closeSession(...); await xg.backpropReward(args.memoryDir, xg.nodeId(sessionId, bestResult.iterations));`.

No change to `optimization_history.json` or `run.json` schemas. `agent_io.json` (Prompts) is written by the same code paths that already know the planner/optimizer/codegen request+response — a 3-line `writeFile` in the loop, also guarded.

### 4.2 Memory Manager — consume the graph for credit assignment

- **`src/gendb/agents/memory-manager/index.mjs`**: pass two new template vars — `{{experience_dir}}` and, per query, `{{session_tree}}` (JSON from `getSessionTree`) — into `user-prompt.md`.
- **`user-prompt.md` edits** (behavior change, not structure):
  - Step 3d → *"For each improving node, diff its `.cpp` against its **`parent_node_id`** (its true base), **not** its list-predecessor."*
  - New step → *"Use `getSiblings` to contrast the winning strategy against sibling attempts that regressed from the **same** parent; record each failed sibling's `strategy` as structured negative evidence (cause → effect) rather than prose."* (This is the paper's causal-contrast credit assignment.)
  - L0 write → add `experience_ref: <best_node_id>` and `session_tree_ref`, making L0 an explicit roll-up (`SUMMARIZES`) of `L-1`.

Everything the agent needs is already file-backed; it can also call the `search-tool.mjs experience` subcommand instead of receiving inlined trees, keeping context small.

---

## 5. Phasing (each phase ships independently; pipeline green throughout)

### Phase (a) — Capture + backfill  *(foundation)*
- **New:** `src/gendb/memory/experience.mjs` (schema + write API + read API §3.4).
- **New:** `scripts/backfill_experience.mjs` — reconstruct trees from every existing `output/**/queries/*/optimization_history.json`. **Parent rule mirrors the live loop:** parent(iter N) = the passing iteration `< N` with the best `timing_ms` for the run's `optimization_target` (i.e., best-so-far by `checkExecutionImprovement`'s own rule). Root = `iter_0`.
- **Wire:** §4.1 capture (guarded).
- **Validate:** run backfill over `output/deprecated/**`; assert on Q24: `parent(iter_5)==iter_2`, `parent(iter_3)==iter_2`, `getSiblings(iter_5) ⊇ {iter_3, iter_4}`, `getBestNode(session)==iter_5`. Add `src/gendb/memory/experience.test.mjs` covering Q24 + a first-iteration-success query (single-node tree) + an all-regression-tail query.

### Phase (b) — Richer cross-session retrieval
- Implement `findSimilarTasks` / `bestSiblingAcrossTasks` on top of `retrieval.mjs` structural features (reuse `classifyQuery`'s Jaccard signature; no new deps).
- **Validate:** for tpc-h `Q1 sf10`, returns prior `Q1` sessions ranked by structural similarity with correct `global_best`.

### Phase (c) — Closed-loop reuse (*warm-start from the best prior sibling*)
- In Phase-2, before `iter_0`, call `bestSiblingAcrossTasks(task)`; if a prior best node exists, seed the optimizer/codegen starting point with its `cpp`+`strategy` and add a `warm_started_from` edge. Slots into the existing `instantiate`/`rebenchmark` action path in `inspectWorkloadState` (`orchestrator.mjs:842`).
- **Validate:** A/B on held-out queries — iterations-to-best and wall-clock **with** vs **without** warm-start, over real reruns.

### Phase (d) — Embeddings & governance *(optional)*
- Replace Jaccard task-similarity with real embeddings behind the same `findSimilarTasks` signature; add retention/pruning of dead branches, `artifacts.available` reconciliation, size caps, and provenance fields. No API change.

---

## 6. Risks, consistency, validation

- **Concurrency (top risk).** Phase-2 optimizes up to `maxConcurrent` queries in parallel. Never reuse `graph.mjs`'s wholesale `edges.json` rewrite — use per-session node files + append-only `edges.jsonl`; make `recordNode`/index updates idempotent so a rebuild after a crash converges. A session only ever writes its own files.
- **Non-breaking guarantee.** All capture behind `args.memoryDir` (off by default), all `try/catch` non-fatal, zero edits to `optimization_history.json`/`run.json` shapes. With memory disabled the run is byte-for-byte unchanged.
- **By-reference fragility.** `output/**` dirs are pruned/gitignored; store enough scalar reward in the node to keep credit-assignment and retrieval meaningful without the `.cpp`, and set `artifacts.available=false` when the referenced path is missing (backfill + Phase-d reconciliation).
- **Parent-reconstruction fidelity.** Backfill must use the *same* best-so-far rule the live loop uses (`checkExecutionImprovement`, per `optimization_target`), or historical trees won't match freshly-captured ones. Encode the escalation-seed rule (`:1523`) too. Cross-check: for runs captured live *and* backfilled, trees must be identical.
- **Reward semantics.** hot vs cold depends on `run.json.optimization_target`; store both, compare only within the same target; treat `validation!="pass"` nodes as non-selectable for `getBestNode`/warm-start but keep them as negative evidence.
- **Validation harness.** `node scripts/backfill_experience.mjs --dry-run` over all `output/**` (report tree shape per query, no writes); `experience.test.mjs` asserts ancestors/siblings/best-node/backprop on the three canonical shapes; Phase (c) ships an A/B script comparing iterations-to-best against real `optimization_history.json` baselines.

---

## Deliverables checklist

| File | Phase | Purpose |
|---|---|---|
| `src/gendb/memory/experience.mjs` | a | schema + capture + retrieval API (§3.4) |
| `src/gendb/memory/experience.test.mjs` | a | Q24 / first-iter / all-regression assertions |
| `scripts/backfill_experience.mjs` | a | reconstruct trees from existing `optimization_history.json` |
| `orchestrator.mjs` (guarded capture, ~6 insertions) | a | live capture (§4.1) |
| `agents/memory-manager/{index.mjs,user-prompt.md}` | a→ | consume tree for parent/sibling credit assignment (§4.2) |
| `memory/index.mjs`, `memory/search-tool.mjs` | b | expose read API + `experience` CLI/agent subcommand |
| `orchestrator.mjs` warm-start hook (`inspectWorkloadState` path) | c | closed-loop reuse |
| embeddings + governance | d | optional |
