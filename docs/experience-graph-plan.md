# Experience Graph: A Self-Evolving Memory System for GenDB

**Paper:** *Experience Graphs: The Data Foundation for Self-Improving Agents* (arXiv:2606.29823)
**Nature:** **Independent** feature. It does **not** reuse, extend, or depend on GenDB's existing HAG/skills memory (`src/gendb/memory/`). It is its own self-evolving memory system with its own store, its own capture, and its own retrieval.
**Backing store:** SQLite via the built-in `node:sqlite` (Node ≥ 22.5; this repo runs v22.22) — **zero new dependencies**. This makes the paper's thesis literal: *search over the experience graph is a database access pattern.*

---

## 0. Are `vector search + relation join + graph traverse` already integrated?

**No. None of the three are implemented — this is greenfield.** The paper's whole point is that these three should be *first-class database operations* over the trajectory data. GenDB today has none of them; even the existing HAG memory only *approximates* them with weaker mechanisms, and it is a separate subsystem we are explicitly **not** reusing:

| Paper access pattern | In GenDB today | In the existing HAG memory (which we do NOT reuse) | This feature delivers |
|---|---|---|---|
| **Vector search** | ✗ none | Fake: `search-tool.mjs` does lowercase **substring** matching over an `embedding_text` string; `classifyQuery` uses **Jaccard** over SQL features. No embeddings, no vectors. | Real embeddings (`BLOB`) + cosine ranking over Tasks/Nodes (optional `sqlite-vec`) |
| **Relation join** | ✗ none | JSON file scans over `nodes/L*/*.json`; no relational model, no joins | Four-table SQL schema joined by foreign keys (`Tasks⋈Sessions⋈Nodes⋈Prompts`) |
| **Graph traverse** | ✗ none | Flat `edges.json` adjacency list; `getConnectedNodes` = one-hop filter, no lineage | Recursive CTE over `nodes.parent_node_id`: ancestors, descendants, siblings, MCTS backprop |

So: the Experience Graph is the component that *introduces* these three as real DB operations. All three are validated to work on `node:sqlite` in this environment (recursive CTE + JOIN confirmed; vectors via BLOB+cosine).

---

## 1. Thesis, and why GenDB is the ideal host

GenDB's optimize loop **is a tree search that we currently flatten and discard.** Each iteration warm-starts from best-so-far — `orchestrator.mjs:1523` / `:1612` literally *"Copy best code as starting point"* — so every `iter_N` has a real parent, but `optimization_history.json` records only a **linear list** and drops the parent edge, the sibling set, and every dead branch.

**Worked example — real data, `output/deprecated/sec-edgar/2026-03-08T22-08-07/queries/Q24`:**

| iter | timing_ms | improved | true parent (best-so-far at start) |
|---|---|---|---|
| 0 | 447.97 | ✓ | — (root) |
| 1 | 2112.34 | ✗ | 0 |
| 2 | **157.13** | ✓ | 0 |
| 3 | 560.45 | ✗ | 2 |
| 4 | 287.84 | ✗ | 2 |
| 5 | **127.38** | ✓ (best) | 2 |

`iter_3`, `iter_4`, `iter_5` are **three siblings off `iter_2`**; `iter_3/4` are dead branches. The linear history cannot express this. The Experience Graph stores it as first-class, queryable state.

This feature does **not** touch the HAG. It coexists as a second, independent memory (both are opt-in via separate flags); they share nothing.

---

## 2. Four-table schema (strict paper fidelity)

One SQLite database, `gendb-experience/experience.db`, four tables mapped onto GenDB artifacts:

| Paper table | GenDB meaning | Backed by |
|---|---|---|
| **Tasks** | An optimization problem: (benchmark, query template, scale factor) | `run.json.workload/scaleFactor` + `queries/<Q>/template.sql` |
| **Sessions** | One run's attempt at one task (the per-query optimize loop in a run) | `run.json` (`runId, model, agentProvider, optimizationTarget`) + hardware fingerprint |
| **Nodes** | One iteration = one search-tree node (attempt + artifact + reward) | `queries/<Q>/iter_<N>/{*.cpp, plan.json, execution_results.json}` + `optimization_history.iterations[N]` |
| **Prompts** | The agent I/O that produced a node | planner/optimizer/codegen request+response (persisted to `iter_<N>/agent_io/`) + telemetry |

```sql
-- src/gendb/experience/schema.sql
CREATE TABLE IF NOT EXISTS tasks (
  task_id            TEXT PRIMARY KEY,       -- <benchmark>__<queryId|template>__sf<N>
  benchmark          TEXT NOT NULL,
  query_id           TEXT NOT NULL,
  scale_factor       INTEGER NOT NULL,
  sql_text           TEXT,
  template_signature TEXT,                   -- JSON structural features
  embedding          BLOB,                   -- vector(sql_text) for vector search
  global_best_node_id TEXT,                  -- FK -> nodes(node_id)
  created_at TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
  session_id         TEXT PRIMARY KEY,       -- <run_id>__<query_id>
  task_id            TEXT NOT NULL REFERENCES tasks(task_id),
  run_id TEXT, query_id TEXT,
  model TEXT, agent_provider TEXT, optimization_target TEXT,
  hardware_fingerprint TEXT,
  root_node_id TEXT, best_node_id TEXT,
  warm_started_from_node_id TEXT,            -- cross-session reuse edge (Phase c)
  status TEXT, started_at TEXT, completed_at TEXT
);
CREATE TABLE IF NOT EXISTS nodes (
  node_id            TEXT PRIMARY KEY,        -- <session_id>__iter_<N>
  session_id         TEXT NOT NULL REFERENCES sessions(session_id),
  task_id            TEXT NOT NULL REFERENCES tasks(task_id),   -- denormalized for task-scoped joins
  parent_node_id     TEXT REFERENCES nodes(node_id),           -- THE search-tree edge (graph traverse)
  iteration          INTEGER,
  reward_ms REAL, hot_ms REAL, cold_ms REAL,
  validation TEXT, improved INTEGER,          -- 0/1
  is_best_in_session INTEGER,
  delta_vs_parent_pct REAL, delta_vs_root_pct REAL,
  visit_count        INTEGER DEFAULT 0,       -- MCTS backprop rollup
  best_descendant_ms REAL,
  categories TEXT,                            -- JSON array
  strategy   TEXT,                            -- optimizer root-cause text
  embedding  BLOB,                            -- vector(strategy) for vector search
  operation_timings TEXT,                     -- JSON
  cpp_path TEXT, plan_path TEXT, exec_results_path TEXT,  -- artifacts BY REFERENCE
  created_at TEXT
);
CREATE TABLE IF NOT EXISTS prompts (
  prompt_id      TEXT PRIMARY KEY,            -- <node_id>__<agent>
  node_id        TEXT NOT NULL REFERENCES nodes(node_id),
  agent          TEXT,                        -- query_planner | query_optimizer | code_generator
  request_path TEXT, response_path TEXT,      -- BY REFERENCE to on-disk logs
  tokens INTEGER, cost_usd REAL, duration_ms REAL,
  created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_nodes_session ON nodes(session_id);
CREATE INDEX IF NOT EXISTS idx_nodes_parent  ON nodes(parent_node_id);
CREATE INDEX IF NOT EXISTS idx_nodes_task    ON nodes(task_id);
CREATE INDEX IF NOT EXISTS idx_sessions_task ON sessions(task_id);
CREATE INDEX IF NOT EXISTS idx_prompts_node  ON prompts(node_id);
```

The search tree is a **parent-pointer forest** (`nodes.parent_node_id`) — no separate edges table; cross-session reuse is `sessions.warm_started_from_node_id`. Large artifacts (cpp/plan/exec/prompt logs) are stored **by reference** (paths into `output/**`), with enough scalar reward inlined that retrieval and credit assignment work even if the referenced files are later pruned.

---

## 3. The three access patterns as the retrieval API

Module `src/gendb/experience/` — completely standalone (imports only `node:sqlite`, `node:fs`, `node:crypto`):

```
src/gendb/experience/
  schema.sql          -- the four-table DDL above
  store.mjs           -- open/init + capture (write side)
  query.mjs           -- the three access patterns (read side)
  embed.mjs           -- pluggable text -> Float32Array (deterministic default; real embeddings in Phase d)
  cli.mjs             -- `node src/gendb/experience/cli.mjs <subcommand>` for humans/agents
```

### 3.1 `store.mjs` — open + capture (write side)
```js
export function openStore(experienceDir)                 // -> { db } ; runs schema.sql (idempotent), WAL mode
export function upsertTask(store, task)                   // dedup by task_id; (re)compute embedding
export function openSession(store, meta)                  // meta: {runId,queryId,benchmark,scaleFactor,model,agentProvider,optimizationTarget,hardwareFingerprint,rootNodeId,warmStartedFromNodeId?}
export function recordNode(store, node)                    // insert node (with parent_node_id); recompute deltas
export function recordPrompt(store, prompt)                // insert one agent I/O row
export function closeSession(store, sessionId, {bestNodeId, status})
export function backpropReward(store, leafNodeId)         // GRAPH TRAVERSE: walk ancestors, bump visit_count, min best_descendant_ms
```

### 3.2 `query.mjs` — the paper's three database access patterns
```js
// ---- GRAPH TRAVERSE (recursive CTE over parent_node_id) ----
export function getAncestors(store, nodeId)              // node -> root path
export function getDescendants(store, nodeId)
export function getSiblings(store, nodeId)               // same parent_node_id, excl. self
export function getSessionTree(store, sessionId)         // {nodes[], edges[]}

// ---- RELATION JOIN (SQL joins across the four tables) ----
export function getBestNodeForTask(store, taskId)        // tasks JOIN sessions JOIN nodes, argmin reward_ms WHERE validation='pass'
export function getWinningPrompt(store, nodeId)          // nodes JOIN prompts -> the agent I/O that produced the best node
export function getTaskLeaderboard(store, benchmark)     // per-task global best across sessions

// ---- VECTOR SEARCH (embedding BLOB + cosine; optional sqlite-vec) ----
export function searchSimilarTasks(store, queryEmbedding, k)      // cross-session task retrieval
export function searchSimilarStrategies(store, queryEmbedding, k) // "who else hit this bottleneck and how"
```

Example — the "best prior solution for a similar task" query the self-improving loop needs is **all three patterns composed**: vector-search Tasks → join to their best Node → traverse that Node's ancestors to replay the winning strategy path.

---

## 4. Minimal, non-breaking wiring into the pipeline

New flag **`--experience-dir <path>`** (separate from `--memory-dir`; the two subsystems are independent and independently optional). All capture is `if (args.experienceDir) { try { … } catch { /* non-fatal */ } }`, so a capture failure never breaks a run and behavior is byte-identical when the flag is off (the default).

**`orchestrator.mjs` capture (additive, ~6 guarded insertions):**
1. iter_0 baseline (`:~1447`): `upsertTask` + `openSession` + `recordNode(parent=null)`.
2. Where best code is copied as base (`:1523`/`:1612`): record `baseIterationForAttempt = bestResult?.iterations ?? 0` — the **true parent**.
3. After the history write in the optimize loop (`:~1983`) and escalation (`:~1594`): `recordNode({ parent_node_id: nodeId(session, baseIterationForAttempt), … })` + `recordPrompt(...)` for planner/optimizer/codegen.
4. Query completion: `closeSession(bestNodeId)` + `backpropReward(bestNodeId)`.

**No changes** to `optimization_history.json`, `run.json`, the HAG, or the Memory Manager. This system is orthogonal to them.

---

## 5. Phasing (independent, pipeline green throughout)

**Phase (a) — Store + capture + backfill.**
- `src/gendb/experience/{schema.sql,store.mjs,embed.mjs}` + guarded `orchestrator.mjs` capture behind `--experience-dir`.
- `scripts/backfill_experience.mjs`: reconstruct the graph from every existing `output/**/optimization_history.json`. **Parent rule mirrors the live loop:** parent(iter N) = passing iteration `< N` with best `reward_ms` for the run's `optimization_target`.
- **Validate:** backfill `output/deprecated/**`; assert on Q24 `getAncestors(iter_5)=[iter_5,iter_2,iter_0]`, `getSiblings(iter_5)⊇{iter_3,iter_4}`, `getBestNodeForTask=iter_5`; `experience.test.mjs` covers Q24 + a single-iteration success + an all-regression tail.

**Phase (b) — The three access patterns (retrieval).**
- `query.mjs` (graph traverse via recursive CTE, relation joins, vector search via cosine over BLOB embeddings) + `cli.mjs`.
- **Validate:** cross-task retrieval returns correct leaderboard; `searchSimilarTasks` ranks a held-out query's neighbors sensibly; recursive-CTE ancestor/sibling results match backfilled trees.

**Phase (c) — Self-evolving closed loop (warm-start from the best prior sibling).**
- Before iter_0: `searchSimilarTasks` → `getBestNodeForTask` → seed the optimizer/codegen with that node's `cpp_path`+`strategy`; set `sessions.warm_started_from_node_id`. This is where the graph *improves the agent*.
- **Validate:** A/B iterations-to-best **with** vs **without** warm-start on real reruns.

**Phase (d) — Real embeddings & governance (optional).**
- Swap `embed.mjs`'s deterministic default for real embeddings (same signature); add retention/pruning of dead branches, `artifacts.available` reconciliation, DB size caps, provenance.

---

## 6. Risks, consistency, validation

- **`node:sqlite` is experimental.** It works on v22.22 (verified: recursive CTE + JOIN). Mitigate: gate entirely behind `--experience-dir`; `openStore` feature-detects `node:sqlite` and disables gracefully (warn, no-op) on older Node. No pipeline dependency on it.
- **Write concurrency.** Phase-2 optimizes up to `maxConcurrentQueries` in parallel *within one Node process*. Use a single `DatabaseSync` in WAL mode and wrap each capture in a short transaction; writes are serialized by SQLite. Never open two writers.
- **Backfill fidelity.** Reconstruction must use the *same* best-so-far rule as the live loop (`checkExecutionImprovement`, per `optimization_target`), or historical trees won't match captured ones. Cross-check: runs captured live *and* backfilled must produce identical trees.
- **By-reference fragility.** `output/**` is gitignored/prunable; inline enough reward to keep credit-assignment and retrieval meaningful without the `.cpp`; mark rows whose artifacts vanished.
- **Reward semantics.** hot vs cold depends on `run.json.optimization_target`; store both, compare only within-target; `validation!='pass'` nodes are non-selectable for best/warm-start but retained as negative evidence.
- **Embedding quality.** Phase (a/b) ship a deterministic structural-feature embedding (offline, no network) so vector search is functional immediately; Phase (d) upgrades to real embeddings behind the same interface.
- **Independence guarantee.** The module imports nothing from `src/gendb/memory/`; disabling or deleting the HAG must not affect it and vice versa. A grep test asserts no cross-imports.

---

## Deliverables checklist

| File | Phase | Purpose |
|---|---|---|
| `src/gendb/experience/schema.sql` | a | four-table DDL |
| `src/gendb/experience/store.mjs` | a | open/init + capture (write side) |
| `src/gendb/experience/embed.mjs` | a | pluggable text→vector |
| `scripts/backfill_experience.mjs` | a | reconstruct graph from existing runs |
| `orchestrator.mjs` (guarded capture, `--experience-dir`) | a | live capture |
| `src/gendb/experience/query.mjs` | b | graph traverse + relation join + vector search |
| `src/gendb/experience/cli.mjs` | b | human/agent query surface |
| `src/gendb/experience/experience.test.mjs` | a/b | Q24 / single-iter / all-regression assertions |
| orchestrator warm-start hook | c | self-evolving closed loop |
| real embeddings + governance | d | optional |
