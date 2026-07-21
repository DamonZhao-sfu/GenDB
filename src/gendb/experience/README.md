# Experience Graph — self-evolving memory (independent feature)

An **independent** self-evolving memory system based on *Experience Graphs: The
Data Foundation for Self-Improving Agents* (arXiv:2606.29823). It does **not**
reuse GenDB's HAG/skills memory (`src/gendb/memory/`) — separate store, separate
capture, separate `--experience-dir` flag.

It makes the optimize loop's **branching search tree** first-class, queryable
state. `optimization_history.json` is a *linear projection* of that tree (each
iteration warm-starts from best-so-far); this module recovers the parent edges,
sibling sets, and dead branches the linear view throws away.

## Backends (pluggable — same async interface)

Open either backend through the unified entry point; the orchestrator and all
callers are backend-agnostic:

```js
import { openExperienceStore } from "./index.mjs";
const store = await openExperienceStore({ backend: "sqlite", dir: "gendb-experience" });
const store = await openExperienceStore({ backend: "postgres", connectionString: "postgres://…" });
// selection precedence: opts.backend → env GENDB_EXPERIENCE_PG → "sqlite"
```

| Backend | Engine | Graph traverse | Relation join | Vector search |
|---|---|---|---|---|
| **sqlite** (default, zero setup) | `node:sqlite` (Node ≥ 22.5) | recursive CTE | SQL joins | **sqlite-vec** exhaustive SIMD KNN (JS-cosine fallback) |
| **postgres** (unified engine) | PostgreSQL + `pg` | recursive CTE | SQL joins | **pgvector HNSW** — real approximate-NN index |

Design doc: [`docs/experience-graph-plan.md`](../../../docs/experience-graph-plan.md).

### ANN / HNSW

- **Postgres backend = real ANN today.** `schema.pg.sql` creates
  `USING hnsw (embedding vector_cosine_ops)` on `tasks` and `nodes`; queries
  `ORDER BY embedding <=> $1` use it (`EXPLAIN` shows `Index Scan using
  idx_nodes_hnsw`). This is the recommended path once the graph outgrows a
  brute-force scan.
- **SQLite backend** uses `sqlite-vec`, which is an *exhaustive* SIMD KNN (no
  ANN index yet — ANN is on sqlite-vec's roadmap). Fine to ~10⁵ vectors. For
  embedded ANN sooner, an `hnswlib-node` sidecar index is a drop-in option
  behind the same `searchSimilar*` functions.

## Four tables (paper-faithful)

`tasks` → `sessions` → `nodes` (parent-pointer forest) → `prompts`. See
[`schema.sql`](./schema.sql). Large artifacts (C++, plans, exec results, prompt
logs) are stored **by reference** (paths into `output/**`).

## Three access patterns (`query.mjs`)

| Pattern | Implementation | Functions |
|---|---|---|
| **Graph traverse** | recursive CTE over `nodes.parent_node_id` | `getAncestors`, `getDescendants`, `getSiblings`, `getChildren`, `getSessionTree` |
| **Relation join** | SQL joins across the four tables | `getBestNodeForTask`, `getWinningPrompt`, `getTaskLeaderboard` |
| **Vector search** | **sqlite-vec** `vec_distance_cosine` (in-engine SIMD KNN) over the `embedding` BLOB columns; JS cosine fallback if the extension can't load | `searchSimilarTasks`, `searchSimilarStrategies` |

MCTS-style `backpropReward` rolls a leaf's reward up its ancestor chain
(`visit_count`, `best_descendant_ms`).

## Files

| File | Purpose |
|---|---|
| `index.mjs` | unified `openExperienceStore({ backend })` dispatcher |
| `schema.sql` / `schema.pg.sql` | four-table DDL (SQLite / Postgres) |
| `store.mjs` | SQLite open/init + capture (sync core) |
| `query.mjs` | SQLite three access patterns (sync core) |
| `backends/sqlite.mjs` | async facade over the SQLite core |
| `backends/postgres.mjs` | Postgres/pgvector backend (async) |
| `embed.mjs` | pluggable text→vector (deterministic offline default) |
| `reconstruct.mjs` | best-so-far parent-pointer reconstruction |
| `experience.test.mjs` | SQLite tests (Q24 / single-iter / all-regression) |
| `experience.pg.test.mjs` | Postgres tests (skips without `GENDB_EXPERIENCE_PG_TEST`) |

## Usage

```bash
# Backfill the graph from existing runs (reconstructs the branching trees)
node scripts/backfill_experience.mjs --root output --out gendb-experience          # SQLite (default)
node scripts/backfill_experience.mjs --root output --backend postgres --pg-url postgres://…/db
node scripts/backfill_experience.mjs --root output --dry-run                        # inspect, no writes

# Run tests
node src/gendb/experience/experience.test.mjs                                       # SQLite
GENDB_EXPERIENCE_PG_TEST=postgres://…/throwaway node src/gendb/experience/experience.pg.test.mjs
```

Live capture during the optimize loop (guarded by `--experience-dir`) is Phase
(a) wiring into `orchestrator.mjs`; closed-loop warm-start reuse is Phase (c).
Status/roadmap: see the design doc.
