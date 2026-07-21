# Experience Graph — self-evolving memory (independent feature)

An **independent** self-evolving memory system based on *Experience Graphs: The
Data Foundation for Self-Improving Agents* (arXiv:2606.29823). It does **not**
reuse GenDB's HAG/skills memory (`src/gendb/memory/`) — separate store, separate
capture, separate flag.

It makes the optimize loop's **branching search tree** first-class, queryable
state. `optimization_history.json` is a *linear projection* of that tree (each
iteration warm-starts from best-so-far); this module recovers the parent edges,
sibling sets, and dead branches the linear view throws away.

## Backing store: PostgreSQL + pgvector

A single PostgreSQL database with the `vector` (pgvector) extension — a genuinely
unified engine for the paper's three access patterns:

| Access pattern | Engine mechanism | API (`query`/store methods) |
|---|---|---|
| **Graph traverse** | recursive CTE over `nodes.parent_node_id` | `getAncestors`, `getDescendants`, `getSiblings`, `getChildren`, `getSessionTree` |
| **Relation join** | SQL joins across `tasks/sessions/nodes/prompts` | `getBestNodeForTask`, `getWinningPrompt`, `getTaskLeaderboard` |
| **Vector search** | pgvector cosine `<=>` over an **HNSW ANN index** | `searchSimilarTasks`, `searchSimilarStrategies` |

The HNSW indexes (`schema.pg.sql`, `USING hnsw (embedding vector_cosine_ops)` on
`tasks` and `nodes`) give real approximate-NN; `EXPLAIN` shows
`Index Scan using idx_nodes_hnsw`. Large artifacts (C++, plans, exec results) are
stored **by reference** (paths into `output/**`). Design doc:
[`docs/experience-graph-plan.md`](../../../docs/experience-graph-plan.md).

## Files

| File | Purpose |
|---|---|
| `index.mjs` | `openExperienceStore({ connectionString })` entry point |
| `schema.pg.sql` | four-table DDL + HNSW indexes |
| `backends/postgres.mjs` | the store (capture + the three access patterns) |
| `ids.mjs` | stable `taskId` / `sessionId` / `nodeId` |
| `embed.mjs` | pluggable text→vector (deterministic offline default) |
| `reconstruct.mjs` | best-so-far parent-pointer reconstruction |
| `ingest.mjs` | history → store (shared by backfill + live capture) |
| `capture.mjs` | `createCapture(args)` — NOOP-safe live capture for the orchestrator |
| `experience.test.mjs` | tests (skip without `GENDB_EXPERIENCE_PG_TEST`) |

## How to run

### 0. Prerequisites — a Postgres with pgvector
```bash
# Debian/Ubuntu example
sudo apt-get install -y postgresql-16 postgresql-16-pgvector
createdb gendb_experience
psql -d gendb_experience -c "CREATE EXTENSION IF NOT EXISTS vector;"
export GENDB_EXPERIENCE_PG="postgres://<user>@<host>:5432/gendb_experience"
```

### 1. Live capture during a GenDB run
```bash
# enable via the flag …
node src/gendb/orchestrator.mjs --benchmark tpc-h --sf 10 \
     --experience-pg "$GENDB_EXPERIENCE_PG"
# … or via env (GENDB_EXPERIENCE_PG); with neither, capture is OFF (default).
```
Each query's search tree is written to Postgres as the query finishes.

### 2. Backfill from existing runs
```bash
node scripts/backfill_experience.mjs --root output --pg-url "$GENDB_EXPERIENCE_PG"
node scripts/backfill_experience.mjs --root output --dry-run     # inspect, no writes
```

### 3. Query the graph
```bash
node --input-type=module -e '
import { openExperienceStore } from "./src/gendb/experience/index.mjs";
const s = await openExperienceStore({});                 // uses GENDB_EXPERIENCE_PG / libpq env
console.log(await s.getTaskLeaderboard("tpc-h"));
console.log((await s.getAncestors("<node_id>")).map(n => n.iteration));
console.log(await s.searchSimilarStrategies("cache-exceeding hash probe", 5));
await s.close();'
```

### 4. Tests (throwaway DB — drops its four tables)
```bash
GENDB_EXPERIENCE_PG_TEST="postgres://user@host/throwaway" \
  node src/gendb/experience/experience.test.mjs
```
