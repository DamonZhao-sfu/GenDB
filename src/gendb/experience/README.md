# Experience Graph — self-evolving memory (independent feature)

An **independent** self-evolving memory system based on *Experience Graphs: The
Data Foundation for Self-Improving Agents* (arXiv:2606.29823). It does **not**
reuse GenDB's HAG/skills memory (`src/gendb/memory/`) — separate store, separate
capture, separate `--experience-dir` flag.

It makes the optimize loop's **branching search tree** first-class, queryable
state. `optimization_history.json` is a *linear projection* of that tree (each
iteration warm-starts from best-so-far); this module recovers the parent edges,
sibling sets, and dead branches the linear view throws away.

## Backing store

A single SQLite database (`<experience-dir>/experience.db`) via the built-in
`node:sqlite` (Node ≥ 22.5) — **zero new dependencies**. Design doc:
[`docs/experience-graph-plan.md`](../../../docs/experience-graph-plan.md).

## Four tables (paper-faithful)

`tasks` → `sessions` → `nodes` (parent-pointer forest) → `prompts`. See
[`schema.sql`](./schema.sql). Large artifacts (C++, plans, exec results, prompt
logs) are stored **by reference** (paths into `output/**`).

## Three access patterns (`query.mjs`)

| Pattern | Implementation | Functions |
|---|---|---|
| **Graph traverse** | recursive CTE over `nodes.parent_node_id` | `getAncestors`, `getDescendants`, `getSiblings`, `getChildren`, `getSessionTree` |
| **Relation join** | SQL joins across the four tables | `getBestNodeForTask`, `getWinningPrompt`, `getTaskLeaderboard` |
| **Vector search** | cosine over embedding BLOBs (`embed.mjs`) | `searchSimilarTasks`, `searchSimilarStrategies` |

MCTS-style `backpropReward` rolls a leaf's reward up its ancestor chain
(`visit_count`, `best_descendant_ms`).

## Files

| File | Purpose |
|---|---|
| `schema.sql` | four-table DDL |
| `store.mjs` | open/init + capture (write side) |
| `query.mjs` | the three access patterns (read side) |
| `embed.mjs` | pluggable text→vector (deterministic offline default) |
| `reconstruct.mjs` | best-so-far parent-pointer reconstruction |
| `experience.test.mjs` | self-contained tests (Q24 / single-iter / all-regression) |

## Usage

```bash
# Backfill the graph from existing runs (reconstructs the branching trees)
node scripts/backfill_experience.mjs --root output --out gendb-experience
node scripts/backfill_experience.mjs --root output --dry-run    # inspect, no writes

# Run tests
node src/gendb/experience/experience.test.mjs
```

Live capture during the optimize loop (guarded by `--experience-dir`) is Phase
(a) wiring into `orchestrator.mjs`; closed-loop warm-start reuse is Phase (c).
Status/roadmap: see the design doc.
