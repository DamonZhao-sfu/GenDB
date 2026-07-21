# Integrating "Experience Graphs" into GenDB's Self-Evolving Memory

Reference: **Experience Graphs: The Data Foundation for Self-Improving Agents**,
Gang Liao, Daniel J. Abadi et al. (Meta Platforms & University of Maryland),
arXiv:2606.29823 (June 2026).

---

## 1. The paper's idea (summary)

Long-horizon *agentic* tasks — code generation, kernel/hardware optimization,
scientific discovery — are a **new database workload**. An agent explores by
generating artifacts, running tools, observing failures, **branching**, and
**repairing** over hundreds of steps. That exploration produces a rich,
structured object the authors call an **experience graph**: executable
artifacts, tool outputs, objective rewards, sibling comparisons, mutable search
statistics, and causal lineage.

Their thesis: **search over an experience graph is a database access pattern.**
So the graph should be *first-class, governed, queryable database state* — not
ephemeral logs. Their system, **Trellis**, makes this concrete:

- **Four-table schema** (algorithm-, task-, and skill-agnostic):
  1. **Tasks** — the problem spec + a *task-description embedding* for
     cross-session vector similarity.
  2. **Sessions** — one search run: who is searching, with which algorithm
     (MCTS, evolutionary, hill-climb…) and model.
  3. **Nodes** — every individual attempt: **parent link**, generated
     **artifact**, execution **output**, **fitness/reward**, evaluation
     **evidence**, and **algorithm-specific stats** (UCB score & visit count for
     MCTS; generation & island for evolutionary search).
  4. **Prompt histories** — the exact messages the LLM saw and produced.
- **Large artifacts by reference**: logs, traces, binaries live in object
  storage; the relational tables stay lean while preserving full lineage.
- **Mixed consistency**: durable node insertion, but *eventual consistency* on
  the mutable statistics (e.g. MCTS backprop walks the ancestor chain updating
  visit count + cumulative reward) — a profile that doesn't map cleanly to
  standard OLTP isolation levels.
- **Payoff**: when the database owns the experience graph, agents become
  *stateless compute*, and crash recovery, horizontal scaling, and a closed-loop
  **training flywheel** emerge as byproducts. Grounded in KernelEvolve at Meta:
  cross-session reuse reached a target speedup ~**10× faster at 52% lower token
  cost**.

Sources: [abs](https://arxiv.org/abs/2606.29823) ·
[html](https://arxiv.org/html/2606.29823v1) ·
[summary](https://arxiviq.substack.com/p/experience-graphs-the-data-foundation)

---

## 2. Proposed prompt (to generate the integration plan)

> You are a systems architect working on **GenDB**, a multi-agent LLM system that
> generates instance-optimized C++ query-execution code. GenDB already has a
> **Self-Evolving Memory System (v3)**: a 6-layer Hierarchical Abstraction Graph
> (HAG) where **L0/L1** are JSON nodes retrieved by SQL structural similarity and
> **L2–L5** are distilled Claude skills. After each run, a **Memory Manager**
> agent performs *differential learning* over each query's linear
> `optimization_history.json` and writes/evolves skills.
>
> Read the paper **"Experience Graphs: The Data Foundation for Self-Improving
> Agents"** (arXiv:2606.29823). Its core claim is that an agent's *branching
> search trajectory* — every attempt, artifact, reward, sibling comparison, and
> causal-lineage edge — should be **first-class, queryable database state** (a
> four-table Tasks/Sessions/Nodes/Prompts schema), not ephemeral logs, because
> "search over the experience graph is a database access pattern."
>
> Produce an implementation plan to integrate this idea into GenDB's memory
> system. The plan must:
> 1. Map the paper's four tables onto GenDB's existing artifacts
>    (`optimization_history.json`, per-iteration C++ dirs, agent prompt logs,
>    `run.json`, HAG L0/L1 nodes).
> 2. Explain **where the experience graph sits relative to the HAG** — argue it
>    is the *raw substrate beneath L0* that L0–L5 are distilled from, and identify
>    what search evidence is currently thrown away (regressions, abandoned
>    branches, sibling attempts).
> 3. Define the schema, storage layout, and retrieval API (ancestors, siblings,
>    best-node, cross-session task similarity, MCTS-style backpropagation),
>    respecting GenDB's file-backed convention and "large artifacts by reference."
> 4. Specify the minimal, **non-breaking** wiring into `orchestrator.mjs` (capture
>    during the optimize loop) and the **Memory Manager** prompt (consume the
>    graph for credit assignment instead of only linear diffs).
> 5. Phase the work: (a) capture + backfill from existing runs, (b) richer
>    cross-session retrieval, (c) closed-loop reuse ("warm-start from the best
>    prior sibling"), (d) optional real embeddings & governance.
> 6. Call out risks, consistency requirements, and how to validate each phase
>    against real `output/**/optimization_history.json` data.
>
> Deliver the plan as phased tasks with concrete file paths and function
> signatures. Keep GenDB's existing pipeline working at every phase.

---

## 3. What GenDB's memory system implements today (v3)

| Area | Status | Where |
|------|--------|-------|
| **6-layer HAG** (L0 instances → L5 principles) | ✅ | `memory/graph.mjs` |
| **L0/L1 structural retrieval** — SQL feature extraction, Jaccard/weighted similarity, 3-tier classify (exact / structural / novel) | ✅ | `memory/retrieval.mjs`, `tools/sql-parser.mjs` |
| **L2–L5 as Claude skills** — semantic discovery via `settingSources:['project']`, auto-loaded | ✅ | `.claude/skills/`, `memory/README.md` |
| **Tier-dependent pre-injection** — exact→fast path (copy best C++), structural→accelerated, novel→full | ✅ | `orchestrator.mjs` (Phase 0/2) |
| **Differential learning** — Memory Manager extracts >30% breakthroughs, regressions→gotchas, distills skills | ✅ | `agents/memory-manager/prompt.md` |
| **Evidence-driven skills** — `evidence.json` counters, code-pattern placeholders, edges (`instance_of`, `uses_operator`…) | ✅ | `agents/memory-manager/*` |
| **Populate/bootstrap/migrate** tooling | ✅ | `memory/{populate,bootstrap,migrate-v2-to-v3}.mjs` |
| **Raw search trajectory as first-class state** | ❌ → **added here** | `memory/experience-graph.mjs` |
| **Cross-session reuse by task similarity (embeddings)** | ⚠️ partial (SQL structural only) → **extended here** | `findSimilarTasks()` |
| **Sibling / dead-end / branch preservation** | ❌ (linear `optimization_history.json` only) → **added here** | ingest models branches |

**Key gap the paper addresses:** GenDB jumps straight from a *linear*
`optimization_history.json` → distilled skills. Every regression, abandoned
branch, and sibling attempt — the exact search evidence needed for good credit
assignment and warm-starting — is discarded. L0 keeps only a flattened
best-of trajectory.

---

## 4. What this change adds (Phase A — capture + backfill)

New module **`src/gendb/memory/experience-graph.mjs`** implements the paper's
four-table schema on GenDB's file-backed conventions, sitting **beneath L0**:

```
Experience Graph (RAW search)        ← NEW: memory/experience-graph.mjs
  tasks.json      — spec + embedding (cross-session vector similarity)
  sessions.json   — algorithm (llm-hill-climb), agent, model
  nodes/<sess>/*  — attempt: parent_id, artifact_ref, reward, eval_evidence,
                    algo_meta{visit_count, cumulative_reward, regression}
        │  (distilled by Memory Manager)
        ▼
L0/L1 HAG  ──►  L2–L5 Skills          ← existing v3
```

Mapping to the paper:

| Paper | GenDB experience graph |
|-------|------------------------|
| Task + embedding | `upsertTask()` + `embedTask()` (offline bag-of-features; pluggable) |
| Session + algorithm | `createSession()` — algorithm `"llm-hill-climb"` |
| Node (parent, artifact, reward, evidence, stats) | `insertNode()` — `artifact_ref` → `iter_N/` C++ dir *by reference* |
| Sibling comparison | `siblings()` — regressions preserved as dead-end siblings |
| Causal lineage | `ancestors()` + `parent_id` chain |
| MCTS backprop (visit + cumulative reward) | `backpropagate()` up the ancestor chain |
| Cross-session retrieval | `findSimilarTasks()` (cosine over task embeddings) |
| Large artifacts by reference | `artifact_ref` / `tool_output` are **paths**, never inlined |

**Backfill tool** `memory/ingest-experience.mjs` converts existing linear
histories into branching graphs. The ingest models GenDB's real control flow:
each iteration derives from the **last accepted** attempt; a **non-improving
iteration becomes a dead-end sibling** rather than being dropped.

Validated against real data (`output/deprecated/sec-edgar`, 42 histories):
attempts that regressed correctly branch off the accepted frontier with
`visit_count = 0`, while the winning path receives backpropagated visits and
cumulative reward — exactly the sibling-comparison + causal-lineage structure
the paper prescribes.

```bash
node src/gendb/memory/ingest-experience.mjs \
  --scan output/deprecated/sec-edgar --benchmark sec-edgar --scale-factor 3 \
  --memory-dir gendb-memory
# → ExperienceGraph: 6 tasks, 83 sessions, 282 nodes
```

---

## 5. Next phases (planned, not yet wired)

- **Phase B — live capture:** in `orchestrator.mjs`'s optimize loop, call
  `insertNode()` per iteration (artifact_ref = the iter dir, prompt.messages_ref
  = the agent transcript) and `backpropagate()` on accept. Non-breaking:
  guarded by `args.memoryDir`.
- **Phase C — Memory-Manager consumption:** give the Memory Manager the
  experience graph so credit assignment uses **sibling deltas and ancestor
  chains** instead of only consecutive linear diffs (better anti-pattern
  detection, higher-signal breakthroughs).
- **Phase D — closed-loop reuse (the flywheel):** on a new query, use
  `findSimilarTasks()` to **warm-start from the best prior sibling's artifact**,
  targeting the paper's "reach target speedup faster at lower token cost."
- **Phase E — real embeddings & governance:** swap `embedTask()` for a real
  embedding provider; add retention/GC policy and provenance for the growing
  node store.
