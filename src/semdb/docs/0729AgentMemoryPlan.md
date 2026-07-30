# SemDB Agent Memory — design study and implementation plan

Part 1 documents how GenDB's `src/gendb/memory/` actually couples its agents.
Part 2 proposes how to give SemDB the same capability without breaking the
PGO branch's contract-first, one-bound-skill-per-role invariants.

---

## Part 1 — How GenDB uses agent memory as inter-agent glue

GenDB's agents never message each other. They are coupled through a durable,
addressable artifact store plus two retrieval channels, with the orchestrator
as the only router.

### 1.1 Roles

| Role | Agent | Phase |
|------|-------|-------|
| Writer | Memory Manager (`agents/memory-manager/`, opus) | Phase 3, post-run |
| Readers | Workload Analyzer, Query Planner, Code Generator, Query Optimizer | Phase 2, per query |
| Router | `orchestrator.mjs` (Phase 0 classification, Phase 2 injection, Phase 3 curation) | — |

### 1.2 Six layers, two retrieval channels

```
L5 Performance Principles   ─┐
L4 Optimization Strategies   │  .claude/skills/<name>/   ← PULL: Claude semantic
L3 Operator Techniques       │                              matching on descriptions
L2 Sub-Structure Patterns   ─┘
L1 Query Templates          ─┐  gendb-memory/graph/nodes/ ← PUSH: deterministic JS
L0 Query Instances          ─┘                              structural retrieval
```

- **Push channel (L0/L1).** `retrieval.mjs:classifyQuery` extracts SQL features
  (`tools/sql-parser.mjs:extractStructuralFeatures`), scores them against every
  L1 node's `structural_signature` with a weighted similarity (joins ×3, table
  Jaccard ×3, 11 boolean features ×1, aggregate overlap ×1), and boosts to 1.0
  when the parameterized template SQL matches exactly. The winning L1's proven
  strategies / anti-patterns and its best L0 instance are rendered to Markdown
  and injected into the agent prompt as `{{memory_pre_injection}}`. Agents
  cannot miss it, and it is capped (`maxPreInjectionTokens: 4000`).
- **Pull channel (L2–L5).** The Memory Manager materializes each learned
  technique as a real Agent-SDK skill directory (`SKILL.md` +
  `code-patterns/*.cpp` + `evidence.json` + `gotchas.md`). `providers/claude.mjs`
  sets `cwd: REPO_ROOT` and `settingSources: ['user','project']`, so any agent
  discovers them natively; `{{memory_catalog}}` only lists what exists. Skill
  invocations are counted (`skillsUsed`) and dumped to `skill_usage.json`.

So specific knowledge is *orchestrator-selected* and forced in; general
knowledge is *agent-selected* and pulled on demand. That split is the whole
trick — it bounds tokens while keeping recall.

### 1.3 Memory carries control flow, not just prompt text

`classifyQuery` returns a tier, and the tier changes the pipeline
(`orchestrator.mjs:2859-2877`, `1215-1274`):

| Tier | Score | Effect |
|------|-------|--------|
| `exact` | ≥ 1.0 | `best_cpp_path` is copied into `iter_0` as the starting implementation; Pass-2 query-guide generation is skipped; prompt says "the reference implementation is already in your working directory" |
| `structural` | ≥ 0.5 | Strategies + edge-linked L3/L2 summaries injected ("accelerated path") |
| `novel` | < 0.5 | Nothing injected ("full path") |

This is stronger coupling than shared text: past runs decide *what the next
agent starts from* and *which agents run at all*.

### 1.4 Edges are the actual agent-to-agent wiring

`instance_of` (L0→L1), `exhibits_pattern` (L1→L2), `uses_operator` (L1→L3),
`implements_strategy` (L3→L4), `exemplifies_principle` (L4→L5).
Retrieval walks *outgoing* `uses_operator` / `exhibits_pattern` from the matched
L1 and *incoming* `instance_of` to find reference instances. Concretely: a
technique the **Code Generator** discovered at iteration 3 of a past run reaches
a future **Query Planner** because the Memory Manager attached it to the L1
template the planner's query now matches.

### 1.5 Write discipline — what keeps the shared context from rotting

- **Differential learning gate.** Classify each query as `NOVEL_SUCCESS` /
  `SIGNIFICANT_IMPROVEMENT` (> `differentialThreshold` = 20%) / `FAMILIAR`
  (skip). Only non-FAMILIAR queries mutate memory.
- **Learn only the non-obvious.** Breakthroughs = >30% improvement between
  consecutive iterations, i.e. things the model did *not* know upfront.
  First-iteration successes are explicitly low priority ("the LLM already knows
  it"). Regressions become `gotchas.md` anti-patterns.
- **Distill, don't copy.** Skills must be benchmark-agnostic, with placeholders
  (`MAX_KEY`, `FACT_ROW_COUNT`), never "TPC-H Q3's optimization".
- **Evidence-driven.** `evidence.json` per skill: query, before_ms, after_ms,
  speedup. No hypotheses.
- **De-dup by construction.** The MM prompt is pre-loaded with existing skills,
  existing L1 templates and existing L4 strategies.
- **Usage feedback.** `skill_usage.json` from the run tells MM which skills were
  actually invoked, so usage itself becomes evidence.

### 1.6 Never let the LLM own the retrieval key

The MM agent writes node JSON directly with `Write`, so the orchestrator repairs
it deterministically afterwards (`orchestrator.mjs:3143-3154`):
`updateIndex()` rebuilds the index from disk, and `patchL1Signatures()` replaces
any string `structural_signature` the LLM wrote with a real feature object
produced by *the same* `extractStructuralFeatures` used at retrieval time.
Writer and reader are thus guaranteed to share one representation.

### 1.7 Memory is an accelerator, never a dependency

Every memory call site is `try/catch` with `(non-fatal)` logging;
`--no-memory` / `memoryDir: null` disables the whole subsystem. Auditability is
explicit: `memory_match.json` per query, `memory_report.json`,
`skill_usage.json`, `memory_update_summary.json` per run.

### 1.8 A second, non-knowledge memory

`storage-pool.mjs` content-addresses physical artifacts —
`sha256(benchmark, sf, schema)` → a pooled `gendb/` storage dir, symlinked into
the new run, skipping Phase-1 ingestion entirely. Knowledge memory and artifact
memory are deliberately separate mechanisms.

### 1.9 Defect found while reading (must not be ported)

`memory/graph.mjs:326,335,352,369` — `loadMemoryNode` cases 2–5 reference
`n.skill_path`, but the local variable is `node`. Loading any L2–L5 node throws
`ReferenceError: n is not defined`. Only L0/L1 paths are exercised today, which
is why it survives.

---

## Part 2 — Plan: agent memory for SemDB

### 2.1 What is different about SemDB (and therefore what cannot be copied)

| GenDB assumption | SemDB reality | Consequence |
|---|---|---|
| Retrieval key = SQL structure | Query = SQL **+** NL predicate over an image/text corpus; the discriminative part is the *semantic site*, not the join graph | Need a SemDB-specific signature (§2.3) |
| Objective = latency (ms), one metric | Per-query objective with a name and direction (`telemetry.refine.objective`: f1 / adjusted_rand_index / ranking / agg-error) | "improvement" and "breakthrough" must be objective- and direction-aware; reuse `pgo-loop.mjs:objective()` |
| Agents discover skills freely | PGO binds **exactly one** skill per role and the prompt says "Do not discover or load any other skill"; `skill-loader.mjs` rejects any frontmatter beyond `name`/`description` | The L2–L5 *pull* channel cannot be adopted as-is (§2.5) |
| MM may read anything in the run dir | **No ground truth may reach a generating agent** (`iteration_feedback` data-boundary invariant) | Memory content is validation-derived only; needs an explicit leak guard + test |
| `workloadDir/runs/<runId>` audit tree | `--out` with per-query dirs, `results.csv`, `_val/`, `_materialized/` | Different artifact placement (§2.7) |

### 2.2 Layer design (5 layers)

| Layer | Name | Storage | Content |
|-------|------|---------|---------|
| L0 | Query instances | HAG JSON | benchmark, query_id, sql, nl, modality, objective{name,value,direction}, metric family, iterations used, replans, optimizer action counts, promoted `plan.json`/helpers/solver paths, primitives used, validation provenance, `data_boundary` |
| L1 | Query templates | HAG JSON | signature (§2.3), proven helper-DAG skeleton, proven strategies, anti-patterns, metric family, per-benchmark instance links |
| L2 | Site patterns | HAG JSON (+ optional skill) | recurring semantic-site shapes, e.g. "image→structured join keyed by OCR'd name, `sampling_unit: pair`", "group-by over a VLM attribute scored by ARI" |
| L3 | Helper recipes | HAG JSON (+ optional skill) | reusable helper implementations as exact compositions of `vadar/predefined*.py` primitives, with thresholds, tie-breaks, and which primitive is wrong for which case — the highest-value layer for SemDB |
| L4 | Optimizer playbook | HAG JSON (+ optional skill) | failure signature → action that fixed it (e.g. "empty trace + FN-dominated ⇒ wrong `sampling_unit` ⇒ REPLAN, not PATCH_CODE"). Directly attacks iteration count |
| L5 | Principles | HAG JSON (+ optional skill) | cross-cutting invariants, e.g. "`value_space` must be closed over the join column's distinct values", "never add a VLM call for a property already decidable by CV" |

Note L2–L5 are *stored as graph nodes first*; skill materialization is optional
and deferred (§2.5), which is the main structural departure from GenDB v3.

### 2.3 The signature — SemDB's retrieval key

Two-stage, because at Phase 0 there is no plan yet.

- **Stage A (plan-independent, used for matching).** From SQL + NL + table
  metadata only, all already computed in the orchestrator:
  `modality` (image/text/mixed), semantic argument shape from `semanticArgs(sql)`,
  operator kind (filter/join/map/group/topk/agg) from the SQL form,
  candidate `sampling_unit` (row vs pair — `tablesInPredicate` already
  distinguishes this), corpus identity, referenced table set, projection arity,
  aggregate/limit/order flags, and a normalized NL predicate fingerprint
  (lowercased, stop-worded, token-set).
  Similarity = weighted agreement, with **modality and operator kind as hard
  gates** (score 0 if either disagrees), then table Jaccard, sampling-unit
  agreement, NL token-set Jaccard, metric-family agreement.
- **Stage B (plan-derived, stored as payload).** From the promoted `plan.json`:
  `semantic_sites[].operator/sampling_unit/output_type/value_space`,
  `helper_dag[].primitive_steps` primitive multiset, `relational_plan` shape,
  `compilability.class`. Used for (a) enriching what gets injected, and (b) an
  optional post-plan affinity check: after the planner emits v1, if a stored L1
  has ≥0.9 plan affinity, the generator prompt may cite the stored helper DAG.

Implemented in one file, `memory/signature.mjs`, and used by **both** the
retrieval path and the deterministic repair pass — the §1.6 lesson.

### 2.4 Read path → control flow

| Tier | Condition | Effect on the PGO loop |
|------|-----------|------------------------|
| `exact` | same benchmark + query id, or stage-A score ≥ 0.98, with a promoted candidate whose objective passed | Warm start: stored `plan.json` is passed to the planner as `previous_plan_path` (so it emits v1 by confirmation, not from scratch), stored helpers+solver copied into `iter_0` as the initial candidate. Validation still re-runs — corpus / scale factor may differ |
| `structural` | 0.55 ≤ score < 0.98 | Inject L1 strategies + anti-patterns, edge-linked L2 site patterns, L3 helper recipes, and L4 playbook entries matching this failure class |
| `novel` | < 0.55 | Inject nothing (optionally L5 principles only, they are cheap) |

Cross-benchmark transfer policy: L0/L1 matches are preferred within the same
benchmark; L3/L4/L5 are allowed to transfer across benchmarks, L0/L1 are not.

### 2.5 Delivery mechanism — injection first, skills later

**v1 = push only.** Everything is rendered into the existing prompt templates as
new sections in `agents/{query-planner,semantic-code-generator,semantic-optimizer}/user-prompt.md`:
`{{memory_pre_injection}}`, `{{memory_catalog}}`, `{{memory_note}}` — the same
placeholder names GenDB uses, so the two orchestrators stay comparable. The
one-bound-skill invariant, `skill-loader.mjs`, and
`tests/test_agent_prompt_structure.mjs` are untouched.

Each system prompt gains one invariant: *prior knowledge is advisory; the
primitive implementation files remain authoritative; never prefer a remembered
recipe over what `vadar/predefined*.py` actually exposes.*

**v2 = optional pull.** When the L3/L4 corpus outgrows the token cap, extend
`skill-loader.mjs` to accept a **role-scoped allowlist** (bound skill + N
retrieved memory skills) rather than free discovery. Deliberately deferred.

### 2.6 Write path — the `memory-curator` agent

New agent `src/semdb/agents/memory-curator/` (distinct name from GenDB's
`memory-manager` to signal a different contract), run once after all queries in
a run, model `opus`, effort `high`, tools `Read/Write/Edit/Glob/Grep`.

Inputs per query dir: `telemetry.json` (objective history, action counts,
plan_versions, metrics), `iter_*/iteration_feedback.json`, `iter_*/plan.json`
(diffed across improving iterations), `iter_*/optimizer_action.json`,
`preflight.json` failures, `diff.json`, `run.log` tails.

Its differential gate, SemDB-flavored:
- objective- and direction-aware improvement, computed by the *same* code as
  `pgo-loop.mjs:defaultImprovement`;
- **breakthrough** = a single iteration that moves the objective by ≥ 30% of the
  remaining headroom (not raw delta — F1 0.9→0.95 matters more than 0.2→0.25);
- **plan-level breakthrough** = a REPLAN that improved the objective ⇒ the
  highest-signal L4 playbook entry available;
- regressions ⇒ anti-patterns on the matched L1 / L3 recipe;
- `FAMILIAR` ⇒ skip.

Output: `memory_update.json` validated against a new
`contracts/memory-update.schema.json`, then applied by **deterministic JS**
(`memory/apply-update.mjs`) rather than by the agent writing nodes directly.
This is an improvement on GenDB, where the agent writes node JSON and the
orchestrator repairs it afterwards. Applying via code gives: schema-checked
nodes, signatures always computed by `signature.mjs`, index always consistent,
and one enforcement point for the leak guard below.

### 2.7 Guards, artifacts, config

- **GT leak guard** (hard requirement): `apply-update.mjs` rejects any node or
  text field containing row-level expected values, and every node carries
  `data_boundary: {source, cert_accessed:false, full_ground_truth_accessed:false}`.
  Aggregate metrics and technique descriptions are allowed; per-row GT is not.
  Covered by a dedicated test.
- **Non-fatal**: every call site `try/catch` + `(non-fatal)` log; `--no-memory`
  and `memoryDir: null` disable everything; `--no-memory` runs stay the reported
  baseline in the paper numbers.
- **Artifacts**: `<query_dir>/memory_match.json`, `<out>/memory_report.json`,
  `<out>/memory_update_summary.json`, `<out>/skill_usage.json` (v2 only).
- **Config** (`semdb.config.mjs`): `memoryDir: null`, and
  ```js
  memory: {
    structuralMatchMinScore: 0.55,
    exactMatchMinScore: 0.98,
    maxPreInjectionTokens: 3500,
    differentialHeadroomThreshold: 0.30,
    crossBenchmarkLayers: [3, 4, 5],
    layers: { L0:"instances", L1:"templates", L2:"site_patterns",
              L3:"helper_recipes", L4:"optimizer_playbook", L5:"principles" },
  }
  ```
  plus `memory_curator` entries in both providers' `agentModels` /
  `agentEffortLevels`.
- **CLI**: `--memory-dir <path>`, `--no-memory`, `--memory-readonly`
  (retrieve but never write — needed for clean A/B runs).

### 2.8 Implementation phases

**Phase 1 — substrate, no agent involved.**
`memory/graph.mjs` (ported, with the §1.9 bug fixed), `memory/signature.mjs`,
`memory/retrieval.mjs`, `memory/index.mjs`, `contracts/memory-node.schema.json`.
Tests: signature determinism, similarity gates (modality/operator hard gates),
tier boundaries, edge walk, index rebuild.
*Exit criteria: `node --test` green; no orchestrator changes yet.*

**Phase 2 — offline backfill (de-risks everything downstream).**
A `memory/populate.mjs` analogue that ingests the ~10 completed run trees
already in `src/semdb/runs/` (mmqa, ecomm, cars, animals, movie) to build an
initial L0/L1 population **offline**, then measures on held-out queries whether
the tier assignment is sane (does ecomm-q4 match a genuinely similar query, or
does everything collapse onto one template as `L1_tpch_Q9` did in GenDB's
`memory_report.json`?). Threshold tuning happens here, before any agent sees a
prompt change.
*Exit criteria: tier distribution over existing runs is defensible; no single L1
absorbs >40% of queries.*

**Phase 3 — read path.**
`{{memory_pre_injection}}` / `{{memory_catalog}}` / `{{memory_note}}` in the
three PGO user prompts, advisory invariant in the three system prompts, token
cap, `memory_match.json` emission, `--memory-dir` wiring in Phase-0 of
`runQueryDirectCore`.
*Exit criteria: prompt-structure tests extended and green; `--dry-run` shows the
injected block; a memory-enabled run reproduces a known query's result.*

**Phase 4 — control flow (warm start).**
Exact-tier warm start (stored plan as `previous_plan_path`, stored
helpers/solver seeded into `iter_0`), guarded so a stale reference that fails
preflight falls back to a cold start rather than failing the query.
*Exit criteria: replaying a previously solved query reaches its best objective
in strictly fewer agent calls.*

**Phase 5 — write path.**
`agents/memory-curator/{index.mjs,prompt.md,user-prompt.md}`,
`contracts/memory-update.schema.json`, `memory/apply-update.mjs`, orchestrator
post-run hook, GT-leak guard + test.
*Exit criteria: a full run writes schema-valid nodes; leak test green; second
run of the same workload classifies as `exact`/`FAMILIAR`.*

**Phase 6 — measurement.**
`scripts/memory_ab.sh`: same workload with `--no-memory` vs `--memory-dir`,
reporting per query: objective, iterations to best, agent calls, agent cost,
tier. Then a `CHANGELOG.md` version entry (Features / Design Rationale /
Notes & Caveats) per the repo convention.
*Exit criteria: a table showing iteration/cost reduction at equal-or-better
objective, or an honest negative result.*

### 2.9 Risks

| Risk | Mitigation |
|------|------------|
| Signature collapses (everything matches one template) | Phase 2 offline tuning before any prompt change; modality + operator hard gates; per-benchmark preference for L0/L1 |
| Negative transfer across benchmarks | `crossBenchmarkLayers: [3,4,5]` — instances and templates never cross |
| Token bloat degrading the planner | Hard `maxPreInjectionTokens`, tier-dependent injection, L5-only for novel |
| GT leakage into agent context | Enforced in `apply-update.mjs`, not in the agent prompt; dedicated test |
| Memory makes results irreproducible | `--memory-readonly` for A/B; `--no-memory` remains the reported baseline; every node records its source run id |
| Curator writes plausible-but-unevidenced knowledge | Every L2–L5 node requires ≥1 evidence entry with a real run id, query id, and before/after objective; nodes failing this are dropped by `apply-update.mjs` |
