# SemDB Agent Memory — code-level implementation plan (v2)

> **Status: IMPLEMENTED (2026-07-29).** Phases 1–6 are built and tested (23/23 SemDB tests
> pass via `scripts/run_semdb_tests.sh`). Phase 7 (the `--no-memory` vs `--memory-dir` A/B
> measurement) is not done; no performance claim is made until it runs.
>
> Decisions taken after review, differing from the plan text below:
> - the skill root is **isolated** per memory directory (§12 Q1 → isolated);
> - the three role skills are **no longer bound** — `agent-runtime/skill-loader.mjs` is
>   deleted and they are published into the discoverable root, each named as the first step
>   in its role's task prompt (§12 Q2 → do not keep binding);
> - the skill root also exists **without** `--memory-dir` (`<out>/_skills`), since the role
>   procedures used to be injected unconditionally and would otherwise disappear;
> - the **codex provider gained a skill capability** (filesystem protocol + usage
>   extraction) rather than only the inline-text fallback;
> - the retrieval key gained a **relational skeleton** component and the weighted
>   similarity path is **capped at 0.95**, after validation found false exact matches;
> - references below the objective floor are **inverted into warnings** and never warm-start.

Companion to `0729AgentMemoryPlan.md` (design study of GenDB's memory system).
This is the reviewable engineering spec.

**v2 changes vs v1:** layer names and the layer→storage split are now **identical
to GenDB**; L2–L5 are materialized as real Agent-SDK skills and the PGO agents
**discover them freely** instead of receiving pushed text. The one-bound-skill
invariant is relaxed accordingly.

Baseline: `agent-pgo-architecture` @ post-refactor tree — PGO agents
`query-planner`, `semantic-code-generator`, `semantic-optimizer` (legacy
`vadar-*` untouched); one primitive library `src/semdb/vadar/predefined.py`;
per-query run dir `<out>/<benchmark>-<query>/` with `telemetry.json`,
`plan.json`, `candidate_manifest.json`, `iter_*/`.

---

## 0. Scope and invariants

**Goal.** Cross-run memory that reduces agent calls and iterations to reach an
equal-or-better objective, via GenDB's two-channel architecture: deterministic
structural retrieval for L0/L1 (push) and native skill discovery for L2–L5
(pull).

**In scope:** PGO path only. Legacy VADAR path untouched.

**Invariants.**

1. `--no-memory` produces byte-identical prompts and an identical tool surface
   to today. Memory is strictly additive.
2. Every memory call site is non-fatal.
3. No ground-truth row ever reaches a memory node or a skill file. Enforced in
   code at the write boundary, not by prompt instruction.
4. The **retrieval key** (L0/L1 signature) is computed by JS, never authored by
   an LLM. This is the one thing GenDB got wrong and had to repair afterwards
   with `patchL1Signatures`.
5. Skill discovery is scoped to SemDB's own skill root (§4.6) — an agent must
   not discover GenDB's C++ skills or the operator's personal global skills.

---

## 1. Layers — identical names to GenDB

```
Layer 5: Performance Principles     ← Skill  (.claude/skills/<name>/)
Layer 4: Optimization Strategies    ← Skill
Layer 3: Operator Techniques        ← Skill
Layer 2: Sub-Structure Patterns     ← Skill
         |
Layer 1: Query Templates            ← HAG    (semdb-memory/graph/nodes/L1/)
         |
Layer 0: Query Instances            ← HAG    (semdb-memory/graph/nodes/L0/)
```

Same names, same storage split, same retrieval split as
`src/gendb/memory/README.md`. What changes is only the *domain content* each
layer holds, because SemDB compiles semantic operators to Python over
`vadar/predefined.py` rather than C++ over a columnar store, and optimizes a
per-query quality objective rather than latency:

| Layer | GenDB content | SemDB content |
|-------|---------------|---------------|
| L0 Query Instances | SQL, timing, best `.cpp` path, trajectory | SQL + NL, modality, objective `{name,value,direction}`, promoted `plan.json`/helpers/solver paths, iteration trajectory |
| L1 Query Templates | structural signature, proven strategies, anti-patterns | semantic signature (§3), proven plan skeleton, proven strategies, anti-patterns |
| L2 Sub-Structure Patterns | recurring operator compositions | recurring **semantic-site shapes** (e.g. image→structured join keyed by an OCR'd name, `sampling_unit: pair`) |
| L3 Operator Techniques | atomic C++ techniques w/ code patterns | atomic **helper recipes**: exact compositions of predefined primitives, with thresholds, tie-breaks, and which primitive is wrong for which case |
| L4 Optimization Strategies | high-level decision guides w/ evidence | **optimizer playbook**: failure signature → the action (`PATCH_CODE`/`REPLAN`) that actually fixed it |
| L5 Performance Principles | cross-cutting perf principles | cross-cutting principles over quality/compilability (e.g. "`value_space` must be closed over the join column's distinct values") |

"Performance" in L5 reads as *objective performance*; the name is kept
deliberately so the two systems stay diffable.

---

## 2. File inventory

```
src/semdb/
├── memory/
│   ├── index.mjs             # initMemory, classifyQuery, getMemorySummary
│   ├── graph.mjs             # node/edge CRUD + index (ported from gendb, bug-fixed)
│   ├── signature.mjs         # NEW: SemDB retrieval key + similarity
│   ├── retrieval.mjs         # tiering + L0/L1 pre-injection + catalog
│   ├── skills.mjs            # NEW: skill root init, lint/quarantine, catalog, usage
│   ├── apply-update.mjs      # deterministic HAG writer + GT-leak guard
│   └── backfill.mjs          # offline L0/L1 population from existing runs/
├── contracts/
│   ├── memory-node.schema.json      # NEW
│   └── memory-update.schema.json    # NEW
├── agents/memory-manager/           # NEW (same role name as GenDB)
│   ├── index.mjs · prompt.md · user-prompt.md
└── tests/
    ├── test_memory_signature.mjs · test_memory_graph.mjs
    ├── test_memory_retrieval.mjs · test_memory_apply_update.mjs
    ├── test_memory_skills.mjs       # NEW: skill lint, quarantine, catalog, isolation
    └── test_memory_injection.mjs
```

Modified: `orchestrator.mjs`, `semdb.config.mjs`,
`agent-runtime/skill-loader.mjs`, the three PGO agents' `index.mjs` +
`prompt.md` + `user-prompt.md`, `gendb/providers/{claude,codex}.mjs` (one new
optional parameter, §4.6), `tests/test_skill_loader.mjs`,
`tests/test_agent_prompt_structure.mjs`, `CHANGELOG.md`.

On-disk layout (default `<out>/../semdb-memory`, override `--memory-dir`):

```
semdb-memory/
├── config.json
├── graph/
│   ├── index.json · edges.json
│   └── nodes/L0..L5/<id>.json         # L2–L5 nodes are lightweight skill refs
├── skill-root/                        # the SDK cwd for PGO agents (§4.6)
│   └── .claude/skills/<skill-name>/
│       ├── SKILL.md
│       ├── code-patterns/<pattern>.py
│       ├── evidence.json
│       └── gotchas.md
└── quarantine/<skill-name>/           # skills that failed lint, kept for diagnosis
```

`.gitignore:55` already ignores `.claude/skills/*/`; add `semdb-memory/` if the
default location lands inside the repo.

---

## 3. `memory/signature.mjs` — the L0/L1 retrieval key

Two-stage, because at Phase 0 there is no plan yet.

### 3.1 Stage A — plan-independent (the matching key)

```js
export const SIGNATURE_VERSION = 1;

/**
 * Retrieval key from SQL + NL + resolved tables only — everything planQuery()
 * already computed, so building it costs no extra I/O.
 * @param {{query, sql, nl, benchmark, corpus:{table,modality}, tables, isImage}} planObj
 */
export function buildQuerySignature(planObj) // → {
//   signature_version, modality, operator_kind, sampling_unit, corpus_table,
//   tables: string[], predicate_tokens: string[], projection_arity: number,
//   flags: {group_by, order_by, limit, aggregate, distinct},
//   sql_template: string, template_hash: string }
```

| Field | Derivation |
|-------|-----------|
| `modality` | `planObj.corpus.modality` |
| `operator_kind` | `join` if `tablesInPredicate(sql,bench).length >= 2`; else `group` / `agg` / `topk` / `filter` / `map` by SQL form |
| `sampling_unit` | `pair` for join, `group` for group, else `row` — mirrors the existing validation split so memory and validation agree |
| `tables` | `tablesInSql(sql,args).map(t=>t.table).sort()` |
| `predicate_tokens` | `semanticArgs(sql) + " " + (nl ?? "")` → lowercase, strip non-alphanum, drop a 40-word stoplist and ≤2-char tokens, unique, sorted |
| `sql_template` | literals → `:p`, whitespace collapsed |
| `template_hash` | `sha256(sql_template + modality).slice(0,16)` |

`metric_family` is excluded from the key: it is produced by
`scenario_metrics.py` *after* evaluation and is unavailable at match time. It is
stored on L0 and used only as a post-hoc tiebreak.

### 3.2 Similarity — hard gates, then weighted agreement

```js
export function signatureSimilarity(a, b) {
  if (a.signature_version !== b.signature_version) return 0;
  if (a.modality !== b.modality) return 0;            // hard gate
  if (a.operator_kind !== b.operator_kind) return 0;  // hard gate
  if (a.template_hash === b.template_hash) return 1;  // same shape, new params
  return 0.20 * (a.sampling_unit === b.sampling_unit ? 1 : 0)
       + 0.20 * jaccard(a.tables, b.tables)
       + 0.35 * jaccard(a.predicate_tokens, b.predicate_tokens)
       + 0.10 * (a.projection_arity === b.projection_arity ? 1 : 0)
       + 0.15 * flagAgreement(a.flags, b.flags);
}
```

The two hard gates are the deliberate departure from GenDB's soft weighting:
their `memory_report.json` shows `L1_tpch_Q9` absorbing 8 of 17 queries, and a
false structural match costs a whole iteration budget of misleading context.

### 3.3 Stage B — plan-derived features (payload, not key)

```js
export function planFeatures(plan)  // {site_operators, sampling_units, output_types,
                                    //  value_space_kinds, helper_names, primitives,
                                    //  relational_ops, compilability}
export function planAffinity(a, b)  // 0.5*jaccard(primitives) + 0.3*jaccard(site_operators)
                                    // + 0.2*(sampling_units equal)
```

Used to enrich L1's `plan_skeleton` and, post-plan, to decide whether the
generator prompt may cite a stored helper DAG.

---

## 4. Modules

### 4.1 `graph.mjs`

Port of `src/gendb/memory/graph.mjs`, three changes:

1. **Fix the ported defect**: `loadMemoryNode` cases 2–5 reference `n.skill_path`
   where the variable is `node` — `ReferenceError` on any L2–L5 load. SemDB
   needs those paths working, since L2–L5 refs are what the catalog renders.
2. `writeNode` no longer calls `updateIndex()` per write (O(n²) on a batch);
   `applyMemoryUpdate` calls it once at the end.
3. Every write goes through ajv against `memory-node.schema.json` (`ajv` is
   already a dependency).

Same exported surface as GenDB's so the two stay diffable.

### 4.2 Node schema

L0/L1 carry full content; **L2–L5 nodes are lightweight skill references**,
exactly as in GenDB:

```jsonc
{ "id": "L3_ocr_name_join", "layer": 3,
  "skill_path": "skill-root/.claude/skills/semdb-ocr-name-join/",
  "skill_name": "semdb-ocr-name-join",
  "summary": "Bind an image→name join through best_ocr_match with a closed value space",
  "benchmark": null,                       // L2–L5 are cross-benchmark by design
  "content": { "description": "...", "preconditions": [], "source_examples": [] },
  "evidence_count": 3,
  "data_boundary": {"source":"select_validation","cert_accessed":false,
                    "full_ground_truth_accessed":false},
  "created_at": "...", "updated_at": "..." }
```

The graph node is the index; the authored prose, code patterns and evidence live
in the skill directory. This is what lets the catalog stay cheap while the skill
body stays rich.

### 4.3 `retrieval.mjs`

```js
/**
 * @returns {{tier, score, matchedL1, matchedL0, warmStart|null,
 *            preInjected: string,   // L0/L1 only — pushed
 *            catalog: string,       // L2–L5 skill summaries — pull hint
 *            inlineSkills: string,  // codex fallback only (§4.7)
 *            signature}}
 */
export async function classifyQuery(planObj, args, memoryDir, config)
```

1. `sig = buildQuerySignature(planObj)`.
2. Score against every L1 (`getNodesByLayer(1)`), same-benchmark preferred on
   ties (0.001 tiebreak, never a correctness role).
3. Tier: `>= 0.98` exact · `>= 0.55` structural · else novel.
4. Exact tier → best L0 by direction-aware objective whose promoted files still
   exist → `warmStart`.
5. `preInjected` renders **L0/L1 only** (strategies, anti-patterns, reference
   result), token-capped.
6. `catalog` renders the L2–L5 skill refs relevant to the matched L1 (via
   `exhibits_pattern` / `uses_operator` edges) plus all L5 — names + one-line
   summaries only. Agents load the bodies themselves.

Cross-benchmark policy: L0/L1 filtered to `node.benchmark === args.benchmark`;
L2–L5 never filtered (`crossBenchmarkLayers: [2,3,4,5]`), which is the whole
point of promoting knowledge out of the instance layers.

### 4.4 `skills.mjs` — the pull channel's plumbing

```js
export async function initSkillRoot(memoryDir)            // mkdir <memoryDir>/skill-root/.claude/skills
export function skillRootFor(memoryDir)                   // absolute path used as the SDK cwd
export async function listSkills(memoryDir)               // [{name, description, dir, evidenceCount}]
export async function lintSkill(dir)                      // → {ok, errors[]}
export async function quarantineSkill(memoryDir, name, errors)
export async function lintAllSkills(memoryDir)            // returns {kept[], quarantined[]}
export function renderCatalog(refs, cap)                  // markdown, capped
export async function recordSkillUsage(outDir, queryId, skillsUsed)
```

`lintSkill` enforces, per skill directory:

- `SKILL.md` exists with parseable YAML frontmatter containing a non-empty
  `name` and `description`; `name` matches `^semdb-[a-z0-9]+(-[a-z0-9]+)*$` and
  equals the directory name;
- description starts with "Use when" or "Load when" (GenDB's convention — it is
  what makes semantic matching work);
- `evidence.json` parses and has ≥1 entry with a real `run_id`, `query_id`, and
  before/after objective;
- any `code-patterns/*.py` parses under `python3 -m py_compile` **only if the
  file is a complete module**; snippets with placeholders are checked by regex
  for balanced fences instead;
- the GT tripwires (§6.3) do not fire on any file in the directory;
- total directory size ≤ 64 KB.

Failures move the directory to `quarantine/<name>/` with a `lint_errors.json`,
so a bad authoring pass degrades to "no skill" rather than poisoning every
future run. Quarantine is the pull channel's equivalent of the schema check on
the push channel — with free discovery, lint is the *only* gate between what the
manager writes and what every future agent reads.

### 4.5 `apply-update.mjs` — deterministic HAG writer

Handles L0/L1 (and the L2–L5 *reference* nodes) from the manager's
`memory_update.json`; the skill *bodies* are authored directly by the agent
(§6). Steps:

1. **Leak guard** (§6.3) — reject the whole update on any hit.
2. **Evidence guard** — drop L2–L5 refs whose `skill_path` does not exist, or
   whose skill failed lint, or with empty evidence.
3. **Signature ownership** — discard any `signature` the agent proposed;
   recompute from stored SQL/NL via `buildQuerySignature`. Writer and reader
   thus share one representation by construction (invariant 4).
4. **L1 merge** — attach each L0 to an L1 with similarity ≥0.98, creating
   `L1_<benchmark>_<operator_kind><modality>_<template_hash>` if absent; merge
   `proven_strategies` / `anti_patterns` as ordered sets capped at 12.
5. **Edges** — `instance_of`, `exhibits_pattern`, `uses_operator`,
   `implements_strategy`, `exemplifies_principle` (GenDB's exact edge types).
6. **Prune** — ≤ `maxNodesPerLayer` per (layer, benchmark), evicting by
   `(evidence_count, updated_at)` ascending; evicted L2–L5 refs also archive
   their skill dir to `quarantine/`.
7. `updateIndex()` once; write `<out>/memory_update_summary.json`.

### 4.6 Enabling free discovery (the core v2 change)

Four coordinated edits:

**(a) `agent-runtime/skill-loader.mjs`** — the bound role skill stays mandatory,
but the sentence that forbids discovery goes. New prompt preamble:

```js
  const prompt = [
    "## Bound procedural skill",
    `This procedural skill is statically bound to the current agent role: \`${values.name}\`.`,
    "Follow it for this invocation — it is your mandatory procedure.",
    "You may additionally load any discovered `semdb-*` memory skill when its description",
    "matches the problem in front of you. Those are advisory prior knowledge, never a",
    "substitute for this procedure or for the authoritative primitive API.",
    "",
    body,
  ].join("\n");
```

`test_skill_loader.mjs` asserts `includes("statically bound")` — still true.

**(b) The three PGO agent configs** — add `"Skill"` to `allowedTools`. Without
it the provider filters the tool out and discovery is dead regardless of
settings. This is the single line that actually turns the pull channel on.

**(c) Provider isolation** — `providers/{claude,codex}.mjs` `runAgent()` takes a
new optional `skillRoot`:

```js
export async function runAgent(name, { ..., skillRoot, settingSources }) {
  ...
  cwd: skillRoot || REPO_ROOT,
  settingSources: settingSources || ['user', 'project'],
```

Defaults preserve GenDB's behavior exactly. SemDB passes
`skillRoot: skillRootFor(memoryDir)` and `settingSources: ['project']`, so a PGO
agent discovers **only** SemDB's learned skills — not GenDB's C++ skills in the
repo root, and not the operator's global skill collection. Free discovery within
a curated namespace; not free discovery of the whole machine.

Because every path in the SemDB prompts is absolute, moving cwd is safe; this is
covered by `test_memory_skills.mjs` asserting that a run with an empty skill root
sees zero discoverable skills.

**(d) `runPhase`** — two independent knobs instead of one:

```js
  const boundSkill = ...                       // existing --no-agent-skills knob
  const memorySkills = args.memoryDir && !args.noMemorySkills;
  const result = await runAgent(agentConfig.name, {
    ...,
    useSkills: Boolean(boundSkill) || memorySkills,
    skillRoot: memorySkills ? skillRootFor(args.memoryDir) : undefined,
    settingSources: memorySkills ? ['project'] : undefined,
    domainSkillsPrompt: boundSkill?.prompt,
  });
  recordSkillUsage(args.out, query, result.skillsUsed);   // → <out>/skill_usage.json
```

### 4.7 Codex fallback

`providers/codex.mjs` has no Skill tool and returns `skillsUsed: {}`
unconditionally. Under `--agent-provider codex`, `classifyQuery` populates
`inlineSkills`: the bodies of the top-K (default 3) edge-linked skills, inlined
as text under a `{{#if memory_inline_skills}}` block, subject to the same token
cap. Information parity, different delivery. Logged once per run:
`[SemDB] memory: codex provider — inlining N skills (no native discovery)`.

### 4.8 `backfill.mjs`

```
node src/semdb/memory/backfill.mjs --memory-dir <dir> --runs <d1,d2,...> [--dry-run]
```

Mechanically synthesizes **L0/L1 only** (no LLM, no skills) from existing
`<out>/<benchmark>-<query>/` trees, for threshold tuning (Phase 2) and cold
start. `--dry-run` prints the tier matrix instead of writing.

---

## 5. Orchestrator integration

Anchored by function name; all in `src/semdb/orchestrator.mjs`.

**5.1 Export helpers.** `semanticArgs` and `tablesInPredicate` become exported so
`signature.mjs` reuses the existing SQL scanning instead of reimplementing it.

**5.2 `parseArgs`.** Add `memoryDir: defaults.memoryDir`, `memoryReadonly:false`,
`noMemorySkills:false` and the flags `--memory-dir <path>`, `--no-memory`,
`--memory-readonly`, `--no-memory-skills`.

**5.3 `main()`.** After `setAgentProvider`: `initMemory` + `initSkillRoot` +
`lintAllSkills` (quarantining anything a previous run left broken) + summary log.
After `runPlans` is finalized, pre-classify every query into
`base.memoryClassifications` and log the tier histogram. After the per-query
loop, run the Memory Manager (§6) when
`memoryReady && !memoryReadonly && !dryRun`.

**5.4 `runQueryDirectCore`.** Write `<runDir>/memory_match.json`; extend
`plannerVars`/`generatorVars`/optimizer vars with
`memory_pre_injection`, `memory_catalog`, `memory_inline_skills`, `memory_note`
— all `""` when memory is off, so every `{{#if}}` collapses and invariant 1
holds.

**5.5 Warm start (exact tier).** `createInitialPlan` passes the stored plan as a
**reference**, not as replan context, via a new
`{{#if memory_reference_plan_path}}` branch (§7.1) — the existing replan branch
demands `plan_version = previous+1`, which would break
`createInitialPlan`'s `plan_version === 1` assertion. `generateCandidate` seeds
`iter_0` with the stored helpers/solver when
`memory.warmStart === true`. If preflight rejects the warm candidate, `iter_0`
is regenerated once cold — a stale reference degrades to today's behavior,
never fails the query.

**5.6 Telemetry.** `telemetry.json` gains:

```jsonc
"memory": { "enabled": true, "tier": "structural", "score": 0.71,
            "matched_l1": "L1_mmqa_joinimage_9f2c1a44", "warm_start": false,
            "skills_available": 12,
            "skills_used": {"semdb-ocr-name-join": 2},
            "injected_tokens": {"pre": 640, "catalog": 210} }
```

`skills_used` is the pull channel's only observability — without it there is no
way to tell whether free discovery is being exercised or ignored.

---

## 6. Memory Manager agent

Named `memory-manager` to match GenDB.

**6.1 `agents/memory-manager/index.mjs`**

```js
export const config = {
  name: "Memory Manager", configKey: "memory_manager",
  promptPath: resolve(here, "prompt.md"),
  userPromptPath: resolve(here, "user-prompt.md"),
  allowedTools: ["Read", "Write", "Edit", "Glob", "Grep"],
};
```

No `skillPath`: it authors skills, it does not consume them. Config:
`agentModels.memory_manager = "opus"` / `"gpt-5.6-luna"`;
`agentEffortLevels.memory_manager = "high"`.

**6.2 Contract — split by ownership.**

- The agent **writes skill directories directly** into
  `<memoryDir>/skill-root/.claude/skills/<semdb-name>/`
  (`SKILL.md`, `code-patterns/*.py`, `evidence.json`, `gotchas.md`) — prose and
  code patterns are exactly what an LLM should own.
- The agent **writes `memory_update.json`** describing L0/L1 content and the
  L2–L5 *reference* nodes + edges. Code applies it (§4.5).
- The agent **never** writes into `graph/`.

Rationale: GenDB lets the agent write node JSON and repairs it afterwards. The
split keeps the retrieval key in code while leaving authoring where it belongs,
and gives lint + schema a single enforcement point each.

**6.3 Differential gate and extraction priorities** (stated in the prompt,
re-checked in code):

- FAMILIAR (L1 match ≥0.98 and no objective gain) → emit nothing.
- NOVEL_SUCCESS (no L1 ≥0.55, run ok, objective non-null) → L0 (+L1 by merge).
- SIGNIFICANT_IMPROVEMENT → gain ≥ `differentialHeadroomThreshold` of remaining
  headroom.

**Headroom-relative breakthrough** — for maximize objectives bounded at 1,
`(after-before)/max(ε,1-before)`; for minimize, `(before-after)/max(ε,before)`.
Raw deltas would rank 0.20→0.25 above 0.90→0.95, which is backwards for quality
metrics. The same formula lives in `apply-update.mjs` so a mis-classified
proposal is rejected.

Priorities: (1) a `REPLAN` that improved the objective → **L4** entry, the
highest-signal artifact available because it encodes "this symptom needs a plan
change, not a patch"; (2) a `PATCH_CODE` that improved ≥ threshold → **L3**
recipe or a new `failure_modes` entry on an existing one; (3) a regression →
`gotchas.md` on the relevant skill + `anti_patterns` on L1; (4) a site shape
recurring across ≥2 L1s → **L2**; (5) a claim supported by ≥2 distinct
benchmarks → **L5**. First-iteration successes produce L0 only.

De-duplication inputs, mirroring GenDB: existing skill names + descriptions,
existing L1 signatures, existing L4 failure signatures, and this run's
`skill_usage.json` (usage counts get merged into each skill's `evidence.json`,
so "which skills actually got loaded" becomes evidence).

**6.4 SKILL.md format** (enforced by `lintSkill`):

```markdown
---
name: semdb-<kebab-name>
description: Use when <specific query/plan/failure situation>. <What it gives you.>
---

## When to Use
## Technique
## Code Patterns      # references code-patterns/*.py, placeholders documented
## Gotchas            # references gotchas.md
```

Code patterns are Python over the predefined API with documented placeholders
(`CORPUS_ROWS`, `CANDIDATE_NAMES`, `SCORE_THRESHOLD`), never a copied solver —
the GenDB "distill, don't copy" rule, transposed.

**6.5 Ground-truth leak guard.** Applied to *both* `memory_update.json` and every
file written under the skill root. Whole-update rejection if any string:
matches `/ground[_ -]?truth|raw_results|\bCERT\b/i`; contains an absolute path
under `groundTruthDir`; carries a `rows`/`mistakes`/`expected` array of per-row
objects; or dumps a long run of row ids. Aggregate metrics and technique prose
are allowed. The manager legitimately runs after final evaluation and therefore
*can* see GT — so the guard must sit at the write boundary, the only place the
agent's output and future prompt content meet.

---

## 7. Prompt changes

**7.1 `query-planner/user-prompt.md`** — after the primitive-API section:

```markdown
{{#if memory_pre_injection}}
{{memory_pre_injection}}
{{/if}}

{{#if memory_catalog}}
{{memory_catalog}}
{{/if}}

{{#if memory_inline_skills}}
{{memory_inline_skills}}
{{/if}}

{{#if memory_reference_plan_path}}
## Reference Plan From a Past Run

A structurally identical query was planned before: `{{memory_reference_plan_path}}`

Read it as a reference, not a specification. This is still the initial plan: keep
`plan_version` at `1` and `parent_plan_version` at `null`. Adopt only what the current
tables and the authoritative primitive API support.
{{/if}}
```

**7.2 `semantic-code-generator/user-prompt.md`** — same three memory blocks after
the plan/manifest inputs.

**7.3 `semantic-optimizer/user-prompt.md`** — same three blocks after
`## Prior Candidate Manifests`. This is where the payoff should be largest: L4
is literally a table of "symptom → action", and action selection is the
optimizer's whole job.

**7.4 The catalog block** rendered by `renderCatalog`:

```markdown
## Available Memory Skills
Learned from past runs. Load one with the Skill tool when its description matches your
situation. They are advisory prior knowledge — the plan schema, table metadata and
`vadar/predefined.py` remain authoritative, and they never contain ground-truth answers.

- **semdb-ocr-name-join** (L3): Use when joining an image corpus to a name column…
- **semdb-closed-value-space** (L5): Use when a helper returns an open string…
```

**7.5 System prompts (all three PGO roles)** — one identical paragraph:

```markdown
## Prior knowledge is advisory

You may discover and load `semdb-*` memory skills, and a "Prior Knowledge" block may
appear in your task. Both summarize past runs; neither is a specification, and either may
be stale or wrong for this query. Never bind a primitive, argument, or return type because
prior knowledge mentioned it — verify it in the authoritative API file first. Prior
knowledge never contains ground-truth answers; do not treat any value in it as a label.
```

---

## 8. Config diff (`semdb.config.mjs`)

```js
  memoryDir: null,
  memory: {
    exactMatchMinScore: 0.98,
    structuralMatchMinScore: 0.55,
    maxPreInjectionTokens: 3000,      // L0/L1 push, per role
    maxCatalogTokens: 700,            // L2–L5 pull hint, per role
    maxInlineSkills: 3,               // codex fallback only
    differentialHeadroomThreshold: 0.30,
    maxNodesPerLayer: 40,
    maxSkills: 30,                    // hard cap on the discoverable namespace
    crossBenchmarkLayers: [2, 3, 4, 5],
    skillNamePrefix: "semdb-",
    settingSources: ["project"],
    warmStart: true,
  },
```

plus `memory_manager` entries in both providers' `agentModels` /
`agentEffortLevels`.

`maxSkills` matters more than it looks: every discoverable skill's description
enters the agent's context whether or not it is loaded, so an uncapped namespace
silently becomes a per-call tax on all three roles.

---

## 9. Tests

Plain top-level-assert `.mjs`, matching the existing convention.

| File | Asserts |
|------|---------|
| `test_memory_signature.mjs` | determinism; param variants share `template_hash`; modality gate → 0; operator gate → 0; jaccard edge cases; `planFeatures` over `runs/mmqa-q2a/plan.json` |
| `test_memory_graph.mjs` | L0–L5 init; write→read; `addEdge` dedup; `getConnectedNodes` both directions; index counts; schema rejection; **L2–L5 render does not throw** (the ported GenDB defect) |
| `test_memory_retrieval.mjs` | tier boundaries 0.55/0.98; same-benchmark preference; L0/L1 never cross benchmarks while L2–L5 do; `warmStart` null when the file is missing; empty memory → novel + empty blocks |
| `test_memory_skills.mjs` | lint accepts a well-formed skill; rejects+quarantines each violation (bad name, missing description, evidence-less, oversized, GT tripwire); catalog respects `maxCatalogTokens`; `skillRootFor` isolation — an agent rooted at the skill root cannot see repo-root `.claude/skills`; `maxSkills` eviction order |
| `test_memory_apply_update.mjs` | GT tripwires reject; clean update applies; agent-proposed signature discarded and recomputed; L2–L5 ref pointing at a quarantined skill is dropped; headroom classification (0.90→0.95 outranks 0.20→0.25); prune order |
| `test_memory_injection.mjs` | each block respects its cap; blocks carry the advisory header; **memory off ⇒ rendered prompt byte-identical to the golden** (invariant 1) |
| `test_skill_loader.mjs` (extend) | bound prompt still says "statically bound"; no longer forbids discovery; explicitly permits `semdb-*` |
| `test_agent_prompt_structure.mjs` (extend) | all three roles carry the three `{{#if}}` memory guards, `"Skill"` in `allowedTools`, and the advisory paragraph |

Add `scripts/run_semdb_tests.sh` (no aggregate runner exists today; `npm test`
points at the GenDB orchestrator).

---

## 10. Phases and exit criteria

| # | Scope | Exit criteria |
|---|-------|---------------|
| 1 | `signature.mjs`, `graph.mjs`, node schema, `backfill.mjs`, tests 1–2 | Tests green; no orchestrator edits; `backfill --dry-run` runs over `runs/{mmqa,ecomm,cars,animals,movie}*` |
| 2 | Threshold tuning on backfilled L0/L1 | **Gate.** No single L1 absorbs >40% of a benchmark's queries; every `exact` pair manually confirmed same-shape. Do not proceed if the signature cannot separate them |
| 3 | Push channel: `retrieval.mjs`, prompts, `parseArgs`, `main()` pre-classification, `memory_match.json`, telemetry | `--no-memory` dry-run diff vs pre-change tree is empty; a memory-on dry-run shows the injected block; one real query reproduces its known objective |
| 4 | **Pull channel**: `skills.mjs`, provider `skillRoot`/`settingSources`, `Skill` in `allowedTools`, skill-loader rewording, catalog, `skill_usage.json`, codex inlining, test 4 | With hand-written seed skills in the skill root, a real PGO run shows a non-empty `skills_used`; an agent rooted at the skill root cannot see repo-root or user skills; codex path inlines instead |
| 5 | Warm start + preflight fallback | Replaying a solved query reaches its best objective in strictly fewer agent calls (from `telemetry.phases`); a corrupted warm reference falls back to cold |
| 6 | Memory Manager + `memory-update.schema.json` + `apply-update.mjs` + hook, test 5 | A full run writes schema-valid nodes and lint-clean skills; leak tests green; re-running the workload yields `exact`/`FAMILIAR` |
| 7 | `scripts/memory_ab.sh` + `CHANGELOG.md` | Table over one benchmark: objective, iterations-to-best, agent calls, cost, tier, skills_used — `--no-memory` vs `--memory-dir`. An honest negative result is acceptable |

Phase 4 deliberately precedes Phase 6: the pull channel is validated with
hand-written seed skills before an agent is allowed to author them, so a
discovery failure and an authoring failure can never be confused.

Estimated diff: ~1700 LOC new (~1000 modules, ~700 tests), ~150 LOC orchestrator,
~25 LOC providers, ~80 lines of prompt text.

---

## 11. Risks specific to free discovery

| Risk | Mitigation |
|------|-----------|
| Namespace pollution — agents discovering GenDB's C++ skills or the operator's global skills | Isolated `skillRoot` + `settingSources:['project']` + `semdb-` prefix; asserted by test |
| Description tax — every skill's description enters context even unloaded | `maxSkills: 30`, eviction by evidence count, catalog token cap |
| A wrong skill is authored once and silently misleads every later run | `lintSkill` + quarantine; evidence required; `gotchas.md` from regressions; `skills_used` telemetry makes influence visible; skills are cheap to delete since the whole root is disposable |
| Non-determinism: the same query now depends on what the agent chose to load | `skills_used` recorded per run; `--no-memory` remains the reported baseline; `--memory-readonly` for A/B |
| Skill content drifting from the primitive API after a `predefined.py` refactor | Lint re-runs at every `initMemory`; add an API-surface hash to `evidence.json` and quarantine skills whose recorded surface no longer matches |
| Prompt injection via skill text | Only the Memory Manager writes there, its output is lint-gated, and the root is not user-writable in normal operation — but note the trust boundary explicitly: a skill body is executed as instructions by three agents |

---

## 12. Open questions for review

1. **Skill-root isolation vs repo-root sharing.** Recommended: isolated root
   (`<memoryDir>/skill-root`). The alternative — writing into the repo's
   `.claude/skills/` like GenDB — makes SemDB and GenDB skills mutually visible.
   Sharing is only right if you want C++ storage techniques to inform semantic
   operator planning; I assume you do not.
2. **Bound role skill: keep mandatory?** Recommended yes — it is the role's
   procedure, not knowledge. The alternative is to publish the three role skills
   into the discoverable root too, which makes the procedure optional. I would
   not.
3. **Warm-start aggressiveness** — seed `iter_0` with the stored solver
   (max saving, risk of carrying a stale helper) vs reference-only. Proposed:
   both behind `memory.warmStart: true | "reference_only"`, decided by Phase 5
   measurement.
4. **Manager cadence** — per run (proposed; needed for L2/L5 cross-query
   synthesis) vs per query.
5. **`predicate_tokens` similarity** — token Jaccard cannot tell "shows a dog"
   from "contains a canine". Proposed: Jaccard for v1, revisit only if Phase 2
   shows it is the binding constraint.
