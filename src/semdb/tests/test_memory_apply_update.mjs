/**
 * The write boundary: leak guard, evidence guard, signature ownership, the
 * headroom rule, the lint gate, and pruning.
 *
 * The Memory Manager runs after final evaluation, so it is the one agent that CAN
 * see ground truth, while everything it writes is later read by the three agents
 * that must not. Every rule below is enforced here rather than in its prompt.
 */

import assert from "node:assert/strict";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import { getNodesByLayer, readAllEdges, readNode } from "../memory/graph.mjs";
import { applyMemoryUpdate, headroomGain, validateUpdate } from "../memory/apply-update.mjs";
import { initMemory } from "../memory/index.mjs";
import { skillRootFor, skillsDirFor } from "../memory/skills.mjs";
import { buildQuerySignature } from "../memory/signature.mjs";

const memoryDir = await mkdtemp(resolve(tmpdir(), "semdb-mem-apply-"));
await initMemory(memoryDir, {});
const skillRoot = skillRootFor(memoryDir);

const SQL = `SELECT t.ID, i.uri FROM mmqa.ap_warrior t, mmqa.images i
WHERE AI.IF(('Determine if the image shows the logo of the racetrack. Racetrack: ', t.Track, i.uri),
  connection_id => 'c')`;

const queryMeta = {
  q2a: {
    sql: SQL, nl: "match racetrack logos",
    corpus: { table: "images", modality: "image" },
    tables: [{ table: "ap_warrior" }, { table: "images" }],
    objective: { name: "f1", value: 0.82, direction: "maximize" },
    iterations: 2, replans: 1,
    actionCounts: { PATCH_CODE: 1, REPLAN: 1, STOP: 1 },
    plan: {
      semantic_sites: [{ site_id: "s1", operator: "sem_join", sampling_unit: "pair", output_type: "boolean", value_space: ["true", "false"] }],
      helper_dag: [{ helper_id: "h1", name: "logo_name", args: [], return_type: "str", depends_on: [], confidence_signal: null, primitive_steps: [{ primitive: "best_ocr_match" }] }],
      relational_plan: ["scan images", "join on predicted name"],
      compilability: { class: "exact", obligations: [], unresolved: [] },
    },
    promoted: { plan_path: "/run/plan.json", helpers_path: "/run/h.py", solver_path: "/run/s.py", candidate_id: "q2a-iter-2" },
    metricFamily: "QueryMetricRetrieval",
    runDir: "/run",
  },
};

const ctx = {
  memoryDir, skillRoot, runId: "2026-07-29T12-00-00", benchmark: "mmqa",
  queries: queryMeta, config: {}, groundTruthDir: "/sembench/files/mmqa/raw_results",
};

const baseUpdate = () => ({
  schema_version: "1.0",
  run_id: ctx.runId,
  benchmark: "mmqa",
  queries: [{
    query_id: "q2a",
    classification: "NOVEL_SUCCESS",
    template: {
      proven_strategies: ["bind the join through best_ocr_match over the closed track-name set"],
      anti_patterns: ["free-form captioning followed by string equality"],
    },
    breakthroughs: [{
      iteration: 1, action: "REPLAN", before: 0.41, after: 0.82,
      technique: "switched the sampling unit from row to pair so the join predicate was scored per pair",
    }],
  }],
  skills: [],
});

async function makeSkill(name, { body, evidence } = {}) {
  const dir = resolve(skillsDirFor(skillRoot), name);
  await mkdir(dir, { recursive: true });
  await writeFile(resolve(dir, "SKILL.md"),
    `---\nname: ${name}\ndescription: Use when an image corpus must be joined to a name column.\n---\n\n`
    + (body || "# Skill\n\n## When to Use\nJoining images to names.\n\n## Technique\nBind through best_ocr_match.\n"));
  await writeFile(resolve(dir, "evidence.json"), JSON.stringify(evidence ?? [{
    run_id: ctx.runId, benchmark: "mmqa", query_id: "q2a", before: 0.41, after: 0.82,
  }], null, 2));
  return dir;
}

// --- schema -----------------------------------------------------------------

assert.throws(() => validateUpdate({ schema_version: "1.0" }), /Invalid memory update/);
assert.throws(
  () => validateUpdate({ ...baseUpdate(), queries: [{ query_id: "q", classification: "MADE_UP" }] }),
  /Invalid memory update/,
);
assert.doesNotThrow(() => validateUpdate(baseUpdate()));

// --- headroom rule ----------------------------------------------------------

const highBase = headroomGain(0.90, 0.95, "maximize");
const lowBase = headroomGain(0.20, 0.25, "maximize");
assert.ok(highBase > lowBase,
  "0.90→0.95 must outrank 0.20→0.25: raw deltas rank quality metrics backwards");
assert.ok(Math.abs(highBase - 0.5) < 1e-9);
assert.ok(headroomGain(null, 0.4) === 1, "a first measurement is all upside");
assert.ok(headroomGain(0.5, 0.25, "minimize") > 0, "minimize objectives improve downward");
assert.equal(headroomGain(0.5, Number.NaN), 0);

// --- happy path -------------------------------------------------------------

const okSummary = await applyMemoryUpdate(baseUpdate(), ctx);
assert.equal(okSummary.applied, true);
assert.equal(okSummary.classifications.NOVEL_SUCCESS, 1);
assert.equal(okSummary.nodes_created.L0, 1);
assert.equal(okSummary.nodes_created.L1, 1);
assert.equal(okSummary.edges_created, 1);

const l0s = await getNodesByLayer(0, memoryDir);
const l1s = await getNodesByLayer(1, memoryDir);
assert.equal(l0s.length, 1);
assert.equal(l1s.length, 1);
assert.equal(l0s[0].content.breakthroughs.length, 1, "the verified breakthrough is kept");
assert.ok(l0s[0].content.breakthroughs[0].headroom_gain > 0.3);
assert.deepEqual(
  l0s[0].signature, buildQuerySignature({
    query: "q2a", sql: SQL, nl: "match racetrack logos", benchmark: "mmqa",
    corpus: queryMeta.q2a.corpus, tables: queryMeta.q2a.tables,
  }),
  "the signature must be recomputed by code, not taken from the agent",
);
assert.equal(l1s[0].content.plan_skeleton.primitives[0], "best_ocr_match");
assert.equal((await readAllEdges(memoryDir))[0].type, "instance_of");

// --- signature ownership: an agent-proposed signature is impossible ---------
// The schema has no signature field at all, so the agent cannot even express one.
assert.throws(
  () => validateUpdate({
    ...baseUpdate(),
    queries: [{ query_id: "q2a", classification: "NOVEL_SUCCESS", signature: { modality: "text" } }],
  }),
  /Invalid memory update/,
  "the update format must not accept a signature",
);

// --- second run merges into the same template ------------------------------

const merged = await applyMemoryUpdate({
  ...baseUpdate(),
  queries: [{
    query_id: "q2a", classification: "SIGNIFICANT_IMPROVEMENT",
    template: { proven_strategies: ["cap the candidate set to the joined table's distinct values"] },
  }],
}, ctx);
assert.equal(merged.nodes_updated, 1, "same signature merges, it does not fork a template");
assert.equal((await getNodesByLayer(1, memoryDir)).length, 1);
const mergedTemplate = (await getNodesByLayer(1, memoryDir))[0];
assert.equal(mergedTemplate.content.proven_strategies.length, 2, "strategies accumulate as a set");
assert.equal(mergedTemplate.content.instance_count, 2);

// --- unverifiable improvement claims are dropped ---------------------------

const weak = await applyMemoryUpdate({
  ...baseUpdate(),
  queries: [{
    query_id: "q2a", classification: "SIGNIFICANT_IMPROVEMENT",
    breakthroughs: [{
      iteration: 1, action: "PATCH_CODE", before: 0.80, after: 0.81,
      technique: "renamed a helper and reordered two independent checks",
    }],
  }],
}, ctx);
assert.ok(
  weak.rejected.some((r) => r.reason === "breakthrough_below_threshold"),
  "a 1-point move on a 0.80 baseline is not a breakthrough",
);

// --- unknown query ----------------------------------------------------------

const unknown = await applyMemoryUpdate({
  ...baseUpdate(),
  queries: [{ query_id: "q99", classification: "NOVEL_SUCCESS" }],
}, ctx);
assert.ok(unknown.rejected.some((r) => r.reason === "unknown_query"),
  "memory may only record queries that actually ran");

// --- ground-truth leak rejects the WHOLE update ----------------------------

const leakCases = [
  ["prose that points at ground truth",
    { notes: "Read the ground truth file to confirm the expected answers." }],
  ["a raw_results path",
    { notes: "Compare with /sembench/files/mmqa/raw_results/ground_truth/Q2.csv" }],
  ["a CERT reference",
    { queries: [{ query_id: "q2a", classification: "NOVEL_SUCCESS",
      template: { proven_strategies: ["consult the CERT split when the score looks wrong"] } }] }],
];
for (const [label, patch] of leakCases) {
  const before = (await getNodesByLayer(0, memoryDir)).length;
  const result = await applyMemoryUpdate({ ...baseUpdate(), ...patch }, ctx);
  assert.equal(result.applied, false, `${label}: must not apply`);
  assert.ok(result.rejected.some((r) => r.reason.startsWith("gt_leak")), `${label}: must report a leak`);
  assert.equal((await getNodesByLayer(0, memoryDir)).length, before,
    `${label}: nothing may be written when a leak is detected`);
}

// A prohibition is still not a leak.
const prohibits = await applyMemoryUpdate({
  ...baseUpdate(),
  notes: "Never read ground truth while iterating; the CERT split stays sealed.",
}, ctx);
assert.equal(prohibits.applied, true, "writing down the boundary rule must be allowed");

// --- skill layers -----------------------------------------------------------

await makeSkill("semdb-ocr-name-join");
const withSkill = await applyMemoryUpdate({
  ...baseUpdate(),
  skills: [{
    name: "semdb-ocr-name-join", layer: 3,
    summary: "Bind an image→name join through best_ocr_match over a closed candidate set",
    description: "…",
    evidence: [{ query_id: "q2a", before: 0.41, after: 0.82, iteration: 1, action: "REPLAN" }],
  }],
}, ctx);
assert.deepEqual(withSkill.skills_accepted, ["semdb-ocr-name-join"]);
const l3 = await readNode("L3_semdb_ocr_name_join", memoryDir);
assert.equal(l3.skill_name, "semdb-ocr-name-join");
assert.equal(l3.evidence[0].run_id, ctx.runId, "evidence is stamped with the real run id");
assert.ok(
  (await readAllEdges(memoryDir)).some((e) => e.target === "L3_semdb_ocr_name_join" && e.type === "uses_operator"),
  "the template must point at the technique, or retrieval can never reach it",
);

// a skill directory that does not exist
const missing = await applyMemoryUpdate({
  ...baseUpdate(),
  skills: [{ name: "semdb-not-written", layer: 3, summary: "claims a skill that was never authored", evidence: [{ query_id: "q2a" }] }],
}, ctx);
assert.ok(missing.rejected.some((r) => r.reason === "skill_missing"));

// a skill whose body leaks → quarantined, never referenced
await makeSkill("semdb-leaky-skill", {
  body: "# S\n\n## When to Use\nx\n\n## Technique\nRead the ground truth csv for the expected labels.\n",
});
const leaky = await applyMemoryUpdate({
  ...baseUpdate(),
  skills: [{ name: "semdb-leaky-skill", layer: 3, summary: "a skill whose body points at labels", evidence: [{ query_id: "q2a" }] }],
}, ctx);
assert.ok(leaky.rejected.some((r) => r.reason === "skill_lint"));
assert.equal(await readNode("L3_semdb_leaky_skill", memoryDir), null,
  "a lint failure must not produce a reference node");
assert.ok(existsSync(resolve(skillRoot, "_quarantine", "semdb-leaky-skill")));

// evidence that names no query from this run
await makeSkill("semdb-unevidenced");
const unevidenced = await applyMemoryUpdate({
  ...baseUpdate(),
  skills: [{ name: "semdb-unevidenced", layer: 4, summary: "cites a query that did not run here", evidence: [{ query_id: "q404" }] }],
}, ctx);
assert.ok(unevidenced.rejected.some((r) => r.reason === "skill_evidence"));

// --- prune ------------------------------------------------------------------

const capped = await applyMemoryUpdate(baseUpdate(), { ...ctx, config: { maxNodesPerLayer: 1, maxSkills: 0 } });
assert.ok(capped.pruned.length > 0, "caps must actually evict");
assert.ok((await getNodesByLayer(0, memoryDir)).length <= 1);
assert.equal((await getNodesByLayer(3, memoryDir)).length, 0, "the skill namespace cap is enforced");

console.log("test_memory_apply_update: PASS");
