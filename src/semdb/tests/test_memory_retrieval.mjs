/** Tiering, cross-benchmark policy, warm-start resolution, and injection caps. */

import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import { addEdge, updateIndex, writeNode } from "../memory/graph.mjs";
import { initMemory, classifyQuery, getMemorySummary } from "../memory/index.mjs";
import { buildQuerySignature } from "../memory/signature.mjs";
import { skillRootFor, skillsDirFor } from "../memory/skills.mjs";
import { capTokens, renderPreInjection } from "../memory/render.mjs";

const memoryDir = await mkdtemp(resolve(tmpdir(), "semdb-mem-retr-"));
const init = await initMemory(memoryDir, {});
assert.equal(init.ready, true);
assert.ok(init.skillRoot.endsWith("skill-root"));

const boundary = {
  source: "select_validation", cert_accessed: false, full_ground_truth_accessed: false,
};

const RACETRACK_SQL = `SELECT t.ID, i.uri FROM mmqa.ap_warrior t, mmqa.images i
WHERE AI.IF(('Determine if the image shows the logo of the racetrack. Racetrack: ', t.Track, i.uri),
  connection_id => 'c')`;
const VESSEL_SQL = `SELECT t.ID, i.uri FROM mmqa.vessels t, mmqa.images i
WHERE AI.IF(('Determine whether the picture depicts a wooden sailing vessel. Registry: ', t.Name, i.uri),
  connection_id => 'c')`;

const planFor = (sql, tables, modality = "image", benchmark = "mmqa") => ({
  query: "qx", sql, nl: null, benchmark,
  corpus: { table: "images", modality },
  tables: tables.map((table) => ({ table })),
});

const LIGHTHOUSE_SQL = `SELECT t.ID, i.uri FROM mmqa.beacons t, mmqa.images i
WHERE AI.IF(('Determine whether the photograph shows a coastal lighthouse beacon. Beacon: ', t.Name, i.uri),
  connection_id => 'c')`;

const racetrack = planFor(RACETRACK_SQL, ["ap_warrior", "images"]);
const vessel = planFor(VESSEL_SQL, ["vessels", "images"]);
const lighthouse = planFor(LIGHTHOUSE_SQL, ["beacons", "images"]);

// --- empty memory -----------------------------------------------------------

let result = await classifyQuery(racetrack, { benchmark: "mmqa", agentProvider: "claude" }, memoryDir, {});
assert.equal(result.tier, "novel");
assert.equal(result.score, 0);
assert.equal(result.blocks.planner, "", "a novel query gets no pushed block");
assert.equal(result.catalog, "", "no learned skills yet → no catalog");
assert.equal(result.warmStart, null);
assert.match(await getMemorySummary(memoryDir), /memory: 0 nodes/);

// --- populate ---------------------------------------------------------------

const runDir = await mkdtemp(resolve(tmpdir(), "semdb-run-"));
const planPath = resolve(runDir, "plan.json");
const solverPath = resolve(runDir, "solve_q2a.py");
const helpersPath = resolve(runDir, "_semantic_helpers_q2a.py");
for (const p of [planPath, solverPath, helpersPath]) await writeFile(p, "{}");

const sig = buildQuerySignature(racetrack);
await writeNode({
  id: "L1_mmqa_joinimage_aaa", layer: 1, signature_version: 1, benchmark: "mmqa",
  signature: sig,
  summary: "image→structured join keyed by an OCR'd name",
  content: {
    proven_strategies: ["bind the join through best_ocr_match over the closed track name set"],
    anti_patterns: ["free-form VLM captioning then string equality"],
    plan_skeleton: { sampling_unit: "pair", helper_names: ["logo_name"], primitives: ["best_ocr_match"] },
  },
  data_boundary: boundary,
}, memoryDir);

await writeNode({
  id: "L0_mmqa_q2a_1", layer: 0, signature_version: 1, benchmark: "mmqa",
  signature: sig,
  summary: "mmqa q2a promoted candidate",
  content: {
    query_id: "q2a", iterations: 2, replans: 0,
    objective: { name: "f1", value: 0.82, direction: "maximize" },
    promoted: { plan_path: planPath, helpers_path: helpersPath, solver_path: solverPath, candidate_id: "q2a-iter-2" },
  },
  data_boundary: boundary,
}, memoryDir);

// A worse instance of the same template — must not win.
await writeNode({
  id: "L0_mmqa_q2a_0", layer: 0, signature_version: 1, benchmark: "mmqa",
  signature: sig,
  summary: "an earlier, worse run",
  content: {
    query_id: "q2a", iterations: 1,
    objective: { name: "f1", value: 0.44, direction: "maximize" },
    promoted: { plan_path: planPath, helpers_path: helpersPath, solver_path: solverPath, candidate_id: "q2a-iter-0" },
  },
  data_boundary: boundary,
}, memoryDir);

// Same shape, different benchmark — must never be matched.
await writeNode({
  id: "L1_ecomm_joinimage_bbb", layer: 1, signature_version: 1, benchmark: "ecomm",
  signature: sig,
  summary: "identical signature, wrong benchmark",
  content: { proven_strategies: ["should never be injected into an mmqa query"] },
  data_boundary: boundary,
}, memoryDir);

await addEdge({ source: "L0_mmqa_q2a_1", target: "L1_mmqa_joinimage_aaa", type: "instance_of" }, memoryDir);
await addEdge({ source: "L0_mmqa_q2a_0", target: "L1_mmqa_joinimage_aaa", type: "instance_of" }, memoryDir);
await updateIndex(memoryDir);

// --- exact tier -------------------------------------------------------------

result = await classifyQuery(racetrack, { benchmark: "mmqa", agentProvider: "claude" }, memoryDir, {});
assert.equal(result.tier, "exact");
assert.equal(result.score, 1);
assert.equal(result.matchedL1, "L1_mmqa_joinimage_aaa");
assert.equal(result.matchedL0, "L0_mmqa_q2a_1", "best instance wins, not the most recent");
assert.equal(result.warmStart.candidateId, "q2a-iter-2");
assert.equal(result.warmStart.planPath, planPath);
assert.match(result.blocks.planner, /exact template/);
assert.match(result.blocks.planner, /best_ocr_match/);
assert.match(result.blocks.planner, /Anti-patterns/);
assert.match(result.blocks.planner, /advisory/i, "pushed context must be marked advisory");
assert.match(result.blocks.generator, /copied into your working/);
assert.ok(
  !result.blocks.planner.includes("should never be injected"),
  "the same-signature ecomm template must not leak into an mmqa query",
);

// --- a failed reference is never imitated -----------------------------------
// Regression: a backfilled run that scored f1=0 was warm-started and its plan shape
// was described as one that "succeeded". Seeding a known-failed solver is worse
// than a cold start, and the claim was simply false.
await writeNode({
  id: "L1_mmqa_failedjoin_zzz", layer: 1, signature_version: 1, benchmark: "mmqa",
  signature: buildQuerySignature(lighthouse),
  summary: "a template whose only instance failed",
  content: {
    proven_strategies: ["the strategy the failed run happened to use"],
    plan_skeleton: { sampling_unit: "pair", helper_names: ["h"], primitives: ["classify"] },
  },
  data_boundary: boundary,
}, memoryDir);
await writeNode({
  id: "L0_mmqa_failed_1", layer: 0, signature_version: 1, benchmark: "mmqa",
  signature: buildQuerySignature(lighthouse),
  summary: "a run that scored nothing",
  content: {
    query_id: "qfail", iterations: 0,
    objective: { name: "f1", value: 0, direction: "maximize" },
    promoted: { plan_path: planPath, helpers_path: helpersPath, solver_path: solverPath, candidate_id: "qfail-iter-0" },
  },
  data_boundary: boundary,
}, memoryDir);
await addEdge({ source: "L0_mmqa_failed_1", target: "L1_mmqa_failedjoin_zzz", type: "instance_of" }, memoryDir);
await updateIndex(memoryDir);

const failed = await classifyQuery(lighthouse, { benchmark: "mmqa", agentProvider: "claude" }, memoryDir, {});
assert.equal(failed.tier, "exact", "the template matches by hash");
assert.equal(failed.referenceIsGood, false, "f1=0 is not a usable reference");
assert.equal(failed.warmStart, null, "a failed candidate must never seed iter_0");
assert.match(failed.blocks.planner, /FAILED/, "the block must say the reference failed");
assert.ok(
  !failed.blocks.planner.includes("Plan shape that succeeded"),
  "a failed run's plan shape must not be presented as a success",
);
assert.ok(
  !failed.blocks.generator.includes("copied into your working"),
  "nothing was seeded, so the generator must not be told otherwise",
);

// --- warm start degrades when the referenced files are gone ------------------

await rm(planPath);
const stale = await classifyQuery(racetrack, { benchmark: "mmqa", agentProvider: "claude" }, memoryDir, {});
assert.equal(stale.tier, "exact", "the template still matches");
assert.equal(stale.warmStart, null, "a dangling reference must fall back to a cold start");
await writeFile(planPath, "{}");

// warmStart can also be disabled outright
const noWarm = await classifyQuery(racetrack, { benchmark: "mmqa", agentProvider: "claude" }, memoryDir, { warmStart: false });
assert.equal(noWarm.warmStart, null);

// --- structural vs novel ----------------------------------------------------

const vesselResult = await classifyQuery(vessel, { benchmark: "mmqa", agentProvider: "claude" }, memoryDir, {});
assert.ok(vesselResult.score > 0 && vesselResult.score < 0.98,
  `same shape, unrelated predicate must not be exact (got ${vesselResult.score})`);
assert.equal(vesselResult.warmStart, null, "only an exact match may warm start");

// threshold behavior is a pure function of the configured bounds
const above = await classifyQuery(vessel, { benchmark: "mmqa", agentProvider: "claude" }, memoryDir,
  { structuralMatchMinScore: vesselResult.score - 0.01 });
assert.equal(above.tier, "structural");
const below = await classifyQuery(vessel, { benchmark: "mmqa", agentProvider: "claude" }, memoryDir,
  { structuralMatchMinScore: vesselResult.score + 0.01 });
assert.equal(below.tier, "novel");
assert.equal(below.blocks.planner, "");

// --- modality gate through the full path ------------------------------------

const textQuery = planFor(RACETRACK_SQL, ["ap_warrior", "images"], "text");
const textResult = await classifyQuery(textQuery, { benchmark: "mmqa", agentProvider: "claude" }, memoryDir, {});
assert.equal(textResult.tier, "novel", "an image template must never match a text corpus");

// --- other benchmark sees nothing -------------------------------------------

const ecommQuery = planFor(RACETRACK_SQL, ["ap_warrior", "images"], "image", "ecomm");
const ecommResult = await classifyQuery(ecommQuery, { benchmark: "cars", agentProvider: "claude" }, memoryDir, {});
assert.equal(ecommResult.tier, "novel");

// --- catalog + codex inlining ----------------------------------------------

const skillDir = resolve(skillsDirFor(skillRootFor(memoryDir)), "semdb-ocr-name-join");
await mkdir(skillDir, { recursive: true });
await writeFile(resolve(skillDir, "SKILL.md"),
  "---\nname: semdb-ocr-name-join\ndescription: Use when joining an image corpus to a name column.\n---\n\n# OCR name join\n\nBind through best_ocr_match.\n");
await writeNode({
  id: "L3_ocr_name_join", layer: 3,
  skill_name: "semdb-ocr-name-join", skill_path: skillDir,
  benchmark: null, signature: null,
  summary: "Bind an image→name join through best_ocr_match",
  content: { description: "..." },
  evidence: [{
    run_id: "r1", benchmark: "mmqa", query_id: "q2a",
    objective: { name: "f1", value: 0.82, direction: "maximize" },
  }],
  data_boundary: boundary,
}, memoryDir);
await addEdge({ source: "L1_mmqa_joinimage_aaa", target: "L3_ocr_name_join", type: "uses_operator" }, memoryDir);

const withSkills = await classifyQuery(racetrack, { benchmark: "mmqa", agentProvider: "claude" }, memoryDir, {});
assert.match(withSkills.catalog, /semdb-ocr-name-join/);
assert.equal(withSkills.skillsAvailable, 1);
assert.deepEqual(withSkills.relevantSkillNames, ["semdb-ocr-name-join"]);
assert.equal(withSkills.inlineSkills, "", "claude discovers skills natively — nothing is inlined");

const codex = await classifyQuery(racetrack, { benchmark: "mmqa", agentProvider: "codex" }, memoryDir, {});
assert.deepEqual(codex.relevantSkillNames, ["semdb-ocr-name-join"]);
assert.match(codex.inlineSkills, /Bind through best_ocr_match/,
  "codex has no Skill tool, so linked skills are inlined instead");
assert.match(codex.inlineSkills, /advisory/i);

// --- token caps -------------------------------------------------------------

assert.equal(capTokens("short", 100), "short");
const long = "line of text\n".repeat(4000);
const capped = capTokens(long, 50);
assert.ok(Math.ceil(capped.length / 4) <= 60);
assert.match(capped, /truncated/);

const bigL1 = {
  id: "L1_x", content: {
    proven_strategies: Array.from({ length: 200 }, (_, i) => `strategy number ${i} with padding text`),
  },
};
const bounded = renderPreInjection(
  { tier: "structural", score: 0.7, l1: bigL1, l0: null, role: "planner", warmStart: null }, 100,
);
assert.ok(Math.ceil(bounded.length / 4) <= 110, "pre-injection respects its cap");

console.log("test_memory_retrieval: PASS");
