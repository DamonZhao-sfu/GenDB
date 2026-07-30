/** HAG CRUD, schema enforcement, and edge-walk tests. */

import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import {
  EDGE_TYPES,
  LAYERS,
  addEdge,
  deleteNode,
  getConnectedNodes,
  getNodesByLayer,
  initGraphDirs,
  readAllEdges,
  readIndex,
  readNode,
  updateIndex,
  validateNode,
  writeNode,
} from "../memory/graph.mjs";
import { buildQuerySignature } from "../memory/signature.mjs";

const dir = await mkdtemp(resolve(tmpdir(), "semdb-mem-graph-"));
await initGraphDirs(dir);

for (const layer of LAYERS) {
  assert.ok(existsSync(resolve(dir, "graph", "nodes", layer)), `${layer} dir created`);
}

const boundary = {
  source: "select_validation",
  cert_accessed: false,
  full_ground_truth_accessed: false,
};

const signature = buildQuerySignature({
  query: "q2a",
  sql: "SELECT t.ID, i.uri FROM mmqa.ap_warrior t, mmqa.images i "
    + "WHERE AI.IF(('does the image show the racetrack logo', t.Track, i.uri), connection_id => 'c')",
  nl: "match racetrack logos",
  benchmark: "mmqa",
  corpus: { table: "images", modality: "image" },
  tables: [{ table: "ap_warrior" }, { table: "images" }],
});

const l0 = {
  id: "L0_mmqa_q2a_20260729",
  layer: 0,
  signature_version: 1,
  benchmark: "mmqa",
  signature,
  summary: "mmqa q2a promoted candidate",
  content: { query_id: "q2a", objective: { name: "f1", value: 0.82, direction: "maximize" } },
  data_boundary: boundary,
  tags: ["image", "join"],
};
const l1 = {
  id: "L1_mmqa_joinimage_abc123",
  layer: 1,
  signature_version: 1,
  benchmark: "mmqa",
  signature,
  summary: "image→structured join keyed by an OCR'd name",
  content: { proven_strategies: ["bind the join to a closed value space"], anti_patterns: [] },
  data_boundary: boundary,
};
const l3 = {
  id: "L3_ocr_name_join",
  layer: 3,
  skill_name: "semdb-ocr-name-join",
  skill_path: "skill-root/.claude/skills/semdb-ocr-name-join/",
  benchmark: null,
  signature: null,
  summary: "Bind an image→name join through best_ocr_match over a closed candidate set",
  content: { description: "..." },
  evidence: [{
    run_id: "2026-07-29T10-11-12", benchmark: "mmqa", query_id: "q2a",
    objective: { name: "f1", value: 0.82, direction: "maximize" },
    before: 0.41, after: 0.82, iteration: 2,
  }],
  evidence_count: 1,
  data_boundary: boundary,
};

await writeNode(l0, dir);
await writeNode(l1, dir);
await writeNode(l3, dir);

// --- round trip -------------------------------------------------------------

const back = await readNode("L0_mmqa_q2a_20260729", dir);
assert.equal(back.id, l0.id);
assert.ok(back.created_at && back.updated_at, "timestamps stamped on write");
assert.deepEqual(back.signature, signature, "signature survives the round trip");
assert.equal(await readNode("L0_does_not_exist", dir), null);

assert.equal((await getNodesByLayer(0, dir)).length, 1);
assert.equal((await getNodesByLayer(3, dir)).length, 1);
assert.equal((await getNodesByLayer(5, dir)).length, 0);

// --- schema enforcement -----------------------------------------------------

await assert.rejects(
  () => writeNode({ ...l0, id: "bogus-id" }, dir),
  /Invalid memory node/,
  "id must follow L<n>_...",
);
await assert.rejects(
  () => writeNode({ ...l0, signature: undefined }, dir),
  /Invalid memory node/,
  "L0 without a signature is rejected",
);
await assert.rejects(
  () => writeNode({ ...l3, evidence: [] }, dir),
  /Invalid memory node/,
  "a skill-layer node with no evidence is rejected",
);
await assert.rejects(
  () => writeNode({ ...l3, skill_name: undefined, skill_path: undefined }, dir),
  /Invalid memory node/,
  "a skill-layer node must point at a skill directory",
);
assert.throws(
  () => validateNode({
    ...l0,
    data_boundary: { source: "select_validation", cert_accessed: true, full_ground_truth_accessed: false },
  }),
  /Invalid memory node/,
  "cert_accessed must be false — the data boundary is structural, not advisory",
);

// --- edges ------------------------------------------------------------------

assert.equal(await addEdge({ source: l0.id, target: l1.id, type: "instance_of" }, dir), true);
assert.equal(
  await addEdge({ source: l0.id, target: l1.id, type: "instance_of" }, dir), false,
  "duplicate edges are dropped",
);
await addEdge({ source: l1.id, target: l3.id, type: "uses_operator" }, dir);
assert.equal((await readAllEdges(dir)).length, 2);

await assert.rejects(
  () => addEdge({ source: l0.id, target: l1.id, type: "made_up" }, dir),
  /Unknown memory edge type/,
);
assert.ok(EDGE_TYPES.includes("exemplifies_principle"));

const instances = await getConnectedNodes(l1.id, "instance_of", dir, "incoming");
assert.deepEqual(instances.map((n) => n.id), [l0.id], "incoming instance_of finds the L0");
const techniques = await getConnectedNodes(l1.id, "uses_operator", dir, "outgoing");
assert.deepEqual(techniques.map((n) => n.id), [l3.id], "outgoing uses_operator finds the L3");
assert.equal((await getConnectedNodes(l1.id, "instance_of", dir, "outgoing")).length, 0);
assert.equal((await getConnectedNodes(l1.id, null, dir, "both")).length, 2);

// An edge pointing at a deleted node must not break the walk.
await deleteNode(l3.id, dir);
assert.deepEqual(await getConnectedNodes(l1.id, "uses_operator", dir, "outgoing"), []);

// --- index ------------------------------------------------------------------

const index = await updateIndex(dir);
assert.equal(index.stats.total, 2, "L3 was deleted");
assert.equal(index.stats.by_layer.L0, 1);
assert.equal(index.stats.by_layer.L1, 1);
assert.ok(index.nodes[l0.id].file.endsWith("L0/L0_mmqa_q2a_20260729.json"));
assert.deepEqual(await readIndex(dir), index);

console.log("test_memory_graph: PASS");
