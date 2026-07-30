/**
 * Hierarchical Abstraction Graph — node/edge CRUD.
 *
 * Ported from src/gendb/memory/graph.mjs, keeping the same layer names, edge
 * types and exported surface so the two systems stay diffable. Three deliberate
 * differences:
 *
 *  1. Every write is validated against contracts/memory-node.schema.json. GenDB
 *     lets its Memory Manager write node JSON directly and repairs the damage
 *     afterwards (patchL1Signatures); here nothing enters the graph unvalidated.
 *  2. writeNode does NOT reindex. GenDB reindexes on every write, which is
 *     O(n^2) when a curation pass writes dozens of nodes; callers invoke
 *     updateIndex() once when they are done.
 *  3. GenDB's loadMemoryNode references an undefined `n` for layers 2-5 and
 *     throws ReferenceError on any skill-layer load. Rendering lives in
 *     render.mjs here and reads `node`.
 */

import { mkdir, readFile, readdir, unlink, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import Ajv2020 from "ajv/dist/2020.js";

const here = dirname(fileURLToPath(import.meta.url));

export const LAYERS = ["L0", "L1", "L2", "L3", "L4", "L5"];
export const LAYER_NAMES = [
  "Query Instances",
  "Query Templates",
  "Sub-Structure Patterns",
  "Operator Techniques",
  "Optimization Strategies",
  "Performance Principles",
];
/** L0/L1 live in the graph; L2-L5 are skills, the graph holds only a reference. */
export const HAG_LAYERS = [0, 1];
export const SKILL_LAYERS = [2, 3, 4, 5];

export const EDGE_TYPES = [
  "instance_of",          // L0 → L1
  "exhibits_pattern",     // L1 → L2
  "uses_operator",        // L1 → L3, L2 → L3
  "implements_strategy",  // L3 → L4
  "exemplifies_principle", // L4 → L5
];

const ajv = new Ajv2020({ allErrors: true, strict: true });
const nodeSchema = JSON.parse(
  await readFile(resolve(here, "..", "contracts", "memory-node.schema.json"), "utf8"),
);
const validateNodeSchema = ajv.compile(nodeSchema);

/** @throws if the node violates the schema. */
export function validateNode(node) {
  if (!validateNodeSchema(node)) {
    const detail = (validateNodeSchema.errors || [])
      .map((e) => `${e.instancePath || "/"} ${e.message}`)
      .join("; ");
    throw new Error(`Invalid memory node ${node?.id ?? "<no id>"}: ${detail}`);
  }
  return node;
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

export async function initGraphDirs(memoryDir) {
  const nodesDir = resolve(memoryDir, "graph", "nodes");
  for (const layer of LAYERS) await mkdir(resolve(nodesDir, layer), { recursive: true });

  const edgesPath = resolve(memoryDir, "graph", "edges.json");
  if (!existsSync(edgesPath)) await writeFile(edgesPath, JSON.stringify([], null, 2));

  const indexPath = resolve(memoryDir, "graph", "index.json");
  if (!existsSync(indexPath)) {
    await writeFile(
      indexPath,
      JSON.stringify({ nodes: {}, stats: { total: 0, by_layer: {} } }, null, 2),
    );
  }
}

// ---------------------------------------------------------------------------
// Nodes
// ---------------------------------------------------------------------------

function layerDirOf(id) {
  return id.split("_")[0];
}

export async function readNode(id, memoryDir) {
  const nodePath = resolve(memoryDir, "graph", "nodes", layerDirOf(id), `${id}.json`);
  try {
    return JSON.parse(await readFile(nodePath, "utf8"));
  } catch {
    return null;
  }
}

/** Create or replace a node. Does not touch the index — call updateIndex() when done. */
export async function writeNode(node, memoryDir) {
  const now = new Date().toISOString();
  const stamped = { ...node, created_at: node.created_at || now, updated_at: now };
  validateNode(stamped);
  const dirPath = resolve(memoryDir, "graph", "nodes", `L${stamped.layer}`);
  await mkdir(dirPath, { recursive: true });
  await writeFile(resolve(dirPath, `${stamped.id}.json`), JSON.stringify(stamped, null, 2));
  return stamped;
}

export async function deleteNode(id, memoryDir) {
  const nodePath = resolve(memoryDir, "graph", "nodes", layerDirOf(id), `${id}.json`);
  try {
    await unlink(nodePath);
  } catch {
    // already gone
  }
}

export async function getNodesByLayer(layer, memoryDir) {
  const dirPath = resolve(memoryDir, "graph", "nodes", `L${layer}`);
  if (!existsSync(dirPath)) return [];
  const files = await readdir(dirPath);
  const nodes = [];
  for (const file of files) {
    if (!file.endsWith(".json")) continue;
    try {
      nodes.push(JSON.parse(await readFile(resolve(dirPath, file), "utf8")));
    } catch {
      // a half-written node must not take down retrieval
    }
  }
  return nodes;
}

export async function getAllNodes(memoryDir) {
  const all = [];
  for (let layer = 0; layer <= 5; layer += 1) {
    all.push(...(await getNodesByLayer(layer, memoryDir)));
  }
  return all;
}

// ---------------------------------------------------------------------------
// Edges
// ---------------------------------------------------------------------------

const edgeSource = (e) => e.source ?? e.from;
const edgeTarget = (e) => e.target ?? e.to;

export async function readAllEdges(memoryDir) {
  try {
    return JSON.parse(await readFile(resolve(memoryDir, "graph", "edges.json"), "utf8"));
  } catch {
    return [];
  }
}

export async function writeEdges(edges, memoryDir) {
  await writeFile(
    resolve(memoryDir, "graph", "edges.json"),
    JSON.stringify(edges, null, 2),
  );
}

/** Add one edge, deduplicating on (source, target, type). */
export async function addEdge(edge, memoryDir) {
  if (!EDGE_TYPES.includes(edge.type)) {
    throw new Error(`Unknown memory edge type: ${edge.type}`);
  }
  const edges = await readAllEdges(memoryDir);
  const src = edgeSource(edge);
  const tgt = edgeTarget(edge);
  const exists = edges.some(
    (e) => edgeSource(e) === src && edgeTarget(e) === tgt && e.type === edge.type,
  );
  if (exists) return false;
  edges.push({
    source: src,
    target: tgt,
    type: edge.type,
    created_at: new Date().toISOString(),
  });
  await writeEdges(edges, memoryDir);
  return true;
}

export async function removeEdgesForNode(nodeId, memoryDir) {
  const edges = await readAllEdges(memoryDir);
  await writeEdges(
    edges.filter((e) => edgeSource(e) !== nodeId && edgeTarget(e) !== nodeId),
    memoryDir,
  );
}

/**
 * Nodes reachable from `nodeId` over edges of `edgeType`.
 * @param {"outgoing"|"incoming"|"both"} direction
 */
export async function getConnectedNodes(nodeId, edgeType, memoryDir, direction = "outgoing") {
  const edges = await readAllEdges(memoryDir);
  const ids = edges
    .filter((e) => {
      if (edgeType && e.type !== edgeType) return false;
      if (direction === "outgoing") return edgeSource(e) === nodeId;
      if (direction === "incoming") return edgeTarget(e) === nodeId;
      return edgeSource(e) === nodeId || edgeTarget(e) === nodeId;
    })
    .map((e) => (edgeSource(e) === nodeId ? edgeTarget(e) : edgeSource(e)));

  const nodes = [];
  for (const id of [...new Set(ids)]) {
    const node = await readNode(id, memoryDir);
    if (node) nodes.push(node);
  }
  return nodes;
}

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

export async function readIndex(memoryDir) {
  try {
    return JSON.parse(await readFile(resolve(memoryDir, "graph", "index.json"), "utf8"));
  } catch {
    return { nodes: {}, stats: { total: 0, by_layer: {} } };
  }
}

/** Rebuild the index from disk. Call once after a batch of writes. */
export async function updateIndex(memoryDir) {
  const index = { nodes: {}, stats: { total: 0, by_layer: {} } };
  for (const layer of LAYERS) {
    const dirPath = resolve(memoryDir, "graph", "nodes", layer);
    if (!existsSync(dirPath)) continue;
    const files = (await readdir(dirPath)).filter((f) => f.endsWith(".json"));
    index.stats.by_layer[layer] = files.length;
    index.stats.total += files.length;
    for (const file of files) {
      index.nodes[file.replace(/\.json$/, "")] = { layer, file: `nodes/${layer}/${file}` };
    }
  }
  await writeFile(
    resolve(memoryDir, "graph", "index.json"),
    JSON.stringify(index, null, 2),
  );
  return index;
}
