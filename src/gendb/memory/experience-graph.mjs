/**
 * GenDB Experience Graph (XG) — first-class search-trajectory store.
 *
 * Implements the data foundation from "Experience Graphs: The Data Foundation
 * for Self-Improving Agents" (Liao, Abadi et al., arXiv:2606.29823), adapted to
 * GenDB's file-backed memory conventions.
 *
 * Where the HAG (graph.mjs) stores DISTILLED knowledge (L0–L5 skills), the
 * Experience Graph stores the RAW search that produced it: every optimization
 * attempt the query-optimizer made — including the dead-ends and regressions
 * that the current pipeline throws away.
 *
 * Four-table relational schema (paper §3):
 *   tasks     — the problem: SQL spec + task embedding for cross-session reuse
 *   sessions  — one search run: who searched, with which algorithm/model
 *   nodes     — every attempt: parent link, artifact ref, reward, eval evidence,
 *               algorithm-specific stats (visit_count, cumulative_reward, ...)
 *   prompts   — the exact messages the LLM saw/produced (by reference)
 *
 * Design fidelity to the paper:
 *   - "Search over experience graphs is a database access pattern": all reads
 *     go through query helpers (bestNode, siblings, ancestors, findSimilarTasks).
 *   - "Large artifacts live in object storage, linked by reference": node.artifact_ref
 *     and prompt.messages_ref are PATHS into output/, never inlined blobs.
 *   - "Eventual consistency on stats, durability on node insertion": nodes are
 *     written durably one file each; the mutable visit/reward counters are
 *     recomputed by backpropagate() and may lag.
 *
 * Storage layout (under <memoryDir>/experience/):
 *   tasks.json                        — { task_id: Task }
 *   sessions.json                     — { session_id: Session }
 *   nodes/<session_id>/<node_id>.json — one Node per file (durable insert)
 */

import { readFile, writeFile, mkdir, readdir } from "fs/promises";
import { resolve } from "path";
import { existsSync } from "fs";

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const xgRoot = (memoryDir) => resolve(memoryDir, "experience");
const tasksPath = (memoryDir) => resolve(xgRoot(memoryDir), "tasks.json");
const sessionsPath = (memoryDir) => resolve(xgRoot(memoryDir), "sessions.json");
const nodesDir = (memoryDir) => resolve(xgRoot(memoryDir), "nodes");

async function readJson(path, fallback) {
  try {
    return JSON.parse(await readFile(path, "utf-8"));
  } catch {
    return fallback;
  }
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

export async function initExperienceGraph(memoryDir) {
  await mkdir(nodesDir(memoryDir), { recursive: true });
  if (!existsSync(tasksPath(memoryDir))) await writeFile(tasksPath(memoryDir), "{}");
  if (!existsSync(sessionsPath(memoryDir))) await writeFile(sessionsPath(memoryDir), "{}");
}

// ---------------------------------------------------------------------------
// Task-description embedding (pluggable)
// ---------------------------------------------------------------------------

/**
 * The paper uses task-description embeddings for cross-session vector similarity.
 * GenDB has no embedding endpoint wired in, so we ship a deterministic, offline
 * bag-of-features embedding: a sparse token/structural signature that supports
 * cosine similarity without any network call. Swap `embedTask` for a real
 * provider (e.g. Voyage/OpenAI) to get semantic recall — the schema is identical.
 */
export function embedTask(spec) {
  const text = [
    spec.benchmark || "",
    `sf${spec.scale_factor ?? ""}`,
    spec.sql || "",
  ].join(" ").toLowerCase();

  const vec = {};
  // structural tokens (SQL keywords) — robust to literal changes
  const tokens = text.match(/[a-z_][a-z0-9_]+/g) || [];
  const KEYWORDS = new Set([
    "select", "from", "where", "group", "order", "having", "join", "inner",
    "left", "right", "sum", "avg", "count", "min", "max", "distinct", "like",
    "between", "case", "when", "in", "exists", "limit", "union", "with",
  ]);
  for (const t of tokens) {
    if (KEYWORDS.has(t) || t.startsWith("sf") || t === spec.benchmark) {
      vec[t] = (vec[t] || 0) + 1;
    }
  }
  return vec;
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (const k in a) { na += a[k] * a[k]; if (b[k]) dot += a[k] * b[k]; }
  for (const k in b) nb += b[k] * b[k];
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ---------------------------------------------------------------------------
// Table: tasks
// ---------------------------------------------------------------------------

/** Create or return a Task for a given SQL spec (idempotent on task_id). */
export async function upsertTask(memoryDir, { task_id, benchmark, scale_factor, sql, query_id }, now) {
  const tasks = await readJson(tasksPath(memoryDir), {});
  const id = task_id || `${benchmark}__${query_id || "q"}__sf${scale_factor}`;
  const spec = { benchmark, scale_factor, sql, query_id };
  tasks[id] = {
    task_id: id,
    spec,
    spec_embedding: embedTask(spec),
    created_at: tasks[id]?.created_at || now,
    updated_at: now,
  };
  await writeFile(tasksPath(memoryDir), JSON.stringify(tasks, null, 2));
  return tasks[id];
}

/**
 * Cross-session retrieval (paper §4): given a new task spec, find prior tasks
 * with similar specifications so their best results can be reused. This is the
 * richer, benchmark-agnostic complement to graph.mjs's SQL structural match.
 */
export async function findSimilarTasks(memoryDir, spec, { topK = 3, minSim = 0.5 } = {}) {
  const tasks = await readJson(tasksPath(memoryDir), {});
  const q = embedTask(spec);
  return Object.values(tasks)
    .map((t) => ({ task: t, similarity: cosine(q, t.spec_embedding || {}) }))
    .filter((r) => r.similarity >= minSim)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, topK);
}

// ---------------------------------------------------------------------------
// Table: sessions
// ---------------------------------------------------------------------------

export async function createSession(memoryDir, { session_id, task_id, algorithm, agent, model, config }, now) {
  const sessions = await readJson(sessionsPath(memoryDir), {});
  const id = session_id || `${task_id}__${now}`;
  sessions[id] = {
    session_id: id,
    task_id,
    algorithm: algorithm || "llm-hill-climb", // GenDB's optimizer is iterative hill-climb w/ repair
    agent: agent || "query-optimizer",
    model: model || null,
    config: config || {},
    started_at: now,
  };
  await writeFile(sessionsPath(memoryDir), JSON.stringify(sessions, null, 2));
  await mkdir(resolve(nodesDir(memoryDir), id), { recursive: true });
  return sessions[id];
}

// ---------------------------------------------------------------------------
// Table: nodes  (durable insert — one file each)
// ---------------------------------------------------------------------------

/**
 * Insert one attempt into the experience graph.
 *
 * @param node.parent_id     link to the attempt this one was derived from (null = root)
 * @param node.artifact_ref  PATH to the generated C++ (not inlined)
 * @param node.tool_output   PATH/summary of compile+run logs (not inlined)
 * @param node.reward        fitness — higher is better (we use speedup vs. baseline)
 * @param node.eval_evidence structured proof: { timing_ms, validation, operation_timings }
 * @param node.algo_meta     algorithm stats: { iteration, categories, regression, ... }
 * @param node.prompt        optional { messages_ref } — prompt history by reference
 */
export async function insertNode(memoryDir, node, now) {
  const dir = resolve(nodesDir(memoryDir), node.session_id);
  await mkdir(dir, { recursive: true });
  const record = {
    node_id: node.node_id,
    session_id: node.session_id,
    task_id: node.task_id,
    parent_id: node.parent_id ?? null,
    artifact_ref: node.artifact_ref ?? null,
    tool_output: node.tool_output ?? null,
    reward: node.reward ?? null,
    eval_evidence: node.eval_evidence ?? {},
    // algorithm-specific mutable stats (paper: UCB/visit_count for MCTS,
    // generation/island for evolutionary search). Filled by backpropagate().
    algo_meta: { visit_count: 0, cumulative_reward: 0, ...(node.algo_meta || {}) },
    prompt: node.prompt ?? null,
    created_at: now,
  };
  await writeFile(resolve(dir, `${node.node_id}.json`), JSON.stringify(record, null, 2));
  return record;
}

export async function readNode(memoryDir, session_id, node_id) {
  return readJson(resolve(nodesDir(memoryDir), session_id, `${node_id}.json`), null);
}

async function writeNodeRecord(memoryDir, record) {
  const dir = resolve(nodesDir(memoryDir), record.session_id);
  await writeFile(resolve(dir, `${record.node_id}.json`), JSON.stringify(record, null, 2));
}

export async function getSessionNodes(memoryDir, session_id) {
  const dir = resolve(nodesDir(memoryDir), session_id);
  if (!existsSync(dir)) return [];
  const files = await readdir(dir);
  const nodes = [];
  for (const f of files) {
    if (f.endsWith(".json")) {
      const n = await readJson(resolve(dir, f), null);
      if (n) nodes.push(n);
    }
  }
  return nodes;
}

// ---------------------------------------------------------------------------
// Search queries (the "database access pattern" the paper argues for)
// ---------------------------------------------------------------------------

/** Ancestor chain from a node up to its root (inclusive of the node). */
export async function ancestors(memoryDir, session_id, node_id) {
  const chain = [];
  let cur = await readNode(memoryDir, session_id, node_id);
  const seen = new Set();
  while (cur && !seen.has(cur.node_id)) {
    seen.add(cur.node_id);
    chain.push(cur);
    cur = cur.parent_id ? await readNode(memoryDir, session_id, cur.parent_id) : null;
  }
  return chain;
}

/** Sibling attempts (same parent) — the paper's "sibling comparisons". */
export async function siblings(memoryDir, session_id, node_id) {
  const node = await readNode(memoryDir, session_id, node_id);
  if (!node) return [];
  const all = await getSessionNodes(memoryDir, session_id);
  return all.filter((n) => n.parent_id === node.parent_id && n.node_id !== node_id);
}

/** Best attempt in a session by reward (ties broken by lowest timing). */
export async function bestNode(memoryDir, session_id) {
  const all = await getSessionNodes(memoryDir, session_id);
  return all
    .filter((n) => n.eval_evidence?.validation !== "fail")
    .sort((a, b) => {
      const dr = (b.reward ?? -Infinity) - (a.reward ?? -Infinity);
      if (dr !== 0) return dr;
      return (a.eval_evidence?.timing_ms ?? Infinity) - (b.eval_evidence?.timing_ms ?? Infinity);
    })[0] || null;
}

/**
 * Backpropagate reward up the ancestor chain (paper §3: after evaluating a
 * node, MCTS walks the ancestor chain and updates each ancestor's visit count
 * and cumulative reward). Tolerates eventual consistency — safe to re-run.
 */
export async function backpropagate(memoryDir, session_id, leaf_node_id) {
  const chain = await ancestors(memoryDir, session_id, leaf_node_id);
  const reward = chain[0]?.reward ?? 0;
  for (const anc of chain) {
    anc.algo_meta = anc.algo_meta || {};
    anc.algo_meta.visit_count = (anc.algo_meta.visit_count || 0) + 1;
    anc.algo_meta.cumulative_reward = (anc.algo_meta.cumulative_reward || 0) + reward;
    anc.algo_meta.mean_reward =
      anc.algo_meta.cumulative_reward / anc.algo_meta.visit_count;
    await writeNodeRecord(memoryDir, anc);
  }
}

// ---------------------------------------------------------------------------
// Ingest: linear optimization_history.json -> branching experience graph
// ---------------------------------------------------------------------------

/**
 * Convert GenDB's existing per-query optimization_history.json into an
 * experience graph. Models the optimizer's real control flow: each iteration is
 * derived from the last ACCEPTED attempt; a non-improving iteration becomes a
 * dead-end sibling (a branch that was tried and abandoned) rather than being
 * silently dropped — which is exactly the search evidence the paper preserves.
 *
 * @returns { task, session, nodeCount, best }
 */
export async function ingestOptimizationHistory(
  memoryDir,
  history,
  { benchmark, scale_factor, sql, query_id, model, artifactDir },
  now
) {
  await initExperienceGraph(memoryDir);
  const stamp = now || new Date().toISOString();

  const task = await upsertTask(memoryDir, { benchmark, scale_factor, sql, query_id }, stamp);
  const session = await createSession(
    memoryDir,
    {
      session_id: `${task.task_id}__${stamp}`,
      task_id: task.task_id,
      algorithm: "llm-hill-climb",
      agent: "query-optimizer",
      model,
      config: { source: "optimization_history.json" },
    },
    stamp
  );

  const iters = history.iterations || [];
  const baseline = iters[0]?.timing_ms || null;

  let lastAccepted = null; // parent for the next attempt
  let count = 0;
  for (const it of iters) {
    const node_id = `${session.session_id}__iter${it.iteration}`;
    const timing = it.timing_ms ?? null;
    // reward = speedup vs. baseline; failed validation is worst-possible
    const reward =
      it.validation === "fail" || timing == null
        ? -1
        : baseline
        ? baseline / timing
        : 0;
    const improved = it.improved === true && it.validation !== "fail";

    await insertNode(
      memoryDir,
      {
        node_id,
        session_id: session.session_id,
        task_id: task.task_id,
        parent_id: lastAccepted,
        artifact_ref: artifactDir ? `${artifactDir}/iter_${it.iteration}` : null,
        tool_output: null,
        reward,
        eval_evidence: {
          timing_ms: timing,
          cold_timing_ms: it.cold_timing_ms ?? null,
          hot_timing_ms: it.hot_timing_ms ?? null,
          validation: it.validation ?? null,
          operation_timings: it.operation_timings ?? null,
          categories: it.categories ?? [],
        },
        algo_meta: {
          iteration: it.iteration,
          improved,
          regression: !improved && it.iteration > 0,
          strategy: it.strategy ? it.strategy.slice(0, 500) : null,
        },
      },
      stamp
    );
    count++;
    // Accepted iterations advance the frontier; regressions branch off and die.
    if (improved || it.iteration === 0) lastAccepted = node_id;
  }

  // Backpropagate from the best leaf so ancestor stats reflect the win.
  const best = await bestNode(memoryDir, session.session_id);
  if (best) await backpropagate(memoryDir, session.session_id, best.node_id);

  return { task, session, nodeCount: count, best };
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

export async function getExperienceGraphSummary(memoryDir) {
  if (!existsSync(xgRoot(memoryDir))) return "ExperienceGraph: disabled";
  const tasks = await readJson(tasksPath(memoryDir), {});
  const sessions = await readJson(sessionsPath(memoryDir), {});
  let nodes = 0;
  for (const sid of Object.keys(sessions)) {
    nodes += (await getSessionNodes(memoryDir, sid)).length;
  }
  return `ExperienceGraph: ${Object.keys(tasks).length} tasks, ${Object.keys(sessions).length} sessions, ${nodes} nodes`;
}
