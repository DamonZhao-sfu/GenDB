/**
 * Experience Graph — read side. Implements the paper's three database access
 * patterns as first-class operations:
 *
 *   GRAPH TRAVERSE — recursive CTE over nodes.parent_node_id
 *   RELATION JOIN  — SQL joins across tasks / sessions / nodes / prompts
 *   VECTOR SEARCH  — cosine over embedding BLOBs (embed.mjs)
 *
 * All functions accept a store handle from openStore(); on a disabled handle
 * they return empty results rather than throwing.
 */

import { embed, cosine, fromBlob, toBlob } from "./embed.mjs";

function rows(store, sql, params = []) {
  if (!store?.enabled) return [];
  return store.db.prepare(sql).all(...params);
}
function row(store, sql, params = []) {
  if (!store?.enabled) return null;
  return store.db.prepare(sql).get(...params) ?? null;
}

// ---------------------------------------------------------------------------
// GRAPH TRAVERSE (recursive CTE)
// ---------------------------------------------------------------------------

export function getNode(store, nodeId) {
  return row(store, `SELECT * FROM nodes WHERE node_id=?`, [nodeId]);
}

/** Ancestor path from node up to the root (node first, root last). */
export function getAncestors(store, nodeId) {
  return rows(
    store,
    `WITH RECURSIVE up(node_id, parent_node_id, depth) AS (
        SELECT node_id, parent_node_id, 0 FROM nodes WHERE node_id=?
        UNION ALL
        SELECT n.node_id, n.parent_node_id, up.depth+1
          FROM nodes n JOIN up ON n.node_id = up.parent_node_id)
     SELECT n.* FROM up JOIN nodes n ON n.node_id = up.node_id ORDER BY up.depth`,
    [nodeId]
  );
}

/** Full subtree rooted at nodeId (self included). */
export function getDescendants(store, nodeId) {
  return rows(
    store,
    `WITH RECURSIVE down(node_id) AS (
        SELECT node_id FROM nodes WHERE node_id=?
        UNION ALL
        SELECT n.node_id FROM nodes n JOIN down ON n.parent_node_id = down.node_id)
     SELECT n.* FROM down JOIN nodes n ON n.node_id = down.node_id WHERE n.node_id != ?`,
    [nodeId, nodeId]
  );
}

/** Direct children of a node. */
export function getChildren(store, nodeId) {
  return rows(store, `SELECT * FROM nodes WHERE parent_node_id=? ORDER BY iteration`, [nodeId]);
}

/** Siblings: nodes sharing the same parent (self excluded). NULL-parent roots are siblings within a session. */
export function getSiblings(store, nodeId) {
  const me = getNode(store, nodeId);
  if (!me) return [];
  if (me.parent_node_id == null) {
    return rows(
      store,
      `SELECT * FROM nodes WHERE session_id=? AND parent_node_id IS NULL AND node_id!=? ORDER BY iteration`,
      [me.session_id, nodeId]
    );
  }
  return rows(
    store,
    `SELECT * FROM nodes WHERE parent_node_id=? AND node_id!=? ORDER BY iteration`,
    [me.parent_node_id, nodeId]
  );
}

/** The whole search tree for one session: nodes + derives-from edges. */
export function getSessionTree(store, sessionId) {
  const nodes = rows(store, `SELECT * FROM nodes WHERE session_id=? ORDER BY iteration`, [sessionId]);
  const edges = nodes
    .filter((n) => n.parent_node_id)
    .map((n) => ({ from: n.node_id, to: n.parent_node_id, type: "derives_from", reward_delta_pct: n.delta_vs_parent_pct }));
  return { nodes, edges };
}

// ---------------------------------------------------------------------------
// RELATION JOIN (joins across the four tables)
// ---------------------------------------------------------------------------

/** Best passing node for a task across ALL its sessions (tasks⋈sessions⋈nodes). */
export function getBestNodeForTask(store, taskId) {
  return row(
    store,
    `SELECT n.* FROM nodes n JOIN sessions s ON n.session_id = s.session_id
       WHERE s.task_id=? AND n.validation='pass' AND n.reward_ms IS NOT NULL
       ORDER BY n.reward_ms ASC LIMIT 1`,
    [taskId]
  );
}

/** The agent I/O rows (prompts) that produced a node (nodes⋈prompts). */
export function getWinningPrompt(store, nodeId) {
  return rows(store, `SELECT * FROM prompts WHERE node_id=? ORDER BY agent`, [nodeId]);
}

/** Per-task global-best leaderboard for a benchmark (tasks⋈nodes). */
export function getTaskLeaderboard(store, benchmark) {
  return rows(
    store,
    `SELECT t.task_id, t.query_id, t.scale_factor, n.node_id AS best_node_id,
            n.reward_ms AS best_ms, n.iteration AS best_iteration
       FROM tasks t LEFT JOIN nodes n ON n.node_id = t.global_best_node_id
      WHERE t.benchmark=? ORDER BY t.query_id`,
    [benchmark]
  );
}

// ---------------------------------------------------------------------------
// VECTOR SEARCH
//   Preferred: sqlite-vec's vec_distance_cosine — an in-engine SIMD KNN scan.
//   Fallback:  JS cosine over decoded BLOBs (when sqlite-vec is unavailable).
//   Both return `score` = cosine similarity in [0,1] (higher = more similar).
// ---------------------------------------------------------------------------

function topKByCosine(candidates, queryVec, k) {
  const scored = [];
  for (const c of candidates) {
    const vec = fromBlob(c.embedding);
    if (!vec || vec.length !== queryVec.length) continue;
    scored.push({ ...c, score: cosine(queryVec, vec) });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

/** KNN via sqlite-vec: ORDER BY vec_distance_cosine ... LIMIT k, pushed into the engine. */
function knnByVec(store, table, queryVec, k, whereExtra = "", whereParams = []) {
  const res = rows(
    store,
    `SELECT *, vec_distance_cosine(embedding, ?) AS _dist
       FROM ${table} WHERE embedding IS NOT NULL ${whereExtra}
       ORDER BY _dist ASC LIMIT ?`,
    [toBlob(queryVec), ...whereParams, k]
  );
  // cosine distance -> similarity, and drop the scratch column
  return res.map(({ _dist, ...r }) => ({ ...r, score: 1 - _dist }));
}

/** Vector-search tasks most similar to a SQL string. Optionally exclude a task_id. */
export function searchSimilarTasks(store, sqlText, k = 5, excludeTaskId = null) {
  if (!store?.enabled) return [];
  const q = embed(sqlText);
  if (store.vec) {
    return knnByVec(store, "tasks", q, k, "AND task_id != ?", [excludeTaskId ?? ""]);
  }
  const cands = rows(store, `SELECT * FROM tasks WHERE embedding IS NOT NULL AND task_id != ?`, [excludeTaskId ?? ""]);
  return topKByCosine(cands, q, k);
}

/** Vector-search nodes whose strategy text is most similar (e.g. "who else hit this bottleneck"). */
export function searchSimilarStrategies(store, strategyText, k = 5) {
  if (!store?.enabled) return [];
  const q = embed(strategyText);
  if (store.vec) {
    return knnByVec(store, "nodes", q, k);
  }
  const cands = rows(store, `SELECT * FROM nodes WHERE embedding IS NOT NULL`);
  return topKByCosine(cands, q, k);
}
