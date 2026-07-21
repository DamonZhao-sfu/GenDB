/**
 * Experience Graph store — open/init + capture (write side).
 *
 * Independent of GenDB's HAG memory (imports nothing from ../memory/).
 * Backed by SQLite via the built-in node:sqlite (Node >= 22.5). Feature-detects
 * gracefully: on older Node (or if node:sqlite is unavailable) openStore returns
 * a disabled handle whose methods are no-ops, so callers never crash.
 */

import { readFileSync } from "node:fs";
import { mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { embed, toBlob } from "./embed.mjs";

const require = createRequire(import.meta.url);
const __dirname = dirname(fileURLToPath(import.meta.url));
const SCHEMA_PATH = resolve(__dirname, "schema.sql");

/** Lazily load node:sqlite; return null if unavailable. */
function loadSqlite() {
  try {
    return require("node:sqlite");
  } catch {
    return null;
  }
}

/**
 * Load the sqlite-vec extension so vector search runs as an in-engine SIMD KNN
 * (vec_distance_cosine) instead of a JS brute-force scan. Returns true on
 * success; on any failure the store keeps working and vector search falls back
 * to the JS cosine path in query.mjs.
 */
function tryLoadVec(db) {
  try {
    const vec = require("sqlite-vec");
    db.enableLoadExtension(true);
    db.loadExtension(vec.getLoadablePath());
    db.enableLoadExtension(false); // re-disable after loading (defense-in-depth)
    return true;
  } catch (e) {
    console.warn(`[experience] sqlite-vec not loaded (${e.message}); vector search uses JS cosine fallback.`);
    return false;
  }
}

/**
 * Open (creating if needed) the experience database at <experienceDir>/experience.db.
 * @returns {{ enabled: boolean, db: object|null, dir: string, vec: boolean }}
 */
export function openStore(experienceDir) {
  const sqlite = loadSqlite();
  if (!sqlite || !sqlite.DatabaseSync) {
    console.warn("[experience] node:sqlite unavailable — experience graph disabled (needs Node >= 22.5).");
    return { enabled: false, db: null, dir: experienceDir, vec: false };
  }
  mkdirSync(experienceDir, { recursive: true });
  const dbPath = resolve(experienceDir, "experience.db");
  let db;
  let vecEnabled = false;
  try {
    // allowExtension is required before loadExtension is permitted.
    db = new sqlite.DatabaseSync(dbPath, { allowExtension: true });
    vecEnabled = tryLoadVec(db);
  } catch {
    // Older node:sqlite without allowExtension — open normally, JS fallback.
    db = new sqlite.DatabaseSync(dbPath);
  }
  const schema = readFileSync(SCHEMA_PATH, "utf-8");
  db.exec(schema);
  return { enabled: true, db, dir: experienceDir, vec: vecEnabled };
}

/** Close the store (safe on a disabled handle). */
export function closeStore(store) {
  if (store?.enabled && store.db) store.db.close();
}

const nowIso = () => new Date().toISOString();

// Identity helpers (pure) -----------------------------------------------------

export function taskId(benchmark, queryId, scaleFactor) {
  return `${benchmark}__${queryId}__sf${scaleFactor}`;
}
export function sessionId(runId, queryId) {
  return `${runId}__${queryId}`;
}
export function nodeId(sessionId_, iteration) {
  return `${sessionId_}__iter_${iteration}`;
}

// Write API -------------------------------------------------------------------

/** Insert or update a task row; (re)computes the SQL embedding. */
export function upsertTask(store, task) {
  if (!store?.enabled) return;
  const emb = toBlob(embed(task.sql_text || task.query_id || ""));
  store.db
    .prepare(
      `INSERT INTO tasks (task_id, benchmark, query_id, scale_factor, sql_text,
          template_signature, embedding, global_best_node_id, created_at, updated_at)
       VALUES (?,?,?,?,?,?,?,?,?,?)
       ON CONFLICT(task_id) DO UPDATE SET
          sql_text=excluded.sql_text,
          template_signature=excluded.template_signature,
          embedding=excluded.embedding,
          updated_at=excluded.updated_at`
    )
    .run(
      task.task_id, task.benchmark, task.query_id, task.scale_factor,
      task.sql_text ?? null, task.template_signature ?? null, emb,
      task.global_best_node_id ?? null, nowIso(), nowIso()
    );
}

/** Insert (or replace) a session row. */
export function openSession(store, s) {
  if (!store?.enabled) return;
  store.db
    .prepare(
      `INSERT INTO sessions (session_id, task_id, run_id, query_id, model,
          agent_provider, optimization_target, hardware_fingerprint,
          root_node_id, best_node_id, warm_started_from_node_id, status,
          started_at, completed_at)
       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
       ON CONFLICT(session_id) DO UPDATE SET
          model=excluded.model, agent_provider=excluded.agent_provider,
          optimization_target=excluded.optimization_target,
          hardware_fingerprint=excluded.hardware_fingerprint,
          root_node_id=excluded.root_node_id,
          warm_started_from_node_id=excluded.warm_started_from_node_id`
    )
    .run(
      s.session_id, s.task_id, s.run_id ?? null, s.query_id ?? null, s.model ?? null,
      s.agent_provider ?? null, s.optimization_target ?? null, s.hardware_fingerprint ?? null,
      s.root_node_id ?? null, s.best_node_id ?? null, s.warm_started_from_node_id ?? null,
      s.status ?? "running", s.started_at ?? nowIso(), s.completed_at ?? null
    );
}

/**
 * Record one iteration node. Computes delta_vs_parent_pct / delta_vs_root_pct
 * from the parent and the session root. Also maintains is_best_in_session and
 * the session's best pointer when this node improves.
 */
export function recordNode(store, node) {
  if (!store?.enabled) return;
  const db = store.db;

  const parentReward = node.parent_node_id
    ? db.prepare(`SELECT reward_ms FROM nodes WHERE node_id=?`).get(node.parent_node_id)?.reward_ms
    : null;
  const rootId = db.prepare(`SELECT root_node_id FROM sessions WHERE session_id=?`).get(node.session_id)?.root_node_id;
  const rootReward = rootId
    ? db.prepare(`SELECT reward_ms FROM nodes WHERE node_id=?`).get(rootId)?.reward_ms
    : null;

  const pct = (a, base) => (a != null && base) ? Math.round(((a - base) / base) * 1000) / 10 : null;
  // Null (not a zero vector) when there is no strategy text, so vec_distance_cosine never sees a zero vector.
  const emb = node.strategy ? toBlob(embed(node.strategy)) : null;

  db.prepare(
    `INSERT INTO nodes (node_id, session_id, task_id, parent_node_id, iteration,
        reward_ms, hot_ms, cold_ms, validation, improved, is_best_in_session,
        delta_vs_parent_pct, delta_vs_root_pct, visit_count, best_descendant_ms,
        categories, strategy, embedding, operation_timings,
        cpp_path, plan_path, exec_results_path, created_at)
     VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
     ON CONFLICT(node_id) DO UPDATE SET
        parent_node_id=excluded.parent_node_id, reward_ms=excluded.reward_ms,
        hot_ms=excluded.hot_ms, cold_ms=excluded.cold_ms, validation=excluded.validation,
        improved=excluded.improved, delta_vs_parent_pct=excluded.delta_vs_parent_pct,
        delta_vs_root_pct=excluded.delta_vs_root_pct, categories=excluded.categories,
        strategy=excluded.strategy, embedding=excluded.embedding,
        operation_timings=excluded.operation_timings, cpp_path=excluded.cpp_path,
        plan_path=excluded.plan_path, exec_results_path=excluded.exec_results_path`
  ).run(
    node.node_id, node.session_id, node.task_id, node.parent_node_id ?? null,
    node.iteration ?? null,
    node.reward_ms ?? null, node.hot_ms ?? null, node.cold_ms ?? null,
    node.validation ?? null, node.improved ? 1 : 0, 0,
    pct(node.reward_ms, parentReward), pct(node.reward_ms, rootReward),
    0, node.reward_ms ?? null,
    node.categories ? JSON.stringify(node.categories) : null,
    node.strategy ?? null, emb,
    node.operation_timings ? JSON.stringify(node.operation_timings) : null,
    node.cpp_path ?? null, node.plan_path ?? null, node.exec_results_path ?? null,
    node.created_at ?? nowIso()
  );

  // Maintain the session's best pointer (lowest passing reward).
  if (node.validation === "pass" && node.reward_ms != null) {
    const curBest = db.prepare(
      `SELECT n.node_id, n.reward_ms FROM sessions s
         LEFT JOIN nodes n ON n.node_id = s.best_node_id
        WHERE s.session_id=?`
    ).get(node.session_id);
    if (!curBest || curBest.reward_ms == null || node.reward_ms < curBest.reward_ms) {
      db.prepare(`UPDATE nodes SET is_best_in_session=0 WHERE session_id=?`).run(node.session_id);
      db.prepare(`UPDATE nodes SET is_best_in_session=1 WHERE node_id=?`).run(node.node_id);
      db.prepare(`UPDATE sessions SET best_node_id=? WHERE session_id=?`).run(node.node_id, node.session_id);
    }
  }
}

/** Record one agent-I/O prompt row for a node. */
export function recordPrompt(store, p) {
  if (!store?.enabled) return;
  store.db.prepare(
    `INSERT INTO prompts (prompt_id, node_id, agent, request_path, response_path,
        tokens, cost_usd, duration_ms, created_at)
     VALUES (?,?,?,?,?,?,?,?,?)
     ON CONFLICT(prompt_id) DO UPDATE SET
        request_path=excluded.request_path, response_path=excluded.response_path,
        tokens=excluded.tokens, cost_usd=excluded.cost_usd, duration_ms=excluded.duration_ms`
  ).run(
    p.prompt_id, p.node_id, p.agent ?? null, p.request_path ?? null, p.response_path ?? null,
    p.tokens ?? null, p.cost_usd ?? null, p.duration_ms ?? null, nowIso()
  );
}

/** Finalize a session and roll its best up to the task's global best. */
export function closeSession(store, sessionId_, { bestNodeId, status } = {}) {
  if (!store?.enabled) return;
  const db = store.db;
  db.prepare(`UPDATE sessions SET status=?, completed_at=?, best_node_id=COALESCE(?, best_node_id) WHERE session_id=?`)
    .run(status ?? "completed", nowIso(), bestNodeId ?? null, sessionId_);

  // Update the task's global best across sessions.
  const row = db.prepare(
    `SELECT n.task_id, n.node_id, n.reward_ms FROM sessions s
        JOIN nodes n ON n.node_id = s.best_node_id
       WHERE s.session_id=?`
  ).get(sessionId_);
  if (row) {
    const cur = db.prepare(
      `SELECT n.node_id, n.reward_ms FROM tasks t
          LEFT JOIN nodes n ON n.node_id = t.global_best_node_id
         WHERE t.task_id=?`
    ).get(row.task_id);
    if (!cur || cur.reward_ms == null || (row.reward_ms != null && row.reward_ms < cur.reward_ms)) {
      db.prepare(`UPDATE tasks SET global_best_node_id=?, updated_at=? WHERE task_id=?`)
        .run(row.node_id, nowIso(), row.task_id);
    }
  }
}

/**
 * MCTS-style backpropagation: propagate a leaf's reward up its ancestor chain,
 * bumping visit_count and lowering best_descendant_ms.
 */
export function backpropReward(store, leafNodeId) {
  if (!store?.enabled) return;
  const db = store.db;
  const leaf = db.prepare(`SELECT reward_ms FROM nodes WHERE node_id=?`).get(leafNodeId);
  if (!leaf) return;
  const reward = leaf.reward_ms;
  let cur = leafNodeId;
  const seen = new Set();
  while (cur && !seen.has(cur)) {
    seen.add(cur);
    db.prepare(
      `UPDATE nodes SET visit_count = visit_count + 1,
          best_descendant_ms = CASE
            WHEN best_descendant_ms IS NULL THEN ?
            WHEN ? IS NOT NULL AND ? < best_descendant_ms THEN ?
            ELSE best_descendant_ms END
        WHERE node_id=?`
    ).run(reward, reward, reward, reward, cur);
    cur = db.prepare(`SELECT parent_node_id FROM nodes WHERE node_id=?`).get(cur)?.parent_node_id;
  }
}
