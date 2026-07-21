/**
 * Experience Graph — PostgreSQL backend (async).
 *
 * A genuinely unified engine for all three access patterns:
 *   graph traverse — recursive CTE over nodes.parent_node_id
 *   relation join   — SQL joins across the four tables
 *   vector search   — pgvector cosine (<=>) accelerated by an HNSW ANN index
 *
 * Requires a reachable PostgreSQL with the `vector` (pgvector) extension.
 * Independent of GenDB's HAG memory.
 */

import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { embed, EMBED_DIM } from "../embed.mjs";

const require = createRequire(import.meta.url);
const __dirname = dirname(fileURLToPath(import.meta.url));
const SCHEMA_PATH = resolve(__dirname, "..", "schema.pg.sql");

const nowIso = () => new Date().toISOString();
/** Encode a Float32Array as a pgvector literal, e.g. "[0.1,0.2,...]". */
function vecLiteral(vecOrText) {
  const v = typeof vecOrText === "string" ? embed(vecOrText) : vecOrText;
  if (!v || v.length !== EMBED_DIM) return null;
  return `[${Array.from(v).join(",")}]`;
}
const pct = (a, base) => (a != null && base) ? Math.round(((a - base) / base) * 1000) / 10 : null;

/**
 * Open a Postgres-backed store.
 * @param {object} opts { connectionString } or { pg: {host,port,user,database,password} }
 * @returns async store handle (same interface as the sqlite backend)
 */
export async function createPostgresBackend(opts = {}) {
  const { Client } = require("pg");
  const client = opts.connectionString
    ? new Client({ connectionString: opts.connectionString })
    : new Client(opts.pg || {});
  await client.connect();
  await client.query(readFileSync(SCHEMA_PATH, "utf-8"));

  const q = (sql, params = []) => client.query(sql, params);
  const all = async (sql, params = []) => (await q(sql, params)).rows;
  const one = async (sql, params = []) => (await q(sql, params)).rows[0] ?? null;

  return {
    backend: "postgres",
    vec: true, // pgvector HNSW always available once the schema loads

    // ---- capture (write) ----
    async upsertTask(t) {
      await q(
        `INSERT INTO tasks (task_id,benchmark,query_id,scale_factor,sql_text,template_signature,embedding,global_best_node_id,created_at,updated_at)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$9)
         ON CONFLICT (task_id) DO UPDATE SET sql_text=EXCLUDED.sql_text,
           template_signature=EXCLUDED.template_signature, embedding=EXCLUDED.embedding, updated_at=EXCLUDED.updated_at`,
        [t.task_id, t.benchmark, t.query_id, t.scale_factor, t.sql_text ?? null,
         t.template_signature ? JSON.stringify(t.template_signature) : null,
         vecLiteral(t.sql_text || t.query_id || ""), t.global_best_node_id ?? null, nowIso()]
      );
    },

    async openSession(s) {
      await q(
        `INSERT INTO sessions (session_id,task_id,run_id,query_id,model,agent_provider,optimization_target,
            hardware_fingerprint,root_node_id,best_node_id,warm_started_from_node_id,status,started_at,completed_at)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
         ON CONFLICT (session_id) DO UPDATE SET model=EXCLUDED.model, agent_provider=EXCLUDED.agent_provider,
            optimization_target=EXCLUDED.optimization_target, hardware_fingerprint=EXCLUDED.hardware_fingerprint,
            root_node_id=EXCLUDED.root_node_id, warm_started_from_node_id=EXCLUDED.warm_started_from_node_id`,
        [s.session_id, s.task_id, s.run_id ?? null, s.query_id ?? null, s.model ?? null,
         s.agent_provider ?? null, s.optimization_target ?? null, s.hardware_fingerprint ?? null,
         s.root_node_id ?? null, s.best_node_id ?? null, s.warm_started_from_node_id ?? null,
         s.status ?? "running", s.started_at ?? nowIso(), s.completed_at ?? null]
      );
    },

    async recordNode(node) {
      const parentReward = node.parent_node_id
        ? (await one(`SELECT reward_ms FROM nodes WHERE node_id=$1`, [node.parent_node_id]))?.reward_ms
        : null;
      const rootId = (await one(`SELECT root_node_id FROM sessions WHERE session_id=$1`, [node.session_id]))?.root_node_id;
      const rootReward = rootId ? (await one(`SELECT reward_ms FROM nodes WHERE node_id=$1`, [rootId]))?.reward_ms : null;
      const emb = node.strategy ? vecLiteral(node.strategy) : null;

      await q(
        `INSERT INTO nodes (node_id,session_id,task_id,parent_node_id,iteration,reward_ms,hot_ms,cold_ms,
            validation,improved,is_best_in_session,delta_vs_parent_pct,delta_vs_root_pct,visit_count,
            best_descendant_ms,categories,strategy,embedding,operation_timings,cpp_path,plan_path,exec_results_path,created_at)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,0,$11,$12,0,$6,$13,$14,$15,$16,$17,$18,$19,$20)
         ON CONFLICT (node_id) DO UPDATE SET parent_node_id=EXCLUDED.parent_node_id, reward_ms=EXCLUDED.reward_ms,
            hot_ms=EXCLUDED.hot_ms, cold_ms=EXCLUDED.cold_ms, validation=EXCLUDED.validation, improved=EXCLUDED.improved,
            delta_vs_parent_pct=EXCLUDED.delta_vs_parent_pct, delta_vs_root_pct=EXCLUDED.delta_vs_root_pct,
            categories=EXCLUDED.categories, strategy=EXCLUDED.strategy, embedding=EXCLUDED.embedding,
            operation_timings=EXCLUDED.operation_timings, cpp_path=EXCLUDED.cpp_path, plan_path=EXCLUDED.plan_path,
            exec_results_path=EXCLUDED.exec_results_path`,
        [node.node_id, node.session_id, node.task_id, node.parent_node_id ?? null, node.iteration ?? null,
         node.reward_ms ?? null, node.hot_ms ?? null, node.cold_ms ?? null, node.validation ?? null,
         node.improved ? 1 : 0, pct(node.reward_ms, parentReward), pct(node.reward_ms, rootReward),
         node.categories ? JSON.stringify(node.categories) : null, node.strategy ?? null, emb,
         node.operation_timings ? JSON.stringify(node.operation_timings) : null,
         node.cpp_path ?? null, node.plan_path ?? null, node.exec_results_path ?? null, node.created_at ?? nowIso()]
      );

      // Maintain session best pointer (lowest passing reward).
      if (node.validation === "pass" && node.reward_ms != null) {
        const cur = await one(
          `SELECT n.node_id, n.reward_ms FROM sessions s LEFT JOIN nodes n ON n.node_id=s.best_node_id WHERE s.session_id=$1`,
          [node.session_id]
        );
        if (!cur || cur.reward_ms == null || node.reward_ms < cur.reward_ms) {
          await q(`UPDATE nodes SET is_best_in_session=0 WHERE session_id=$1`, [node.session_id]);
          await q(`UPDATE nodes SET is_best_in_session=1 WHERE node_id=$1`, [node.node_id]);
          await q(`UPDATE sessions SET best_node_id=$1 WHERE session_id=$2`, [node.node_id, node.session_id]);
        }
      }
    },

    async recordPrompt(p) {
      await q(
        `INSERT INTO prompts (prompt_id,node_id,agent,request_path,response_path,tokens,cost_usd,duration_ms,created_at)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
         ON CONFLICT (prompt_id) DO UPDATE SET request_path=EXCLUDED.request_path, response_path=EXCLUDED.response_path,
            tokens=EXCLUDED.tokens, cost_usd=EXCLUDED.cost_usd, duration_ms=EXCLUDED.duration_ms`,
        [p.prompt_id, p.node_id, p.agent ?? null, p.request_path ?? null, p.response_path ?? null,
         p.tokens ?? null, p.cost_usd ?? null, p.duration_ms ?? null, nowIso()]
      );
    },

    async closeSession(sessionId, { bestNodeId, status } = {}) {
      await q(`UPDATE sessions SET status=$1, completed_at=$2, best_node_id=COALESCE($3,best_node_id) WHERE session_id=$4`,
        [status ?? "completed", nowIso(), bestNodeId ?? null, sessionId]);
      const row = await one(
        `SELECT n.task_id, n.node_id, n.reward_ms FROM sessions s JOIN nodes n ON n.node_id=s.best_node_id WHERE s.session_id=$1`,
        [sessionId]);
      if (row) {
        const cur = await one(
          `SELECT n.node_id, n.reward_ms FROM tasks t LEFT JOIN nodes n ON n.node_id=t.global_best_node_id WHERE t.task_id=$1`,
          [row.task_id]);
        if (!cur || cur.reward_ms == null || (row.reward_ms != null && row.reward_ms < cur.reward_ms)) {
          await q(`UPDATE tasks SET global_best_node_id=$1, updated_at=$2 WHERE task_id=$3`, [row.node_id, nowIso(), row.task_id]);
        }
      }
    },

    async backpropReward(leafNodeId) {
      const leaf = await one(`SELECT reward_ms FROM nodes WHERE node_id=$1`, [leafNodeId]);
      if (!leaf) return;
      const reward = leaf.reward_ms;
      let cur = leafNodeId;
      const seen = new Set();
      while (cur && !seen.has(cur)) {
        seen.add(cur);
        // LEAST ignores NULL inputs, so this is min-with-null-safety.
        await q(
          `UPDATE nodes SET visit_count=visit_count+1,
             best_descendant_ms = LEAST(best_descendant_ms, $1::double precision)
           WHERE node_id=$2`, [reward, cur]);
        cur = (await one(`SELECT parent_node_id FROM nodes WHERE node_id=$1`, [cur]))?.parent_node_id;
      }
    },

    // ---- GRAPH TRAVERSE (recursive CTE) ----
    getNode: (id) => one(`SELECT * FROM nodes WHERE node_id=$1`, [id]),
    getAncestors: (id) => all(
      `WITH RECURSIVE up(node_id,parent_node_id,depth) AS (
          SELECT node_id,parent_node_id,0 FROM nodes WHERE node_id=$1
          UNION ALL SELECT n.node_id,n.parent_node_id,up.depth+1 FROM nodes n JOIN up ON n.node_id=up.parent_node_id)
       SELECT n.* FROM up JOIN nodes n ON n.node_id=up.node_id ORDER BY up.depth`, [id]),
    getDescendants: (id) => all(
      `WITH RECURSIVE down(node_id) AS (
          SELECT node_id FROM nodes WHERE node_id=$1
          UNION ALL SELECT n.node_id FROM nodes n JOIN down ON n.parent_node_id=down.node_id)
       SELECT n.* FROM down JOIN nodes n ON n.node_id=down.node_id WHERE n.node_id<>$1`, [id]),
    getChildren: (id) => all(`SELECT * FROM nodes WHERE parent_node_id=$1 ORDER BY iteration`, [id]),
    async getSiblings(id) {
      const me = await one(`SELECT session_id,parent_node_id FROM nodes WHERE node_id=$1`, [id]);
      if (!me) return [];
      if (me.parent_node_id == null) {
        return all(`SELECT * FROM nodes WHERE session_id=$1 AND parent_node_id IS NULL AND node_id<>$2 ORDER BY iteration`, [me.session_id, id]);
      }
      return all(`SELECT * FROM nodes WHERE parent_node_id=$1 AND node_id<>$2 ORDER BY iteration`, [me.parent_node_id, id]);
    },
    async getSessionTree(sessionId) {
      const nodes = await all(`SELECT * FROM nodes WHERE session_id=$1 ORDER BY iteration`, [sessionId]);
      const edges = nodes.filter((n) => n.parent_node_id).map((n) => ({ from: n.node_id, to: n.parent_node_id, type: "derives_from", reward_delta_pct: n.delta_vs_parent_pct }));
      return { nodes, edges };
    },

    // ---- RELATION JOIN ----
    getBestNodeForTask: (taskId) => one(
      `SELECT n.* FROM nodes n JOIN sessions s ON n.session_id=s.session_id
         WHERE s.task_id=$1 AND n.validation='pass' AND n.reward_ms IS NOT NULL
         ORDER BY n.reward_ms ASC LIMIT 1`, [taskId]),
    getWinningPrompt: (nodeId) => all(`SELECT * FROM prompts WHERE node_id=$1 ORDER BY agent`, [nodeId]),
    getTaskLeaderboard: (benchmark) => all(
      `SELECT t.task_id,t.query_id,t.scale_factor,n.node_id AS best_node_id,n.reward_ms AS best_ms,n.iteration AS best_iteration
         FROM tasks t LEFT JOIN nodes n ON n.node_id=t.global_best_node_id WHERE t.benchmark=$1 ORDER BY t.query_id`, [benchmark]),

    // ---- VECTOR SEARCH (pgvector HNSW, cosine) ----
    async searchSimilarTasks(sqlText, k = 5, excludeTaskId = null) {
      const lit = vecLiteral(sqlText);
      const res = await all(
        `SELECT *, 1-(embedding <=> $1::vector) AS score FROM tasks
           WHERE embedding IS NOT NULL AND task_id <> $2 ORDER BY embedding <=> $1::vector LIMIT $3`,
        [lit, excludeTaskId ?? "", k]);
      return res;
    },
    async searchSimilarStrategies(strategyText, k = 5) {
      const lit = vecLiteral(strategyText);
      return all(
        `SELECT *, 1-(embedding <=> $1::vector) AS score FROM nodes
           WHERE embedding IS NOT NULL ORDER BY embedding <=> $1::vector LIMIT $2`,
        [lit, k]);
    },

    async close() { await client.end(); },
  };
}
