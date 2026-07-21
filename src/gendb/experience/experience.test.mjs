#!/usr/bin/env node
/**
 * Experience Graph tests — validates the three access patterns on the
 * Postgres/pgvector backend: graph traverse (recursive CTE), relation join, and
 * vector search (HNSW ANN index), plus parent reconstruction and MCTS backprop.
 *
 * Requires a reachable Postgres with pgvector. Set the connection string in
 * GENDB_EXPERIENCE_PG_TEST (a THROWAWAY database — the test drops its four
 * tables at start). Without it, the test SKIPS (exit 0) so CI stays green on
 * machines without Postgres.
 *
 *   GENDB_EXPERIENCE_PG_TEST=postgres://user@host/db node src/gendb/experience/experience.test.mjs
 */

import assert from "node:assert";
import { createRequire } from "node:module";
import { openExperienceStore, taskId, sessionId, nodeId } from "./index.mjs";
import { reconstructParents } from "./reconstruct.mjs";

const require = createRequire(import.meta.url);
const URL = process.env.GENDB_EXPERIENCE_PG_TEST;

if (!URL) {
  console.log("SKIP: set GENDB_EXPERIENCE_PG_TEST to a throwaway Postgres DB to run PG backend tests.");
  process.exit(0);
}

let passed = 0;
const eq = (a, b, msg) => { assert.deepStrictEqual(a, b, msg); passed++; };
const near = (a, b, tol, msg) => { assert.ok(Math.abs(a - b) <= tol, `${msg} (${a} vs ${b})`); passed++; };

const Q24 = [
  { iteration: 0, improved: true, hot_timing_ms: 448, validation: "pass", strategy: "initial implementation" },
  { iteration: 1, improved: false, hot_timing_ms: 2112, validation: "pass", strategy: "aggregate anti-join key first, order by hash region" },
  { iteration: 2, improved: true, hot_timing_ms: 157, validation: "pass", strategy: "partitioned exact membership storage extension, 524288 partitions" },
  { iteration: 3, improved: false, hot_timing_ms: 560, validation: "pass", strategy: "scatter into partition buffers, aggregate per partition" },
  { iteration: 4, improved: false, hot_timing_ms: 287, validation: "pass", strategy: "k-way merge date postings, direct uom column read" },
  { iteration: 5, improved: true, hot_timing_ms: 127, validation: "pass", strategy: "group pre keys by adsh into dense offsets plus sorted payload in LLC" },
];
const SINGLE = [{ iteration: 0, improved: true, hot_timing_ms: 80, validation: "pass", strategy: "initial" }];
const ALL_REGRESS = [
  { iteration: 0, improved: true, hot_timing_ms: 100, validation: "pass", strategy: "initial" },
  { iteration: 1, improved: false, hot_timing_ms: 200, validation: "pass", strategy: "bad idea A" },
  { iteration: 2, improved: false, hot_timing_ms: 150, validation: "pass", strategy: "bad idea B" },
];

async function resetTables(url) {
  const { Client } = require("pg");
  const c = new Client({ connectionString: url });
  await c.connect();
  await c.query("DROP TABLE IF EXISTS prompts,nodes,sessions,tasks CASCADE");
  await c.end();
}

async function loadTrajectory(store, benchmark, queryId, sf, iters) {
  const tId = taskId(benchmark, queryId, sf);
  const sId = sessionId("RUN", queryId);
  const parents = reconstructParents(iters, "hot");
  await store.upsertTask({ task_id: tId, benchmark, query_id: queryId, scale_factor: sf, sql_text: `SELECT * FROM ${queryId}` });
  await store.openSession({ session_id: sId, task_id: tId, run_id: "RUN", query_id: queryId, optimization_target: "hot", root_node_id: nodeId(sId, iters[0].iteration) });
  let best = null, bestMs = Infinity;
  for (const it of iters) {
    const pi = parents.get(it.iteration);
    await store.recordNode({
      node_id: nodeId(sId, it.iteration), session_id: sId, task_id: tId,
      parent_node_id: pi == null ? null : nodeId(sId, pi),
      iteration: it.iteration, reward_ms: it.hot_timing_ms, hot_ms: it.hot_timing_ms,
      validation: it.validation, improved: it.improved, strategy: it.strategy, categories: [],
    });
    if (it.validation === "pass" && it.hot_timing_ms < bestMs) { bestMs = it.hot_timing_ms; best = nodeId(sId, it.iteration); }
  }
  await store.closeSession(sId, { bestNodeId: best });
  if (best) await store.backpropReward(best);
  return { tId, sId };
}

async function main() {
  await resetTables(URL);
  const store = await openExperienceStore({ backend: "postgres", connectionString: URL });
  console.log(`  backend: ${store.backend}, vector: ${store.vec ? "pgvector HNSW" : "none"}`);

  const { tId, sId } = await loadTrajectory(store, "sec-edgar", "Q24", 3, Q24);
  const n = (i) => nodeId(sId, i);

  // GRAPH TRAVERSE (recursive CTE)
  eq((await store.getAncestors(n(5))).map((r) => r.iteration), [5, 2, 0], "ancestors(iter5)");
  eq((await store.getSiblings(n(5))).map((r) => r.iteration).sort((a, b) => a - b), [3, 4], "siblings(iter5)");
  eq((await store.getChildren(n(2))).map((r) => r.iteration).sort((a, b) => a - b), [3, 4, 5], "children(iter2)");
  eq((await store.getDescendants(n(0))).length, 5, "descendants(root) count");

  // RELATION JOIN
  const best = await store.getBestNodeForTask(tId);
  eq(best.iteration, 5, "bestNodeForTask=iter5");
  eq((await store.getNode(n(5))).is_best_in_session, 1, "iter5 is_best_in_session");

  // deltas + backprop
  near((await store.getNode(n(5))).delta_vs_parent_pct, -19.1, 0.5, "iter5 delta_vs_parent");
  near(Number((await store.getNode(n(0))).best_descendant_ms), 127, 0.001, "root best_descendant via backprop");

  // VECTOR SEARCH (pgvector HNSW)
  const sim = await store.searchSimilarStrategies("partitioned exact membership storage extension in LLC", 3);
  assert.ok(sim.length > 0 && Number(sim[0].score) > 0, "vector search returns ranked strategies"); passed++;
  assert.ok(Number(sim[0].score) >= Number(sim[sim.length - 1].score), "ranked by descending similarity"); passed++;

  // single-iteration + all-regression shapes
  const single = await loadTrajectory(store, "tpc-h", "Q1", 10, SINGLE);
  eq((await store.getSiblings(nodeId(single.sId, 0))).length, 0, "single-iter: no siblings");
  eq((await store.getBestNodeForTask(single.tId)).iteration, 0, "single-iter: best=iter0");

  const regress = await loadTrajectory(store, "tpc-h", "Q9", 10, ALL_REGRESS);
  eq((await store.getBestNodeForTask(regress.tId)).iteration, 0, "all-regression: best stays iter0");
  eq((await store.getSiblings(nodeId(regress.sId, 1))).map((r) => r.iteration).sort((a, b) => a - b), [2], "all-regression: siblings(iter1)=[2]");

  await store.close();
  console.log(`\n✅ experience.test.mjs — ${passed} assertions passed`);
}

main().catch((e) => { console.error(e); process.exit(1); });
