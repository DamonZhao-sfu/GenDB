#!/usr/bin/env node
/**
 * Experience Graph — self-contained tests. No dependency on output/**.
 *
 * Run: node src/gendb/experience/experience.test.mjs
 *
 * Covers:
 *   1. Parent reconstruction on three canonical trajectory shapes
 *      (Q24-style branching, single-iteration success, all-regression tail).
 *   2. The three access patterns on a store built from the Q24 shape:
 *      graph traverse (ancestors/siblings/children), relation join (best node),
 *      vector search (cosine over strategy embeddings), + MCTS backprop.
 */

import assert from "node:assert";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  openStore, closeStore, taskId, sessionId, nodeId,
  upsertTask, openSession, recordNode, closeSession, backpropReward,
} from "./store.mjs";
import * as q from "./query.mjs";
import { reconstructParents } from "./reconstruct.mjs";

let passed = 0;
const check = (cond, msg) => { assert.ok(cond, msg); passed++; };
const eq = (a, b, msg) => { assert.deepStrictEqual(a, b, msg); passed++; };

// --- Canonical trajectory fixtures ------------------------------------------
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

// --- 1. Parent reconstruction -----------------------------------------------
function testReconstruct() {
  const p24 = reconstructParents(Q24, "hot");
  eq([...p24.entries()], [[0, null], [1, 0], [2, 0], [3, 2], [4, 2], [5, 2]], "Q24 parents");

  const pS = reconstructParents(SINGLE, "hot");
  eq([...pS.entries()], [[0, null]], "single-iter parents");

  const pR = reconstructParents(ALL_REGRESS, "hot");
  eq([...pR.entries()], [[0, null], [1, 0], [2, 0]], "all-regression parents");
}

// --- helper: load a trajectory into a store --------------------------------
function loadTrajectory(store, benchmark, queryId, sf, iters) {
  const tId = taskId(benchmark, queryId, sf);
  const sId = sessionId("RUN", queryId);
  const parents = reconstructParents(iters, "hot");
  upsertTask(store, { task_id: tId, benchmark, query_id: queryId, scale_factor: sf, sql_text: `SELECT * FROM ${queryId}` });
  openSession(store, { session_id: sId, task_id: tId, run_id: "RUN", query_id: queryId, optimization_target: "hot", root_node_id: nodeId(sId, iters[0].iteration) });
  let best = null, bestMs = Infinity;
  for (const it of iters) {
    const pi = parents.get(it.iteration);
    recordNode(store, {
      node_id: nodeId(sId, it.iteration), session_id: sId, task_id: tId,
      parent_node_id: pi == null ? null : nodeId(sId, pi),
      iteration: it.iteration, reward_ms: it.hot_timing_ms, hot_ms: it.hot_timing_ms,
      validation: it.validation, improved: it.improved, strategy: it.strategy, categories: [],
    });
    if (it.validation === "pass" && it.hot_timing_ms < bestMs) { bestMs = it.hot_timing_ms; best = nodeId(sId, it.iteration); }
  }
  closeSession(store, sId, { bestNodeId: best });
  if (best) backpropReward(store, best);
  return { tId, sId };
}

// --- 2. Access patterns ------------------------------------------------------
function testAccessPatterns(store) {
  const { tId, sId } = loadTrajectory(store, "sec-edgar", "Q24", 3, Q24);
  const n = (i) => nodeId(sId, i);

  // GRAPH TRAVERSE
  eq(q.getAncestors(store, n(5)).map((r) => r.iteration), [5, 2, 0], "ancestors(iter5)");
  eq(q.getSiblings(store, n(5)).map((r) => r.iteration).sort(), [3, 4], "siblings(iter5)");
  eq(q.getChildren(store, n(2)).map((r) => r.iteration).sort(), [3, 4, 5], "children(iter2)");
  eq(q.getDescendants(store, n(0)).length, 5, "descendants(root) count");

  // RELATION JOIN
  const best = q.getBestNodeForTask(store, tId);
  eq(best.iteration, 5, "bestNodeForTask=iter5");
  eq(q.getNode(store, n(5)).is_best_in_session, 1, "iter5 is_best_in_session");
  const lb = q.getTaskLeaderboard(store, "sec-edgar").find((r) => r.query_id === "Q24");
  eq(lb.best_iteration, 5, "leaderboard Q24 best iteration");

  // deltas + backprop
  const node5 = q.getNode(store, n(5));
  // (127-157)/157 ≈ -19.1% for the rounded fixture timings.
  check(node5.delta_vs_parent_pct < -18 && node5.delta_vs_parent_pct > -20, "iter5 delta_vs_parent ≈ -19%");
  const root = q.getNode(store, n(0));
  eq(root.best_descendant_ms, 127, "root best_descendant_ms via backprop");
  check(root.visit_count >= 1, "root visited by backprop");

  // VECTOR SEARCH (sqlite-vec engine KNN when available, else JS cosine)
  console.log(`  vector engine: ${store.vec ? "sqlite-vec (vec_distance_cosine)" : "JS cosine fallback"}`);
  const sim = q.searchSimilarStrategies(store, "partitioned exact membership storage extension in LLC", 3);
  check(sim.length > 0 && sim[0].score > 0, "vector search returns ranked strategies");
  check(sim.every((r) => r.score >= -1.0001 && r.score <= 1.0001), "vector scores are cosine similarities in [-1,1]");
  check(sim[0].score >= sim[sim.length - 1].score, "vector results ranked by descending similarity");

  // single-iteration + all-regression shapes
  const single = loadTrajectory(store, "tpc-h", "Q1", 10, SINGLE);
  eq(q.getSiblings(store, nodeId(single.sId, 0)).length, 0, "single-iter: no siblings");
  eq(q.getBestNodeForTask(store, single.tId).iteration, 0, "single-iter: best=iter0");

  const regress = loadTrajectory(store, "tpc-h", "Q9", 10, ALL_REGRESS);
  eq(q.getBestNodeForTask(store, regress.tId).iteration, 0, "all-regression: best stays iter0");
  eq(q.getSiblings(store, nodeId(regress.sId, 1)).map((r) => r.iteration).sort(), [2], "all-regression: siblings(iter1)=[2]");
}

function main() {
  testReconstruct();
  const dir = mkdtempSync(join(tmpdir(), "xg-test-"));
  const store = openStore(dir);
  if (!store.enabled) { console.error("SKIP: node:sqlite unavailable (needs Node >= 22.5)"); process.exit(0); }
  try {
    testAccessPatterns(store);
  } finally {
    closeStore(store);
    rmSync(dir, { recursive: true, force: true });
  }
  console.log(`\n✅ experience.test.mjs — ${passed} assertions passed`);
}

main();
