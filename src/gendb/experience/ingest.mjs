/**
 * Ingest one query's optimization history into the Experience Graph store.
 *
 * Shared by the offline backfill (scripts/backfill_experience.mjs) and by live
 * capture (capture.mjs) so both produce identical graphs. Reconstructs the
 * branching search tree from the linear iterations[] using the loop's own
 * best-so-far rule (reconstruct.mjs), then writes task + session + nodes and
 * runs MCTS backprop.
 */

import { existsSync } from "node:fs";
import { join } from "node:path";
import { taskId, sessionId, nodeId } from "./ids.mjs";
import { rewardOf, reconstructParents } from "./reconstruct.mjs";

/**
 * @param store  an open Experience store (from openExperienceStore)
 * @param opts {
 *   benchmark, scaleFactor, runId, queryId, target,   // identity + reward target
 *   sqlText?,                                          // task SQL (optional)
 *   iterations,                                        // optimization_history.iterations[]
 *   queryDir?,                                         // dir holding iter_<n>/ artifacts (optional)
 *   session?                                           // extra session metadata (model, provider, hw, timestamps)
 * }
 * @returns { sessionId, bestNodeId, nodeCount }
 */
export async function ingestHistory(store, opts) {
  const {
    benchmark, scaleFactor, runId, queryId, target = "hot",
    sqlText = null, iterations = [], queryDir = null, session = {},
  } = opts;
  if (!iterations.length) return { sessionId: null, bestNodeId: null, nodeCount: 0 };

  const tId = taskId(benchmark, queryId, scaleFactor);
  const sId = sessionId(runId, queryId);
  const parents = reconstructParents(iterations, target);
  const rootNodeId = nodeId(sId, iterations[0]?.iteration ?? 0);

  await store.upsertTask({
    task_id: tId, benchmark, query_id: queryId, scale_factor: scaleFactor, sql_text: sqlText,
  });
  await store.openSession({
    session_id: sId, task_id: tId, run_id: runId, query_id: queryId,
    optimization_target: target, root_node_id: rootNodeId,
    model: session.model ?? null, agent_provider: session.agent_provider ?? null,
    hardware_fingerprint: session.hardware_fingerprint ?? null,
    status: session.status ?? "captured",
    started_at: session.started_at ?? null, completed_at: session.completed_at ?? null,
  });

  let bestNodeId = null, bestReward = Infinity, nodeCount = 0;
  for (const it of iterations) {
    const parentIter = parents.get(it.iteration);
    const reward = rewardOf(it, target);
    const iterDir = queryDir ? join(queryDir, `iter_${it.iteration}`) : null;
    const cppName = `${queryId.toLowerCase()}.cpp`;
    await store.recordNode({
      node_id: nodeId(sId, it.iteration), session_id: sId, task_id: tId,
      parent_node_id: parentIter == null ? null : nodeId(sId, parentIter),
      iteration: it.iteration, reward_ms: reward,
      hot_ms: it.hot_timing_ms ?? null, cold_ms: it.cold_timing_ms ?? null,
      validation: it.validation ?? null, improved: !!it.improved,
      categories: it.categories ?? [], strategy: it.strategy ?? null,
      operation_timings: it.operation_timings ?? null,
      cpp_path: iterDir && existsSync(join(iterDir, cppName)) ? join(iterDir, cppName) : null,
      plan_path: iterDir && existsSync(join(iterDir, "plan.json")) ? join(iterDir, "plan.json") : null,
      exec_results_path: iterDir && existsSync(join(iterDir, "execution_results.json")) ? join(iterDir, "execution_results.json") : null,
    });
    if ((it.validation ?? "pass") === "pass" && reward != null && reward < bestReward) {
      bestReward = reward; bestNodeId = nodeId(sId, it.iteration);
    }
    nodeCount++;
  }

  await store.closeSession(sId, { bestNodeId, status: session.status ?? "captured" });
  if (bestNodeId) await store.backpropReward(bestNodeId);
  return { sessionId: sId, bestNodeId, nodeCount };
}
