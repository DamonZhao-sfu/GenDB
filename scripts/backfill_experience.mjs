#!/usr/bin/env node
/**
 * Backfill the Experience Graph from existing GenDB runs.
 *
 * Scans output/** for run.json + queries/<Q>/optimization_history.json and
 * reconstructs the branching search tree that the linear history flattens away.
 *
 * PARENT RULE (mirrors the live optimize loop, orchestrator.mjs:1523/1612 —
 * "Copy best code as starting point"): the parent of iteration N is the
 * best-so-far iteration at the moment N started.
 *
 * Usage:
 *   node scripts/backfill_experience.mjs [--root <output-dir>] [--dry-run]
 *                                        [--backend sqlite|postgres]
 *                                        [--out <dir>] [--pg-url <conn>]
 *
 * Defaults: --backend sqlite  --out gendb-experience/  --root output/
 *   Postgres: --backend postgres --pg-url postgres://... (or env GENDB_EXPERIENCE_PG)
 */

import { readFileSync, existsSync, readdirSync, statSync } from "node:fs";
import { resolve, dirname, basename, join } from "node:path";
import { fileURLToPath } from "node:url";
import { openExperienceStore, taskId, sessionId, nodeId } from "../src/gendb/experience/index.mjs";
import { rewardOf, reconstructParents } from "../src/gendb/experience/reconstruct.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, "..");

function parseArgs(argv) {
  const args = {
    out: resolve(REPO_ROOT, "gendb-experience"), root: resolve(REPO_ROOT, "output"),
    dryRun: false, backend: null, pgUrl: null,
  };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === "--out" && argv[i + 1]) args.out = resolve(argv[++i]);
    else if (argv[i] === "--root" && argv[i + 1]) args.root = resolve(argv[++i]);
    else if (argv[i] === "--backend" && argv[i + 1]) args.backend = argv[++i];
    else if (argv[i] === "--pg-url" && argv[i + 1]) args.pgUrl = argv[++i];
    else if (argv[i] === "--dry-run") args.dryRun = true;
  }
  return args;
}

function findHistories(root) {
  const out = [];
  const walk = (dir) => {
    let entries;
    try { entries = readdirSync(dir); } catch { return; }
    for (const e of entries) {
      const p = join(dir, e);
      let st;
      try { st = statSync(p); } catch { continue; }
      if (st.isDirectory()) walk(p);
      else if (e === "optimization_history.json") out.push(p);
    }
  };
  walk(root);
  return out;
}

const readJson = (p) => JSON.parse(readFileSync(p, "utf-8"));

function findRunJson(historyPath) {
  const qDir = dirname(historyPath);
  const runRoot = dirname(dirname(qDir));
  const rj = join(runRoot, "run.json");
  return existsSync(rj) ? { runJsonPath: rj, runRoot } : { runJsonPath: null, runRoot };
}

async function main() {
  const args = parseArgs(process.argv);
  const backend = args.backend || (args.pgUrl ? "postgres" : "sqlite");
  console.log(`[backfill] root=${args.root} backend=${backend} out=${args.out} dryRun=${args.dryRun}`);

  const histories = findHistories(args.root);
  console.log(`[backfill] found ${histories.length} optimization_history.json files`);

  let store = null;
  if (!args.dryRun) {
    store = await openExperienceStore({ backend, dir: args.out, connectionString: args.pgUrl });
    console.log(`[backfill] backend=${store.backend} vector=${store.vec ? "ann/knn" : "none"}`);
  }

  let nSessions = 0, nNodes = 0, nSkipped = 0;
  const summaries = [];

  for (const histPath of histories) {
    let hist;
    try { hist = readJson(histPath); } catch { nSkipped++; continue; }
    const iterations = hist.iterations || [];
    if (iterations.length === 0) { nSkipped++; continue; }

    const { runJsonPath, runRoot } = findRunJson(histPath);
    let run = {};
    if (runJsonPath) { try { run = readJson(runJsonPath); } catch {} }

    const benchmark = run.workload || basename(dirname(dirname(runRoot))) || "unknown";
    const scaleFactor = run.scaleFactor ?? 0;
    const runId = run.runId || basename(runRoot);
    const target = run.optimizationTarget || "hot";
    const queryId = hist.query_id || basename(dirname(histPath));

    const tId = taskId(benchmark, queryId, scaleFactor);
    const sId = sessionId(runId, queryId);

    let sqlText = null;
    const templatePath = join(dirname(histPath), "template.sql");
    if (existsSync(templatePath)) { try { sqlText = readFileSync(templatePath, "utf-8"); } catch {} }

    const parents = reconstructParents(iterations, target);
    const rootNodeId = nodeId(sId, iterations[0]?.iteration ?? 0);

    if (store) {
      await store.upsertTask({ task_id: tId, benchmark, query_id: queryId, scale_factor: scaleFactor, sql_text: sqlText });
      await store.openSession({
        session_id: sId, task_id: tId, run_id: runId, query_id: queryId,
        model: run.model, agent_provider: run.agentProvider, optimization_target: target,
        hardware_fingerprint: run.hardwareFingerprint ?? null, root_node_id: rootNodeId,
        status: "backfilled", started_at: run.startedAt, completed_at: run.completedAt,
      });
    }

    let bestNodeId = null, bestReward = Infinity;
    for (const it of iterations) {
      const parentIter = parents.get(it.iteration);
      const reward = rewardOf(it, target);
      const iterDir = join(dirname(histPath), `iter_${it.iteration}`);
      if (store) {
        await store.recordNode({
          node_id: nodeId(sId, it.iteration), session_id: sId, task_id: tId,
          parent_node_id: parentIter == null ? null : nodeId(sId, parentIter),
          iteration: it.iteration, reward_ms: reward,
          hot_ms: it.hot_timing_ms ?? null, cold_ms: it.cold_timing_ms ?? null,
          validation: it.validation ?? null, improved: !!it.improved,
          categories: it.categories ?? [], strategy: it.strategy ?? null,
          operation_timings: it.operation_timings ?? null,
          cpp_path: existsSync(join(iterDir, `${queryId.toLowerCase()}.cpp`)) ? join(iterDir, `${queryId.toLowerCase()}.cpp`) : null,
          plan_path: existsSync(join(iterDir, "plan.json")) ? join(iterDir, "plan.json") : null,
          exec_results_path: existsSync(join(iterDir, "execution_results.json")) ? join(iterDir, "execution_results.json") : null,
        });
      }
      if ((it.validation ?? "pass") === "pass" && reward != null && reward < bestReward) {
        bestReward = reward; bestNodeId = nodeId(sId, it.iteration);
      }
      nNodes++;
    }

    if (store) {
      await store.closeSession(sId, { bestNodeId, status: "backfilled" });
      if (bestNodeId) await store.backpropReward(bestNodeId);
    }
    nSessions++;
    summaries.push({ session: sId, iters: iterations.length, best: bestNodeId, bestMs: Number.isFinite(bestReward) ? bestReward : null });
  }

  const nRuns = new Set(summaries.map((s) => s.session.split("__")[0])).size;
  console.log(`[backfill] runs=${nRuns} sessions=${nSessions} nodes=${nNodes} skipped=${nSkipped}`);
  if (args.dryRun) {
    for (const s of summaries.slice(0, 20)) {
      console.log(`  ${s.session}: ${s.iters} iters, best=${s.best} (${s.bestMs != null ? Math.round(s.bestMs) + "ms" : "n/a"})`);
    }
    if (summaries.length > 20) console.log(`  … and ${summaries.length - 20} more`);
  }
  if (store) { await store.close(); if (backend === "sqlite") console.log(`[backfill] wrote ${resolve(args.out, "experience.db")}`); }
}

main().catch((e) => { console.error(e); process.exit(1); });
