#!/usr/bin/env node
/**
 * Backfill the Experience Graph (PostgreSQL) from existing GenDB runs.
 *
 * Scans output/** for run.json + queries/<Q>/optimization_history.json and
 * reconstructs the branching search tree the linear history flattens away
 * (same ingest path as live capture).
 *
 * Usage:
 *   node scripts/backfill_experience.mjs [--root <output-dir>] [--pg-url <conn>] [--dry-run]
 *
 * Connection: --pg-url, or env GENDB_EXPERIENCE_PG, or standard libpq env vars.
 */

import { readFileSync, existsSync, readdirSync, statSync } from "node:fs";
import { resolve, dirname, basename, join } from "node:path";
import { fileURLToPath } from "node:url";
import { openExperienceStore } from "../src/gendb/experience/index.mjs";
import { ingestHistory } from "../src/gendb/experience/ingest.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, "..");

function parseArgs(argv) {
  const args = { root: resolve(REPO_ROOT, "output"), dryRun: false, pgUrl: null };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === "--root" && argv[i + 1]) args.root = resolve(argv[++i]);
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
  const runRoot = dirname(dirname(dirname(historyPath)));
  const rj = join(runRoot, "run.json");
  return existsSync(rj) ? { runJsonPath: rj, runRoot } : { runJsonPath: null, runRoot };
}

async function main() {
  const args = parseArgs(process.argv);
  console.log(`[backfill] root=${args.root} dryRun=${args.dryRun}`);

  const histories = findHistories(args.root);
  console.log(`[backfill] found ${histories.length} optimization_history.json files`);

  let store = null;
  if (!args.dryRun) {
    store = await openExperienceStore({ connectionString: args.pgUrl });
    console.log(`[backfill] backend=${store.backend} vector=${store.vec ? "pgvector HNSW" : "none"}`);
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

    let sqlText = null;
    const templatePath = join(dirname(histPath), "template.sql");
    if (existsSync(templatePath)) { try { sqlText = readFileSync(templatePath, "utf-8"); } catch {} }

    if (store) {
      const res = await ingestHistory(store, {
        benchmark, scaleFactor, runId, queryId, target, sqlText,
        iterations, queryDir: dirname(histPath),
        session: { model: run.model, agent_provider: run.agentProvider, status: "backfilled",
                   started_at: run.startedAt, completed_at: run.completedAt },
      });
      nNodes += res.nodeCount;
      summaries.push({ session: res.sessionId, iters: iterations.length, best: res.bestNodeId });
    } else {
      nNodes += iterations.length;
      summaries.push({ session: `${runId}__${queryId}`, iters: iterations.length, best: null });
    }
    nSessions++;
  }

  const nRuns = new Set(summaries.map((s) => (s.session || "").split("__")[0])).size;
  console.log(`[backfill] runs=${nRuns} sessions=${nSessions} nodes=${nNodes} skipped=${nSkipped}`);
  if (args.dryRun) {
    for (const s of summaries.slice(0, 20)) console.log(`  ${s.session}: ${s.iters} iters`);
    if (summaries.length > 20) console.log(`  … and ${summaries.length - 20} more`);
  }
  if (store) await store.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
