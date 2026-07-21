/**
 * Ingest existing GenDB runs into the Experience Graph.
 *
 * Walks a run's output directory, finds every query's optimization_history.json,
 * and materializes the full search trajectory as a queryable experience graph
 * (see experience-graph.mjs). This backfills the raw search evidence that the
 * current pipeline distills-then-discards.
 *
 * Usage:
 *   node src/gendb/memory/ingest-experience.mjs \
 *     --run output/tpc-h-sf10-3.19/runs/2026-03-19T23-08-43 \
 *     --benchmark tpc-h --scale-factor 10 \
 *     [--memory-dir gendb-memory]
 *
 *   # or point straight at a directory tree containing optimization_history.json:
 *   node src/gendb/memory/ingest-experience.mjs --scan output/deprecated/sec-edgar
 */

import { readFile, readdir } from "fs/promises";
import { resolve } from "path";
import { existsSync } from "fs";
import {
  ingestOptimizationHistory,
  getExperienceGraphSummary,
  findSimilarTasks,
} from "./experience-graph.mjs";

function parseArgs(argv) {
  const args = { memoryDir: "gendb-memory", benchmark: "unknown", scaleFactor: 0 };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === "--run" && argv[i + 1]) args.run = argv[++i];
    else if (argv[i] === "--scan" && argv[i + 1]) args.scan = argv[++i];
    else if (argv[i] === "--memory-dir" && argv[i + 1]) args.memoryDir = argv[++i];
    else if (argv[i] === "--benchmark" && argv[i + 1]) args.benchmark = argv[++i];
    else if (argv[i] === "--scale-factor" && argv[i + 1]) args.scaleFactor = Number(argv[++i]);
  }
  return args;
}

/** Recursively find optimization_history.json files under a root. */
async function findHistories(root, acc = []) {
  let entries;
  try {
    entries = await readdir(root, { withFileTypes: true });
  } catch {
    return acc;
  }
  for (const e of entries) {
    const p = resolve(root, e.name);
    if (e.isDirectory()) await findHistories(p, acc);
    else if (e.name === "optimization_history.json") acc.push(p);
  }
  return acc;
}

async function main() {
  const args = parseArgs(process.argv);
  const root = args.run || args.scan;
  if (!root) {
    console.error("Provide --run <dir> or --scan <dir>");
    process.exit(1);
  }

  const histories = await findHistories(root);
  if (histories.length === 0) {
    console.error(`No optimization_history.json found under ${root}`);
    process.exit(1);
  }
  console.log(`Found ${histories.length} query histories under ${root}`);

  let ingested = 0;
  for (const hp of histories) {
    try {
      const history = JSON.parse(await readFile(hp, "utf-8"));
      const queryId = history.query_id || "Q?";
      const artifactDir = resolve(hp, "..");
      // Try to read the query SQL if it sits next to the history.
      let sql = "";
      const sqlCandidates = ["query.sql", `${queryId}.sql`, "input.sql"];
      for (const c of sqlCandidates) {
        const sp = resolve(artifactDir, c);
        if (existsSync(sp)) { sql = await readFile(sp, "utf-8"); break; }
      }

      const res = await ingestOptimizationHistory(
        args.memoryDir,
        history,
        {
          benchmark: args.benchmark,
          scale_factor: args.scaleFactor,
          sql,
          query_id: queryId,
          artifactDir,
        }
      );
      const bestMs = res.best?.eval_evidence?.timing_ms;
      console.log(
        `  ${queryId}: ${res.nodeCount} attempts → best ${bestMs != null ? Math.round(bestMs) + "ms" : "n/a"} (reward ${res.best?.reward?.toFixed(2) ?? "n/a"})`
      );
      ingested++;
    } catch (err) {
      console.warn(`  Skipped ${hp}: ${err.message}`);
    }
  }

  console.log(`\nIngested ${ingested}/${histories.length} query histories.`);
  console.log(await getExperienceGraphSummary(args.memoryDir));

  // Demonstrate cross-session retrieval on the last ingested spec.
  const sample = await findSimilarTasks(args.memoryDir, { benchmark: args.benchmark, scale_factor: args.scaleFactor, sql: "select sum from lineitem group by order" }, { topK: 3, minSim: 0 });
  if (sample.length) {
    console.log("\nCross-session similarity probe (top matches for a sample spec):");
    for (const s of sample) console.log(`  ${s.task.task_id}  sim=${s.similarity.toFixed(2)}`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
