#!/usr/bin/env node
/**
 * Backfill a unified per-query results CSV from an existing GenDB run — without
 * re-running the pipeline. Reads the run's telemetry.json (for per-query
 * generation cost/wall time and run metadata) and each
 * queries/<Qid>/iter_<N>/execution_results.json (for timing + runtime phase
 * breakdown), and emits the same results.csv the orchestrator now writes.
 *
 * Usage:
 *   node scripts/results-to-csv.mjs <run_dir> [--model <name>] \
 *        [--benchmark <b>] [--sf <n>] [--out <path>]
 *
 *   <run_dir>  a run audit directory, e.g.
 *              output/tpc-h-sf10/runs/2026-07-20T23-31-28
 *
 * benchmark/sf/model are taken from telemetry.json when present, then from the
 * run_dir path (workload dir "<benchmark>-sf<N>"), then from CLI flags.
 * The CSV is written to <run_dir>/results.csv unless --out is given.
 */

import { readFileSync, readdirSync, existsSync, writeFileSync } from "fs";
import { resolve, basename, dirname } from "path";

function parseArgs(argv) {
  const out = { runDir: null, model: null, benchmark: null, sf: null, out: null };
  const rest = [];
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--model" && argv[i + 1]) out.model = argv[++i];
    else if (a === "--benchmark" && argv[i + 1]) out.benchmark = argv[++i];
    else if (a === "--sf" && argv[i + 1]) out.sf = argv[++i];
    else if (a === "--out" && argv[i + 1]) out.out = argv[++i];
    else rest.push(a);
  }
  out.runDir = rest[0] || null;
  return out;
}

function readJSON(path) {
  try { return JSON.parse(readFileSync(path, "utf-8")); } catch { return null; }
}

const csvEscape = (v) => {
  const s = v === null || v === undefined ? "" : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

/** Natural numeric sort for query ids like Q1, Q2, ..., Q10, Q22. */
function queryIdSort(a, b) {
  const na = parseInt(String(a).replace(/\D/g, ""), 10);
  const nb = parseInt(String(b).replace(/\D/g, ""), 10);
  if (Number.isNaN(na) || Number.isNaN(nb)) return String(a).localeCompare(String(b));
  return na - nb;
}

function main() {
  const args = parseArgs(process.argv);
  if (!args.runDir) {
    console.error("Usage: node scripts/results-to-csv.mjs <run_dir> [--model <name>] [--benchmark <b>] [--sf <n>] [--out <path>]");
    process.exit(1);
  }
  const runDir = resolve(args.runDir);
  if (!existsSync(runDir)) {
    console.error(`Run directory not found: ${runDir}`);
    process.exit(1);
  }

  const telemetry = readJSON(resolve(runDir, "telemetry.json")) || {};

  // Resolve metadata: telemetry.json > path > CLI flags.
  // Path shape: .../output/<benchmark>-sf<N>/runs/<timestamp>
  let benchmark = telemetry.benchmark || args.benchmark || null;
  let sf = telemetry.scale_factor != null ? telemetry.scale_factor : args.sf;
  const workloadDirName = basename(dirname(dirname(runDir))); // "<benchmark>-sf<N>"
  const m = workloadDirName.match(/^(.*)-sf(\d+)$/);
  if (m) {
    if (!benchmark) benchmark = m[1];
    if (sf == null) sf = m[2];
  }
  const model = telemetry.model || args.model || "unknown";
  const provider = telemetry.provider || "unknown";
  const runId = basename(runDir);

  // Locate the queries directory (run audit dir, or a workload dir fallback).
  let queriesDir = resolve(runDir, "queries");
  if (!existsSync(queriesDir)) queriesDir = resolve(dirname(dirname(runDir)), "queries");
  if (!existsSync(queriesDir)) {
    console.error(`No queries/ directory found under ${runDir}`);
    process.exit(1);
  }

  const queryIds = readdirSync(queriesDir)
    .filter((d) => existsSync(resolve(queriesDir, d)) && /^Q?\d+/i.test(d))
    .sort(queryIdSort);

  const columns = [
    "benchmark", "sf", "model", "provider", "run_id", "query_id", "status",
    "best_time_ms", "best_iter", "num_iterations", "per_iter_ms",
    "gen_wall_ms", "gen_cost_usd", "phase_breakdown",
  ];
  const rows = [columns.join(",")];

  let sumBest = 0;
  for (const qid of queryIds) {
    const qDir = resolve(queriesDir, qid);
    // Collect iterations by index.
    const iters = new Map();
    for (const d of readdirSync(qDir)) {
      const im = d.match(/^iter_(\d+)$/);
      if (!im) continue;
      const exec = readJSON(resolve(qDir, d, "execution_results.json"));
      if (!exec) continue;
      iters.set(parseInt(im[1], 10), {
        timing_ms: exec.timing_ms,
        validation: exec.validation?.status || "-",
        operation_timings: exec.hot_operation_timings || exec.operation_timings || null,
      });
    }
    if (iters.size === 0) continue;

    const maxIter = Math.max(...iters.keys()) + 1;
    let bestTime = null, bestIter = -1, bestOps = null;
    const perIter = [];
    let anyFail = false, count = 0;
    for (let i = 0; i < maxIter; i++) {
      const it = iters.get(i);
      if (!it) { perIter.push(""); continue; }
      count++;
      perIter.push(it.timing_ms != null ? Math.round(it.timing_ms) : "");
      if (it.validation === "fail") anyFail = true;
      if (it.validation === "pass" && it.timing_ms != null && (bestTime === null || it.timing_ms < bestTime)) {
        bestTime = it.timing_ms; bestIter = i; bestOps = it.operation_timings || null;
      }
    }

    const phaseObj = telemetry.phases?.[`query_${qid}`];
    const genWall = phaseObj?.total_ms;
    const genCost = phaseObj
      ? Object.values(phaseObj.agents || {}).reduce((s, a) => s + (a.cost_usd || 0), 0)
      : null;
    const phaseBreakdown = bestOps
      ? Object.entries(bestOps).map(([k, v]) => `${k}=${v}`).join(";")
      : "";
    const status = bestTime !== null ? "PASS" : (anyFail ? "FAIL" : "-");
    if (bestTime !== null) sumBest += bestTime;

    rows.push([
      benchmark, sf, model, provider, runId, qid, status,
      bestTime !== null ? Math.round(bestTime) : "",
      bestIter >= 0 ? bestIter : "",
      count,
      perIter.join(";"),
      genWall != null ? Math.round(genWall) : "",
      genCost != null ? genCost.toFixed(4) : "",
      phaseBreakdown,
    ].map(csvEscape).join(","));
  }

  rows.push([
    benchmark, sf, model, provider, runId, "TOTAL", "",
    Math.round(sumBest), "", "", "",
    telemetry.total_wall_clock_ms != null ? Math.round(telemetry.total_wall_clock_ms) : "",
    telemetry.total_cost_usd != null ? telemetry.total_cost_usd.toFixed(4) : "",
    "",
  ].map(csvEscape).join(","));

  const outPath = args.out ? resolve(args.out) : resolve(runDir, "results.csv");
  writeFileSync(outPath, rows.join("\n") + "\n");
  console.log(`Wrote ${queryIds.length} queries to ${outPath}`);
}

main();
