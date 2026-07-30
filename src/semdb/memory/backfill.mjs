/**
 * Offline L0/L1 population from completed run trees. No LLM, no skills.
 *
 * Two jobs:
 *  1. Cold start — a memory directory built from runs that already happened.
 *  2. Threshold validation — the reason this exists before any prompt changes.
 *     GenDB's own memory_report.json shows one template (`L1_tpch_Q9`) absorbing 8
 *     of 17 queries; a signature that collapses like that would hand queries each
 *     other's solvers. `--dry-run` prints the leave-one-out tier matrix so the
 *     collapse is visible before agents ever see an injected block.
 *
 * Usage:
 *   node src/semdb/memory/backfill.mjs --sembench-dir <dir> --runs <d1,d2,...> \
 *        [--memory-dir <dir>] [--dry-run] [--structural-min 0.55] [--exact-min 0.98]
 *
 * Each --runs entry is an orchestrator `--out` directory; its `<benchmark>-<query>`
 * subdirectories are the runs. Nested layouts (runs/, runs/ecomm_new/, ...) are
 * scanned one level deep.
 */

import { readFile, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { BENCHMARKS, SUPPORTED } from "../benchmarks.mjs";
import { aliasMap } from "../sql-features.mjs";
import { addEdge, updateIndex, writeNode } from "./graph.mjs";
import { initGraphDirs } from "./graph.mjs";
import {
  buildQuerySignature,
  planFeatures,
  signatureHash,
  signatureSimilarity,
} from "./signature.mjs";

const here = dirname(fileURLToPath(import.meta.url));

const DATA_BOUNDARY = {
  source: "select_validation",
  cert_accessed: false,
  full_ground_truth_accessed: false,
};

function parseArgs(argv) {
  const args = {
    memoryDir: null, sembenchDir: null, runs: [], dryRun: false,
    structuralMin: 0.55, exactMin: 0.98,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === "--memory-dir" && argv[i + 1]) args.memoryDir = resolve(argv[++i]);
    else if (a === "--sembench-dir" && argv[i + 1]) args.sembenchDir = resolve(argv[++i]);
    else if (a === "--runs" && argv[i + 1]) {
      args.runs = argv[++i].split(",").map((s) => resolve(s.trim())).filter(Boolean);
    } else if (a === "--dry-run") args.dryRun = true;
    else if (a === "--structural-min" && argv[i + 1]) args.structuralMin = Number(argv[++i]);
    else if (a === "--exact-min" && argv[i + 1]) args.exactMin = Number(argv[++i]);
  }
  if (!args.runs.length) throw new Error("backfill needs --runs <dir[,dir...]>");
  if (!args.dryRun && !args.memoryDir) {
    throw new Error("backfill needs --memory-dir unless --dry-run");
  }
  return args;
}

async function readJson(path) {
  try {
    return JSON.parse(await readFile(path, "utf8"));
  } catch {
    return null;
  }
}

/** Every `<benchmark>-<query>` directory under a run root, one level deep. */
async function findRunDirs(root) {
  const found = [];
  if (!existsSync(root)) return found;
  const match = (name) => {
    for (const bench of SUPPORTED) {
      if (name.startsWith(`${bench}-`)) return { benchmark: bench, query: name.slice(bench.length + 1) };
    }
    return null;
  };
  for (const entry of await readdir(root, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name.startsWith("_")) continue;
    const dir = resolve(root, entry.name);
    const hit = match(entry.name);
    if (hit) {
      found.push({ dir, ...hit });
      continue;
    }
    for (const sub of await readdir(dir, { withFileTypes: true }).catch(() => [])) {
      if (!sub.isDirectory() || sub.name.startsWith("_")) continue;
      const subHit = match(sub.name);
      if (subHit) found.push({ dir: resolve(dir, sub.name), ...subHit });
    }
  }
  return found;
}

/** SQL + NL for one query, from the SemBench tree. */
async function loadQueryText(sembenchDir, benchmark, query) {
  const cfg = BENCHMARKS[benchmark];
  if (!sembenchDir || !cfg) return null;
  const queryDir = resolve(sembenchDir, "files", benchmark, cfg.queryDir);
  const sqlPath = resolve(queryDir, `${query}.sql`);
  if (!existsSync(sqlPath)) return null;
  const sql = await readFile(sqlPath, "utf8");
  const nlRaw = await readJson(resolve(queryDir, "..", "natural_language", `${query}.json`));
  const nl = typeof nlRaw === "string"
    ? nlRaw
    : (nlRaw?.nl_question || nlRaw?.question || nlRaw?.description || null);
  return { sql, nl };
}

/** Best objective actually achieved, direction-aware. */
function bestObjective(telemetry) {
  const refine = telemetry?.refine || {};
  const name = refine.objective || "objective";
  const direction = refine.objective_direction === "minimize" ? "minimize" : "maximize";
  const values = (refine.objective_history || [])
    .map((entry) => Number(entry?.value))
    .filter((v) => Number.isFinite(v));
  if (!values.length) {
    const fallback = Number(telemetry?.metrics?.accuracy ?? telemetry?.metrics?.f1);
    return { name, value: Number.isFinite(fallback) ? fallback : null, direction };
  }
  const value = direction === "minimize" ? Math.min(...values) : Math.max(...values);
  return { name, value, direction };
}

/** Corpus modality, from the benchmark's table config. */
function corpusModality(benchmark, corpusTable) {
  const cfg = BENCHMARKS[benchmark] || {};
  const desc = cfg.tables?.[corpusTable];
  if (desc?.modality) return desc.modality;
  if ((cfg.imageTables || []).includes(corpusTable)) return "image";
  if (/image/i.test(corpusTable || "")) return "image";
  return "text";
}

/** Mechanically derive one L0 candidate from a finished run directory. */
async function readRun({ dir, benchmark, query }, sembenchDir) {
  const telemetry = await readJson(resolve(dir, "telemetry.json"));
  if (!telemetry) return { skipped: `${benchmark}-${query}: no telemetry.json` };
  const text = await loadQueryText(sembenchDir, benchmark, query);
  if (!text) return { skipped: `${benchmark}-${query}: query SQL not found` };

  const corpusTable = telemetry.corpus || null;
  const modality = corpusModality(benchmark, corpusTable);
  const tables = [...new Set(Object.values(aliasMap(text.sql, benchmark)))].map((t) => ({ table: t }));
  const signature = buildQuerySignature({
    query, sql: text.sql, nl: text.nl, benchmark,
    corpus: { table: corpusTable, modality },
    tables,
  });

  const objective = bestObjective(telemetry);
  const plan = await readJson(resolve(dir, "plan.json"));
  const helpers = resolve(dir, `_semantic_helpers_${query}.py`);
  const solver = resolve(dir, `solve_${query}.py`);
  const manifest = resolve(dir, "candidate_manifest.json");

  return {
    benchmark,
    query,
    dir,
    signature,
    objective,
    telemetry,
    node: {
      id: `L0_${benchmark}_${query.replace(/[^A-Za-z0-9]/g, "_")}_${signatureHash(signature)}`,
      layer: 0,
      signature_version: signature.signature_version,
      benchmark,
      signature,
      summary: `${benchmark} ${query} promoted candidate`
        + ` (${objective.name}=${objective.value ?? "n/a"})`,
      content: {
        query_id: query,
        sql: text.sql,
        nl: text.nl,
        modality,
        corpus_table: corpusTable,
        objective,
        metric_family: telemetry.metrics?.metric_family ?? null,
        iterations: telemetry.refine?.iterations ?? 0,
        replans: telemetry.replans ?? 0,
        action_counts: telemetry.optimizer_actions ?? null,
        plan_versions: telemetry.plan_versions ?? null,
        plan_features: plan ? planFeatures(plan) : null,
        promoted: {
          plan_path: existsSync(resolve(dir, "plan.json")) ? resolve(dir, "plan.json") : null,
          helpers_path: existsSync(helpers) ? helpers : null,
          solver_path: existsSync(solver) ? solver : null,
          manifest_path: existsSync(manifest) ? manifest : null,
          candidate_id: telemetry.best_candidate_id ?? null,
        },
        val: {
          mode: telemetry.validation?.mode ?? null,
          n: telemetry.validation?.sampled_ids ?? null,
        },
        source_run_dir: dir,
      },
      data_boundary: DATA_BOUNDARY,
      tags: [modality, signature.operator_kind, benchmark],
    },
  };
}

/** Group runs into templates; the best instance per template wins. */
function buildTemplates(runs, exactMin) {
  const templates = [];
  for (const run of runs) {
    let target = templates.find(
      (t) => t.benchmark === run.benchmark
        && signatureSimilarity(t.signature, run.signature) >= exactMin,
    );
    if (!target) {
      target = {
        benchmark: run.benchmark,
        signature: run.signature,
        id: `L1_${run.benchmark}_${run.signature.operator_kind}${run.signature.modality}`
          + `_${signatureHash(run.signature)}`,
        instances: [],
      };
      templates.push(target);
    }
    target.instances.push(run);
  }
  return templates;
}

/**
 * Leave-one-out matrix: for each query, its best match among templates built from
 * OTHER queries. This is the number that matters — a template matching its own
 * instance is trivially exact and says nothing about generalization.
 */
function tierMatrix(runs, { structuralMin, exactMin }) {
  const rows = [];
  const absorbed = new Map();
  for (const run of runs) {
    const others = runs.filter((r) => r !== run && r.benchmark === run.benchmark);
    let best = null;
    for (const other of others) {
      const score = signatureSimilarity(run.signature, other.signature);
      if (!best || score > best.score) best = { score, other };
    }
    const score = best?.score ?? 0;
    const tier = score >= exactMin ? "exact" : (score >= structuralMin ? "structural" : "novel");
    rows.push({ run, tier, score, match: best?.other ?? null });
    if (tier !== "novel" && best) {
      const key = `${best.other.benchmark}-${best.other.query}`;
      absorbed.set(key, (absorbed.get(key) || 0) + 1);
    }
  }
  return { rows, absorbed };
}

function report(runs, matrix, args) {
  const byBenchmark = new Map();
  for (const { run, tier, score, match } of matrix.rows) {
    if (!byBenchmark.has(run.benchmark)) byBenchmark.set(run.benchmark, []);
    byBenchmark.get(run.benchmark).push({ run, tier, score, match });
  }

  let worstShare = 0;
  for (const [benchmark, rows] of [...byBenchmark.entries()].sort()) {
    console.log(`\n=== ${benchmark} (${rows.length} queries) ===`);
    const counts = { exact: 0, structural: 0, novel: 0 };
    for (const row of rows.sort((a, b) => b.score - a.score)) {
      counts[row.tier] += 1;
      const target = row.match ? `${row.match.query}` : "-";
      console.log(`  ${row.run.query.padEnd(6)} ${row.tier.padEnd(10)}`
        + ` ${row.score.toFixed(2)}  nearest=${target}`
        + `  [${row.run.signature.modality}/${row.run.signature.operator_kind}]`);
    }
    console.log(`  tiers: ${counts.exact} exact, ${counts.structural} structural, ${counts.novel} novel`);

    // Concentration: how many of this benchmark's queries land on one nearest template.
    const local = new Map();
    for (const row of rows) {
      if (row.tier === "novel" || !row.match) continue;
      const key = row.match.query;
      local.set(key, (local.get(key) || 0) + 1);
    }
    const top = [...local.entries()].sort((a, b) => b[1] - a[1])[0];
    if (top) {
      const share = top[1] / rows.length;
      worstShare = Math.max(worstShare, share);
      console.log(`  most-absorbing template: ${top[0]} takes ${top[1]}/${rows.length}`
        + ` (${(share * 100).toFixed(0)}%)`);
    }
  }

  console.log("\n--- gate ---");
  const pass = worstShare <= 0.40;
  console.log(`worst single-template share: ${(worstShare * 100).toFixed(0)}% `
    + `(gate: <= 40%) → ${pass ? "PASS" : "FAIL"}`);
  return pass;
}

async function main() {
  const args = parseArgs(process.argv);

  const discovered = [];
  for (const root of args.runs) discovered.push(...(await findRunDirs(root)));
  // Deduplicate identical benchmark+query across run roots, keeping the better objective.
  const runs = [];
  const skipped = [];
  for (const found of discovered) {
    const parsed = await readRun(found, args.sembenchDir);
    if (parsed.skipped) {
      skipped.push(parsed.skipped);
      continue;
    }
    const existing = runs.findIndex(
      (r) => r.benchmark === parsed.benchmark && r.query === parsed.query,
    );
    if (existing < 0) {
      runs.push(parsed);
      continue;
    }
    const keepNew = parsed.objective.direction === "minimize"
      ? Number(parsed.objective.value ?? Infinity) < Number(runs[existing].objective.value ?? Infinity)
      : Number(parsed.objective.value ?? -Infinity) > Number(runs[existing].objective.value ?? -Infinity);
    if (keepNew) runs[existing] = parsed;
  }

  console.log(`[backfill] ${discovered.length} run dirs, ${runs.length} usable,`
    + ` ${skipped.length} skipped`);
  for (const reason of skipped.slice(0, 10)) console.log(`  skip: ${reason}`);
  if (skipped.length > 10) console.log(`  ... ${skipped.length - 10} more skipped`);
  if (!runs.length) return;

  const scored = runs.filter((r) => Number.isFinite(Number(r.objective.value)) && Number(r.objective.value) > 0);
  console.log(`[backfill] ${scored.length}/${runs.length} runs actually scored above zero;`
    + ` the rest are recorded but will not be used as references or warm starts.`);

  const matrix = tierMatrix(runs, args);
  const pass = report(runs, matrix, args);

  if (args.dryRun) {
    console.log("\n[backfill] --dry-run: nothing written.");
    process.exitCode = pass ? 0 : 1;
    return;
  }

  await initGraphDirs(args.memoryDir);
  const templates = buildTemplates(runs, args.exactMin);
  let nodes = 0;
  let edges = 0;
  for (const template of templates) {
    const best = template.instances.reduce((a, b) => (
      a.objective.direction === "minimize"
        ? (Number(b.objective.value ?? Infinity) < Number(a.objective.value ?? Infinity) ? b : a)
        : (Number(b.objective.value ?? -Infinity) > Number(a.objective.value ?? -Infinity) ? b : a)
    ));
    const features = best.node.content.plan_features;
    await writeNode({
      id: template.id,
      layer: 1,
      signature_version: template.signature.signature_version,
      benchmark: template.benchmark,
      signature: template.signature,
      summary: `${template.benchmark} ${template.signature.modality}`
        + ` ${template.signature.operator_kind} template`
        + ` (${template.instances.length} instance(s))`,
      content: {
        template_name: `${template.signature.modality}-${template.signature.operator_kind}`,
        // Backfill is mechanical: it can state the plan shape that worked, but it
        // must not invent strategies or anti-patterns. Those need the Memory
        // Manager's differential analysis, which reads iteration trajectories.
        proven_strategies: [],
        anti_patterns: [],
        plan_skeleton: features
          ? {
            sampling_unit: features.sampling_units?.[0] ?? template.signature.sampling_unit,
            helper_names: features.helper_names ?? [],
            primitives: features.primitives ?? [],
            relational_ops: features.relational_ops ?? [],
          }
          : null,
        best_instance_id: best.node.id,
        instance_count: template.instances.length,
        instance_queries: template.instances.map((r) => r.query),
        metric_families: [...new Set(
          template.instances.map((r) => r.node.content.metric_family).filter(Boolean),
        )],
        origin: "backfill",
      },
      data_boundary: DATA_BOUNDARY,
      tags: [template.signature.modality, template.signature.operator_kind, template.benchmark],
    }, args.memoryDir);
    nodes += 1;

    for (const instance of template.instances) {
      await writeNode(instance.node, args.memoryDir);
      nodes += 1;
      if (await addEdge(
        { source: instance.node.id, target: template.id, type: "instance_of" },
        args.memoryDir,
      )) edges += 1;
    }
  }
  const index = await updateIndex(args.memoryDir);
  console.log(`\n[backfill] wrote ${nodes} nodes, ${edges} edges → ${args.memoryDir}`);
  console.log(`[backfill] index: ${index.stats.total} nodes (`
    + `${Object.entries(index.stats.by_layer).filter(([, c]) => c).map(([l, c]) => `${l}:${c}`).join(", ")})`);
}

const invokedDirectly = process.argv[1]
  && basename(process.argv[1]) === "backfill.mjs";
if (invokedDirectly) {
  main().catch((error) => {
    console.error(`[backfill] failed: ${error.message}`);
    process.exit(1);
  });
}

export { buildTemplates, findRunDirs, readRun, tierMatrix };
