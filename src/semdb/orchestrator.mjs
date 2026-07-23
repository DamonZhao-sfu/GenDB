/**
 * SemDB Orchestrator — compile SemBench semantic operators into relational
 * programs via three agents, reusing GenDB's provider/agent plumbing.
 *
 * The work is amortized PER CORPUS, not per query:
 *   Phase A  Schema Designer : ONCE per corpus, seeing ALL its queries  -> _corpus/<c>/schema.json
 *   Phase B  Extractor       : ONCE per corpus (small model)            -> _corpus/<c>/<c>_attrs.json
 *   Phase C  Code Generator  : PER query (reuses the corpus schema+attrs)-> compiled_<q>.py
 *   Execute + evaluate       : per query -> results, telemetry.json, results.csv (P/R/F1)
 *
 * Queries are grouped by their extract-side "corpus" (e.g. all logo joins share
 * the images corpus; q3a–g share the movie-text corpus), so the expensive schema
 * design + extraction run once and every query over that corpus reuses them.
 *
 * Usage:
 *   node src/semdb/orchestrator.mjs \
 *        [--query q3a] \                      # omit to process ALL *.sql in --query-dir
 *        --query-dir /.../mmqa/query/bigquery \
 *        --data-dir  /.../mmqa/data/sf_200 \
 *        --ground-truth-dir /.../mmqa/raw_results/ground_truth \
 *        [--endpoint http://localhost:8000/v1] [--force] [--dry-run]
 *
 * --ground-truth-dir (or --run) executes extraction + compiled query + scoring.
 * --force re-runs the cached corpus schema/extraction. --dry-run just prints the
 * rendered prompts. GPU-free demos live in ./poc/ and ./examples/.
 */

import { readFile, writeFile, mkdir, readdir } from "fs/promises";
import { existsSync } from "fs";
import { resolve, dirname, basename } from "path";
import { fileURLToPath } from "url";
import { spawnSync } from "child_process";

import {
  renderTemplate,
  runAgent,
  readJSON,
  setAgentProvider,
} from "../gendb/shared.mjs";
import { defaults, getAgentModel, getAgentEffort } from "./semdb.config.mjs";
import { BENCHMARKS, SUPPORTED, tableDesc, tableFile } from "./benchmarks.mjs";
import { config as schemaDesignerConfig } from "./agents/schema-designer/index.mjs";
import { config as extractorConfig } from "./agents/extractor/index.mjs";

/** SQL table-qualifier prefix for a benchmark (e.g. mmqa, cars_dataset). */
function benchPrefix(bench) { return (BENCHMARKS[bench] && BENCHMARKS[bench].prefix) || bench; }
function prefixRe(bench, tail) {
  return new RegExp(String.raw`\b${benchPrefix(bench)}\.(\w+)` + (tail || ""), "gi");
}
import { config as codeGeneratorConfig } from "./agents/code-generator/index.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
  const args = {
    query: null,
    benchmark: null,     // inferred from --query-dir/--data-dir path if not given
    querySource: defaults.querySource,
    scaleFactor: null,   // sf_<N> data subdir (null = flat data/)
    sembenchDir: null,   // repo root; query-dir/data-dir derived from it if given
    queryDir: null,      // .../files/<b>/query/<dialect>
    dataDir: null,       // .../files/<b>/data/<sf>
    imageDir: null,      // .../data/<sf>/images  (defaults to <dataDir>/images)
    out: resolve(__dirname, "runs"),
    agentProvider: defaults.agentProvider,
    modelOverride: null,   // force one model for all agents (testing)
    groundTruthDir: null,  // SemBench raw_results/ground_truth
    telemetryCsv: null,    // append the telemetry+metrics row here
    // Execution of the downstream Python steps (extract → compiled query → eval).
    run: false,            // force-run; auto-enabled when --ground-truth-dir is set
    noRun: false,          // disable auto-run
    endpoint: null,        // vLLM/OpenAI base URL for extraction (+ residual)
    apiKey: "EMPTY",
    concurrency: 8,        // in-flight extract.py --endpoint requests (vLLM batches server-side)
    extractModel: null,    // small VLM/LLM id for TEXT corpora (config extraction.small*Model)
    clipModel: null,       // CLIP id for IMAGE corpora (config extraction.clipModel)
    theta: null,
    predCols: "0,1",       // predicted columns to compare vs GT tuple order
    force: false,          // re-run corpus schema design + extraction even if cached
    dryRun: false,
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--query" && argv[i + 1]) args.query = argv[++i];
    else if (a === "--benchmark" && argv[i + 1]) args.benchmark = argv[++i];
    else if (a === "--agent-provider" && argv[i + 1]) args.agentProvider = argv[++i];
    else if (a === "--model" && argv[i + 1]) args.modelOverride = argv[++i];
    else if (a === "--query-source" && argv[i + 1]) args.querySource = argv[++i];
    else if (a === "--sf" && argv[i + 1]) args.scaleFactor = argv[++i];
    else if (a === "--sembench-dir" && argv[i + 1]) args.sembenchDir = resolve(argv[++i]);
    else if (a === "--query-dir" && argv[i + 1]) args.queryDir = resolve(argv[++i]);
    else if (a === "--data-dir" && argv[i + 1]) args.dataDir = resolve(argv[++i]);
    else if (a === "--image-dir" && argv[i + 1]) args.imageDir = resolve(argv[++i]);
    else if (a === "--ground-truth-dir" && argv[i + 1]) args.groundTruthDir = resolve(argv[++i]);
    else if (a === "--telemetry-csv" && argv[i + 1]) args.telemetryCsv = resolve(argv[++i]);
    else if (a === "--out" && argv[i + 1]) args.out = resolve(argv[++i]);
    else if (a === "--run") args.run = true;
    else if (a === "--no-run") args.noRun = true;
    else if (a === "--endpoint" && argv[i + 1]) args.endpoint = argv[++i];
    else if (a === "--api-key" && argv[i + 1]) args.apiKey = argv[++i];
    else if (a === "--concurrency" && argv[i + 1]) args.concurrency = parseInt(argv[++i], 10);
    else if (a === "--extract-model" && argv[i + 1]) args.extractModel = argv[++i];
    else if (a === "--clip-model" && argv[i + 1]) args.clipModel = argv[++i];
    else if (a === "--theta" && argv[i + 1]) args.theta = argv[++i];
    else if (a === "--pred-cols" && argv[i + 1]) args.predCols = argv[++i];
    else if (a === "--force") args.force = true;
    else if (a === "--dry-run") args.dryRun = true;
  }
  // Infer benchmark / sembench root / scale-factor from explicit dir paths, so
  // `--data-dir .../files/cars/data/sf_9836` works WITHOUT --benchmark/--sf.
  const infer = (p) => {
    const m = p && p.match(/\/files\/([^/]+)(?:\/|$)/);
    return m && SUPPORTED.includes(m[1]) ? m[1] : null;
  };
  if (!args.benchmark) args.benchmark = infer(args.queryDir) || infer(args.dataDir) || defaults.benchmark;
  if (!args.sembenchDir) {
    const src = args.dataDir || args.queryDir || "";
    const k = src.indexOf("/files/");
    if (k > 0) args.sembenchDir = src.slice(0, k);
  }
  if (!args.scaleFactor && args.dataDir) {
    const m = basename(args.dataDir).match(/^sf_(.+)$/);
    if (m) args.scaleFactor = m[1];
  }
  const b = BENCHMARKS[args.benchmark];
  // Derive dirs from the sembench root + benchmark config when not given explicitly.
  if (args.sembenchDir) {
    const base = resolve(args.sembenchDir, "files", args.benchmark);
    args.queryDir = args.queryDir || resolve(base, b ? b.queryDir : `query/${args.querySource}`);
    if (!args.dataDir) {
      args.dataDir = (b && b.dataLayout === "sf" && args.scaleFactor)
        ? resolve(base, "data", `sf_${args.scaleFactor}`)
        : resolve(base, "data");
    }
    if (!args.groundTruthDir && b) args.groundTruthDir = resolve(base, b.gtDir);
  }
  // Image root varies by scenario (imageBase): under <dataDir> ("sf"), at the
  // sembench root for repo-relative path columns ("root"), or absolute ("absolute").
  if (args.dataDir && !args.imageDir) {
    const ib = b ? b.imageBase : "sf";
    if (ib === "absolute") args.imageDir = null;                 // path column is absolute
    else if (ib === "root") args.imageDir = args.sembenchDir || null;
    else args.imageDir = resolve(args.dataDir, (b && b.imageRoot) || "images");
  }
  return args;
}

/** Load a SemBench query's SQL from the query folder, and its NL intent if present. */
async function loadQuery(args) {
  if (!args.queryDir) {
    return {
      sql: "-- (no --query-dir given; built-in Q7 airline-logo example)\n" +
        'SELECT t.Airlines, i.uri\nFROM mmqa.tampa_international_airport t, mmqa.images i\n' +
        'WHERE AI.IF(STRUCT("...is this the logo of the airline?...", t.Airlines, i.uri));',
      nl: "Join airlines to images where the image shows that airline's logo.",
    };
  }
  const sqlPath = resolve(args.queryDir, `${args.query}.sql`);
  const sql = await readFile(sqlPath, "utf-8");
  // NL intent lives a sibling folder over: .../query/natural_language/<q>.json
  const nlPath = resolve(args.queryDir, "..", "natural_language", `${args.query}.json`);
  const nlRaw = existsSync(nlPath) ? await readJSON(nlPath) : null;
  const nl = nlRaw?.question || nlRaw?.nl || nlRaw?.description ||
    (nlRaw ? JSON.stringify(nlRaw) : "(no natural_language/*.json found)");
  return { sql, nl };
}

/** Count data rows (excluding header) in a CSV. Returns null if unreadable. */
async function countRows(path) {
  try {
    const lines = (await readFile(path, "utf-8")).split(/\r?\n/).filter((l) => l.trim().length);
    return Math.max(0, lines.length - 1);
  } catch {
    return null;
  }
}

/** Read the header row of every CSV table in the data dir → a schema summary. */
async function readTableHeaders(dataDir) {
  if (!dataDir || !existsSync(dataDir)) {
    return "(no --data-dir given; provide it so the Designer sees real columns)";
  }
  const entries = await readdir(dataDir);
  const lines = [];
  for (const name of entries.filter((n) => n.toLowerCase().endsWith(".csv")).sort()) {
    try {
      const head = (await readFile(resolve(dataDir, name), "utf-8")).split(/\r?\n/, 1)[0];
      lines.push(`- ${name}: ${head}`);
    } catch { /* skip unreadable */ }
  }
  const hasImages = existsSync(resolve(dataDir, "images"));
  if (hasImages) lines.push(`- images/: (directory of image files)`);
  return lines.length ? lines.join("\n") : "(no CSV tables found in data dir)";
}

/** Resolve a SemBench ground-truth file (Q2a.json for query q2a) + its row count. */
async function resolveGroundTruth(gtDir, query, sf) {
  if (!gtDir || !query) return null;
  const n = query.replace(/^[qQ]/, "");
  const caps = [`${query}`, `Q${n}`, `q${n}`, query.toUpperCase(),
    `${query[0].toUpperCase()}${query.slice(1)}`];
  // JSON (mmqa) first, then CSV (every other scenario), incl. scale-suffixed CSV.
  const json = caps.map((c) => `${c}.json`);
  const csv = caps.flatMap((c) => sf ? [`${c}_${sf}.csv`, `${c}.csv`] : [`${c}.csv`]);
  for (const name of [...json, ...csv]) {
    const p = resolve(gtDir, name);
    if (!existsSync(p)) continue;
    if (name.endsWith(".json")) {
      const gt = await readJSON(p);
      const rows = Array.isArray(gt?.ground_truth) ? gt.ground_truth.length
        : (Array.isArray(gt) ? gt.length : null);
      return { file: p, count: rows, format: "json", question: gt?.nl_question || null };
    }
    return { file: p, count: await countRows(p), format: "csv", question: null };
  }
  return null;
}

/** Which SemBench table(s) does this query touch? Config-driven from the SQL. */
function tablesInSql(sql, args) {
  const { benchmark: bench, scaleFactor: sf } = args;
  const dataDir = args.tableDir || args.dataDir;   // materialized CSVs for parquet benchmarks
  const names = [...sql.matchAll(prefixRe(bench))].map((m) => m[1]);
  const uniq = [...new Set(names)];
  return uniq.map((t) => {
    const d = tableDesc(bench, t);
    const file = d ? tableFile(d, sf) : `${t}.csv`;
    const path = (dataDir && file) ? resolve(dataDir, file) : (file || "");
    const modality = (d && d.modality) || (/image/i.test(t) ? "image" : "text");
    return { table: t, path, modality,
             isImages: modality === "image", isAudio: modality === "audio",
             col: (d && d.col) || null, key: (d && d.key) || null };
  });
}

/** alias → table map from FROM/JOIN clauses (<prefix>.table [AS] alias). */
function aliasMap(sql, bench) {
  const m = {};
  for (const x of sql.matchAll(prefixRe(bench, String.raw`\s+(?:AS\s+)?(\w+)`))) m[x[2]] = x[1];
  return m;
}

/** The argument text of the AI.IF / AI.GENERATE call (up to connection_id). */
function semanticArgs(sql) {
  const i = sql.search(/AI\.(IF|GENERATE)\s*\(/i);
  if (i < 0) return "";
  const j = sql.toLowerCase().indexOf("connection_id", i);
  return sql.slice(i, j < 0 ? Math.min(i + 500, sql.length) : j);
}

/** Tables whose alias is referenced inside the semantic predicate. */
function tablesInPredicate(sql, bench) {
  const amap = aliasMap(sql, bench);
  const arg = semanticArgs(sql);
  const used = Object.keys(amap)
    .filter((al) => new RegExp(`\\b${al}\\.`).test(arg))
    .map((al) => amap[al]);
  return [...new Set(used)];
}

/** First header line of a CSV, or "" if unreadable. */
async function headerOf(path) {
  try { return (await readFile(path, "utf-8")).split(/\r?\n/, 1)[0]; } catch { return ""; }
}

/**
 * Naive (original) semantic-call count, by operator:
 *   sem_join   (predicate spans ≥2 tables): |left| × |right|
 *   sem_filter / sem_map (one table):       |table|
 * Row counts read from the CSVs.
 */
async function computeNaive(sql, args) {
  const { benchmark: bench, scaleFactor: sf } = args;
  const dataDir = args.tableDir || args.dataDir;
  let tabs = tablesInPredicate(sql, bench);
  if (tabs.length === 0) tabs = [...new Set(Object.values(aliasMap(sql, bench)))];
  const counts = {};
  for (const t of tabs) {
    const d = tableDesc(bench, t);
    const file = d ? tableFile(d, sf) : `${t}.csv`;
    counts[t] = (dataDir && file) ? await countRows(resolve(dataDir, file)) : null;
  }
  if (tabs.length >= 2 && Object.values(counts).every((c) => c != null)) {
    const naive = Object.values(counts).reduce((a, b) => a * b, 1);
    return { type: "join", naive, tables: tabs, counts };
  }
  const t0 = tabs[0];
  return { type: tabs.length >= 2 ? "join" : "filter", naive: counts[t0] ?? null, tables: tabs, counts };
}

/**
 * Choose the "corpus" table to extract from, using the config modality: prefer an
 * IMAGE table referenced in the predicate, then a TEXT one. AUDIO-only queries have
 * no supported corpus — the caller detects `modality === "audio"` and skips them.
 */
async function chooseCorpus(sql, tables, args) {
  const used = new Set(tablesInPredicate(sql, args.benchmark));
  const inPred = tables.filter((t) => used.has(t.table));
  const pick = (cands) =>
    cands.find((t) => t.modality === "image") ||
    cands.find((t) => t.modality === "text") || null;
  return pick(inPred) || pick(tables) ||
    tables.find((t) => t.modality === "audio") ||   // audio-only → caller skips
    tables[tables.length - 1] || { table: "corpus", path: "", modality: "text", isImages: false };
}

/** Distinct non-empty values of a CSV column — the value space for CLIP classify. */
async function distinctValues(path, column) {
  try {
    const lines = (await readFile(path, "utf-8")).split(/\r?\n/).filter((l) => l.length);
    if (!lines.length) return [];
    const idx = lines[0].split(",").map((c) => c.trim()).indexOf(column);
    if (idx < 0) return [];
    const vals = new Set();
    for (const line of lines.slice(1)) {
      const cell = (line.split(",")[idx] || "").trim();
      if (cell) vals.add(cell);
    }
    return [...vals].sort();
  } catch { return []; }
}

/** Fill each attribute's `labels` from its `labels_from: "table.column"` — the DB
 *  value space — so CLIP classify returns a real structured FIELD value, not a score.
 *  For a logo↔named-table join (mmqa q2a/q7), labels_from points at the structured
 *  side's name column. Mutates `schema`; returns true if it changed anything. */
async function resolveLabelsFrom(schema, args) {
  let changed = false;
  const tableDir = args.tableDir || args.dataDir;
  for (const a of schema.attributes || []) {
    const lf = a.extractor && a.extractor.labels_from;
    if (!lf || (a.extractor.labels && a.extractor.labels.length)) continue;
    // Accept "table.column" OR a prefixed "<benchmark>.table.column" (as the SQL writes
    // it): the LAST segment is the column, the one before it is the table.
    const parts = String(lf).split(".");
    const col = parts.pop();
    const tbl = parts.pop();
    const d = tableDesc(args.benchmark, tbl);
    const file = d ? tableFile(d, args.scaleFactor) : `${tbl}.csv`;
    const vals = await distinctValues(resolve(tableDir, file), col);
    if (vals.length) {
      a.extractor.labels = vals; changed = true;
      console.log(`[SemDB] labels_from ${lf}: ${vals.length} value(s) -> attr '${a.name}'`);
    } else {
      console.warn(`[SemDB] labels_from ${lf}: no values at ${resolve(tableDir, file)}`);
    }
  }
  return changed;
}

/** List query ids (<name>.sql → <name>) in a query dir, sorted. */
async function listQueries(dir) {
  if (!dir) return [];
  const files = await readdir(dir);
  return files.filter((f) => f.toLowerCase().endsWith(".sql"))
    .map((f) => f.replace(/\.sql$/i, "")).sort();
}

async function runPhase(agentConfig, vars, runDir, args) {
  const systemPrompt = await readFile(agentConfig.promptPath, "utf-8");
  const template = await readFile(agentConfig.userPromptPath, "utf-8");
  const userPrompt = renderTemplate(template, vars);

  if (args.dryRun) {
    console.log(`\n[SemDB] --- ${agentConfig.name} (dry-run) ---`);
    console.log(userPrompt);
    return { dryRun: true };
  }

  const result = await runAgent(agentConfig.name, {
    systemPrompt,
    userPrompt,
    allowedTools: agentConfig.allowedTools,
    model: args.modelOverride || getAgentModel(agentConfig.configKey, args.agentProvider),
    effortLevel: getAgentEffort(agentConfig.configKey, args.agentProvider),
    configName: agentConfig.configKey,
    cwd: runDir,
    timeoutMs: defaults.agentTimeoutMs,
    useSkills: false,
  });
  if (result.error) throw new Error(`${agentConfig.name} failed: ${result.error}`);
  return result;
}

/** Build a phase-telemetry recorder bound to a phases[] array. */
function makeRecorder(args, phases) {
  return (name, r) => {
    if (!r || r.dryRun) return r;
    phases.push({
      phase: name,
      model: args.modelOverride || getAgentModel(name, args.agentProvider),
      duration_ms: r.durationMs || 0,
      tokens: r.tokens || {},
      cost_usd: r.costUsd || 0,
      llm_calls: r.numTurns || 1,   // internal turns this agent made (min 1)
    });
    return r;
  };
}

function agentModelsLine(args) {
  const m = (k) => args.modelOverride || getAgentModel(k, args.agentProvider);
  return `designer=${m("schema_designer")}, extractor=${m("extractor")}, codegen=${m("code_generator")}`;
}

/** Corpus columns + the small model for extraction. Prefer the benchmark config's
 *  extract column / key; fall back to header heuristics for unknown tables. */
async function corpusCols(corpus, args) {
  let cols = [];
  try { cols = (await headerOf(corpus.path)).split(",").map((c) => c.trim()); } catch { /* unreadable */ }
  const isImage = corpus.isImages;
  const key = corpus.key || cols[0] || (isImage ? "uri" : "id");
  const textCol = (!isImage && corpus.col) || cols.find((c) => /text|description|overview|body|content|summary|symptoms|review|complaint|plot|display/i.test(c)) || cols[cols.length - 1] || "text";
  const imageCol = (isImage && corpus.col) || cols.find((c) => /image|uri|path|file|ref/i.test(c)) || cols[0] || "uri";
  return {
    cols,
    idCol: key,
    textCol,
    imageCol,
    extractModel: args.extractModel || (isImage ? defaults.extraction.smallImageModel : defaults.extraction.smallTextModel),
  };
}

// ---------------------------------------------------------------------------
// Per-query planning: which corpus (extract side), structured side, operator.
// ---------------------------------------------------------------------------
async function planQuery(args, query) {
  const { sql, nl } = await loadQuery({ ...args, query });
  const tables = tablesInSql(sql, args);
  const corpus = await chooseCorpus(sql, tables, args);
  // structured side = a non-corpus, non-image/audio (relational) table if any
  const structured = tables.find((t) => t.modality === "structured" && t.path !== corpus.path)
    || tables.find((t) => !t.isImages && !t.isAudio && t.path !== corpus.path)
    || tables.find((t) => t.path !== corpus.path) || corpus;
  const plan = await computeNaive(sql, args);
  return { query, sql, nl, tables, corpus, structured,
           isImage: corpus.modality === "image", isAudio: corpus.modality === "audio", plan };
}

// ---------------------------------------------------------------------------
// Corpus-level Phase A + B — run ONCE per corpus, shared by all its queries.
// ---------------------------------------------------------------------------
async function ensureCorpus(args, corpus, corpusQueries) {
  const corpusDir = resolve(args.out, "_corpus", corpus.table);
  await mkdir(corpusDir, { recursive: true });
  const schemaPath = resolve(corpusDir, "schema.json");
  const attrsPath = resolve(corpusDir, `${corpus.table}_attrs.json`);
  const isImage = corpus.isImages;
  const modality = isImage ? "image" : "text";
  const { cols, idCol, textCol, imageCol, extractModel } = await corpusCols(corpus, args);
  const doRun = !args.dryRun && (args.run || !!args.groundTruthDir) && !args.noRun;

  const phases = [];
  const record = makeRecorder(args, phases);

  console.log(`\n[SemDB] ===== CORPUS ${corpus.table} [${modality}] — ${corpusQueries.length} quer${corpusQueries.length === 1 ? "y" : "ies"}: ${corpusQueries.map((p) => p.query).join(", ")} =====`);

  // Phase A — Schema Designer ONCE, seeing ALL queries over this corpus so the
  // schema covers every attribute they need (skip if cached unless --force).
  if (!existsSync(schemaPath) || args.force) {
    const tableHeaders = await readTableHeaders(args.dataDir);
    const querySqls = corpusQueries.map((p) => `-- ${p.query}\n${p.sql}`).join("\n\n");
    record("schema_designer", await runPhase(schemaDesignerConfig, {
      query_id: `${corpus.table} [${corpusQueries.map((p) => p.query).join(",")}]`,
      query_sql: querySqls,
      query_nl: corpusQueries.map((p) => p.nl).filter(Boolean).join(" | "),
      table_schemas: tableHeaders,
      corpus_name: corpus.table,
      corpus_size: "(rows in " + basename(corpus.path || "corpus") + ")",
      modality,
      structured_name: "(varies per query)",
      structured_size: "N/A",
      schema_path: schemaPath,
      existing_schema: "",
    }, corpusDir, args));
  } else {
    console.log(`[SemDB] reuse cached corpus schema: ${schemaPath}`);
  }

  const schema = args.dryRun ? null : await readJSON(schemaPath);
  // Bake the DB value space into any `labels_from` attribute (structured field
  // extraction: CLIP classify over a column's distinct values → a real field value).
  if (schema && !args.dryRun && await resolveLabelsFrom(schema, args)) {
    await writeFile(schemaPath, JSON.stringify(schema, null, 2));
  }
  if (schema && schema.decomposable === false) {
    console.log(`[SemDB] corpus ${corpus.table} not decomposable: ${schema.rationale}`);
  }

  // Phase B — Extractor ONCE, in two steps mirroring Phase C (Code Generator):
  //   1. GENERATE a thin per-corpus driver (agent) -> extract_<corpus>.py that
  //      implements the semextract ExtractDriver hooks (column mapping, prompt +
  //      context columns, preprocessing) and calls semextract.run(...).
  //   2. EXECUTE it (orchestrator) on the full corpus -> attrs + <attrs>.meta.json.
  // Both cache per corpus (amortized across all its queries).
  const driverPath = resolve(corpusDir, `extract_${corpus.table}.py`);
  if (!existsSync(driverPath) || args.force) {          // agent artifact step (like schema design); runPhase handles --dry-run
    record("extractor", await runPhase(extractorConfig, {
      corpus_name: corpus.table,
      schema_json: schema ? JSON.stringify(schema, null, 2) : "{{corpus schema.json}}",
      schema_path: schemaPath,
      header: cols.join(", "),
      modality,
      id_col: idCol, text_col: textCol, image_col: imageCol,
      image_dir: args.imageDir || "",
      small_model: extractModel,
      escalation_model: defaults.extraction.escalationImageModel,
      corpus_manifest: corpus.path,
      corpus_size: `(rows in ${basename(corpus.path)})`,
      driver_path: driverPath,
      attrs_path: attrsPath,
      semextract_path: resolve(__dirname, "semextract.py"),
    }, corpusDir, args));
  } else if (existsSync(driverPath)) {
    console.log(`[SemDB] reuse cached corpus extractor driver: ${driverPath}`);
  }
  if (doRun && !args.dryRun && existsSync(driverPath) && (!existsSync(attrsPath) || args.force)) {
    // IMAGE corpora extract locally via semvision (tiered CV+CLIP proxies) — the model
    // is a CLIP id (NOT the VLM --extract-model) and NO vLLM endpoint is needed. TEXT
    // corpora keep the endpoint path with the VLM --extract-model.
    const imageModel = args.clipModel || defaults.extraction.clipModel;
    const exArgs = [driverPath, corpus.path, attrsPath, "--schema", schemaPath,
      "--model", (isImage ? imageModel : extractModel),
      ...(isImage ? ["--image-dir", args.imageDir] : []),
      ...(!isImage && args.endpoint ? ["--endpoint", args.endpoint, "--api-key", args.apiKey,
                           "--concurrency", String(args.concurrency)] : []),
      ...(args.theta != null ? ["--theta", String(args.theta)] : [])];
    console.log(`\n[SemDB] Extracting corpus ${corpus.table} (once): python3 ${exArgs.join(" ")}`);
    const ex = spawnSync("python3", exArgs, { stdio: "inherit" });
    if (ex.status !== 0) console.warn(`[SemDB] extraction exited ${ex.status}.`);
  } else if (existsSync(attrsPath)) {
    console.log(`[SemDB] reuse cached corpus attrs: ${attrsPath}`);
  }

  const extMeta = await readJSON(attrsPath + ".meta.json");
  const sd = phases.filter((p) => p.phase === "schema_designer");
  const ext = phases.filter((p) => p.phase === "extractor");
  const corpusTelemetry = {
    corpus: corpus.table,
    modality,
    queries: corpusQueries.map((p) => p.query),
    query_count: corpusQueries.length,
    schema_design: {
      ms: sd.reduce((s, p) => s + p.duration_ms, 0),
      calls: sd.reduce((s, p) => s + p.llm_calls, 0),
      cost_usd: sd.reduce((s, p) => s + p.cost_usd, 0),
      model: agentModelsLine(args),
    },
    extractor_codegen: {
      ms: ext.reduce((s, p) => s + p.duration_ms, 0),
      calls: ext.reduce((s, p) => s + p.llm_calls, 0),
      cost_usd: ext.reduce((s, p) => s + p.cost_usd, 0),
    },
    extraction: {
      sec: extMeta?.elapsed_sec ?? null,
      calls: extMeta?.llm_calls ?? null,
      model: extractModel,
    },
  };
  if (!args.dryRun) await writeFile(resolve(corpusDir, "corpus_telemetry.json"), JSON.stringify(corpusTelemetry, null, 2));
  return { corpusDir, schemaPath, attrsPath, schema, corpusTelemetry, idCol, textCol, imageCol, extractModel, isImage, modality };
}

// ---------------------------------------------------------------------------
// Per-query Phase C + execute + evaluate (reuses the corpus schema + attrs).
// ---------------------------------------------------------------------------
async function runQueryCodegen(args, planObj, art, csvPath) {
  const { query, sql, structured, plan, corpus } = planObj;
  const wallStart = Date.now();
  const runDir = resolve(args.out, `${args.benchmark}-${query}`);
  await mkdir(runDir, { recursive: true });
  const codePath = resolve(runDir, `compiled_${query}.py`);
  const resultsCsv = resolve(runDir, `${query}_results.csv`);
  const { schemaPath, attrsPath, schema, corpusTelemetry, extractModel } = art;
  const doRun = !args.dryRun && (args.run || !!args.groundTruthDir) && !args.noRun;

  const phases = [];
  const record = makeRecorder(args, phases);

  console.log(`\n[SemDB] ---- ${query}  (corpus ${corpus.table}, ${plan.type}) ----`);

  // Phase C — Code Generator (per query; reuses the corpus schema + attrs).
  record("code_generator", await runPhase(codeGeneratorConfig, {
    query_id: query,
    query_sql: sql,
    schema_json: schema ? JSON.stringify(schema, null, 2) : "{{corpus schema.json}}",
    attrs_path: attrsPath,
    attrs_columns: "(schema attributes + conf)",
    structured_path: structured.path || "(structured table path)",
    structured_columns: "(see table headers)",
    code_path: codePath,
    code_basename: `compiled_${query}.py`,
  }, runDir, args));

  if (doRun && existsSync(codePath) && existsSync(attrsPath)) {
    const cqArgs = [codePath, structured.path, attrsPath, resultsCsv,
      ...(args.endpoint ? ["--endpoint", args.endpoint, "--api-key", args.apiKey, "--model", extractModel] : [])];
    console.log(`\n[SemDB] Running compiled query: python3 ${cqArgs.join(" ")}`);
    const cq = spawnSync("python3", cqArgs, { stdio: "inherit" });
    if (cq.status !== 0) console.warn(`[SemDB] compiled query exited ${cq.status} — check its CLI signature.`);
  } else if (doRun) {
    console.warn(`[SemDB] skipping compiled query — missing ${existsSync(codePath) ? attrsPath : codePath}`);
  }

  if (args.dryRun) return null;

  // --- Per-query telemetry (codegen + residual) + shared corpus (amortized) ---
  const cg = phases.find((p) => p.phase === "code_generator") || { duration_ms: 0, cost_usd: 0, llm_calls: 0, tokens: {} };
  const cqMeta = await readJSON(resolve(runDir, `compiled_${query}.meta.json`));
  const residualCalls = cqMeta?.residual_calls || 0;
  const N = await countRows(corpus.path);
  const extractionCalls = corpusTelemetry.extraction.calls ?? (N ?? 0);
  const schemaCalls = corpusTelemetry.schema_design.calls || 0;
  const K = corpusTelemetry.query_count || 1;
  const naiveCalls = plan.naive;
  const compiledExecCalls = extractionCalls + residualCalls;
  const reduction = (naiveCalls && compiledExecCalls) ? Number((naiveCalls / compiledExecCalls).toFixed(1)) : null;
  const amortizedTotal = cg.llm_calls + residualCalls + (schemaCalls + extractionCalls) / K;
  const codegenCost = cg.cost_usd;
  const amortizedCost = codegenCost + (corpusTelemetry.schema_design.cost_usd || 0) / K;
  const gt = await resolveGroundTruth(args.groundTruthDir, query, args.scaleFactor);

  const report = {
    query, corpus: corpus.table, provider: args.agentProvider, operator: plan.type,
    wall_clock_ms: Date.now() - wallStart,
    per_query: {
      codegen_ms: cg.duration_ms, codegen_calls: cg.llm_calls,
      codegen_cost_usd: Number(codegenCost.toFixed(4)),
      codegen_tokens: (cg.tokens.input || 0) + (cg.tokens.output || 0),
      compiled_query_sec: cqMeta?.elapsed_sec ?? null, residual_calls: residualCalls,
    },
    shared_corpus: {
      corpus: corpus.table, query_count: K,
      schema_design_calls: schemaCalls,
      schema_design_cost_usd: Number((corpusTelemetry.schema_design.cost_usd || 0).toFixed(4)),
      extraction_calls: extractionCalls, extraction_sec: corpusTelemetry.extraction.sec,
      amortized_schema_design_calls: Number((schemaCalls / K).toFixed(2)),
      amortized_extraction_calls: Number((extractionCalls / K).toFixed(2)),
    },
    llm_calls: {
      schema_design: schemaCalls, extraction: extractionCalls,
      codegen: cg.llm_calls, residual: residualCalls,
      amortized_total: Number(amortizedTotal.toFixed(2)),
    },
    naive_llm_calls: naiveCalls ?? null,
    compiled_execution_calls: compiledExecCalls,
    call_reduction: reduction,
    total_estimated_cost_usd: Number(amortizedCost.toFixed(4)),
    ground_truth: gt ? { file: gt.file, count: gt.count } : null,
    phases,
  };
  const telePath = resolve(runDir, "telemetry.json");
  await writeFile(telePath, JSON.stringify(report, null, 2));

  // --- Summary print ---
  console.log(`\n[SemDB] === ${query} ===`);
  console.log(`[SemDB]   code_generator     ${(cg.duration_ms / 1000).toFixed(1)}s  ${cg.llm_calls} calls  $${cg.cost_usd.toFixed(4)}  (${report.phases[0]?.model || ""})`);
  console.log(`[SemDB]   shared/corpus      schema_design ${schemaCalls} + extraction ${extractionCalls} calls  ÷ ${K} queries  (amortized ${(schemaCalls / K).toFixed(1)}+${(extractionCalls / K).toFixed(1)})`);
  console.log(`[SemDB]   residual           ${residualCalls}`);
  const naiveExpr = plan.type === "join"
    ? `= ${plan.tables.map((t) => plan.counts[t]).join(" × ")} (${plan.tables.join(" × ")})`
    : (plan.naive != null ? `= ${plan.naive} rows` : "");
  console.log(`[SemDB]   ORIGINAL (naive)   ${naiveCalls != null ? naiveCalls : "?"} [${plan.type}] ${naiveExpr}`);
  console.log(`[SemDB]   COMPILED exec      ${compiledExecCalls} (extraction ${extractionCalls} + residual ${residualCalls})${reduction ? `  → ${reduction}× fewer` : ""}`);

  // --- Score against ground truth + append CSV ---
  if (doRun && gt && existsSync(resultsCsv)) {
    const evArgs = [resolve(__dirname, "evaluate.py"), "--telemetry", telePath,
      "--ground-truth", gt.file, "--pred", resultsCsv, "--pred-cols", args.predCols,
      "--query", query, "--benchmark", args.benchmark, "--csv", csvPath,
      ...(args.groundTruthDir ? ["--ground-truth-dir", args.groundTruthDir] : []),
      ...(args.scaleFactor ? ["--sf", String(args.scaleFactor)] : [])];
    const ev = spawnSync("python3", evArgs, { stdio: "inherit" });
    if (ev.status !== 0) console.warn(`[SemDB] evaluate.py exited ${ev.status}.`);
    const scored = await readJSON(telePath);
    const m = scored?.metrics;
    if (m) {
      console.log(`[SemDB]   METRICS            precision=${m.precision}  recall=${m.recall}  F1=${m.f1}  (tp=${m.tp} fp=${m.fp} fn=${m.fn})`);
      console.log(`[SemDB]   saved -> ${telePath} and ${csvPath}`);
    }
  } else if (gt) {
    console.log(`[SemDB]   (compiled output not found — score later with evaluate.py --pred ${resultsCsv})`);
  }
  return report;
}

async function main() {
  const base = parseArgs(process.argv);
  setAgentProvider(base.agentProvider);
  const csvPath = base.telemetryCsv || resolve(base.out, "results.csv");

  const queries = base.query ? [base.query] : await listQueries(base.queryDir);
  if (queries.length === 0) {
    console.error("[SemDB] no query given and no *.sql found in --query-dir.");
    process.exit(1);
  }
  console.log(`[SemDB] provider=${base.agentProvider} models: ${agentModelsLine(base)}`);
  console.log(`[SemDB] processing ${queries.length} quer${queries.length === 1 ? "y" : "ies"}: ${queries.join(", ")}`);

  // 0) PARQUET benchmarks (ecomm): materialize tables + image manifest to CSV ONCE,
  //    then resolve table files from that dir (images stay in <dataDir>/images).
  base.tableDir = base.dataDir;
  const bcfg = BENCHMARKS[base.benchmark];
  if (bcfg && bcfg.parquet && base.dataDir && !base.dryRun) {
    const matDir = resolve(base.out, "_materialized", `${base.benchmark}_sf${base.scaleFactor || "flat"}`);
    const mArgs = [resolve(__dirname, "materialize.py"), base.benchmark, base.dataDir, matDir,
      ...(base.force ? ["--force"] : [])];
    console.log(`[SemDB] materializing parquet → CSV: python3 ${mArgs.join(" ")}`);
    const mp = spawnSync("python3", mArgs, { stdio: "inherit" });
    if (mp.status !== 0) console.warn(`[SemDB] materialize.py exited ${mp.status} (needs pandas — run under gendb/sembench env).`);
    else base.tableDir = matDir;
  }

  // 1) Plan every query and group by corpus (the extract-side table).
  const allPlans = [];
  for (const q of queries) {
    try { allPlans.push(await planQuery(base, q)); }
    catch (e) { console.error(`[SemDB] [${q}] plan failed: ${e.message}`); }
  }
  // Skip queries we can't run: audio corpora, or a corpus whose table file didn't
  // resolve (usually a wrong/missing --benchmark so the SQL prefix matched nothing).
  const plans = allPlans.filter((p) => {
    if (p.isAudio || p.corpus.modality === "audio") {
      console.warn(`[SemDB] [${p.query}] SKIP: audio-modality corpus '${p.corpus.table}' is not supported.`);
      return false;
    }
    if (!p.corpus.path || !existsSync(p.corpus.path)) {
      console.warn(`[SemDB] [${p.query}] SKIP: corpus table not found ('${p.corpus.table}' → '${p.corpus.path || "<empty>"}'). `
        + `Check --benchmark (got '${base.benchmark}') and --sf so the SQL prefix '${benchPrefix(base.benchmark)}.' matches.`);
      return false;
    }
    return true;
  });
  const corpora = new Map();
  for (const p of plans) {
    const key = p.corpus.table;
    if (!corpora.has(key)) corpora.set(key, { corpus: p.corpus, queries: [] });
    corpora.get(key).queries.push(p);
  }
  console.log(`[SemDB] ${corpora.size} corpus/corpora: ${[...corpora.entries()].map(([k, v]) => `${k}(${v.queries.length})`).join(", ")}`);

  // 2) Schema Designer + Extractor ONCE per corpus (shared by its queries).
  const corpusArt = new Map();
  for (const [key, info] of corpora) {
    try { corpusArt.set(key, await ensureCorpus(base, info.corpus, info.queries)); }
    catch (e) { console.error(`[SemDB] corpus ${key} failed: ${e.message}`); }
  }

  // 3) Code Generator + execute + evaluate PER query.
  const summary = [];
  for (const p of plans) {
    console.log(`\n[SemDB] ==================== ${p.query} ====================`);
    const art = corpusArt.get(p.corpus.table);
    if (!art) { summary.push({ q: p.query, error: "corpus artifacts missing" }); continue; }
    try {
      await runQueryCodegen(base, p, art, csvPath);
      const tele = await readJSON(resolve(base.out, `${base.benchmark}-${p.query}`, "telemetry.json"));
      if (tele?.metrics) summary.push({ q: p.query, ...tele.metrics });
    } catch (e) {
      console.error(`[SemDB] [${p.query}] failed: ${e.message}`);
      summary.push({ q: p.query, error: e.message });
    }
  }

  // 4) Workload summary.
  console.log(`\n[SemDB] ==================== SUMMARY ====================`);
  for (const s of summary) {
    console.log(s.error
      ? `[SemDB]   ${s.q.padEnd(6)} FAILED — ${s.error}`
      : `[SemDB]   ${s.q.padEnd(6)} P=${s.precision} R=${s.recall} F1=${s.f1}`);
  }
  const scored = summary.filter((s) => s.f1 != null);
  if (scored.length) {
    const mean = (k) => (scored.reduce((a, s) => a + s[k], 0) / scored.length).toFixed(4);
    console.log(`[SemDB]   ${"MEAN".padEnd(6)} P=${mean("precision")} R=${mean("recall")} F1=${mean("f1")}  (${scored.length} scored)`);
  }
  console.log(`[SemDB]   metrics CSV -> ${csvPath}`);
}

main().catch((err) => {
  console.error("[SemDB] Fatal:", err.message);
  process.exit(1);
});
