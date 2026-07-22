/**
 * SemDB Orchestrator — compile a SemBench semantic operator into a relational
 * program via three agents, reusing GenDB's provider/agent plumbing.
 *
 *   Phase A  Schema Designer  : query + table headers   -> schema.json
 *   Phase B  Extractor        : schema + corpus         -> <corpus>_attrs.json  (once, shared)
 *   Phase C  Code Generator   : query + schema + attrs  -> compiled_<q>.py
 *   Verify   compiled result  == naive oracle, report model-call reduction
 *
 * You provide two paths — the SemBench query folder and the data folder — and a
 * query id. The agents read the REAL .sql file and the REAL table headers and
 * emit artifacts that reference your real files.
 *
 * Usage:
 *   node src/semdb/orchestrator.mjs \
 *        --query q3a \
 *        --query-dir /localhome/hza214/SemBench/files/mmqa/query/bigquery \
 *        --data-dir  /localhome/hza214/SemBench/files/mmqa/data/sf_200 \
 *        [--out <dir>] [--dry-run]
 *
 * `--dry-run` prints the resolved plan + rendered prompts without spawning agents
 * (no Claude credentials needed) — use it to see exactly what each agent receives.
 * The self-contained, GPU-free demonstration lives in ./poc/ and ./examples/.
 */

import { readFile, writeFile, mkdir, readdir } from "fs/promises";
import { existsSync } from "fs";
import { resolve, dirname, basename } from "path";
import { fileURLToPath } from "url";

import {
  renderTemplate,
  runAgent,
  readJSON,
  setAgentProvider,
} from "../gendb/shared.mjs";
import { defaults, getAgentModel, getAgentEffort } from "./semdb.config.mjs";
import { config as schemaDesignerConfig } from "./agents/schema-designer/index.mjs";
import { config as extractorConfig } from "./agents/extractor/index.mjs";
import { config as codeGeneratorConfig } from "./agents/code-generator/index.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
  const args = {
    query: null,
    benchmark: defaults.benchmark,
    querySource: defaults.querySource,
    sembenchDir: null,   // repo root; query-dir/data-dir derived from it if given
    queryDir: null,      // .../files/<b>/query/<dialect>
    dataDir: null,       // .../files/<b>/data/<sf>
    imageDir: null,      // .../data/<sf>/images  (defaults to <dataDir>/images)
    out: resolve(__dirname, "runs"),
    agentProvider: defaults.agentProvider,
    modelOverride: null,   // force one model for all agents (testing)
    dryRun: false,
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--query" && argv[i + 1]) args.query = argv[++i];
    else if (a === "--benchmark" && argv[i + 1]) args.benchmark = argv[++i];
    else if (a === "--agent-provider" && argv[i + 1]) args.agentProvider = argv[++i];
    else if (a === "--model" && argv[i + 1]) args.modelOverride = argv[++i];
    else if (a === "--query-source" && argv[i + 1]) args.querySource = argv[++i];
    else if (a === "--sembench-dir" && argv[i + 1]) args.sembenchDir = resolve(argv[++i]);
    else if (a === "--query-dir" && argv[i + 1]) args.queryDir = resolve(argv[++i]);
    else if (a === "--data-dir" && argv[i + 1]) args.dataDir = resolve(argv[++i]);
    else if (a === "--image-dir" && argv[i + 1]) args.imageDir = resolve(argv[++i]);
    else if (a === "--out" && argv[i + 1]) args.out = resolve(argv[++i]);
    else if (a === "--dry-run") args.dryRun = true;
  }
  // Derive query/data dirs from the repo root when only --sembench-dir is given.
  if (args.sembenchDir) {
    const base = resolve(args.sembenchDir, "files", args.benchmark);
    args.queryDir = args.queryDir || resolve(base, "query", args.querySource);
    args.dataDir = args.dataDir || resolve(base, "data");
  }
  if (args.dataDir && !args.imageDir) args.imageDir = resolve(args.dataDir, "images");
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

/** Which SemBench table(s) does this query touch? Best-effort from the SQL. */
function tablesInSql(sql, dataDir) {
  const names = [...sql.matchAll(/mmqa\.(\w+)/g)].map((m) => m[1]);
  const uniq = [...new Set(names)];
  return uniq.map((t) => {
    const csv = dataDir ? resolve(dataDir, `${t}.csv`) : `${t}.csv`;
    return { table: t, path: csv, isImages: /image/i.test(t) };
  });
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

async function main() {
  const args = parseArgs(process.argv);
  setAgentProvider(args.agentProvider);
  const wallStart = Date.now();

  const runDir = resolve(args.out, `${args.benchmark}-${args.query || "example"}`);
  await mkdir(runDir, { recursive: true });

  // Telemetry: one entry per agent phase (time, tokens, cost).
  const telemetry = { phases: [] };
  const record = (name, r) => {
    if (!r || r.dryRun) return r;
    telemetry.phases.push({
      phase: name,
      provider: args.agentProvider,
      model: args.modelOverride || getAgentModel(name, args.agentProvider),
      duration_ms: r.durationMs || 0,
      tokens: r.tokens || {},
      cost_usd: r.costUsd || 0,
    });
    return r;
  };

  const { sql, nl } = await loadQuery(args);
  const tableHeaders = await readTableHeaders(args.dataDir);
  const tables = tablesInSql(sql, args.dataDir);
  const imageTable = tables.find((t) => t.isImages);
  const unstructured = imageTable || tables[tables.length - 1] || { table: "corpus", path: "" };

  const schemaPath = resolve(runDir, "schema.json");
  const attrsPath = resolve(runDir, `${unstructured.table}_attrs.json`);
  const codePath = resolve(runDir, `compiled_${args.query || "q"}.py`);

  console.log(`[SemDB] benchmark=${args.benchmark} query=${args.query || "example"}`);
  console.log(`[SemDB] provider=${args.agentProvider} models: designer=${args.modelOverride || getAgentModel("schema_designer", args.agentProvider)}, extractor=${args.modelOverride || getAgentModel("extractor", args.agentProvider)}, codegen=${args.modelOverride || getAgentModel("code_generator", args.agentProvider)}`);
  console.log(`[SemDB] query-dir: ${args.queryDir || "(none)"}`);
  console.log(`[SemDB] data-dir:  ${args.dataDir || "(none)"}`);
  console.log(`[SemDB] tables in SQL: ${tables.map((t) => t.table).join(", ") || "(none parsed)"}`);
  console.log(`[SemDB] run dir:   ${runDir}`);
  console.log(`[SemDB] query:\n${sql}\n`);

  // Phase A — Schema Designer (sees the real SQL + real table headers)
  record("schema_designer", await runPhase(schemaDesignerConfig, {
    query_id: args.query || "example",
    query_sql: sql,
    query_nl: nl,
    table_schemas: tableHeaders,
    corpus_name: unstructured.table,
    corpus_size: "(rows in " + basename(unstructured.path || "corpus") + ")",
    modality: imageTable ? "image" : "text",
    structured_name: (tables.find((t) => !t.isImages) || {}).table || "(structured side)",
    structured_size: "N/A",
    schema_path: schemaPath,
    existing_schema: "",
  }, runDir, args));

  const schema = args.dryRun ? null : await readJSON(schemaPath);
  if (schema && schema.decomposable === false) {
    console.log(`[SemDB] Not decomposable: ${schema.rationale}. Fall back to naive execution.`);
    return;
  }

  // Phase B — Extractor (small model, once per corpus; reused by every query over it)
  const isImage = !!imageTable;
  record("extractor", await runPhase(extractorConfig, {
    corpus_name: unstructured.table,
    schema_json: schema ? JSON.stringify(schema, null, 2) : "{{schema.json from Phase A}}",
    corpus_manifest: unstructured.path || "(corpus table path)",
    modality: isImage ? "image" : "text",
    corpus_size: "(all rows)",
    small_model: isImage ? defaults.extraction.smallImageModel : defaults.extraction.smallTextModel,
    escalation_model: defaults.extraction.escalationImageModel,
    attrs_path: attrsPath,
  }, runDir, args));

  // Phase C — Code Generator (compiled program references the real table paths)
  const structured = tables.find((t) => !t.isImages) || unstructured;
  record("code_generator", await runPhase(codeGeneratorConfig, {
    query_id: args.query || "q",
    query_sql: sql,
    schema_json: schema ? JSON.stringify(schema, null, 2) : "{{schema.json from Phase A}}",
    attrs_path: attrsPath,
    attrs_columns: "(schema attributes + conf)",
    structured_path: structured.path || "(structured table path)",
    structured_columns: "(see table headers)",
    code_path: codePath,
    code_basename: `compiled_${args.query || "q"}.py`,
  }, runDir, args));

  // --- Telemetry: agent-stage time + estimated cost -------------------------
  if (!args.dryRun) {
    // "code execution" = the extraction + compiled-query runtime. Those run in
    // separate scripts (extract.py, compiled_<q>.py), each of which writes an
    // elapsed_sec into its own <out>.meta.json. Merge them if present.
    const codeExec = { extraction_sec: null, compiled_query_sec: null };
    const extMeta = await readJSON(attrsPath + ".meta.json");
    if (extMeta?.elapsed_sec != null) codeExec.extraction_sec = extMeta.elapsed_sec;
    const cqMeta = await readJSON(resolve(runDir, `compiled_${args.query || "q"}.meta.json`));
    if (cqMeta?.elapsed_sec != null) codeExec.compiled_query_sec = cqMeta.elapsed_sec;

    const agentMs = telemetry.phases.reduce((s, p) => s + p.duration_ms, 0);
    const costUsd = telemetry.phases.reduce((s, p) => s + p.cost_usd, 0);
    const totalTok = telemetry.phases.reduce((s, p) =>
      s + (p.tokens.input || 0) + (p.tokens.output || 0), 0);
    const codeExecMs = 1000 * ((codeExec.extraction_sec || 0) + (codeExec.compiled_query_sec || 0));

    const report = {
      query: args.query, provider: args.agentProvider,
      wall_clock_ms: Date.now() - wallStart,
      agent_stage_ms: agentMs,
      code_execution_ms: codeExecMs || null,
      code_execution: codeExec,
      total_estimated_cost_usd: Number(costUsd.toFixed(4)),
      total_agent_tokens: totalTok,
      phases: telemetry.phases,
    };
    await writeFile(resolve(runDir, "telemetry.json"), JSON.stringify(report, null, 2));

    console.log(`\n[SemDB] === Telemetry ===`);
    for (const p of telemetry.phases) {
      console.log(`[SemDB]   ${p.phase.padEnd(16)} ${(p.duration_ms / 1000).toFixed(1)}s  `
        + `${((p.tokens.input || 0) + (p.tokens.output || 0))} tok  $${p.cost_usd.toFixed(4)}  (${p.model})`);
    }
    console.log(`[SemDB]   ${"AGENT STAGE TOTAL".padEnd(16)} ${(agentMs / 1000).toFixed(1)}s  ${totalTok} tok  $${costUsd.toFixed(4)}`);
    console.log(`[SemDB]   code execution     ${codeExecMs ? (codeExecMs / 1000).toFixed(1) + "s" : "(run extract.py + compiled query to populate)"}`);
    console.log(`[SemDB]   WALL CLOCK         ${((Date.now() - wallStart) / 1000).toFixed(1)}s`);
    console.log(`[SemDB]   telemetry -> ${resolve(runDir, "telemetry.json")}`);
  }

  console.log(`\n[SemDB] Done. Artifacts in ${runDir}`);
  console.log(`[SemDB] Extract with:  python3 src/semdb/extract.py --schema ${schemaPath} \\`);
  console.log(`[SemDB]                  --table ${unstructured.path} --modality ${isImage ? "image" : "text"} \\`);
  console.log(`[SemDB]                  ${isImage ? `--image-dir ${args.imageDir} ` : ""}--out ${attrsPath}`);
  console.log(`[SemDB] Then run:      python3 ${codePath}`);
}

main().catch((err) => {
  console.error("[SemDB] Fatal:", err.message);
  process.exit(1);
});
