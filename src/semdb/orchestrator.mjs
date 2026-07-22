/**
 * SemDB Orchestrator — compile a SemBench semantic operator into a relational
 * program via three agents, reusing GenDB's provider/agent plumbing.
 *
 *   Phase A  Schema Designer  : query + schemas        -> schema.json
 *   Phase B  Extractor        : schema + corpus        -> <corpus>_attrs.json   (once, shared)
 *   Phase C  Code Generator   : query + schema + attrs -> compiled_<q>.py
 *   Verify   compiled result  == naive M×N oracle, report model-call reduction
 *
 * The agent runner, template renderer, and telemetry come straight from
 * src/gendb/shared.mjs — SemDB adds only the three prompts and this wiring.
 *
 * Usage:
 *   node src/semdb/orchestrator.mjs --query q7 [--benchmark mmqa]
 *        [--sembench-dir <path to SemBench checkout>] [--out <dir>] [--dry-run]
 *
 * `--dry-run` prints the resolved plan and prompts without spawning agents —
 * useful without API credentials or a SemBench checkout. The self-contained,
 * GPU-free demonstration of the compiled execution lives in ./poc/run_poc.sh.
 */

import { readFile, writeFile, mkdir } from "fs/promises";
import { existsSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

import {
  renderTemplate,
  runAgent,
  readJSON,
  setAgentProvider,
} from "../gendb/shared.mjs";
import { defaults } from "./semdb.config.mjs";
import { config as schemaDesignerConfig } from "./agents/schema-designer/index.mjs";
import { config as extractorConfig } from "./agents/extractor/index.mjs";
import { config as codeGeneratorConfig } from "./agents/code-generator/index.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
  const args = {
    query: null,
    benchmark: defaults.benchmark,
    querySource: defaults.querySource,
    sembenchDir: null,
    out: resolve(__dirname, "runs"),
    agentProvider: defaults.agentProvider,
    dryRun: false,
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--query" && argv[i + 1]) args.query = argv[++i];
    else if (a === "--benchmark" && argv[i + 1]) args.benchmark = argv[++i];
    else if (a === "--query-source" && argv[i + 1]) args.querySource = argv[++i];
    else if (a === "--sembench-dir" && argv[i + 1]) args.sembenchDir = resolve(argv[++i]);
    else if (a === "--out" && argv[i + 1]) args.out = resolve(argv[++i]);
    else if (a === "--dry-run") args.dryRun = true;
  }
  return args;
}

/** Load a SemBench query's SQL and natural-language intent from a checkout. */
async function loadQuery(args) {
  if (!args.sembenchDir) {
    return {
      sql: "-- (no --sembench-dir given; using the built-in Q7 airline-logo example)\n" +
        'SELECT t.Airlines, i.uri\nFROM mmqa.tampa_international_airport t, mmqa.images i\n' +
        'WHERE AI.IF(STRUCT("...is this the logo of the airline?...", t.Airlines, i.uri));',
      nl: "Join airlines to images where the image shows that airline's logo.",
    };
  }
  const base = resolve(args.sembenchDir, "files", args.benchmark, "query");
  const sqlPath = resolve(base, args.querySource, `${args.query}.sql`);
  const nlPath = resolve(base, "natural_language", `${args.query}.json`);
  const sql = await readFile(sqlPath, "utf-8");
  const nlRaw = await readJSON(nlPath);
  const nl = nlRaw?.question || nlRaw?.nl || nlRaw?.description || JSON.stringify(nlRaw);
  return { sql, nl };
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
    model: defaults.agentModels[agentConfig.configKey] || agentConfig.model,
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

  const runDir = resolve(args.out, `${args.benchmark}-${args.query || "example"}`);
  await mkdir(runDir, { recursive: true });

  const { sql, nl } = await loadQuery(args);
  const schemaPath = resolve(runDir, "schema.json");
  const attrsPath = resolve(runDir, "img_attrs.json");
  const codePath = resolve(runDir, `compiled_${args.query || "q"}.py`);

  console.log(`[SemDB] benchmark=${args.benchmark} query=${args.query || "example"}`);
  console.log(`[SemDB] run dir: ${runDir}`);
  console.log(`[SemDB] query:\n${sql}\n`);

  // Phase A — Schema Designer
  await runPhase(schemaDesignerConfig, {
    query_id: args.query || "example",
    query_sql: sql,
    query_nl: nl,
    table_schemas: "(provide via --sembench-dir; see poc/data for the example shape)",
    corpus_name: "images",
    corpus_size: "N",
    modality: "image",
    structured_name: "tampa_international_airport",
    structured_size: "M",
    schema_path: schemaPath,
    existing_schema: "",
  }, runDir, args);

  const schema = args.dryRun ? null : await readJSON(schemaPath);
  if (schema && schema.decomposable === false) {
    console.log(`[SemDB] Query not decomposable: ${schema.rationale}. Falling back to naive execution.`);
    return;
  }

  // Phase B — Extractor (once per corpus; reused by every query over it)
  await runPhase(extractorConfig, {
    corpus_name: "images",
    schema_json: schema ? JSON.stringify(schema, null, 2) : "{{schema.json from Phase A}}",
    corpus_manifest: "poc/data/images.csv",
    modality: "image",
    corpus_size: "N",
    small_model: defaults.extraction.smallImageModel,
    escalation_model: defaults.extraction.escalationImageModel,
    attrs_path: attrsPath,
  }, runDir, args);

  // Phase C — Code Generator
  await runPhase(codeGeneratorConfig, {
    query_id: args.query || "q",
    query_sql: sql,
    schema_json: schema ? JSON.stringify(schema, null, 2) : "{{schema.json from Phase A}}",
    attrs_path: attrsPath,
    attrs_columns: "uri, logo_brand, conf, ocr_text",
    structured_path: "poc/data/airlines.csv",
    structured_columns: "Airlines, Destinations",
    code_path: codePath,
    code_basename: `compiled_${args.query || "q"}.py`,
  }, runDir, args);

  console.log(`\n[SemDB] Done. Artifacts in ${runDir}`);
  console.log(`[SemDB] Verify equivalence with: (see poc/run_poc.sh for the reference harness)`);
}

main().catch((err) => {
  console.error("[SemDB] Fatal:", err.message);
  process.exit(1);
});
