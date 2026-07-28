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

import { readFile, writeFile, mkdir, readdir, copyFile } from "fs/promises";
import { existsSync, readFileSync, writeFileSync, realpathSync } from "fs";
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
import { config as vadarSignatureConfig } from "./agents/vadar-signature/index.mjs";
import { config as vadarApiConfig } from "./agents/vadar-api/index.mjs";
import { config as vadarProgramConfig } from "./agents/vadar-program/index.mjs";
import { config as vadarSolverConfig } from "./agents/vadar-solver/index.mjs";
import { config as queryPlannerConfig } from "./agents/query-planner/index.mjs";
import { config as semanticCodeGeneratorConfig } from "./agents/semantic-code-generator/index.mjs";
import { config as semanticOptimizerConfig } from "./agents/semantic-optimizer/index.mjs";
import { loadBoundAgentSkill } from "./agent-runtime/skill-loader.mjs";
import {
  assertPlanGeneratable,
  finalizeCandidateManifest,
  readAndValidateOptimizerAction,
  readAndValidatePlan,
  writeJsonAtomic,
} from "./agent-runtime/contracts.mjs";
import { runPgoLoop } from "./agent-runtime/pgo-loop.mjs";
import { assertSelectValidationPayload } from "./agent-runtime/feedback.mjs";

/** SQL table-qualifier prefix for a benchmark (e.g. mmqa, cars_dataset). */
function benchPrefix(bench) { return (BENCHMARKS[bench] && BENCHMARKS[bench].prefix) || bench; }
function prefixRe(bench, tail) {
  return new RegExp(String.raw`\b${benchPrefix(bench)}\.(\w+)` + (tail || ""), "gi");
}
import { config as codeGeneratorConfig } from "./agents/code-generator/index.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));

export function parseArgs(argv) {
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
    noGroundTruth: false,  // production mode: never derive/read GT or compute F1
    telemetryCsv: null,    // append the telemetry+metrics row here
    // Execution of the downstream Python steps (extract → compiled query → eval).
    run: false,            // force-run; auto-enabled when --ground-truth-dir is set
    noRun: false,          // disable auto-run
    endpoint: null,        // vLLM/OpenAI base URL for extraction (+ residual)
    apiKey: "EMPTY",
    concurrency: 8,        // in-flight extract.py --endpoint requests (vLLM batches server-side)
    extractModel: null,    // small VLM/LLM id for TEXT corpora (config extraction.small*Model)
    clipModel: null,       // CLIP id for IMAGE corpora (config extraction.clipModel)
    caption: false,        // OpImgCap: caption the image corpus once (needs --endpoint)
    captionModel: null,    // VLM id for captioning (config extraction.captionModel)
    theta: null,
    predCols: "0,1",       // predicted columns to compare vs GT tuple order
    force: false,          // re-run corpus schema design + extraction even if cached
    dryRun: false,
    imageOnly: false,      // run only queries whose corpus is an image table
    direct: false,         // DIRECT mode: skip Schema Designer + extract/compile split;
                           // VADAR 3 agents (Signature→API→Solver) write ONE end-to-end
                           // solve_<q>.py that calls the local vision API and answers the query.
    agentArchitecture: defaults.directAgentArchitecture,
    maxReplans: defaults.maxReplans,
    enableAgentSkills: defaults.enableAgentSkills,
    directOptionsSpecified: false,
    maxIterations: defaults.maxRefineIterations,
    noRefine: false,
    valFile: null,
    valRate: null,          // build a val set at this sampling rate instead of --val-file
    // Retained only so older command lines get an explicit warning. Join validation
    // now samples the complete pair population; pruning changes the estimand.
    valPairTop: null,
    valCertRate: null,      // sealed CERT half, as a fraction of the corpus
    valMethod: "stratified",
    valSeed: 7,
    valStrataK: 5,
    valScoreTilt: 2.0,
    valCallSite: null,      // which AI call site to label, for a multi-predicate query
    oracleModel: null,      // labeling model; defaults to the strong image/text model
    valPlanOnly: false,     // build/cardinality-check validation frames; no Oracle/agents
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
    else if (a === "--no-ground-truth") args.noGroundTruth = true;
    else if (a === "--telemetry-csv" && argv[i + 1]) args.telemetryCsv = resolve(argv[++i]);
    else if (a === "--out" && argv[i + 1]) args.out = resolve(argv[++i]);
    else if (a === "--run") args.run = true;
    else if (a === "--no-run") args.noRun = true;
    else if (a === "--endpoint" && argv[i + 1]) args.endpoint = argv[++i];
    else if (a === "--api-key" && argv[i + 1]) args.apiKey = argv[++i];
    else if (a === "--concurrency" && argv[i + 1]) args.concurrency = parseInt(argv[++i], 10);
    else if (a === "--caption") args.caption = true;
    else if (a === "--caption-model" && argv[i + 1]) args.captionModel = argv[++i];
    else if (a === "--extract-model" && argv[i + 1]) args.extractModel = argv[++i];
    else if (a === "--clip-model" && argv[i + 1]) args.clipModel = argv[++i];
    else if (a === "--theta" && argv[i + 1]) args.theta = argv[++i];
    else if (a === "--pred-cols" && argv[i + 1]) args.predCols = argv[++i];
    else if (a === "--force") args.force = true;
    else if (a === "--image-only") args.imageOnly = true;
    else if (a === "--direct") args.direct = true;
    else if (a === "--agent-architecture" && argv[i + 1]) {
      args.agentArchitecture = argv[++i];
      args.directOptionsSpecified = true;
    }
    else if (a === "--max-replans" && argv[i + 1]) {
      args.maxReplans = Number(argv[++i]);
      args.directOptionsSpecified = true;
    }
    else if (a === "--no-agent-skills") {
      args.enableAgentSkills = false;
      args.directOptionsSpecified = true;
    }
    else if (a === "--dry-run") args.dryRun = true;
    else if (a === "--max-iterations" && argv[i + 1]) args.maxIterations = parseInt(argv[++i], 10);
    else if (a === "--val-file" && argv[i + 1]) args.valFile = resolve(argv[++i]);
    else if (a === "--val-rate" && argv[i + 1]) args.valRate = parseFloat(argv[++i]);
    else if (a === "--val-cert-rate" && argv[i + 1]) args.valCertRate = parseFloat(argv[++i]);
    else if (a === "--val-method" && argv[i + 1]) args.valMethod = argv[++i];
    else if (a === "--val-seed" && argv[i + 1]) args.valSeed = parseInt(argv[++i], 10);
    else if (a === "--val-strata-k" && argv[i + 1]) args.valStrataK = parseInt(argv[++i], 10);
    else if (a === "--val-score-tilt" && argv[i + 1]) args.valScoreTilt = parseFloat(argv[++i]);
    else if (a === "--val-call-site" && argv[i + 1]) args.valCallSite = parseInt(argv[++i], 10);
    else if (a === "--oracle-model" && argv[i + 1]) args.oracleModel = argv[++i];
    else if (a === "--val-plan-only") args.valPlanOnly = true;
    else if (a === "--val-pair-top" && argv[i + 1]) args.valPairTop = parseInt(argv[++i], 10);
    else if (a === "--no-refine") args.noRefine = true;
  }
  if (!["legacy", "pgo"].includes(args.agentArchitecture)) {
    throw new Error(
      `--agent-architecture must be "legacy" or "pgo" (got "${args.agentArchitecture}")`,
    );
  }
  if (!Number.isInteger(args.maxReplans) || args.maxReplans < 0) {
    throw new Error("--max-replans must be a non-negative integer");
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
    if (!args.noGroundTruth && !args.groundTruthDir && b) {
      args.groundTruthDir = resolve(base, b.gtDir);
    }
  }
  // Image root varies by scenario (imageBase): under <dataDir> ("sf"), at the
  // sembench root for repo-relative path columns ("root"), or absolute ("absolute").
  if (args.dataDir && !args.imageDir) {
    const ib = b ? b.imageBase : "sf";
    if (ib === "absolute") args.imageDir = null;                 // path column is absolute
    else if (ib === "root") args.imageDir = args.sembenchDir || null;
    else args.imageDir = resolve(args.dataDir, (b && b.imageRoot) || "images");
  }
  if (args.noGroundTruth) args.groundTruthDir = null;
  return args;
}

/** Fail once, before planning every query, when a requested sf directory is absent. */
export async function validateDataDirectory(args) {
  if (!args.dataDir || existsSync(args.dataDir)) return;
  let available = [];
  try {
    const entries = await readdir(dirname(args.dataDir), { withFileTypes: true });
    available = entries
      .filter((entry) => entry.isDirectory() && /^sf_/.test(entry.name))
      .map((entry) => entry.name.slice(3))
      .sort((a, b) => Number(a) - Number(b));
  } catch { /* the parent itself is missing */ }
  const requested = Number((basename(args.dataDir).match(/^sf_(\d+)$/) || [])[1]);
  const suggestion = available.length
    ? [...available].sort((a, b) =>
        Math.abs(Number(a) - requested) - Math.abs(Number(b) - requested))[0]
    : null;
  const hint = available.length
    ? ` Available scale factors: ${available.join(", ")}.`
    : "";
  throw new Error(
    `data directory does not exist: ${args.dataDir}.${hint}`
    + (suggestion ? ` Nearest available value: --sf ${suggestion}.` : ""));
}

/** Verify an explicitly requested oracle model exists before paying for any work. */
export async function validateEndpointModel(endpoint, model, apiKey = "EMPTY") {
  if (!endpoint || !model) return;
  const url = endpoint.replace(/\/+$/, "") + "/models";
  let response;
  try {
    response = await fetch(url, {
      headers: { Authorization: `Bearer ${apiKey}` },
      signal: AbortSignal.timeout(5000),
    });
  } catch (error) {
    throw new Error(`cannot reach oracle endpoint ${url}: ${error.message}`);
  }
  if (!response.ok) {
    throw new Error(`oracle endpoint ${url} returned HTTP ${response.status}`);
  }
  let payload;
  try { payload = await response.json(); }
  catch { throw new Error(`oracle endpoint ${url} did not return JSON`); }
  const available = (payload?.data || []).map((entry) => entry.id).filter(Boolean);
  if (!available.includes(model)) {
    throw new Error(
      `oracle model '${model}' is not served by ${endpoint}. `
      + `Available model(s): ${available.join(", ") || "(none)"}. `
      + `Restart the endpoint with --model ${model}, or pass a served --oracle-model.`
    );
  }
}

/** Normalize old F1-only outcomes and new metric-aware outcomes. */
export function semdbObjective(outcome) {
  const candidate = outcome?.objective;
  const hasNumber = (value) => value !== null && value !== undefined && value !== ""
    && Number.isFinite(Number(value));
  if (candidate && hasNumber(candidate.value)) {
    return {
      name: candidate.name || "objective",
      value: Number(candidate.value),
      direction: candidate.direction === "minimize" ? "minimize" : "maximize",
      details: candidate.details || null,
    };
  }
  if (hasNumber(outcome?.f1)) {
    return { name: "f1", value: Number(outcome.f1), direction: "maximize", details: null };
  }
  return { name: candidate?.name || "objective", value: null,
           direction: candidate?.direction === "minimize" ? "minimize" : "maximize",
           details: candidate?.details || null };
}

/**
 * Improvement selector — correctness-first, then the query's typed objective.
 * Legacy callers with only `f1` remain supported.
 */
export function checkSemdbImprovement(prev, next) {
  const prevOk = prev && prev.status === "ok";
  const nextOk = next && next.status === "ok";
  if (prevOk && !nextOk) return false;     // regressed to a crash/empty
  if (!prevOk && nextOk) return true;      // fixed a crash/empty
  if (prevOk && nextOk) {
    const before = semdbObjective(prev);
    const after = semdbObjective(next);
    if (before.value == null) return after.value != null;
    if (after.value == null || before.name !== after.name
        || before.direction !== after.direction) return false;
    const improved = before.direction === "minimize"
      ? after.value < before.value
      : after.value > before.value;
    if (improved) return true;
    // Query-level objectives such as top1 are coarse. On an exact tie, prefer the
    // candidate with better operator fidelity; never let the surrogate override a
    // genuine query-metric regression.
    if (after.value === before.value
        && prev.f1 != null && next.f1 != null
        && Number.isFinite(Number(prev.f1)) && Number.isFinite(Number(next.f1))) {
      return Number(next.f1) > Number(prev.f1);
    }
    return false;
  }
  return false;                            // both broken → no improvement
}

/**
 * Iteration-budget gate. `history` = [{ iter, f1, status, improved }].
 *
 * Validation quality selects the best candidate but never shortens the requested
 * experiment. A perfect score on a finite validation sample can be accidental
 * (mmqa q7 scored 1.0 on 40 sampled pairs and 0.1569 on the full join), while a
 * stalled candidate can still be repaired by a later independent agent call.
 */
export function shouldContinueSemdb(history, iteration, maxIter, stallThreshold) {
  if (iteration > maxIter) return { action: "stop", reason: "Max iterations reached" };
  const last = history[history.length - 1];
  if (last && last.status !== "ok") return { action: "continue", reason: "Fix runtime failure first" };
  return { action: "continue", reason: "Iteration budget remains" };
}

/** Reconcile DIRECT-mode wall time without treating every non-agent millisecond as
 * generated-code execution. All stages are sequential in this orchestrator, so the
 * residual is scoring, preflight, artifact I/O, and other orchestration overhead. */
export function directTimingBreakdown(
  wallMs, agentMs, validationSamplingLlmMs, codeExecutionRuns = [],
) {
  const codeExecutionMs = codeExecutionRuns.reduce(
    (total, run) => total + (Number(run?.duration_ms) || 0), 0);
  return {
    agent_stage_ms: Number(agentMs) || 0,
    validation_sampling_llm_ms: Number(validationSamplingLlmMs) || 0,
    code_execution_ms: codeExecutionMs,
    other_overhead_ms: (Number(wallMs) || 0)
      - (Number(agentMs) || 0)
      - (Number(validationSamplingLlmMs) || 0)
      - codeExecutionMs,
  };
}

/**
 * VADAR-generated runtime code is strictly offline. Code-generation agents may run
 * before this point, but the emitted Python must not call a model/network endpoint or
 * use an endpoint-backed semantic-judgement wrapper.
 */
const VADAR_RUNTIME_FORBIDDEN = [
  ["semantic judge API", /\bjudge\s*\(|\b(?:vlm_judge|gen_endpoint)\s*\(/i],
  // Endpoint-backed modules. semvqa (OpImgVQA) and semcaption (OpImgCap) DO call a VLM
  // by design — they belong to the extraction/residual layer, which runs outside this
  // guard. Naming them here keeps generated VADAR code from importing its way around it.
  ["endpoint-backed module",
    /\b(?:semtext|semvqa|semcaption|TextPatch|get_ctx|img_vqa)\b/i],
  ["endpoint/API credential",
    /--endpoint\b|--api-key\b|\.endpoint\b|\.api_?key\b|\b(?:endpoint|api_?key)\s*(?:=|[,):])/i],
  ["OpenAI client", /\b(?:from|import)\s+openai\b|\bOpenAI\s*\(/i],
  ["HTTP/network client", /\b(?:requests|httpx|aiohttp|urllib|socket)\b/i],
  ["shell/network escape", /\b(?:subprocess|Popen|urlopen|curl)\b|\bos\.system\s*\(/i],
];

export function offlineVadarViolations(source) {
  return VADAR_RUNTIME_FORBIDDEN
    .filter(([, pattern]) => pattern.test(source))
    .map(([label]) => label);
}

function validateOfflineVadarFile(path) {
  const violations = offlineVadarViolations(readFileSync(path, "utf-8"));
  if (violations.length) {
    throw new Error(`Refusing to execute non-offline VADAR code ${path}: ${violations.join(", ")}`);
  }
}

/**
 * Static compile gate — runs BEFORE the solver is executed so a syntax error or an
 * undefined name costs one cheap static pass instead of a full corpus run, and so the
 * agent gets an exact location plus source context instead of a truncated stderr tail.
 * Complements validateOfflineVadarFile, which checks compliance rather than compilation.
 *
 * Returns { ok, stage, text, report } and never throws: a broken/missing preflight.py
 * must not stop code generation, so an unusable checker reports ok with stage "skipped".
 */
/**
 * Build (or reuse) an oracle-labeled validation set for one query.
 *
 * Cached under `<out>/_val/<bench>-<query>/`, keyed by the design that produced it, so
 * re-running a query does not re-pay for labels. The key includes the oracle model and
 * the call site because either one changing means the labels answer a different
 * question; it includes the rate and seed because those change which rows were drawn.
 *
 * Returns the path to select.json, or null when no val set could be built. Never
 * throws: a query whose predicate is pairwise (a join) has no per-row label frame, and
 * that must degrade to the existing full-ground-truth scoring rather than kill the run.
 */
export function buildValSet(args, query, spec) {
  const key = [args.valMethod, args.valRate, args.valCertRate ?? 0, args.valSeed,
    args.valStrataK, args.valScoreTilt, spec.oracleModel,
    args.valCallSite ?? "auto",
    // Sampling v2 adds a probability-valid certainty stratum at the score-ranked
    // head; it must not reuse a pre-v2 draw with a different inclusion design.
    "oracleframes-v3",
    spec.importanceBy || "default-score",
    ...(spec.textCols || []),
    // A pairwise design samples a different frame with different keys, so it must
    // never reuse a per-row cache entry (or vice versa).
    ...(spec.pairwise ? ["pairwise", spec.frameKey || "full-frame"] : []),
  ].join("_").replace(/[^\w.-]/g, "");
  const dir = resolve(args.out, "_val", `${args.benchmark}-${query}`, key);
  const selectPath = resolve(dir, "select.json");
  if (existsSync(selectPath)) {
    console.log(`[SemDB] [${query}] reusing cached validation set ${selectPath}`);
    return selectPath;
  }
  // A PAIRWISE frame already carries its own score column (build_pairs.py computed it
  // with one matmul), so importance reads that column instead of re-encoding, and the
  // images come from --pair-image-cols rather than a single --image-col.
  const pairMode = !!spec.pairwise;
  const importanceBy = spec.importanceBy || (pairMode ? "column:pair_score"
    : (spec.isImage ? "clip-similarity" : "query-similarity"));
  const bvArgs = [resolve(__dirname, "build_valset.py"),
    "--corpus", spec.corpusCsv, "--id-col", spec.idCol,
    "--query", query, "--attr", "answer", "--sql", spec.sqlPath,
    "--method", args.valMethod, "--rate", String(args.valRate),
    ...(args.valCertRate ? ["--cert-rate", String(args.valCertRate)] : []),
    "--seed", String(args.valSeed), "--strata-k", String(args.valStrataK),
    "--score-tilt", String(args.valScoreTilt),
    "--strata-by", "score-decile",
    "--importance-by", importanceBy,
    ...(pairMode
      ? ["--pairwise", "--pair-image-cols",
         (spec.pairImageCols ?? ["file2"]).join(","),
         "--image-dir", spec.imageDir || "", "--clip-model", spec.clipModel]
      : (spec.isImage ? ["--image-col", spec.imageCol, "--image-dir", spec.imageDir,
                         "--clip-model", spec.clipModel] : [])),
    ...(spec.textCols || []).flatMap((c) => ["--text-col", c]),
    ...(args.valCallSite != null ? ["--call-site", String(args.valCallSite)] : []),
    "--label-source", "oracle", "--endpoint", spec.endpoint,
    "--oracle-model", spec.oracleModel, "--api-key", args.apiKey || "EMPTY",
    "--concurrency", String(args.concurrency ?? 8),
    // One cache per (benchmark, query) rather than per design: raising the rate then
    // re-pays only for rows never labeled before.
    "--label-cache", resolve(args.out, "_val", `${args.benchmark}-${query}`, "labels.json"),
    "--out", dir];
  console.log(`\n[SemDB] [${query}] building validation set (rate=${args.valRate}, `
    + `${args.valMethod}, oracle=${spec.oracleModel})`);
  const proc = spawnSync("python3", bvArgs, { stdio: "inherit" });
  if (proc.status !== 0 || !existsSync(selectPath)) {
    console.warn(`[SemDB] [${query}] validation set not built (build_valset.py exited `
      + `${proc.status}); falling back to full ground-truth scoring.`);
    return null;
  }
  return selectPath;
}

/** The AI call sites of a query, via predicate.py --json. [] when unreadable — the
 *  caller then behaves as it did before shapes were consulted. */
export function callSites(sqlPath) {
  const pr = spawnSync("python3", [resolve(__dirname, "predicate.py"), sqlPath, "--json"],
    { encoding: "utf-8" });
  if (pr.status !== 0) return [];
  try { return JSON.parse(pr.stdout) || []; } catch { return []; }
}

/** Versioned candidate-domain plan from validation_plan.py. */
export function validationPlan(sqlPath, benchmark, query) {
  const pr = spawnSync("python3", [
    resolve(__dirname, "validation_plan.py"), sqlPath,
    "--benchmark", benchmark || "", "--query", query || "",
  ], { encoding: "utf-8" });
  if (pr.status !== 0) {
    throw new Error(`validation_plan.py exited ${pr.status}: ${(pr.stderr || "").trim()}`);
  }
  try { return JSON.parse(pr.stdout); }
  catch { throw new Error(`validation_plan.py returned invalid JSON for ${query}`); }
}

/** Execute EComm's ordinary CTE/filter prefix before a semantic self join. */
export function buildDeterministicSelfJoinRows(args, query, sqlPath) {
  if (args.benchmark !== "ecomm") {
    throw new Error(`automatic self-join validation is not implemented for `
      + `benchmark '${args.benchmark}'`);
  }
  const products = resolve(args.tableDir || args.dataDir, "ecomm_products.csv");
  if (!existsSync(products)) {
    throw new Error(`normalized EComm product view not found: ${products}`);
  }
  const dir = resolve(args.out, "_val", `${args.benchmark}-${query}`);
  const out = resolve(dir, "deterministic_rows.csv");
  const fbArgs = [
    resolve(__dirname, "frame_builder.py"),
    "--benchmark", args.benchmark, "--query", query,
    "--sql", sqlPath, "--products", products, "--out", out,
  ];
  console.log(`\n[SemDB] [${query}] executing deterministic self-join prefix`);
  const proc = spawnSync("python3", fbArgs, { stdio: "inherit" });
  if (proc.status !== 0 || !existsSync(out)) {
    throw new Error(`frame_builder.py exited ${proc.status}; no deterministic `
      + `self-join frame was produced`);
  }
  return out;
}

/** Materialize an ordered self-join population after deterministic filtering. */
export function buildSelfPairFrame(args, query, spec) {
  const dir = resolve(args.out, "_val", `${args.benchmark}-${query}`);
  const mode = spec.textCol ? "text" : "image";
  const diagonal = spec.includeDiagonal ? "diag" : "nodiag";
  const out = resolve(dir, `pairs_self_${mode}_ordered_${diagonal}.csv`);
  const bpArgs = [
    resolve(__dirname, "build_pairs.py"),
    "--corpus", spec.corpusCsv, "--id-col", spec.idCol,
    ...(spec.textCol ? ["--text-col", spec.textCol]
      : ["--image-col", spec.imageCol]),
    ...(spec.imageDir ? ["--image-dir", spec.imageDir] : []),
    "--clip-model", spec.clipModel, "--ordered",
    ...(spec.includeDiagonal ? ["--include-diagonal"] : []),
    "--out", out,
  ];
  console.log(`\n[SemDB] [${query}] building ordered self-join pair frame `
    + `(${mode}, diagonal=${spec.includeDiagonal ? "included" : "excluded"})`);
  const proc = spawnSync("python3", bpArgs, { stdio: "inherit" });
  if (proc.status !== 0 || !existsSync(out)) {
    throw new Error(`build_pairs.py exited ${proc.status}; no self-join pair frame `
      + `was produced`);
  }
  return out;
}

/**
 * Materialize the CROSS-TABLE pair frame for a join query whose AI predicate is the
 * join condition (mmqa q2a/q7: a structured table joined to the image table). Returns
 * the frame CSV path, or null if it could not be built.
 *
 * The frame is the sampling unit for a pairwise val set: its id column is
 * "<left_id>-<image_id>", the composite key SemBench's own join ground truth lists.
 * Cached next to the val set so re-running a query does not re-encode the corpus.
 */
export function buildPairFrame(args, query, spec) {
  const dir = resolve(args.out, "_val", `${args.benchmark}-${query}`);
  const out = resolve(dir, `pairs${spec.top ? `_top${spec.top}` : ""}.csv`);
  if (existsSync(out)) {
    console.log(`[SemDB] [${query}] reusing cached pair frame ${out}`);
    return out;
  }
  const bpArgs = [resolve(__dirname, "build_pairs.py"),
    "--corpus", spec.leftCsv, "--id-col", spec.leftIdCol, "--text-col", spec.leftTextCol,
    "--right", spec.rightCsv, "--right-id-col", spec.rightIdCol,
    "--right-image-col", spec.rightImageCol,
    ...(spec.imageDir ? ["--right-image-dir", spec.imageDir] : []),
    "--clip-model", spec.clipModel,
    ...(spec.top ? ["--top", String(spec.top)] : []),
    "--out", out];
  console.log(`\n[SemDB] [${query}] building cross-table pair frame `
    + `(${spec.leftTextCol} x ${spec.rightImageCol})`);
  const proc = spawnSync("python3", bpArgs, { stdio: "inherit" });
  if (proc.status !== 0 || !existsSync(out)) {
    console.warn(`[SemDB] [${query}] pair frame not built (build_pairs.py exited `
      + `${proc.status}).`);
    return null;
  }
  return out;
}

/** The column an AI call site reads for one alias, from predicate.py's `columns`
 *  ("t.Track", "i.uri", …). This is what the predicate actually asks about, which a
 *  header heuristic cannot recover. Null when the alias contributes no column. */
export function predicateCol(site, alias) {
  if (!site || !alias) return null;
  const hit = (site.columns || []).find((c) => c.split(".")[0] === alias);
  return hit ? hit.split(".").slice(1).join(".") : null;
}

/** All ordinary-text columns consumed by a per-row AI predicate, resolved against
 * the actual corpus header. Supports both alias.column and unqualified columns.
 * Header order is retained so the rendered row description is stable and natural. */
export function predicateTextCols(site, header) {
  if (!site || !header) return [];
  const available = String(header).split(",").map((c) => c.trim()).filter(Boolean);
  const wanted = new Set((site.columns || []).map((ref) => {
    const parts = String(ref).split(".");
    return parts[parts.length - 1].toLowerCase();
  }));
  return available.filter((c) => wanted.has(c.toLowerCase()));
}

/** The column a query SELECTs from its structured side — the left key of the pair, and
 *  what SemBench's join ground truth lists (q2a -> ID, q7 -> Airlines). Null when the
 *  SELECT does not project that alias. */
export function selectedKeyCol(sql, alias) {
  if (!sql || !alias) return null;
  // Every SELECT..FROM segment, not just the first: a CTE query (mmqa q2b) opens with
  // the CTE's own SELECT, whose aliases are not the outer query's. Take the first
  // segment that actually projects this alias.
  const col = new RegExp(`\\b${alias}\\.([A-Za-z_]\\w*)`);
  for (const m of sql.matchAll(/\bselect\b([\s\S]*?)\bfrom\b/gi)) {
    const hit = m[1].match(col);
    if (hit) return hit[1];
  }
  return null;
}

export function runPreflight(paths, outPath) {
  const files = paths.filter((p) => p && existsSync(p));
  if (!files.length) return { ok: true, stage: "skipped", text: "", report: null };
  const pf = spawnSync("python3", [resolve(__dirname, "preflight.py"), ...files,
    ...(outPath ? ["--out", outPath] : []), "--quiet"], { encoding: "utf-8" });
  if (pf.error || pf.status === null || pf.status > 1) {
    console.warn(`[SemDB] preflight unavailable (${pf.error?.message || `exit ${pf.status}`}); skipping.`);
    return { ok: true, stage: "skipped", text: "", report: null };
  }
  const report = outPath && existsSync(outPath) ? JSON.parse(readFileSync(outPath, "utf-8")) : null;
  return { ok: pf.status === 0, stage: report?.stage ?? null,
           text: renderPreflightText(report), report };
}

/** Lines worth showing an agent: everything else in a run log is progress noise.
 *  `\w*(error|exception)` rather than `\berror\b` because the lines that matter most
 *  are Python exception names — KeyError, ValueError — where no word boundary
 *  precedes "Error". */
const RUN_LOG_SIGNALS =
  /\w*(?:error|exception)\b|\b(?:warn|warning|traceback|fail|failed|failure|fallback|retry|retries|skip|skipped|timeout|missing|unmatched|unknown)\b|^\s*File "/i;

/** The marker the solver prompt contract requires on every diagnostic line. */
const SOLVE_MARKER = /^\s*\[solve\]/;

/**
 * Reduce a captured run log to the handful of lines an agent can act on.
 *
 * Three reductions, in order, because a raw log is both too long to paste into a
 * prompt and mostly repetition:
 *   1. keep only signal lines (see RUN_LOG_SIGNALS), plus the final `tailLines`
 *      lines whatever they say — a crash message rarely matches a keyword and the
 *      end of the log is where it lands;
 *   2. collapse identical lines into one with a count, so "no keyword matched"
 *      repeated 180 times costs one line instead of drowning everything else;
 *   3. cap the result and say how much was dropped, so a truncated view never
 *      reads as a complete one.
 *
 * Pure: takes text, returns { lines, total, omitted }.
 */
export function filterRunLog(text, opts = {}) {
  const { maxLines = 25, tailLines = 8, maxLineChars = 200, maxStructured = 40 } = opts;
  const all = String(text || "").split("\n").map((l) => l.trimEnd()).filter((l) => l.trim());
  if (!all.length) return { lines: [], total: 0, omitted: 0 };

  const collapse = (lines) => {
    const counts = new Map();
    for (const line of lines) {
      const key = line.length > maxLineChars ? `${line.slice(0, maxLineChars)}…` : line;
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    return [...counts].map(([line, n]) => (n > 1 ? `${line}   (× ${n})` : line));
  };

  // The prompt contract makes the solver emit its branch counts and totals AFTER
  // the row loop, so they sit at the end of a log whose middle may be thousands of
  // per-row lines. Selecting them by marker rather than by position guarantees the
  // one part of the log we actually specified survives the tail window and the cap.
  //
  // The contract also splits its own output by shape: per-row diagnostics carry an
  // `id=`, aggregates do not. Aggregates are bounded (<=8 branches plus totals) and
  // are the highest-value lines, so they are kept whole and the per-row ones absorb
  // the truncation — a solver that ignores the per-reason cap must not be able to
  // push its own summary out of the report.
  const marked = all.filter((l) => SOLVE_MARKER.test(l));
  const summary = collapse(marked.filter((l) => !/\bid=/.test(l))).slice(0, maxStructured);
  const perRow = collapse(marked.filter((l) => /\bid=/.test(l)))
    .slice(0, Math.max(0, maxStructured - summary.length));
  const structured = [...perRow, ...summary];

  const rest = all.filter((l) => !SOLVE_MARKER.test(l));
  const tailFrom = Math.max(0, rest.length - tailLines);
  const collapsedRest = collapse(rest.filter((line, i) => i >= tailFrom || RUN_LOG_SIGNALS.test(line)));
  const room = Math.max(0, maxLines - structured.length);

  return {
    lines: [...structured, ...collapsedRest.slice(0, room)],
    total: all.length,
    omitted: Math.max(0, collapsedRest.length - room),
  };
}

/**
 * Run a Python program, persisting its output to `<iterDir>/run.log` and timing it.
 *
 * stdout was previously `inherit`, which put it on the terminal and nowhere else —
 * so whatever the generated program reported about its own execution was gone by
 * the time feedback was assembled. Both streams are now captured; stdout is echoed
 * afterwards to keep the console readable, at the cost of it arriving at the end of
 * the run rather than during it.
 */
export function runPythonLogged(argv, logPath, label) {
  const started = process.hrtime.bigint();
  const proc = spawnSync("python3", argv, {
    stdio: ["inherit", "pipe", "pipe"],
    encoding: "utf-8",
    maxBuffer: 64 * 1024 * 1024,
  });
  const execMs = Number((process.hrtime.bigint() - started) / 1000000n);
  const stdout = proc.stdout || "";
  // ENOBUFS (output past maxBuffer) or a spawn failure leaves the streams null;
  // keep the reason in the log instead of reporting an empty run.
  const stderr = proc.stderr || (proc.error ? `[SemDB] ${label} spawn error: ${proc.error.message}` : "");
  if (stdout) process.stdout.write(stdout);
  try {
    writeFileSync(logPath, `$ python3 ${argv.join(" ")}\n\n${stdout}\n--- stderr ---\n${stderr}\n`);
  } catch (error) {
    console.warn(`[SemDB] could not write ${logPath}: ${error.message}`);
  }
  return { proc, stdout, stderr, execMs, logPath };
}

/** Mirror of preflight.render_text on the JS side, so the feedback block does not
 *  need a second Python round-trip. */
export function renderPreflightText(report) {
  if (!report || report.ok) return "";
  const lines = [`COMPILE FAILED at stage \`${report.stage}\``];
  for (const err of report.errors || []) {
    const where = `${err.file}:${err.line}${err.col ? `:${err.col}` : ""}`;
    lines.push(`${err.error_class} at ${where} — ${err.message}`);
    lines.push(...(err.context || []));
  }
  const warn = report.warnings || [];
  if (warn.length) {
    lines.push(`Also flagged (non-fatal, ${warn.length}):`);
    lines.push(...warn.slice(0, 5).map((w) => `  ${w.error_class} at ${w.file}:${w.line} — ${w.message}`));
  }
  return lines.join("\n");
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

async function runPhase(agentConfig, vars, runDir, args, opts = {}) {
  const systemPromptPath = opts.systemPromptPath || agentConfig.promptPath;
  const userPromptPath = opts.userPromptPath || agentConfig.userPromptPath;
  const systemPrompt = await readFile(systemPromptPath, "utf-8");
  const template = await readFile(userPromptPath, "utf-8");
  const userPrompt = renderTemplate(template, vars);
  const skillRequested = args.enableAgentSkills === false
    ? false
    : (opts.useSkills ?? agentConfig.useSkills ?? args.enableAgentSkills ?? false);
  const skillEnabled = Boolean(agentConfig.skillPath && skillRequested);
  const boundSkill = skillEnabled && agentConfig.skillPath
    ? await loadBoundAgentSkill(agentConfig, true)
    : null;
  const domainSkillsPrompt = boundSkill?.prompt;

  if (args.dryRun) {
    console.log(`\n[SemDB] --- ${agentConfig.name} (dry-run) ---`);
    console.log(`[SemDB] bound skill: ${boundSkill?.name || "(disabled)"}`);
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
    useSkills: skillEnabled,
    domainSkillsPrompt,
  });
  if (result.error) throw new Error(`${agentConfig.name} failed: ${result.error}`);
  return result;
}

/** Render the per-iteration feedback block appended to a generating agent's prompt.
 *  Runtime failures short-circuit to a fix-first block; otherwise metrics + FP/FN. */
export function renderFeedback(prev) {
  const currentObjective = semdbObjective(prev);
  const objectiveLabel = currentObjective.name === "f1" ? "F1" : currentObjective.name;
  const objectiveText = currentObjective.value == null
    ? `${objectiveLabel}=n/a`
    : `${objectiveLabel}=${currentObjective.value} (${currentObjective.direction})`;
  const histLines = (prev.history || [])
    .map((h) => {
      const objective = semdbObjective(h);
      return `  iter ${h.iter}: ${objective.name}=${objective.value == null ? "n/a" : objective.value}`
        + ` ${h.status.toUpperCase()}${h.improved ? " (improved)" : ""}`;
    })
    .join("\n");
  // What the program itself reported while running. Empty for a program that
  // printed nothing, in which case the section is dropped rather than shown blank.
  const log = prev.runLog || { lines: [], total: 0, omitted: 0 };
  const logBlock = log.lines.length
    ? [`## RUNTIME LOG — what your program printed (${log.total} lines, showing ${log.lines.length}${log.omitted ? `, ${log.omitted} more omitted` : ""})`,
       ...log.lines.map((l) => `  ${l}`)].join("\n")
    : "";
  const timing = prev.execMs == null ? "" : ` in ${(prev.execMs / 1000).toFixed(1)}s`;
  if (prev.status !== "ok") {
    // A compile-gate failure never executed, so calling it a crash would send the
    // agent looking for a runtime cause that does not exist.
    const compileFailed = prev.stage === "compile";
    return [
      compileFailed
        ? "\n## LAST ITERATION DID NOT COMPILE — FIX THIS FIRST, NOTHING ELSE"
        : "\n## LAST RUN FAILED — FIX THIS FIRST",
      compileFailed
        ? "The program was NOT executed, so there are no quality numbers this round."
        : `The program ${prev.status === "empty" ? "produced no output rows" : "crashed"}.`,
      "```",
      (prev.stderrTail || "(no stderr captured)"),
      "```",
      compileFailed
        ? "Fix the error above. Do not change anything else."
        : "Diagnose and fix the error before any accuracy work.",
      histLines ? `\n## HISTORY\n${histLines}` : "",
    ].join("\n");
  }
  // Per-row validation mode: diff carries `mistakes` (id/text/predicted/expected).
  if (prev.diff && Array.isArray(prev.diff.mistakes)) {
    const pm = prev.metrics || {};
    const rows = (prev.diff.mistakes || [])
      .map((r) => `  - id=${r.id} predicted=${r.predicted == null ? "MISSING" : r.predicted} expected=${r.expected}  text="${(r.text || "").slice(0, 160)}"`)
      .join("\n") || "  (none)";
    // Accuracy alone is a bad objective at these base rates: with 2% positives,
    // answering `false` everywhere scores 98%. Show the per-class numbers so the
    // agent optimizes recall on the rare class rather than the majority label.
    const q = pm.quality || null;
    const num = (v) => (v == null ? "n/a" : v);
    const fidelityAccuracy = pm.accuracy ?? (
      Number.isFinite(Number(pm.correct)) && Number.isFinite(Number(pm.n)) && Number(pm.n) > 0
        ? Number((Number(pm.correct) / Number(pm.n)).toFixed(4))
        : null
    );
    const qualityLine = q
      ? `## CLASS BREAKDOWN — precision=${num(q.precision)} recall=${num(q.recall)} F1=${num(q.f1)}`
        + `  (tp=${q.tp} fp=${q.fp} fn=${q.fn})`
        + (pm.weighted
            ? `\nThese are corpus estimates: the validation rows were drawn at UNEQUAL rates`
              + ` (${pm.design?.weight_spread}x spread) and are weighted back to the`
              + ` ${pm.design?.N ?? "?"}-row corpus. Raw counts over the labeled rows are`
              + ` precision=${num(pm.unweighted?.precision)} recall=${num(pm.unweighted?.recall)}.`
            : "")
        + (q.recall === 0
            ? `\nRECALL IS ZERO — the program finds none of the positives. A program that`
              + ` always answers "no" would score the same accuracy. Fix this first.`
            : "")
      : "";
    return [
      `\n## LAST RUN — VALIDATION OBJECTIVE ${objectiveText}${timing}`,
      `## PER-ROW INFERENCE accuracy=${fidelityAccuracy ?? "n/a"} — FIDELITY `
        + `(${pm.correct ?? "?"}/${pm.n ?? "?"} labeled rows correct)`,
      qualityLine,
      `## MISLABELED ROWS — ${prev.diff.n_mistakes ?? 0} total, showing ${(prev.diff.mistakes || []).length}:`,
      rows,
      logBlock,
      histLines ? `## HISTORY\n${histLines}` : "",
      "Each row above was inferred WRONG for the query's key attribute. Diagnose WHY:",
      "wrong value-space mapping, an over/under-broad judge/classify prompt, a bad",
      "threshold, or the wrong attribute entirely. Revise the judge/classify/extract",
      "call in the solver. Edit the existing program in place. Keep writing trace_<q>.json.",
    ].join("\n");
  }
  const m = prev.metrics || {};
  const d = prev.diff || {};
  if ([
    "ari", "adjusted_rand_index", "top1", "macro_f1",
    "relative_error", "mape", "spearman", "spearman_correlation",
  ].includes(currentObjective.name)) {
    const detail = currentObjective.details
      ? `\n## OBJECTIVE DETAILS\n${JSON.stringify(currentObjective.details, null, 2)}`
      : "";
    return [
      `\n## LAST RUN — QUERY METRIC ${objectiveText}${timing}`,
      `## QUERY METRICS\n${JSON.stringify(m, null, 2)}`,
      detail,
      logBlock,
      histLines ? `## HISTORY\n${histLines}` : "",
      `Improve the query's ${currentObjective.name} objective. `
        + (currentObjective.direction === "minimize"
          ? "A smaller value is better."
          : "A larger value is better."),
      "Use the metric details and runtime log to revise the generated program.",
    ].filter(Boolean).join("\n");
  }
  const fmt = (rows) => (rows && rows.length)
    ? rows.map((r) => `  - ${typeof r === "object" ? JSON.stringify(r) : r}`).join("\n")
    : "  (none)";
  return [
    `\n## LAST RUN — ${objectiveText} (P=${m.precision} R=${m.recall}, tp=${m.tp} fp=${m.fp} fn=${m.fn})${timing}`,
    `## FALSE POSITIVES (predicted, but wrong) — ${d.fp_total ?? 0} total, showing ${(d.false_positives || []).length}:`,
    fmt(d.false_positives),
    `## FALSE NEGATIVES (missed) — ${d.fn_total ?? 0} total, showing ${(d.false_negatives || []).length}:`,
    fmt(d.false_negatives),
    logBlock,
    histLines ? `## HISTORY\n${histLines}` : "",
    "Diagnose WHY these are wrong and revise the code. Common causes: wrong threshold,",
    "wrong label/value mapping, over-broad predicate, wrong join key, CLIP/LLM prompt",
    "too long or off-target. Edit the existing program in place.",
  ].join("\n");
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
  //      composes a local modality API and calls its offline execution engine.
  //   2. EXECUTE it (orchestrator) on the full corpus -> attrs + <attrs>.meta.json.
  // Both cache per corpus (amortized across all its queries).
  const driverPath = resolve(corpusDir, `extract_${corpus.table}.py`);
  if (!existsSync(driverPath) || args.force) {          // agent artifact step (like schema design); runPhase handles --dry-run
    if (isImage) {
      // IMAGE corpora: strict VADAR 3-agent dynamic-API synthesis (arXiv 2502.06787).
      // Signature -> API -> Program (each a codex/claude agent) compose the predefined
      // vision API (vadar/predefined.py) into extract(image); the program calls
      // vadar_engine.run over the corpus. All 3 share the corpus queries + schema.
      const corpusSql = corpusQueries.map((p) => `-- ${p.query}\n${p.sql}`).join("\n\n");
      const schemaJson = schema ? JSON.stringify(schema, null, 2) : "{{corpus schema.json}}";
      const sigPath = resolve(corpusDir, "_vadar_signatures.txt");
      const helpersPath = resolve(corpusDir, "_vadar_helpers.py");
      const common = { corpus_name: corpus.table, semdb_dir: __dirname };
      record("vadar_signature", await runPhase(vadarSignatureConfig, {
        ...common, query_sql: corpusSql, schema_json: schemaJson, sig_path: sigPath,
      }, corpusDir, args));
      record("vadar_api", await runPhase(vadarApiConfig, {
        ...common, sig_path: sigPath, helpers_path: helpersPath,
      }, corpusDir, args));
      if (!args.dryRun && existsSync(helpersPath)) validateOfflineVadarFile(helpersPath);
      record("vadar_program", await runPhase(vadarProgramConfig, {
        ...common, schema_json: schemaJson, helpers_path: helpersPath, driver_path: driverPath,
        header: cols.join(", "), id_col: idCol, image_col: imageCol,
      }, corpusDir, args));
      if (!args.dryRun && existsSync(driverPath)) validateOfflineVadarFile(driverPath);
    } else {
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
      if (!args.dryRun && existsSync(driverPath)) validateOfflineVadarFile(driverPath);
    }
  } else if (existsSync(driverPath)) {
    console.log(`[SemDB] reuse cached corpus extractor driver: ${driverPath}`);
  }
  if (doRun && !args.dryRun && existsSync(driverPath) && (!existsSync(attrsPath) || args.force)) {
    // Generated extraction drivers run locally. Image drivers use CV+CLIP; text drivers
    // use deterministic string/regex helpers. Neither generated runtime receives an
    // endpoint or API key.
    const imageModel = args.clipModel || defaults.extraction.clipModel;
    validateOfflineVadarFile(driverPath);
    if (isImage) {
      const helpersPath = resolve(corpusDir, "_vadar_helpers.py");
      if (existsSync(helpersPath)) validateOfflineVadarFile(helpersPath);
    }
    const exArgs = [driverPath, corpus.path, attrsPath, "--schema", schemaPath,
      "--model", (isImage ? imageModel : extractModel),
      ...(isImage ? ["--image-dir", args.imageDir] : []),
      ...(args.theta != null ? ["--theta", String(args.theta)] : [])];
    console.log(`\n[SemDB] Extracting corpus ${corpus.table} (once): python3 ${exArgs.join(" ")}`);
    const ex = spawnSync("python3", exArgs, { stdio: "inherit" });
    if (ex.status !== 0) console.warn(`[SemDB] extraction exited ${ex.status}.`);
  } else if (existsSync(attrsPath)) {
    console.log(`[SemDB] reuse cached corpus attrs: ${attrsPath}`);
  }

  // --- OpImgCap: one caption per corpus image, amortized like the attribute table ----
  // A caption is the cross-modal proxy that lets the cheap TEXT side answer an image
  // predicate. It needs a VLM, so it cannot live in the (offline) generated program —
  // it runs here, once per CORPUS, and every query in the family reads the column.
  const captionsPath = resolve(corpusDir, "captions.json");
  if (args.caption && isImage && doRun && !args.dryRun) {
    if (!args.endpoint) {
      console.warn("[SemDB] --caption needs --endpoint (OpImgCap is a VLM call); skipping.");
    } else if (existsSync(captionsPath) && !args.force) {
      console.log(`[SemDB] reuse cached corpus captions: ${captionsPath}`);
    } else {
      const capArgs = [resolve(__dirname, "semcaption.py"), corpus.path, captionsPath,
        "--id-col", idCol, "--image-col", imageCol,
        ...(args.imageDir ? ["--image-dir", args.imageDir] : []),
        "--model", args.captionModel || defaults.extraction.captionModel,
        "--endpoint", args.endpoint, "--concurrency", String(args.concurrency ?? 8)];
      console.log(`\n[SemDB] Captioning corpus ${corpus.table} (once): python3 ${capArgs.join(" ")}`);
      const cap = spawnSync("python3", capArgs, { stdio: "inherit" });
      if (cap.status !== 0) console.warn(`[SemDB] captioning exited ${cap.status}.`);
    }
  }

  const extMeta = await readJSON(attrsPath + ".meta.json");
  const capMeta = await readJSON(captionsPath + ".meta.json");
  const sd = phases.filter((p) => p.phase === "schema_designer");
  const ext = phases.filter((p) => p.phase === "extractor" || p.phase.startsWith("vadar_"));
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
    // Amortized across the whole query family, like extraction — not per query.
    captioning: capMeta ? {
      sec: capMeta.elapsed_sec ?? null,
      calls: capMeta.llm_calls ?? null,
      rows: capMeta.rows ?? null,
      model: args.captionModel || defaults.extraction.captionModel,
    } : null,
  };
  if (!args.dryRun) await writeFile(resolve(corpusDir, "corpus_telemetry.json"), JSON.stringify(corpusTelemetry, null, 2));
  return { corpusDir, schemaPath, attrsPath, captionsPath, schema, corpusTelemetry, idCol, textCol, imageCol, extractModel, isImage, modality };
}

/** Per-row validation scoring for one iteration: run evaluate.py --score-inference on
 *  the solver's trace_<q>.json vs val.json, read a typed query objective plus fidelity
 *  diagnostics. status "ok" requires a trace file. */
async function scoreInference(args, query, iterDir, corpusCsv, valFile, diffPath) {
  const tracePath = resolve(iterDir, `trace_${query}.json`);
  if (!existsSync(tracePath)) return { status: "empty", f1: null, metrics: null, diff: null };
  const evArgs = [resolve(__dirname, "evaluate.py"), "--score-inference",
    "--trace", tracePath, "--val-file", valFile,
    "--query", query, "--benchmark", args.benchmark,
    "--emit-diff", diffPath, "--diff-cap", String(defaults.refineSampleCap),
    ...(corpusCsv ? ["--corpus-csv", corpusCsv] : [])];
  const ev = spawnSync("python3", evArgs, { stdio: "inherit" });
  if (ev.status !== 0) console.warn(`[SemDB] evaluate.py --score-inference exited ${ev.status}.`);
  const out = await readJSON(diffPath);
  if (!out) return { status: "ok", f1: null, metrics: null, diff: null };
  // `estimates` is present only for an unequal-probability design; when it is, IT is
  // the number that describes the corpus and the unweighted accuracy does not.
  const est = out.estimates || out.unweighted || null;
  const weighted = !!(out.design && out.design.weighted && out.estimates);
  return {
    status: "ok", f1: weighted ? (est.f1 ?? out.accuracy) : out.accuracy,
    objective: out.objective || null,
    metrics: {
      accuracy: out.accuracy, n: out.n, correct: out.correct,
      quality: est, unweighted: out.unweighted || null,
      design: out.design || null, weighted,
    },
    diff: { mistakes: out.mistakes || [], n_mistakes: out.n_mistakes ?? 0, sampled: out.sampled },
  };
}

/** Run evaluate.py (with --emit-diff) for one iteration and read the outcome back.
 *  Returns { status, f1, metrics, diff, stderrTail }. status: "ok"|"empty" (crash is
 *  detected by the caller from the run step). Only writes results.csv when finalize. */
async function scoreWithDiff(args, query, planObj, telePath, resultsCsv, diffPath, csvPath, finalize) {
  if (args.dryRun) return { status: "ok", f1: null, metrics: null, diff: null };
  if (!existsSync(resultsCsv)) return { status: "empty", f1: null, metrics: null, diff: null };
  const gt = await resolveGroundTruth(args.groundTruthDir, query, args.scaleFactor);
  if (!gt) return { status: "ok", f1: null, metrics: null, diff: null };
  const evArgs = [resolve(__dirname, "evaluate.py"), "--telemetry", telePath,
    "--ground-truth", gt.file, "--pred", resultsCsv, "--pred-cols", args.predCols,
    "--query", query, "--benchmark", args.benchmark,
    "--emit-diff", diffPath, "--diff-cap", String(defaults.refineSampleCap),
    ...(finalize ? ["--csv", csvPath] : ["--csv", resolve(dirname(diffPath), "_scratch_results.csv")]),
    ...(args.groundTruthDir ? ["--ground-truth-dir", args.groundTruthDir] : []),
    ...(args.scaleFactor ? ["--sf", String(args.scaleFactor)] : [])];
  const ev = spawnSync("python3", evArgs, { stdio: "inherit" });
  if (ev.status !== 0) console.warn(`[SemDB] evaluate.py exited ${ev.status}.`);
  // Read metrics from the DIFF json (it carries f1/precision/recall/tp/fp/fn), NOT from
  // telePath: during iterations telemetry.json does not exist yet, so evaluate.py can't
  // persist metrics into it. The diff file is always written when there is ground truth.
  const diff = await readJSON(diffPath);
  if (!diff) return { status: "ok", f1: null, metrics: null, diff: null };
  const metrics = {
    ...(diff.query_metrics || {}),
    f1: diff.f1 ?? diff.query_metrics?.f1 ?? null,
    precision: diff.precision ?? diff.query_metrics?.precision ?? null,
    recall: diff.recall ?? diff.query_metrics?.recall ?? null,
    tp: diff.tp ?? null, fp: diff.fp ?? diff.fp_total ?? null, fn: diff.fn ?? diff.fn_total ?? null,
  };
  return { status: "ok", f1: metrics.f1, objective: diff.objective || null, metrics, diff };
}

/**
 * GenDB-style per-query refinement loop. iter_0 = genFirst(); iters 1..maxIter =
 * regen(feedback) → keep-or-rollback. Best code wins; the best iteration's results
 * CSV is copied back to `resultsCsv`. No-op (single shot) when maxIter is 0.
 *
 * Closure contract (each generates code AND runs it, returning the RUN outcome;
 * scoreIter then scores that run — no shared mutable state between them):
 *   genFirst(iterDir, iterCode, iterCsv) -> { status: "ok"|"crash"|"empty", stderr }
 *   regen(iterDir, iterCode, iterCsv, feedback) -> { status, stderr }
 *   scoreIter(iterDir, iterCode, iterCsv, runOutcome) -> { status, f1, metrics, diff, stderrTail }
 */
async function refineLoop({ args, query, runDir, codeBasename, resultsCsv, genFirst, regen, scoreIter }) {
  const maxIter = args.noRefine ? 0 : (args.maxIterations ?? defaults.maxRefineIterations);
  const iter0Dir = resolve(runDir, "iter_0");
  await mkdir(iter0Dir, { recursive: true });
  const iter0Code = resolve(iter0Dir, codeBasename);
  const iter0Csv = resolve(iter0Dir, basename(resultsCsv));

  const run0 = await genFirst(iter0Dir, iter0Code, iter0Csv);
  let best = { iter: 0, dir: iter0Dir, code: iter0Code, csv: iter0Csv,
               outcome: await scoreIter(iter0Dir, iter0Code, iter0Csv, run0) };
  const historyEntry = (iteration, outcome, improved) => {
    const objective = semdbObjective(outcome);
    return {
      iter: iteration, f1: outcome.f1, objective,
      objective_name: objective.name, objective_value: objective.value,
      objective_direction: objective.direction,
      status: outcome.status, improved,
    };
  };
  const history = [historyEntry(0, best.outcome, true)];

  // GLOBAL CONSTRAINT: refinement engages only with a measurable typed objective.
  // Single-shot when iter_0 ran successfully but no validation/final metric is available.
  // A crash/empty WITH an evaluator keeps iterating (fix-first),
  // matching shouldContinueSemdb, which returns "continue" when the last run is not "ok".
  // --dry-run never actually executes iter_0 (genFirst/runCompiled are no-ops), so its
  // status is "empty" rather than "ok" — treat that as no-signal too, otherwise the loop
  // would try to seed iter_1 from a compiled_<query>.py that dry-run never wrote.
  const noSignal = args.dryRun
    || (best.outcome.status === "ok" && semdbObjective(best.outcome).value == null);
  const effectiveMaxIter = noSignal ? 0 : maxIter;
  if (noSignal && maxIter > 0) {
    console.log(`[SemDB] [${query}] no measurable objective — single-shot, skipping refinement.`);
  }

  for (let iteration = 1; iteration <= effectiveMaxIter; iteration++) {
    const decision = shouldContinueSemdb(history, iteration, maxIter, defaults.refineStallThreshold);
    console.log(`[SemDB] [${query}] --- refine ${iteration}/${maxIter} --- ${decision.action}: ${decision.reason}`);
    if (decision.action === "stop") break;

    const itDir = resolve(runDir, `iter_${iteration}`);
    await mkdir(itDir, { recursive: true });
    const itCode = resolve(itDir, codeBasename);
    const itCsv = resolve(itDir, basename(resultsCsv));
    // seed from the best code so far
    await writeFile(itCode, await readFile(best.code, "utf-8"));

    const feedback = renderFeedback({
      status: best.outcome.status, f1: best.outcome.f1, objective: best.outcome.objective,
      metrics: best.outcome.metrics,
      diff: best.outcome.diff, stderrTail: best.outcome.stderrTail,
      stage: best.outcome.stage, history,
    });
    const run = await regen(itDir, itCode, itCsv, feedback);
    const outcome = await scoreIter(itDir, itCode, itCsv, run);
    const improved = checkSemdbImprovement(best.outcome, outcome);
    history.push(historyEntry(iteration, outcome, improved));
    if (improved) {
      const before = semdbObjective(best.outcome);
      const after = semdbObjective(outcome);
      console.log(`[SemDB] [${query}] iter ${iteration} improved `
        + `(${before.name} ${before.value} → ${after.value}). Keeping.`);
      best = { iter: iteration, dir: itDir, code: itCode, csv: itCsv, outcome };
    } else {
      console.log(`[SemDB] [${query}] iter ${iteration} did not improve. Rolling back.`);
    }
  }
  // Promote the best iteration's artifacts to the run root.
  if (existsSync(best.code)) await writeFile(resolve(runDir, codeBasename), await readFile(best.code, "utf-8"));
  if (existsSync(best.csv)) await writeFile(resultsCsv, await readFile(best.csv, "utf-8"));
  return { bestIter: best.iter, bestF1: best.outcome.f1,
           bestObjective: semdbObjective(best.outcome), stopReason: history, history };
}

// ---------------------------------------------------------------------------
// Per-query Phase C + execute + evaluate (reuses the corpus schema + attrs).
// ---------------------------------------------------------------------------
async function runQueryCodegen(args, planObj, art, csvPath) {
  const { query, sql, structured, plan, corpus } = planObj;
  const wallStart = Date.now();
  const runDir = resolve(args.out, `${args.benchmark}-${query}`);
  await mkdir(runDir, { recursive: true });
  const resultsCsv = resolve(runDir, `${query}_results.csv`);
  const { schemaPath, attrsPath, schema, corpusTelemetry, extractModel } = art;
  const doRun = !args.dryRun && (args.run || !!args.groundTruthDir) && !args.noRun;

  const phases = [];
  const record = makeRecorder(args, phases);

  console.log(`\n[SemDB] ---- ${query}  (corpus ${corpus.table}, ${plan.type}) ----`);

  const codeBasename = `compiled_${query}.py`;
  const telePath = resolve(runDir, "telemetry.json");

  const cgVars = (iterCode, querySql) => ({
    query_id: query, query_sql: querySql,
    schema_json: schema ? JSON.stringify(schema, null, 2) : "{{corpus schema.json}}",
    attrs_path: attrsPath, attrs_columns: "(schema attributes + conf)",
    structured_path: structured.path || "(structured table path)",
    structured_columns: "(see table headers)",
    code_path: iterCode, code_basename: codeBasename,
  });

  const runCompiled = (iterDir, iterCode, iterCsv) => {
    if (!(doRun && existsSync(iterCode) && existsSync(attrsPath))) return { status: "empty", stderr: "" };
    const pre = runPreflight([iterCode], resolve(iterDir, "preflight.json"));
    if (!pre.ok) {
      console.warn(`[SemDB] [${query}] preflight failed (${pre.stage}) — not executing.\n${pre.text}`);
      return { status: "crash", stage: "compile", preflight: pre.report, stderr: pre.text };
    }
    const cqArgs = [iterCode, structured.path, attrsPath, iterCsv,
      ...(args.endpoint ? ["--endpoint", args.endpoint, "--api-key", args.apiKey, "--model", extractModel] : [])];
    console.log(`\n[SemDB] Running compiled query: python3 ${cqArgs.join(" ")}`);
    const { proc, stdout, stderr, execMs, logPath } =
      runPythonLogged(cqArgs, resolve(iterDir, "run.log"), "compiled query");
    const out = { stderr, stdout, execMs, logPath };
    if (proc.status !== 0) {
      console.warn(`[SemDB] compiled query exited ${proc.status} (${(execMs / 1000).toFixed(1)}s) — log: ${logPath}`);
      return { status: "crash", ...out };
    }
    console.log(`[SemDB] compiled query ok (${(execMs / 1000).toFixed(1)}s) — log: ${logPath}`);
    return { status: "ok", ...out };
  };

  const genFirst = async (iterDir, iterCode, iterCsv) => {
    record("code_generator", await runPhase(codeGeneratorConfig, cgVars(iterCode, sql), iterDir, args));
    return runCompiled(iterDir, iterCode, iterCsv);
  };
  const regen = async (iterDir, iterCode, iterCsv, feedback) => {
    record("code_generator", await runPhase(codeGeneratorConfig, cgVars(iterCode, sql + "\n\n" + feedback), iterDir, args));
    return runCompiled(iterDir, iterCode, iterCsv);
  };
  const scoreIter = async (iterDir, iterCode, iterCsv, run) => {
    const diffPath = resolve(iterDir, "diff.json");
    const scored = await scoreWithDiff(args, query, planObj, telePath, iterCsv, diffPath, csvPath, false);
    const status = run.status === "crash" ? "crash" : (existsSync(iterCsv) ? "ok" : "empty");
    return { ...scored, status, stage: run.stage ?? null,
             execMs: run.execMs ?? null,
             runLog: filterRunLog([run.stdout, run.stderr].filter(Boolean).join("\n")),
             stderrTail: (run.stderr || "").split("\n").slice(-40).join("\n") };
  };

  const { bestIter, bestF1, bestObjective, history } = await refineLoop({
    args, query, runDir, codeBasename, resultsCsv,
    genFirst, regen, scoreIter,
  });

  if (args.dryRun) return null;

  // --- Per-query telemetry (codegen + residual) + shared corpus (amortized) ---
  // record("code_generator", ...) pushes one phase per iteration; sum across all of them
  // rather than taking only the first (iter_0) via .find, which would undercount cost.
  const cgPhases = phases.filter((p) => p.phase === "code_generator");
  const cg = {
    duration_ms: cgPhases.reduce((s, p) => s + (p.duration_ms || 0), 0),
    cost_usd:    cgPhases.reduce((s, p) => s + (p.cost_usd || 0), 0),
    llm_calls:   cgPhases.reduce((s, p) => s + (p.llm_calls || 0), 0),
    tokens: cgPhases.reduce((t, p) => ({ input: (t.input || 0) + (p.tokens?.input || 0),
                                         output: (t.output || 0) + (p.tokens?.output || 0) }), {}),
  };
  const cqMeta = await readJSON(resolve(runDir, `iter_${bestIter}`, `compiled_${query}.meta.json`));
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
    refine: { iterations: history.length - 1, best_iteration: bestIter,
              max_iterations: args.noRefine ? 0 : args.maxIterations,
              objective: bestObjective.name,
              objective_direction: bestObjective.direction,
              objective_history: history.map((item) => item.objective),
              f1_history: history },
    phases,
  };
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

  // (3) finalize: score the promoted best CSV → merge metrics into telemetry + append results.csv row
  if (doRun) {
    const gt = await resolveGroundTruth(args.groundTruthDir, query, args.scaleFactor);
    if (gt && existsSync(resultsCsv)) {
      await scoreWithDiff(args, query, planObj, telePath, resultsCsv, resolve(runDir, "diff.json"), csvPath, true);
    }
  }

  // --- Score against ground truth + append CSV (re-read telePath, merged by finalize above) ---
  if (doRun && gt && existsSync(resultsCsv)) {
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

/** Pick the filename + absolute-path columns from an image-manifest header. */
function pickImageCols(header) {
  const cols = header.split(",").map((c) => c.trim());
  const find = (res) => cols.find((c) => res.some((re) => re.test(c))) || "";
  const filepath = find([/filepath/i, /image_path/i, /\buri\b/i, /\bpath\b/i]);
  const filename = find([/filename/i, /\bfile\b/i, /\bimage\b/i]) || filepath;
  return { filename, filepath };
}

/**
 * DIRECT mode (per query): the strict VADAR 3 agents (Signature → API → Solver) write ONE
 * end-to-end `solve_<q>.py` that composes the LOCAL vision API and answers the whole query
 * (join/filter included) — NO Schema Designer, NO extract/attrs/compile split. Execute it,
 * then score its result CSV exactly like the compiled path.
 */
async function runQueryDirectCore(args, planObj, csvPath, architecture) {
  const { query, sql, nl, tables, corpus, structured } = planObj;
  const wallStart = Date.now();
  const runDir = resolve(args.out, `${args.benchmark}-${query}`);
  await mkdir(runDir, { recursive: true });
  const solvePath = resolve(runDir, `solve_${query}.py`);
  const resultsCsv = resolve(runDir, `${query}_results.csv`);
  const doRun = !args.dryRun && (args.run || !!args.groundTruthDir) && !args.noRun;

  const phases = [];
  const record = makeRecorder(args, phases);
  console.log(`\n[SemDB] ---- ${query}  (DIRECT ${architecture}: solver over ${corpus.table}) ----`);

  // Describe every referenced table (header + path) for the solver.
  const tableLines = [];
  for (const t of tables) {
    const kind = t.isImages ? "image manifest" : (t.modality || "table");
    tableLines.push(`- ${t.table} (${kind}): path=${t.path}\n    columns: ${await headerOf(t.path)}`);
  }
  const isImage = corpus.isImage || corpus.modality === "image";
  const directSqlPath = args.queryDir
    ? resolve(args.queryDir, `${query}.sql`)
    : null;
  const directSites = directSqlPath ? callSites(directSqlPath) : [];
  const directPairSite = directSites.find((site) => site.shape === "pairwise");
  const queryPairKind = directPairSite
    ? (new Set(directPairSite.bases || []).size < 2 ? "self" : "cross")
    : "";
  let imgFilenameCol = "", imgFilepathCol = "", imageDir = "";
  if (isImage) {
    const imgHeader = await headerOf(corpus.path);
    ({ filename: imgFilenameCol, filepath: imgFilepathCol } = pickImageCols(imgHeader));
    imageDir = args.imageDir || (corpus.path ? resolve(dirname(corpus.path), "images") : "");
  }

  const codeBasename = `solve_${query}.py`;
  const clipModel = args.clipModel || defaults.extraction.clipModel;
  const dataDir = args.tableDir || args.dataDir;
  const telePath = resolve(runDir, "telemetry.json");

  // Per-row validation mode: a sampled, oracle-labeled val.json is the only GT the
  // refinement loop reads. Either supplied with --val-file or built here from
  // --val-rate. Image corpora are included: the solver honours --only-ids and writes
  // trace_<q>.json in both modalities.
  let valFile = args.valFile;
  let valPairwise = false;      // the val set keys on "<left_id>-<image_id>"
  let valPairKind = "";         // "cross" | "self"; affects the solver trace contract
  let valPairIncludeDiagonal = false;
  let valCorpusCsv = corpus.path;
  let valRowIdCol = "";
  let valLeftIdCol = "";        // structured-side key column of a pair id
  let valKeyExample = "";       // a real pair id, shown to the solver verbatim
  const buildsValidation = !valFile && !!args.valRate && !args.noRefine;
  const validationStarted = Date.now();
  if (!valFile && args.valRate && !args.noRefine) {
    if (!args.endpoint && !args.valPlanOnly) {
      console.warn(`[SemDB] [${query}] --val-rate needs --endpoint (the oracle is a `
        + `model call); falling back to full ground-truth scoring.`);
    } else if (!args.queryDir) {
      console.warn(`[SemDB] [${query}] --val-rate needs --query-dir (the oracle labels `
        + `against the query SQL); falling back to full ground-truth scoring.`);
    } else {
      // DIRECT mode never calls ensureCorpus, so the corpus id/text/image columns are
      // not already resolved here — derive them the same way ensureCorpus does.
      const { idCol, textCol, imageCol } = await corpusCols(corpus, args);
      valRowIdCol = idCol;
      const sqlPath = resolve(args.queryDir, `${query}.sql`);
      const oracleModel = args.oracleModel
        || (isImage ? defaults.extraction.strongImageModel : defaults.extraction.smallTextModel);
      // A query whose AI predicate IS the join condition has no per-row label frame:
      // the sampling unit is the (structured row, image) pair. Materialize that frame
      // first, then sample it as an ordinary corpus keyed on "<id1>-<id2>".
      const sites = callSites(sqlPath);
      if (sites.length > 1 && args.valPlanOnly) {
        const vplan = validationPlan(sqlPath, args.benchmark, query);
        if (vplan.candidate?.unit === "tuple") {
          const deterministic = buildDeterministicSelfJoinRows(
            args, query, sqlPath);
          const deterministicMeta = await readJSON(
            deterministic + ".meta.json");
          const rows = deterministicMeta?.output_rows;
          const arity = vplan.candidate.arity;
          if (!Number.isInteger(rows) || !Number.isInteger(arity)) {
            throw new Error(`[SemDB] [${query}] tuple validation plan has no `
              + `logical row count/arity.`);
          }
          const population = rows ** arity;
          const sampleN = Math.ceil(population * args.valRate);
          console.log(`[SemDB] [${query}] multi-site validation population: `
            + `${rows}^${arity} = ${population} ordered candidate tuples; `
            + `rate=${args.valRate} -> ${sampleN} validation tuples.`);
          console.log(`[SemDB] [${query}] ${sites.length} AI sites compose via `
            + `${vplan.candidate.composition?.kind || "typed composition"}.`);
          console.log(`[SemDB] [${query}] validation plan complete; `
            + `multi-site Oracle bundle/execution is the next implementation phase.`);
          return {
            query, validation_plan: vplan, deterministic_rows: rows,
            candidate_population: population, sample_n: sampleN,
          };
        }
        console.log(`[SemDB] [${query}] typed/grouped multi-site plan: `
          + `${(vplan.candidate?.kinds || []).join(" + ")}. Candidate-frame `
          + `materialization is deferred to the typed-operator phase.`);
        return { query, validation_plan: vplan };
      }
      const pairSite = sites.find((s) => s.shape === "pairwise");
      const selectedSite = args.valCallSite != null ? sites[args.valCallSite] : null;
      const rowSite = selectedSite?.shape === "per_row"
        ? selectedSite
        : (sites.filter((s) => s.shape === "per_row").length === 1
            ? sites.find((s) => s.shape === "per_row") : null);
      const semanticTextCols = isImage ? [] :
        predicateTextCols(rowSite, await headerOf(corpus.path));
      // Cross-table vs SELF-join is decided by the call site's own `bases`, not by the
      // plan's table list. ecomm q9 pairs p1/p2 which BOTH resolve to
      // `product_selection`; the plan still offers `styles` as a structured side, so a
      // path test would happily build a styles x IMAGES frame for a predicate that
      // never looks at styles. Distinct bases is the real question.
      const distinctBases = pairSite ? new Set(pairSite.bases || []).size : 0;
      const crossTable = pairSite && isImage && distinctBases >= 2
        && structured && structured.path && structured.path !== corpus.path;
      if (pairSite && distinctBases < 2) {
        const vplan = validationPlan(sqlPath, args.benchmark, query);
        if (vplan.candidate?.unit !== "pair") {
          throw new Error(`[SemDB] [${query}] validation plan did not produce a `
            + `single pair candidate domain.`);
        }
        const deterministic = buildDeterministicSelfJoinRows(
          args, query, sqlPath);
        // The normalized description column contains embedded newlines, so the
        // generic newline counter is not a CSV-record counter here. frame_builder
        // writes the logical row count after DuckDB execution.
        const deterministicMeta = await readJSON(deterministic + ".meta.json");
        const filteredRows = deterministicMeta?.output_rows;
        if (!Number.isInteger(filteredRows)) {
          throw new Error(`[SemDB] [${query}] deterministic frame metadata has no `
            + `logical output_rows count.`);
        }
        valPairIncludeDiagonal = !!vplan.candidate.include_diagonal;
        const pairPopulation = valPairIncludeDiagonal
          ? (filteredRows ?? 0) * (filteredRows ?? 0)
          : (filteredRows ?? 0) * Math.max(0, (filteredRows ?? 0) - 1);
        const pairSample = Math.ceil(pairPopulation * args.valRate);
        console.log(`[SemDB] [${query}] self-join validation population: `
          + `${filteredRows} filtered rows -> ${pairPopulation} ordered pairs `
          + `(diagonal=${valPairIncludeDiagonal ? "included" : "excluded"}); `
          + `rate=${args.valRate} -> ${pairSample} validation rows.`);
        const frame = buildSelfPairFrame(args, query, {
          corpusCsv: deterministic, idCol: "id",
          ...(isImage ? { imageCol: "filename", imageDir }
            : { textCol: "semantic_text" }),
          clipModel, includeDiagonal: valPairIncludeDiagonal,
        });
        if (args.valPlanOnly) {
          console.log(`[SemDB] [${query}] validation plan complete; `
            + `--val-plan-only skips Oracle labeling and agent execution.`);
          return {
            query,
            validation_plan: vplan,
            deterministic_rows: filteredRows,
            candidate_population: pairPopulation,
            sample_n: pairSample,
            frame,
          };
        }
        valCorpusCsv = frame;
        valFile = buildValSet(args, query, {
          corpusCsv: frame, idCol: "pair_id", sqlPath,
          isImage, pairwise: true,
          pairImageCols: isImage ? ["file1", "file2"] : [],
          textCols: isImage ? [] : ["text1", "text2"],
          imageDir, clipModel, endpoint: args.endpoint, oracleModel,
          importanceBy: "column:pair_score",
          frameKey: `self-ordered-${valPairIncludeDiagonal ? "diag" : "nodiag"}`,
        });
        valPairwise = !!valFile;
        valPairKind = "self";
        valLeftIdCol = "id";
        const firstRow = (await readFile(frame, "utf-8")).split(/\r?\n/)[1] || "";
        valKeyExample = firstRow.split(",")[0] || "<left_id>-<right_id>";
      } else if (crossTable) {
        const leftAlias = (pairSite.aliases || [])[0];
        // Pair/trace identity is a physical row identity, not necessarily a projected
        // value. q7 projects Airlines, but 200 rows contain only 135 distinct airline
        // names; using Airlines as the pair id makes keys collide. The configured key
        // (or first column, row_id for q7) is unique while the solver may still project
        // Airlines in its result CSV.
        const leftPhysical = await corpusCols(structured, args);
        const leftIdCol = leftPhysical.idCol;
        // The text the predicate ASKS ABOUT, from the call site's own column list —
        // not a header heuristic. q2a's predicate reads t.Track (the racetrack name);
        // guessing from the header picks the last column, `Condition` ("Firm"/"Fast"),
        // and CLIP then scores logos against track surface conditions.
        const leftTextCol = predicateCol(pairSite, leftAlias)
          || (await corpusCols(structured, args)).textCol;
        console.log(`[SemDB] [${query}] pairwise call site (${pairSite.reason}) — `
          + `pair frame ${structured.table}.${leftTextCol} x ${corpus.table}`);
        // The validation population for a join is the complete Cartesian product.
        // Pruning to a global CLIP top-K changes the estimand and made q7 look perfect
        // on validation (40/40) while scoring 0.1569 F1 over all 40,000 pairs.
        // `build_valset --rate r` therefore draws ceil(|L| * |R| * r) rows.
        if (args.valPairTop) {
          console.warn(`[SemDB] [${query}] ignoring deprecated --val-pair-top `
            + `${args.valPairTop}: join validation samples the full pair population.`);
        }
        const leftRows = await countRows(structured.path);
        const rightRows = await countRows(corpus.path);
        const pairPopulation = (leftRows ?? 0) * (rightRows ?? 0);
        const pairSample = Math.ceil(pairPopulation * args.valRate);
        console.log(`[SemDB] [${query}] join validation population: ${leftRows} × `
          + `${rightRows} = ${pairPopulation} pairs; rate=${args.valRate} -> `
          + `${pairSample} validation rows.`);
        const frame = buildPairFrame(args, query, {
          leftCsv: structured.path, leftIdCol, leftTextCol,
          rightCsv: corpus.path,
          rightIdCol: imgFilenameCol || imageCol,
          rightImageCol: imgFilepathCol || imgFilenameCol || imageCol,
          imageDir, clipModel, top: null,
        });
        if (frame) {
          valCorpusCsv = frame;
          valFile = buildValSet(args, query, {
            corpusCsv: frame, idCol: "pair_id", sqlPath, isImage: true,
            pairwise: true, pairImageCols: ["file2"], textCols: ["text1"],
            imageDir, clipModel, endpoint: args.endpoint, oracleModel,
            importanceBy: "column:pair_score",
            frameKey: "full-frame",
          });
          valPairwise = !!valFile;
          valPairKind = "cross";
          valLeftIdCol = leftIdCol;
          // A real key from the frame beats a described one: the solver copies it.
          const firstRow = (await readFile(frame, "utf-8")).split(/\r?\n/)[1] || "";
          valKeyExample = firstRow.split(",")[0] || `<${leftIdCol}>-<image filename>`;
        }
      } else {
        const textCols = semanticTextCols.length ? semanticTextCols : [textCol];
        valFile = buildValSet(args, query, {
          corpusCsv: corpus.path, idCol,
          sqlPath, isImage,
          imageCol: imgFilenameCol || imageCol, imageDir, clipModel,
          textCols: isImage ? [] : textCols,
          // Multi-field relational predicates need semantic relevance: TF-IDF cannot
          // connect Frankfurt with Germany/Europe, while the local CLIP encoder can.
          importanceBy: !isImage && textCols.length > 1
            ? "clip-text-similarity" : undefined,
          endpoint: args.endpoint, oracleModel,
        });
      }
      // NO SILENT DEGRADATION. Without a val set the refinement loop scores against
      // the full ground truth, and renderFeedback then shows the agent GT-derived
      // FALSE POSITIVES / FALSE NEGATIVES rows verbatim — the agent would be reading
      // the test set. Asking for --val-rate and getting GT-driven iteration instead is
      // exactly the failure this must not have, so fail loudly instead.
      if (!valFile) {
        throw new Error(
          `[SemDB] [${query}] --val-rate was given but NO validation set could be built.\n`
          + `  Refusing to continue: without it every iteration would be scored against the\n`
          + `  FULL ground truth and the agent's feedback would contain ground-truth rows.\n`
          + `  Fix the val-set build above, or re-run WITHOUT --val-rate to accept\n`
          + `  ground-truth-driven refinement deliberately.`);
      }
    }
  }
  const validationSamplingLlmMs = buildsValidation
    ? Date.now() - validationStarted
    : 0;
  const valMode = !!valFile && !args.noRefine;
  const val = valMode ? await readJSON(valFile) : null;
  if (architecture === "pgo" && valMode) {
    assertSelectValidationPayload(val, valFile);
  }
  const validationTelemetry = valMode ? {
    mode: valPairwise ? "pairwise" : "per_row",
    pair_kind: valPairwise ? valPairKind : null,
    population: val?.design?.N ?? null,
    sampled_ids: val?.ids?.length ?? null,
    labeled_rows: val?.labels ? Object.keys(val.labels).length : null,
    oracle: val?.provenance?.oracle?.cost ?? null,
    build_and_label_ms: validationSamplingLlmMs,
  } : null;
  const valIdsPath = resolve(runDir, "_val_ids.txt");
  if (valMode) {
    if (!val || !val.labels) throw new Error(`[SemDB] --val-file ${valFile} has no "labels"`);
    await writeFile(valIdsPath, Object.keys(val.labels).join("\n") + "\n");
    console.log(`[SemDB] [${query}] ${valPairwise ? "pairwise" : "per-row"} `
      + `validation mode: ${Object.keys(val.labels).length} labeled rows from ${valFile}`);
  }
  const tracePairKind = valPairKind || queryPairKind;
  if (tracePairKind === "cross" && !valLeftIdCol && structured) {
    valLeftIdCol = (await corpusCols(structured, args)).idCol;
  }
  if (tracePairKind === "self" && !valPairKind && directSqlPath) {
    try {
      valPairIncludeDiagonal = Boolean(
        validationPlan(directSqlPath, args.benchmark, query).candidate?.include_diagonal,
      );
    } catch {
      // The Planner still receives the ordered-pair contract; unresolved diagonal
      // semantics must then be made explicit in plan.json.
    }
  }

  // Text corpora select BOTH the text system prompt AND the text user prompt (the shared
  // user prompts are image-specific and would otherwise make the text solver emit image
  // code). Image corpora leave both undefined → runPhase falls back to the image prompts.
  const sysPrompts = isImage
    ? {}
    : { sig: vadarSignatureConfig.promptPathText, api: vadarApiConfig.promptPathText, solver: vadarSolverConfig.promptPathText };
  const userPrompts = isImage
    ? {}
    : { sig: vadarSignatureConfig.userPromptPathText, api: vadarApiConfig.userPromptPathText, solver: vadarSolverConfig.userPromptPathText };

  const codeExecutionRuns = [];
  const runSolver = (
    iterDir,
    iterCode,
    iterCsv,
    onlyIds = null,
    candidateHelpersPath = null,
  ) => {
    if (!existsSync(iterCode)) return { status: "empty", stderr: "" };
    const helpersPath = candidateHelpersPath
      || resolve(runDir, "iter_0", `_vadar_helpers_${query}.py`);
    try {
      validateOfflineVadarFile(iterCode);
      if (existsSync(helpersPath)) validateOfflineVadarFile(helpersPath);
    } catch (error) {
      const stderr = String(error && error.message ? error.message : error);
      console.warn(`[SemDB] ${stderr}`);
      return { status: "crash", stderr };
    }
    // Compile gate: a syntax error or an undefined name must not cost a corpus run.
    const pre = runPreflight([iterCode, helpersPath], resolve(iterDir, "preflight.json"));
    if (!pre.ok) {
      console.warn(`[SemDB] [${query}] preflight failed (${pre.stage}) — not executing.\n${pre.text}`);
      return { status: "crash", stage: "compile", preflight: pre.report, stderr: pre.text };
    }
    if (!doRun) return { status: "empty", stderr: "" };
    const sArgs = isImage
      ? [iterCode, iterCsv, "--data-dir", dataDir, ...(imageDir ? ["--image-dir", imageDir] : []),
         "--clip-model", clipModel,
         // Without this an image solver runs the whole corpus every iteration and the
         // validation set saves nothing.
         ...(onlyIds ? ["--only-ids", onlyIds] : [])]
      : [iterCode, iterCsv, "--data-dir", dataDir,
         ...(onlyIds ? ["--only-ids", onlyIds] : [])];
    console.log(`\n[SemDB] Running direct solver: python3 ${sArgs.join(" ")}`);
    const { proc, stdout, stderr, execMs, logPath } =
      runPythonLogged(sArgs, resolve(iterDir, "run.log"), "direct solver");
    const iterMatch = basename(iterDir).match(/^iter_(\d+)$/);
    codeExecutionRuns.push({
      iteration: iterMatch ? Number(iterMatch[1]) : null,
      scope: onlyIds
        ? "validation_iteration"
        : (iterMatch ? "iteration_full_corpus" : "final_full_corpus"),
      duration_ms: execMs,
      status: proc.status === 0 ? "ok" : "crash",
    });
    const out = { stderr, stdout, execMs, logPath };
    if (proc.status !== 0) {
      console.warn(`[SemDB] direct solver exited ${proc.status} (${(execMs / 1000).toFixed(1)}s) — log: ${logPath}`);
      return { status: "crash", ...out };
    }
    console.log(`[SemDB] direct solver ok (${(execMs / 1000).toFixed(1)}s) — log: ${logPath}`);
    return { status: "ok", ...out };
  };

  // The trace key follows the query's shape. A pairwise (join) query is scored per
  // PAIR — a per-image trace cannot be matched against a pair-keyed val set at all —
  // so the contract is stated explicitly rather than left to the agent to infer.
  const crossPairTraceContract =
    [`This query's AI predicate IS the join condition, so it is scored PER PAIR.`,
       ``,
       `- \`<key>\` is \`"<physical_row_id>-<image_filename>"\` — the structured row's`,
       `  \`${valLeftIdCol}\` value, a literal \`-\`, then the image filename`,
       `  (e.g. \`"${valKeyExample}"\`). This key is for sampled predicate validation;`,
       `  the final result must still project the columns named by the SQL SELECT list.`,
       `- Write ONE trace entry per (structured row, image) pair you evaluate, NOT one`,
       `  per image. The same image paired with two structured rows is two entries.`,
       `- \`--only-ids\` holds those pair keys. Apply it AFTER forming the pairs, not`,
       `  when loading the manifest:`,
       ``,
       "```python",
       `for srow in structured_rows:`,
       `    for irow in image_rows:`,
       `        key = f"{srow['${valLeftIdCol}']}-{irow['${imgFilenameCol}']}"`,
       `        if only is not None and key not in only:`,
       `            continue          # skip the visual call entirely`,
       `        trace[key] = "true" if <predicate holds> else "false"`,
       "```",
       ``,
       `  With \`--only-ids\` the loop must run the vision model ONLY for listed pairs —`,
       `  that is the whole point of the sampled validation set.`].join("\n");
  const selfPairTraceContract =
    [`This query is a SEMANTIC SELF-JOIN and is scored PER ORDERED PAIR.`,
     ``,
     `- \`<key>\` is \`"<left_id>-<right_id>"\`, using the physical product \`id\``,
     `  on both sides (e.g. \`"${valKeyExample}"\`). Do not use the loop index or`,
     `  image path. The final result still follows the SQL SELECT projection.`,
     `- The pair domain is ordered: \`a-b\` and \`b-a\` are distinct trace keys.`,
     `- ${valPairIncludeDiagonal
       ? "Diagonal keys such as `a-a` ARE part of this query and must be evaluated."
       : "Diagonal keys such as `a-a` are excluded by the deterministic SQL predicate."}`,
     `- Apply every ordinary CTE/join/filter first, then form pairs. Apply`,
     `  \`--only-ids\` to the PAIR KEY after pair formation:`,
     ``,
     "```python",
     `for left in filtered_rows:`,
     `    for right in filtered_rows:`,
     `        key = f"{left['id']}-{right['id']}"`,
     `        if only is not None and key not in only:`,
     `            continue`,
     `        trace[key] = "true" if <semantic pair predicate holds> else "false"`,
     "```",
     ``,
     `- Write one trace entry for every listed pair, including false decisions. With`,
     `  \`--only-ids\`, do semantic inference only for listed pair keys.`].join("\n");
  const rowTraceContract = isImage
    ? [`This query is scored PER ROW.`,
       ``,
       `- \`<key>\` is the image manifest's primary-key value (a string) —`,
       `  the \`${valRowIdCol || imgFilenameCol}\` column of \`${corpus.table}\`.`,
       `- \`--only-ids\` holds those row ids; apply it as a membership filter right`,
       `  after the manifest is loaded:`,
       ``,
       "```python",
       `if only is not None:`,
       `    rows = [r for r in rows if str(r["${valRowIdCol || imgFilenameCol}"]).strip() in only]`,
       "```"].join("\n")
    : [`This query is scored PER ROW.`,
       ``,
       `- \`<key>\` is the \`${valRowIdCol || "primary-key"}\` value of the text`,
       `  corpus row, converted to a string.`,
       `- Apply \`--only-ids\` immediately after loading rows, before semantic`,
       `  inference, and write one true/false or extracted-value trace entry for`,
       `  every listed row key.`].join("\n");
  const traceContract = tracePairKind
    ? (tracePairKind === "self" ? selfPairTraceContract : crossPairTraceContract)
    : rowTraceContract;
  const agentTraceContract = valKeyExample
    ? traceContract.split(valKeyExample).join(
        tracePairKind === "self"
          ? "<left_id>-<right_id>"
          : "<physical_row_id>-<image_filename>",
      )
    : traceContract;

  // Solver template vars differ by modality: image gets manifest cols, text does not.
  const solverVars = (iterCode, querySql, helpersPath) => isImage
    ? { query_id: query, query_sql: querySql, query_nl: nl || "(none)", semdb_dir: __dirname,
        tables_doc: tableLines.join("\n"),
        image_table: corpus.table, image_path: corpus.path,
        image_filename_col: imgFilenameCol, image_filepath_col: imgFilepathCol,
        image_dir: imageDir, helpers_path: helpersPath, solve_path: iterCode,
        trace_contract: traceContract }
    : { query_id: query, query_sql: querySql, query_nl: nl || "(none)", semdb_dir: __dirname,
        tables_doc: tableLines.join("\n"), helpers_path: helpersPath,
        solve_path: iterCode, trace_contract: traceContract };

  // The 3 agents (Signature → API → Solver) generate iter_0's solve_<q>.py.
  const gen3Agents = async (iterDir, iterCode, querySql) => {
    const sigPath = resolve(iterDir, `_vadar_signatures_${query}.txt`);
    const helpersPath = resolve(iterDir, `_vadar_helpers_${query}.py`);
    const common = { corpus_name: corpus.table, semdb_dir: __dirname };
    record("vadar_signature", await runPhase(vadarSignatureConfig, {
      ...common, query_sql: querySql,
      schema_json: "(DIRECT mode: no schema; read value spaces from the CSVs at runtime)",
      sig_path: sigPath,
    }, iterDir, args, { systemPromptPath: sysPrompts.sig, userPromptPath: userPrompts.sig }));
    record("vadar_api", await runPhase(vadarApiConfig, {
      ...common, sig_path: sigPath, helpers_path: helpersPath,
    }, iterDir, args, { systemPromptPath: sysPrompts.api, userPromptPath: userPrompts.api }));
    if (!args.dryRun && existsSync(helpersPath)) validateOfflineVadarFile(helpersPath);
    record("vadar_solver", await runPhase(vadarSolverConfig,
      solverVars(iterCode, querySql, helpersPath),
      iterDir, args, { systemPromptPath: sysPrompts.solver, userPromptPath: userPrompts.solver }));
    if (!args.dryRun && existsSync(iterCode)) validateOfflineVadarFile(iterCode);
  };

  const regenSolver = async (iterDir, iterCode, feedback) => {
    // Helpers were generated ONCE into iter_0 (refineLoop seeds only code forward) — read
    // from iter_0, not the current iterDir which has no helpers file.
    const helpersPath = resolve(runDir, "iter_0", `_vadar_helpers_${query}.py`);
    if (existsSync(helpersPath)) validateOfflineVadarFile(helpersPath);
    const vars = solverVars(iterCode, sql + "\n\n" + feedback,
      existsSync(helpersPath) ? helpersPath : "(seed helpers from iter_0)");
    record("vadar_solver", await runPhase(vadarSolverConfig, vars, iterDir, args,
      { systemPromptPath: sysPrompts.solver, userPromptPath: userPrompts.solver }));
    if (!args.dryRun && existsSync(iterCode)) validateOfflineVadarFile(iterCode);
  };

  const genFirst = async (iterDir, iterCode, iterCsv) => {
    await gen3Agents(iterDir, iterCode, sql);
    return runSolver(iterDir, iterCode, iterCsv, valMode ? valIdsPath : null);
  };
  const regen = async (iterDir, iterCode, iterCsv, feedback) => {
    await regenSolver(iterDir, iterCode, feedback);
    return runSolver(iterDir, iterCode, iterCsv, valMode ? valIdsPath : null);
  };
  const scoreIter = async (iterDir, iterCode, iterCsv, run) => {
    const diffPath = resolve(iterDir, "diff.json");
    if (valMode) {
      // Pairwise: mistakes are looked up in the PAIR frame, not the image manifest —
      // its ids are what the val file and the solver's trace key on.
      const scored = await scoreInference(args, query, iterDir, valCorpusCsv, valFile, diffPath);
      // "ok" requires a trace to score; without one the agent must fix-first.
      const traceOk = existsSync(resolve(iterDir, `trace_${query}.json`));
      const status = run.status === "crash" ? "crash" : (traceOk ? "ok" : "empty");
      return { ...scored, status, stage: run.stage ?? null,
               execMs: run.execMs ?? null,
               runLog: filterRunLog([run.stdout, run.stderr].filter(Boolean).join("\n")),
               stderrTail: (run.stderr || "").split("\n").slice(-40).join("\n") };
    }
    const scored = await scoreWithDiff(args, query, planObj, telePath, iterCsv, diffPath, csvPath, false);
    const status = run.status === "crash" ? "crash" : (existsSync(iterCsv) ? "ok" : "empty");
    return { ...scored, status, stage: run.stage ?? null,
             execMs: run.execMs ?? null,
             runLog: filterRunLog([run.stdout, run.stderr].filter(Boolean).join("\n")),
             stderrTail: (run.stderr || "").split("\n").slice(-40).join("\n") };
  };

  const primitiveFiles = isImage
    ? [resolve(__dirname, "vadar", "predefined.py")]
    : [resolve(__dirname, "vadar", "predefined_text.py")];
  const plannerVars = (planPath, previousPlanPath = "", optimizerActionPath = "") => ({
    query_id: query,
    query_sql: sql,
    query_nl: nl || "(none)",
    modality: isImage ? "image" : "text",
    tables_doc: tableLines.join("\n"),
    local_primitive_files: primitiveFiles.map((path) => `- ${path}`).join("\n"),
    trace_contract: agentTraceContract,
    plan_path: planPath,
    previous_plan_path: previousPlanPath,
    optimizer_action_path: optimizerActionPath,
  });
  const generatorVars = ({
    planPath,
    parentManifestPath = "",
    actionPath = "",
    helpersPath,
    solverPath,
    manifestPath,
  }) => ({
    query_id: query,
    plan_path: planPath,
    parent_candidate_manifest_path: parentManifestPath,
    optimizer_action_path: actionPath,
    helpers_path: helpersPath,
    solve_path: solverPath,
    manifest_draft_path: manifestPath,
    tables_doc: tableLines.join("\n"),
    semdb_dir: __dirname,
    runtime_args: isImage ? " --image-dir <dir> --clip-model <model>" : "",
  });

  const createInitialPlan = async ({ iterDir, planPath }) => {
    record("query_planner", await runPhase(
      queryPlannerConfig,
      plannerVars(planPath),
      iterDir,
      args,
    ));
    const plan = await readAndValidatePlan(planPath, { queryId: query });
    if (plan.plan_version !== 1 || plan.parent_plan_version !== null) {
      throw new Error(
        `[SemDB] [${query}] initial plan must use plan_version=1 and parent_plan_version=null`,
      );
    }
    return { plan, planPath };
  };
  const replan = async ({
    iterDir,
    planPath,
    previousPlan,
    previousPlanPath,
    actionPath,
  }) => {
    record("query_planner", await runPhase(
      queryPlannerConfig,
      plannerVars(planPath, previousPlanPath, actionPath),
      iterDir,
      args,
    ));
    const plan = await readAndValidatePlan(planPath, {
      queryId: query,
      previousPlan,
    });
    return { plan, planPath };
  };
  const generateCandidate = async ({
    iteration,
    iterDir,
    plan,
    planPath,
    action,
    actionPath = "",
    parentCandidate,
  }) => {
    assertPlanGeneratable(plan);
    const helperPath = resolve(iterDir, `_semantic_helpers_${query}.py`);
    const solverPath = resolve(iterDir, codeBasename);
    const manifestPath = resolve(iterDir, "candidate_manifest.json");
    if (action?.action === "PATCH_CODE" && parentCandidate) {
      await copyFile(parentCandidate.helperPath, helperPath);
      await copyFile(parentCandidate.solverPath, solverPath);
    }
    record("semantic_code_generator", await runPhase(
      semanticCodeGeneratorConfig,
      generatorVars({
        planPath,
        parentManifestPath: parentCandidate?.manifestPath || "",
        actionPath,
        helpersPath: helperPath,
        solverPath,
        manifestPath,
      }),
      iterDir,
      args,
    ));
    if (!existsSync(helperPath) || !existsSync(solverPath) || !existsSync(manifestPath)) {
      throw new Error(
        `[SemDB] [${query}] Generator did not create a complete candidate in ${iterDir}`,
      );
    }
    validateOfflineVadarFile(helperPath);
    validateOfflineVadarFile(solverPath);
    const manifestDraft = await readJSON(manifestPath);
    if (!manifestDraft || typeof manifestDraft !== "object") {
      throw new Error(`[SemDB] [${query}] Generator wrote an invalid manifest draft`);
    }
    const candidateId = `${query}-iter-${iteration}`;
    const manifest = await finalizeCandidateManifest(manifestPath, {
      ...manifestDraft,
      candidate_id: candidateId,
      query_id: query,
      iteration,
      plan_version: plan.plan_version,
      parent_candidate_id: parentCandidate?.candidate_id || null,
      trigger_action: action?.action || "INITIAL",
      artifacts: {
        plan: basename(planPath),
        helpers: basename(helperPath),
        solver: basename(solverPath),
      },
    }, {
      queryId: query,
      candidateId,
      iteration,
      planVersion: plan.plan_version,
      parentCandidateId: parentCandidate?.candidate_id || null,
      triggerAction: action?.action || "INITIAL",
    });
    return {
      candidate_id: candidateId,
      iteration,
      iterDir,
      plan,
      planPath,
      helperPath,
      solverPath,
      manifest,
      manifestPath,
      csvPath: resolve(iterDir, basename(resultsCsv)),
    };
  };
  const optimize = async ({
    iteration,
    planPath,
    candidate,
    feedbackPath,
    history: optimizerHistory,
    remainingIterationBudget,
    remainingReplanBudget,
  }) => {
    const iterDir = resolve(runDir, `iter_${iteration}`);
    const actionPath = resolve(iterDir, "optimizer_action.json");
    const historyManifestPaths = optimizerHistory
      .map((entry) => entry.manifest_path)
      .filter(Boolean)
      .map((path) => `- ${path}`)
      .join("\n") || "(none)";
    record("semantic_optimizer", await runPhase(
      semanticOptimizerConfig,
      {
        query_id: query,
        plan_path: candidate.planPath || planPath,
        candidate_manifest_path: candidate.manifestPath,
        iteration_feedback_path: feedbackPath,
        history_manifest_paths: historyManifestPaths,
        optimizer_action_path: actionPath,
        remaining_iteration_budget: remainingIterationBudget,
        remaining_replan_budget: remainingReplanBudget,
      },
      iterDir,
      args,
    ));
    const actionObject = await readAndValidateOptimizerAction(actionPath, {
      queryId: query,
      candidateId: candidate.candidate_id,
    });
    return { actionObject, actionPath };
  };
  const executePgoCandidate = async (candidate) => runSolver(
    candidate.iterDir,
    candidate.solverPath,
    candidate.csvPath,
    valMode ? valIdsPath : null,
    candidate.helperPath,
  );
  const scorePgoCandidate = async (candidate, run) => {
    if (valMode) {
      return scoreIter(
        candidate.iterDir,
        candidate.solverPath,
        candidate.csvPath,
        run,
      );
    }
    const status = run.status === "crash"
      ? "crash"
      : (existsSync(candidate.csvPath) ? "ok" : "empty");
    return {
      status,
      stage: run.stage ?? null,
      execMs: run.execMs ?? null,
      f1: null,
      objective: { name: "f1", value: null, direction: "maximize" },
      metrics: null,
      diff: null,
      runLog: filterRunLog([run.stdout, run.stderr].filter(Boolean).join("\n")),
      stderrTail: (run.stderr || "").split("\n").slice(-40).join("\n"),
    };
  };
  const promotePgoCandidate = async (candidate) => {
    const promotions = [
      [candidate.planPath, resolve(runDir, "plan.json")],
      [candidate.helperPath, resolve(runDir, basename(candidate.helperPath))],
      [candidate.solverPath, solvePath],
      [candidate.manifestPath, resolve(runDir, "candidate_manifest.json")],
    ];
    if (existsSync(candidate.csvPath)) promotions.push([candidate.csvPath, resultsCsv]);
    for (const [source, target] of promotions) await copyFile(source, target);
  };

  let loopResult;
  if (architecture === "pgo") {
    if (args.dryRun) {
      const iter0Dir = resolve(runDir, "iter_0");
      const iter1Dir = resolve(runDir, "iter_1");
      await mkdir(iter0Dir, { recursive: true });
      await mkdir(iter1Dir, { recursive: true });
      const dryPlanPath = resolve(iter0Dir, "plan.json");
      const dryHelperPath = resolve(iter0Dir, `_semantic_helpers_${query}.py`);
      const drySolverPath = resolve(iter0Dir, codeBasename);
      const dryManifestPath = resolve(iter0Dir, "candidate_manifest.json");
      record("query_planner", await runPhase(
        queryPlannerConfig,
        plannerVars(dryPlanPath),
        iter0Dir,
        args,
      ));
      record("semantic_code_generator", await runPhase(
        semanticCodeGeneratorConfig,
        generatorVars({
          planPath: dryPlanPath,
          helpersPath: dryHelperPath,
          solverPath: drySolverPath,
          manifestPath: dryManifestPath,
        }),
        iter0Dir,
        args,
      ));
      record("semantic_optimizer", await runPhase(
        semanticOptimizerConfig,
        {
          query_id: query,
          plan_path: dryPlanPath,
          candidate_manifest_path: dryManifestPath,
          iteration_feedback_path: resolve(iter0Dir, "iteration_feedback.json"),
          history_manifest_paths: `- ${dryManifestPath}`,
          optimizer_action_path: resolve(iter1Dir, "optimizer_action.json"),
          remaining_iteration_budget: args.maxIterations,
          remaining_replan_budget: args.maxReplans,
        },
        iter1Dir,
        args,
      ));
      return null;
    }
    loopResult = await runPgoLoop({
      args: { ...args, refineSampleCap: defaults.refineSampleCap },
      query: { query_id: query },
      runDir,
      createInitialPlan,
      replan,
      generateCandidate,
      optimize,
      executeCandidate: executePgoCandidate,
      scoreCandidate: scorePgoCandidate,
      promoteCandidate: promotePgoCandidate,
      isImprovement: checkSemdbImprovement,
      hasValidationSignal: valMode && doRun,
    });
  } else {
    if (args.dryRun) { await gen3Agents(runDir, solvePath, sql); return null; }
    loopResult = await refineLoop({
      args, query, runDir, codeBasename, resultsCsv, genFirst, regen, scoreIter,
    });
  }
  const {
    bestIter,
    bestObjective,
    history,
    replansUsed = 0,
    actionCounts = { PATCH_CODE: 0, REPLAN: 0, STOP: 0 },
    planVersions = architecture === "pgo" ? 1 : 0,
    bestCandidate = null,
  } = loopResult;

  // Val mode: the loop scored the frozen solver on the LABELED sub-corpus only. Now run
  // the promoted best solver over the FULL corpus (NO --only-ids) to produce the real
  // result CSV. No-leakage: the reported output covers all rows, not just labeled ones.
  if (valMode && doRun) {
    const bestCode = resolve(runDir, codeBasename);
    console.log(`[SemDB] [${query}] val mode: final full-corpus run of the frozen solver.`);
    const promotedHelpers = architecture === "pgo"
      ? resolve(runDir, `_semantic_helpers_${query}.py`)
      : null;
    runSolver(runDir, bestCode, resultsCsv, null, promotedHelpers);
  }

  // DIRECT timing is measured at its actual boundaries. In particular, do not infer
  // code execution as wall-agent: that residual also contains validation construction,
  // scoring, preflight, and artifact I/O.
  const gt = await resolveGroundTruth(args.groundTruthDir, query, args.scaleFactor);
  const agentMs = phases.reduce((s, p) => s + p.duration_ms, 0);
  const agentCalls = phases.reduce((s, p) => s + p.llm_calls, 0);
  const agentCost = phases.reduce((s, p) => s + p.cost_usd, 0);
  const wallMs = Date.now() - wallStart;
  const timingBreakdown = directTimingBreakdown(
    wallMs, agentMs, validationSamplingLlmMs, codeExecutionRuns);
  const report = {
    query, corpus: corpus.table, provider: args.agentProvider, operator: "direct",
    mode: "direct", wall_clock_ms: wallMs,
    agent_architecture: architecture,
    plan_versions: planVersions,
    replans: replansUsed,
    optimizer_actions: actionCounts,
    best_candidate_id: bestCandidate?.candidate_id ?? null,
    direct: { agent_stage_ms: agentMs, agent_calls: agentCalls,
              agent_cost_usd: Number(agentCost.toFixed(4)),
              timing_breakdown_ms: timingBreakdown,
              code_execution_runs: codeExecutionRuns },
    naive_llm_calls: planObj.plan.naive ?? null,
    total_estimated_cost_usd: Number(agentCost.toFixed(4)),
    ground_truth: gt ? { file: gt.file, count: gt.count } : null,
    validation: validationTelemetry,
    optimizer_data_boundary: architecture === "pgo" ? {
      source: valMode && doRun ? "select_validation" : "none",
      cert_accessed: false,
      full_ground_truth_accessed: false,
    } : null,
    // "f1_fallback" means a val set WAS asked for and could not be built, so every
    // iteration was scored on the full ground truth — the tuning set was the test set.
    refine: { mode: valMode
                ? "per_row_val"
                : (architecture === "pgo" ? "single_shot_no_select"
                  : (args.valRate ? "metric_fallback" : "metric")),
              objective: bestObjective.name,
              objective_direction: bestObjective.direction,
              objective_history: history.map((item) => item.objective),
              ...(valMode ? { val_file: valFile,
                              val_n: Object.keys(val.labels).length,
                              val_shape: valPairwise ? "pairwise" : "per_row",
                              ...(valPairwise ? { pair_frame: valCorpusCsv } : {}) } : {}),
              iterations: history.length - 1, best_iteration: bestIter,
              max_iterations: args.noRefine ? 0 : args.maxIterations,
              f1_history: history },
    phases,
  };
  await writeFile(telePath, JSON.stringify(report, null, 2));

  console.log(`\n[SemDB] === ${query} (DIRECT) ===`);
  console.log(`[SemDB]   3-agent solver     ${(agentMs / 1000).toFixed(1)}s  ${agentCalls} calls  $${agentCost.toFixed(4)}`);
  console.log(`[SemDB]   validation build  ${(validationSamplingLlmMs / 1000).toFixed(1)}s  (sampling + oracle LLM)`);
  console.log(`[SemDB]   code execution total  ${(timingBreakdown.code_execution_ms / 1000).toFixed(1)}s  ${codeExecutionRuns.length} runs`);
  for (const run of codeExecutionRuns) {
    const iter = run.iteration == null ? "final" : `iter_${run.iteration}`;
    console.log(`[SemDB]     ${iter.padEnd(8)} ${run.scope.padEnd(24)} `
      + `${(run.duration_ms / 1000).toFixed(3)}s  ${run.status}`);
  }
  console.log(`[SemDB]   other overhead    ${(timingBreakdown.other_overhead_ms / 1000).toFixed(1)}s  (preflight + scoring + I/O)`);

  // (3) finalize: score the promoted best CSV → merge metrics into telemetry + append results.csv row
  if (doRun) {
    const gt = await resolveGroundTruth(args.groundTruthDir, query, args.scaleFactor);
    if (gt && existsSync(resultsCsv)) {
      await scoreWithDiff(args, query, planObj, telePath, resultsCsv, resolve(runDir, "diff.json"), csvPath, true);
    }
  }

  // --- Score against ground truth + append CSV (re-read telePath, merged by finalize above) ---
  if (doRun && gt && existsSync(resultsCsv)) {
    const scored = await readJSON(telePath);
    const m = scored?.metrics;
    if (m) {
      console.log(`[SemDB]   METRICS            precision=${m.precision}  recall=${m.recall}  F1=${m.f1}  (tp=${m.tp} fp=${m.fp} fn=${m.fn})`);
      console.log(`[SemDB]   saved -> ${telePath} and ${csvPath}`);
    }
  } else if (gt) {
    console.log(`[SemDB]   (solver output not found — score later with evaluate.py --pred ${resultsCsv})`);
  }
  return report;
}

export async function runQueryDirectLegacy(args, planObj, csvPath) {
  return runQueryDirectCore(args, planObj, csvPath, "legacy");
}

export async function runQueryDirectPgo(args, planObj, csvPath) {
  return runQueryDirectCore(args, planObj, csvPath, "pgo");
}

async function runQueryDirect(args, planObj, csvPath) {
  return args.agentArchitecture === "legacy"
    ? runQueryDirectLegacy(args, planObj, csvPath)
    : runQueryDirectPgo(args, planObj, csvPath);
}

async function main() {
  const base = parseArgs(process.argv);
  await validateDataDirectory(base);
  if (!base.direct && base.directOptionsSpecified) {
    console.warn(
      "[SemDB] --agent-architecture/--max-replans/--no-agent-skills "
      + "only affect --direct and are ignored.",
    );
  }
  if (base.valRate && !base.valPlanOnly && base.endpoint && base.oracleModel) {
    await validateEndpointModel(base.endpoint, base.oracleModel, base.apiKey);
  }
  setAgentProvider(base.agentProvider);
  const csvPath = base.telemetryCsv || resolve(base.out, "results.csv");

  // --query takes ONE id or a list (comma/space separated): --query q2,q4  OR  --query "q2 q4".
  const queries = base.query
    ? base.query.split(/[,\s]+/).map((s) => s.trim()).filter(Boolean)
    : await listQueries(base.queryDir);
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
    if (mp.status !== 0) {
      throw new Error(`materialize.py exited ${mp.status} `
        + `(verify the scale directory and pandas/pyarrow installation)`);
    }
    const required = [
      "styles_details.csv", "styles.csv", "IMAGES.csv", "ecomm_products.csv",
    ];
    const missing = required.filter((name) => !existsSync(resolve(matDir, name)));
    if (missing.length) {
      throw new Error(`parquet materialization produced no ${missing.join(", ")} in ${matDir}`);
    }
    base.tableDir = matDir;
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
    const audioTables = p.tables.filter((table) => table.modality === "audio");
    if (audioTables.length) {
      console.warn(`[SemDB] [${p.query}] SKIP: query references unsupported audio `
        + `table(s): ${audioTables.map((table) => table.table).join(", ")}.`);
      return false;
    }
    if (!p.corpus.path || !existsSync(p.corpus.path)) {
      console.warn(`[SemDB] [${p.query}] SKIP: corpus table not found ('${p.corpus.table}' → '${p.corpus.path || "<empty>"}'). `
        + `Check --benchmark (got '${base.benchmark}') and --sf so the SQL prefix '${benchPrefix(base.benchmark)}.' matches.`);
      return false;
    }
    if (base.valRate && base.valCallSite == null && base.queryDir) {
      const sites = callSites(resolve(base.queryDir, `${p.query}.sql`));
      if (sites.length > 1 && !base.valPlanOnly) {
        console.warn(`[SemDB] [${p.query}] SKIP: query has ${sites.length} AI call `
          + `sites but one per-predicate validation frame cannot score the complete `
          + `multi-predicate query. Run it explicitly with --val-call-site N, or omit `
          + `--val-rate.`);
        return false;
      }
    }
    return true;
  });
  // --image-only: keep only queries whose chosen corpus is an image table.
  const runPlans = base.imageOnly ? plans.filter((p) => p.isImage) : plans;
  if (base.imageOnly) {
    console.log(`[SemDB] --image-only: ${runPlans.length}/${plans.length} image quer${runPlans.length === 1 ? "y" : "ies"}`
      + (runPlans.length ? `: ${runPlans.map((p) => p.query).join(", ")}` : ""));
    if (!runPlans.length) { console.log("[SemDB] no image queries in this scenario — nothing to run."); return; }
  }
  const corpora = new Map();
  for (const p of runPlans) {
    const key = p.corpus.table;
    if (!corpora.has(key)) corpora.set(key, { corpus: p.corpus, queries: [] });
    corpora.get(key).queries.push(p);
  }
  console.log(`[SemDB] ${corpora.size} corpus/corpora: ${[...corpora.entries()].map(([k, v]) => `${k}(${v.queries.length})`).join(", ")}`);

  // DIRECT mode: skip Schema Designer + extract/compile; the VADAR 3 agents write one
  // end-to-end solver per query. Otherwise: Schema Designer + Extractor once per corpus.
  const summary = [];
  if (base.direct) {
    console.log(
      `[SemDB] DIRECT mode (${base.agentArchitecture}): agent solver per query `
      + "(no schema design, no extract/compile split).",
    );
    for (const p of runPlans) {
      console.log(`\n[SemDB] ==================== ${p.query} ====================`);
      try {
        await runQueryDirect(base, p, csvPath);
        const tele = await readJSON(resolve(base.out, `${base.benchmark}-${p.query}`, "telemetry.json"));
        if (tele?.metrics) summary.push({ q: p.query, ...tele.metrics });
      } catch (e) {
        console.error(`[SemDB] [${p.query}] failed: ${e.message}`);
        summary.push({ q: p.query, error: e.message });
      }
    }
  } else {
  // 2) Schema Designer + Extractor ONCE per corpus (shared by its queries).
  const corpusArt = new Map();
  for (const [key, info] of corpora) {
    try { corpusArt.set(key, await ensureCorpus(base, info.corpus, info.queries)); }
    catch (e) { console.error(`[SemDB] corpus ${key} failed: ${e.message}`); }
  }

  // 3) Code Generator + execute + evaluate PER query.
  for (const p of runPlans) {
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

/** Compare two paths as the FILESYSTEM sees them, not as strings.
 *
 *  `resolve()` normalizes but does not follow symlinks, while `import.meta.url` is
 *  always the real path (node resolves the module specifier). When the repo is reached
 *  through a symlinked prefix — /localhome/hza214 -> /local-scratch/localhome/hza214
 *  here — invoking the file by its ABSOLUTE symlinked path made the two sides differ,
 *  so main() silently never ran: no output, exit 0. A relative path happened to work
 *  because process.cwd() is already resolved, which is what made this so easy to miss.
 */
function samePath(a, b) {
  try { return realpathSync(a) === realpathSync(b); } catch { return resolve(a) === resolve(b); }
}

// Only run when invoked directly (not when imported by tests).
if (process.argv[1] && samePath(process.argv[1], fileURLToPath(import.meta.url))) {
  main().catch((err) => {
    console.error("[SemDB] Fatal:", err.message);
    process.exit(1);
  });
}
