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
import { existsSync, readFileSync, writeFileSync, realpathSync, statSync } from "fs";
import { createHash } from "crypto";
import { resolve, dirname, basename } from "path";
import { fileURLToPath } from "url";
import { spawnSync } from "child_process";

import {
  renderTemplate,
  runAgent,
  runStructuredAgent,
  readJSON,
  setAgentProvider,
} from "../gendb/shared.mjs";
import { defaults, getAgentModel, getAgentEffort } from "./semdb.config.mjs";
import { BENCHMARKS, SUPPORTED, tableDesc, tableFile } from "./benchmarks.mjs";
import { config as vadarSignatureConfig } from "./agents/vadar-signature/index.mjs";
import { config as vadarApiConfig } from "./agents/vadar-api/index.mjs";
import { config as vadarSolverConfig } from "./agents/vadar-solver/index.mjs";
import { config as queryPlannerConfig } from "./agents/query-planner/index.mjs";
import { config as semanticCodeGeneratorConfig } from "./agents/semantic-code-generator/index.mjs";
import { config as semanticOptimizerConfig } from "./agents/semantic-optimizer/index.mjs";
import { config as memoryManagerConfig } from "./agents/memory-manager/index.mjs";
import { applyMemoryUpdate } from "./memory/apply-update.mjs";
import { getNodesByLayer } from "./memory/graph.mjs";
import {
  classifyQuery,
  getMemorySummary,
  initMemory,
  initSkillRoot,
  listSkills,
  lintAllSkills,
  publishRoleSkills,
  recordSkillUsage,
  resolveSkillRoot,
  skillsDirFor,
} from "./memory/index.mjs";
import {
  assertPlanGeneratable,
  finalizeCandidateManifest,
  lintImagePlan,
  readAndValidateOptimizerAction,
  readAndValidatePlan,
  writeJsonAtomic,
  writeTextAtomic,
} from "./agent-runtime/contracts.mjs";
import { runPgoLoop } from "./agent-runtime/pgo-loop.mjs";
import {
  prepareAgentRole,
  prepareStructuredRole,
  supportsStructuredRole,
  validateStructuredOptimizerEvidence,
} from "./agent-runtime/structured-role.mjs";
import {
  assertSelectValidationPayload,
  summarizeTraceArtifact,
} from "./agent-runtime/feedback.mjs";
import {
  buildPlannerTableProfile,
  lintPlanAgainstTableProfile,
} from "./agent-runtime/context-profile.mjs";
import { classifyValidationCapability } from "./validation_capability.mjs";

// Shared SQL scanners (also used by memory/signature.mjs, so the retrieval key is
// derived from exactly the same view of a query as the pipeline's own).
import {
  aliasMap,
  benchPrefix,
  prefixRe,
  semanticArgs,
  tablesInPredicate,
} from "./sql-features.mjs";

export { aliasMap };

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
    // The VADAR agents (Signature→API→Solver) write ONE end-to-end solve_<q>.py over
    // the vadar/ operator library. This is the only pipeline — the Schema-Designer +
    // extract/attrs/compile split was removed.
    agentArchitecture: defaults.directAgentArchitecture,
    agentExecution: defaults.agentExecution,
    structuredReasoningEffort: null,
    plannerMaxOutputTokens: null,
    optimizerMaxOutputTokens: null,
    maxReplans: defaults.maxReplans,
    enableAgentSkills: defaults.enableAgentSkills,
    memoryDir: defaults.memoryDir,   // cross-run memory; null disables it entirely
    memoryReadonly: false,           // retrieve but never write — for clean A/B runs
    noMemorySkills: false,           // keep L0/L1 push, drop the learned-skill namespace
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
    semanticPlanOnly: false, // PGO Planner artifact only; skip Generator/Optimizer/run
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--query" && argv[i + 1]) args.query = argv[++i];
    else if (a === "--benchmark" && argv[i + 1]) args.benchmark = argv[++i];
    else if (a === "--agent-provider" && argv[i + 1]) args.agentProvider = argv[++i];
    else if (a === "--base-url" && argv[i + 1]) process.env.VLLM_BASE_URL = argv[++i];
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
    else if (a === "--direct") { /* the only pipeline; accepted for compatibility */ }
    else if (a === "--agent-architecture" && argv[i + 1]) args.agentArchitecture = argv[++i];
    else if (a === "--agent-execution" && argv[i + 1]) args.agentExecution = argv[++i];
    else if (a === "--structured-reasoning-effort" && argv[i + 1]) {
      args.structuredReasoningEffort = argv[++i];
    }
    else if (a === "--planner-max-output-tokens" && argv[i + 1]) {
      args.plannerMaxOutputTokens = Number(argv[++i]);
    }
    else if (a === "--optimizer-max-output-tokens" && argv[i + 1]) {
      args.optimizerMaxOutputTokens = Number(argv[++i]);
    }
    else if (a === "--max-replans" && argv[i + 1]) args.maxReplans = Number(argv[++i]);
    else if (a === "--no-agent-skills") args.enableAgentSkills = false;
    else if (a === "--memory-dir" && argv[i + 1]) args.memoryDir = resolve(argv[++i]);
    else if (a === "--no-memory") args.memoryDir = null;
    else if (a === "--memory-readonly") args.memoryReadonly = true;
    else if (a === "--no-memory-skills") args.noMemorySkills = true;
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
    else if (a === "--semantic-plan-only") args.semanticPlanOnly = true;
    else if (a === "--val-pair-top" && argv[i + 1]) args.valPairTop = parseInt(argv[++i], 10);
    else if (a === "--no-refine") args.noRefine = true;
  }
  if (!["legacy", "pgo"].includes(args.agentArchitecture)) {
    throw new Error(
      `--agent-architecture must be "legacy" or "pgo" (got "${args.agentArchitecture}")`,
    );
  }
  if (args.semanticPlanOnly && args.agentArchitecture !== "pgo") {
    throw new Error("--semantic-plan-only requires --agent-architecture pgo");
  }
  if (!Number.isInteger(args.maxReplans) || args.maxReplans < 0) {
    throw new Error("--max-replans must be a non-negative integer");
  }
  if (!["agent", "structured"].includes(args.agentExecution)) {
    throw new Error(`--agent-execution must be "agent" or "structured" (got "${args.agentExecution}")`);
  }
  if (
    args.structuredReasoningEffort
    && !["low", "medium", "xhigh"]
      .includes(args.structuredReasoningEffort)
  ) {
    throw new Error(
      `--structured-reasoning-effort must be low, medium, or xhigh for Qwen3.8 `
      + `(got "${args.structuredReasoningEffort}")`,
    );
  }
  for (const [flag, value] of [
    ["--planner-max-output-tokens", args.plannerMaxOutputTokens],
    ["--optimizer-max-output-tokens", args.optimizerMaxOutputTokens],
  ]) {
    if (value != null && (!Number.isInteger(value) || value <= 0)) {
      throw new Error(`${flag} must be a positive integer`);
    }
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
    // Kept so scenario-level sidecars (supg_targets.json, the SUPG oracle's withheld
    // labels) can be found without re-deriving the path at every use site.
    args.scenarioRoot = base;
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
  if (!endpoint || !model) return model;
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
  const entries = payload?.data || [];
  const match = entries.find((entry) => entry?.id === model)
    || entries.find((entry) => entry?.root === model);
  const available = entries.map((entry) => entry?.id).filter(Boolean);
  if (!match?.id) {
    throw new Error(
      `oracle model '${model}' is not served by ${endpoint}. `
      + `Available model(s): ${available.join(", ") || "(none)"}. `
      + `Restart the endpoint with --model ${model}, or pass a served --oracle-model.`
    );
  }
  return match.id;
}

/** Normalize old F1-only outcomes and new metric-aware outcomes. */
export function semdbObjective(outcome) {
  const candidate = outcome?.objective;
  const hasNumber = (value) => value !== null && value !== undefined && value !== ""
    && Number.isFinite(Number(value));
  // Once an evaluator emits a typed objective it is authoritative, including an
  // explicit null such as query_metric_unavailable.  Falling back to the legacy F1
  // field in that case would silently optimize an operator surrogate.
  if (candidate && typeof candidate === "object") {
    return {
      name: candidate.name || "objective",
      value: hasNumber(candidate.value) ? Number(candidate.value) : null,
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

/** Render a scored query's OWN metric family.
 *
 *  SemBench does not score every query with precision/recall/F1: a grouping query is
 *  scored by adjusted Rand index, an aggregate by relative error, a top-k by Spearman.
 *  The summary used to print a hardcoded `P=… R=… F1=…`, so those queries reported
 *  `P=undefined R=undefined F1=undefined` even though telemetry held their real score.
 *  Dispatch on `metric` the same way `metric_objective()` does in evaluate.py.
 *
 *  Returns { key, label, value } — `key` groups queries that share a metric so the
 *  workload mean is taken WITHIN a family instead of averaging an ARI against an F1.
 */
export function formatQueryMetric(m) {
  const num = (v, d = 4) => (v === null || v === undefined || v === "" || !Number.isFinite(Number(v))
    ? null
    : Number(Number(v).toFixed(d)));
  const metric = String(m?.metric || "").toLowerCase();
  const pick = (...keys) => {
    for (const k of keys) { const v = num(m?.[k]); if (v !== null) return v; }
    return null;
  };
  if (metric === "adjusted-rand-index" || metric === "ari") {
    const ari = pick("adjusted_rand_index", "ari");
    return { key: "ari", label: `ARI=${ari ?? "n/a"}`, value: ari };
  }
  if (metric === "aggregation") {
    const rel = pick("relative_error");
    const mape = pick("mape", "mean_absolute_percentage_error");
    const abs = pick("absolute_error");
    return {
      key: "relative_error",
      label: `rel_err=${rel ?? "n/a"}`
        + (mape === null ? "" : ` MAPE=${mape}%`)
        + (abs === null ? "" : ` abs_err=${abs}`),
      value: rel,
    };
  }
  if (metric === "ranking") {
    const sp = pick("spearman_correlation", "spearman");
    const kt = pick("kendall_tau", "kendall");
    return {
      key: "spearman",
      label: `spearman=${sp ?? "n/a"}` + (kt === null ? "" : ` kendall=${kt}`),
      value: sp,
    };
  }
  // Retrieval / classification families (retrieval_f1, f1-score, id_set_f1) —
  // the macro_classification variant is a macro F1 over classes, not a retrieval F1.
  const variant = String(m?.metric_variant || m?.variant || "");
  const f1 = pick("f1", "f1_score");
  if (f1 === null && metric === "") {
    return { key: "unscored", label: "no metric recorded", value: null };
  }
  const name = variant === "macro_classification" || metric === "macro_f1" ? "macroF1" : "F1";
  const p = pick("precision");
  const r = pick("recall");
  const counts = [m?.tp, m?.fp, m?.fn].every((v) => v !== null && v !== undefined)
    ? `  (tp=${m.tp} fp=${m.fp} fn=${m.fn})` : "";
  return {
    key: name === "macroF1" ? "macro_f1" : "f1",
    label: `P=${p ?? "n/a"} R=${r ?? "n/a"} ${name}=${f1 ?? "n/a"}${counts}`,
    value: f1,
  };
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
        && Number.isFinite(Number(prev.f1)) && Number.isFinite(Number(next.f1))
        && Number(next.f1) !== Number(prev.f1)) {
      return Number(next.f1) > Number(prev.f1);
    }
    // Still tied — and at objective 0 that is the common case, because "crashed on
    // every row" and "ran correctly but matched nothing" both score 0. Fall back to
    // execution health, which does separate them: a candidate that loses fewer rows to
    // exceptions is strictly the better thing to keep iterating from, even when the
    // score has not moved yet.
    if (after.value === before.value) {
      const beforeRate = runErrorRate(prev.health);
      const afterRate = runErrorRate(next.health);
      if (beforeRate !== null && afterRate !== null && afterRate !== beforeRate) {
        return afterRate < beforeRate;
      }
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
  // by design — they belong to the oracle-labelling / residual layer, which lives OUTSIDE
  // `vadar/` and runs outside this guard. Naming them here keeps generated code from
  // importing its way around it.
  ["endpoint-backed module",
    /\b(?:semvqa|semcaption|semextract|TextPatch|get_ctx|img_vqa)\b/i],
  ["endpoint/API credential",
    /--endpoint\b|--api-key\b|\.endpoint\b|\.api_?key\b|\b(?:endpoint|api_?key)\s*(?:=|[,):])/i],
  ["OpenAI client", /\b(?:from|import)\s+openai\b|\bOpenAI\s*\(/i],
  // urllib.parse is pure string processing and is needed to reproduce SQL URI
  // basename operations over offline metadata. Block transport APIs, not parsing.
  ["HTTP/network client",
    /\b(?:requests|httpx|aiohttp|urllib3|socket)\b|urllib\s*\.\s*request|\burlretrieve\b/i],
  ["shell/network escape", /\b(?:subprocess|Popen|urlopen|curl)\b|\bos\.system\s*\(/i],
];

export function offlineVadarViolations(source) {
  return VADAR_RUNTIME_FORBIDDEN
    .filter(([, pattern]) => pattern.test(source))
    .map(([label]) => label);
}

export function localImportRootViolations(sources, semdbDir = __dirname) {
  const combined = sources.join("\n");
  const violations = [];
  const bareLocal = /^\s*(?:from|import)\s+vadar\b/im;
  const packaged = /^\s*(?:from|import)\s+semdb(?:\.|\b)/im;
  if (bareLocal.test(combined) && !combined.includes(JSON.stringify(semdbDir))) {
    violations.push(`bare SemDB imports require sys.path root ${semdbDir}`);
  }
  const parent = dirname(semdbDir);
  if (packaged.test(combined) && !combined.includes(JSON.stringify(parent))) {
    violations.push(`semdb.* imports require sys.path root ${parent}`);
  }
  return violations;
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
 * Returns the path to select.json, or null when no val set could be built. The caller
 * fails closed when --val-rate was explicitly requested, preventing accidental
 * full-ground-truth feedback.
 */
/**
 * The SUPG contract for one query, or null outside the supg benchmark.
 *
 * `supg_targets.json` is written by build_supg_scenario.py and is the single source of
 * truth for which target grades a query (PT -> precision, RT -> recall), how many
 * oracle labels the generated program may spend, and where the withheld labels live.
 */
export function supgSpec(args, query) {
  if (args.benchmark !== "supg" || !args.scenarioRoot) return null;
  const path = resolve(args.scenarioRoot, "supg_targets.json");
  if (!existsSync(path)) return null;
  let all;
  try { all = JSON.parse(readFileSync(path, "utf-8")); }
  catch { return null; }
  const spec = all?.[query];
  if (!spec) return null;
  return {
    ...spec,
    oracleLabelsPath: spec.oracle_labels
      ? resolve(args.scenarioRoot, spec.oracle_labels) : null,
  };
}

/**
 * Install the metered SUPG oracle next to a solver so `import supg_oracle` resolves via
 * sys.path[0]. The module is copied verbatim (so every run executes the reviewed source)
 * and the per-run wiring goes in a sibling JSON it reads at import.
 */
export function installSupgOracle(execDir, { labelsCsv, budget, dataset, query, ledger }) {
  const source = resolve(__dirname, "data", "supg", "supg_oracle.py");
  if (!existsSync(source)) {
    throw new Error(`supg_oracle.py not found at ${source}; the replay corpora cannot `
      + `be labelled without it.`);
  }
  writeFileSync(resolve(execDir, "supg_oracle.py"), readFileSync(source, "utf-8"));
  writeFileSync(resolve(execDir, "supg_oracle_config.json"), JSON.stringify({
    labels_csv: labelsCsv, budget, dataset, query, ledger,
  }, null, 2) + "\n");
}

export function validationCorpusFingerprint(corpusPath) {
  try {
    const stat = statSync(corpusPath);
    const identity = [
      realpathSync(corpusPath),
      String(stat.size),
      String(stat.mtimeMs),
    ].join("\0");
    return createHash("sha256").update(identity).digest("hex").slice(0, 16);
  } catch {
    return createHash("sha256")
      .update(resolve(corpusPath))
      .digest("hex")
      .slice(0, 16);
  }
}

/** Count data rows (excluding the header) in a solver's result CSV.
 *
 *  Returns null when the file is missing or unreadable, which the caller reports as
 *  "unknown" rather than as zero — a missing file is a crash, not an empty predicate.
 */
export function countCsvDataRows(path) {
  try {
    if (!path || !existsSync(path)) return null;
    const text = readFileSync(path, "utf-8");
    if (!text.trim()) return 0;
    const lines = text.split("\n").filter((line) => line.trim() !== "");
    return Math.max(0, lines.length - 1);      // drop the header
  } catch { return null; }
}

/** Make a validation-design key safe to use as a directory name.
 *
 *  The key concatenates every component of the sampling design, including the oracle's
 *  answer choices. mmqa q2b's choices are the 14-colour value space, which pushed the
 *  name past the 255-byte filename limit and crashed build_valset.py with ENAMETOOLONG
 *  — the query then failed outright, because --val-rate refuses to fall back to
 *  ground-truth scoring.
 *
 *  Long keys keep a readable prefix and carry their identity in a hash of the WHOLE key,
 *  so two designs still never collide. Keys that already fit are returned untouched, so
 *  validation sets cached before this change still hit.
 */
export function valDesignDirName(key, maxLength = 150) {
  if (key.length <= maxLength) return key;
  const digest = createHash("sha256").update(key).digest("hex").slice(0, 16);
  return `${key.slice(0, maxLength - digest.length - 1)}-${digest}`;
}

export function buildValSet(args, query, spec, attempt = {}) {
  const valRate = Math.max(
    attempt.rate ?? args.valRate,
    Number(spec.minimumRate ?? 0),
  );
  const corpusFingerprint = validationCorpusFingerprint(spec.corpusCsv);
  const key = [args.valMethod, valRate, args.valCertRate ?? 0, args.valSeed,
    args.valStrataK, args.valScoreTilt, spec.oracleModel,
    args.valCallSite ?? "auto",
    // A SUPG replay draw is labelled from the withheld column at a FIXED size, not by
    // a model at a rate; reusing a model-labelled cache entry would answer a different
    // question with a different cost.
    ...(spec.labelSource ? [spec.labelSource, `valn-${spec.valN ?? "rate"}`] : []),
    // Sampling v2 adds a probability-valid certainty stratum at the score-ranked
    // head; it must not reuse a pre-v2 draw with a different inclusion design.
    "oracleframes-v5",
    `corpus-${corpusFingerprint}`,
    spec.importanceBy || "default-score",
    ...(spec.textCols || []),
    // A pairwise design samples a different frame with different keys, so it must
    // never reuse a per-row cache entry (or vice versa).
    ...(spec.pairwise ? ["pairwise", spec.frameKey || "full-frame"] : []),
    ...(spec.oracleQuestion ? ["joint-oracle-v1", spec.oracleLabelType || "text",
      ...(spec.oracleChoices || [])] : []),
  ].join("_").replace(/[^\w.-]/g, "");
  const dir = resolve(args.out, "_val", `${args.benchmark}-${query}`, valDesignDirName(key));
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
    "--query", query, "--attr", "answer",
    ...(spec.oracleQuestion
      ? ["--query-nl", spec.oracleQuestion,
         "--label-type", spec.oracleLabelType || "text",
         ...(spec.oracleChoices
           ? ["--label-choices-json", JSON.stringify(spec.oracleChoices)] : [])]
      : ["--sql", spec.sqlPath]),
    "--method", args.valMethod,
    // SUPG replay draws a fixed number of labels (its oracle budget), not a rate: a
    // rate over 973k night_street rows would silently spend 48,000 oracle labels.
    ...(spec.valN ? ["--n", String(spec.valN)] : ["--rate", String(valRate)]),
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
    ...(spec.labelSource === "supg-oracle"
      ? ["--label-source", "supg-oracle", "--oracle-labels", spec.oracleLabels]
      : ["--label-source", "oracle", "--endpoint", spec.endpoint,
         "--oracle-model", spec.oracleModel, "--api-key", args.apiKey || "EMPTY"]),
    "--concurrency", String(args.concurrency ?? 8),
    // One cache per (benchmark, query) rather than per design: raising the rate then
    // re-pays only for rows never labeled before.
    "--label-cache", resolve(
      args.out, "_val", `${args.benchmark}-${query}`,
      `labels-${corpusFingerprint}.json`,
    ),
    "--out", dir];
  console.log(`\n[SemDB] [${query}] building validation set `
    + `(${spec.valN ? `n=${spec.valN}` : `rate=${valRate}`}, ${args.valMethod}, `
    + `oracle=${spec.labelSource === "supg-oracle"
        ? "SUPG label column (metered)" : spec.oracleModel})`);
  const proc = spawnSync("python3", bvArgs, { stdio: "inherit" });
  if (proc.status !== 0 || !existsSync(selectPath)) {
    console.warn(`[SemDB] [${query}] validation set not built (build_valset.py exited `
      + `${proc.status}).`);
    let failure = null;
    try { failure = JSON.parse(readFileSync(resolve(dir, "failure.json"), "utf-8")); }
    catch { /* A non-class-balance failure is not retryable here. */ }
    const maxSelectRate = Math.max(0, 1 - Number(args.valCertRate ?? 0));
    if (failure?.reason_code === "single_class_select" && valRate < maxSelectRate) {
      const nextRate = Math.min(maxSelectRate, Math.max(valRate * 2, valRate + 0.01));
      console.warn(`[SemDB] [${query}] SELECT labels are single-class at rate=${valRate}; `
        + `retrying adaptively at rate=${nextRate}. Existing Oracle labels are cached.`);
      return buildValSet(args, query, spec, { rate: nextRate });
    }
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
  if (args.benchmark === "movie") {
    const reviews = resolve(args.tableDir || args.dataDir, "Reviews.csv");
    if (!existsSync(reviews)) throw new Error(`movie review corpus not found: ${reviews}`);
    const sql = readFileSync(sqlPath, "utf-8");
    const match = sql.match(
      /\bWHERE\s+([A-Za-z_]\w*)\.id\s*=\s*'([^']+)'/i,
    );
    if (!match) {
      throw new Error(`[SemDB] [${query}] cannot prove the movie self-join's `
        + `deterministic row filter`);
    }
    const dir = resolve(args.out, "_val", `${args.benchmark}-${query}`);
    const out = resolve(dir, "deterministic_rows.csv");
    const pfArgs = [
      resolve(__dirname, "physical_frame.py"),
      "--corpus", reviews, "--out", out,
      "--text-col", "reviewText",
      "--filter-col", "id", "--filter-value", match[2],
    ];
    console.log(`\n[SemDB] [${query}] executing deterministic movie self-join prefix`);
    const proc = spawnSync("python3", pfArgs, { stdio: "inherit" });
    if (proc.status !== 0 || !existsSync(out)) {
      throw new Error(`physical_frame.py exited ${proc.status}; no deterministic `
        + `movie self-join frame was produced`);
    }
    return out;
  }
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

/** Execute EComm q8's deterministic CTE on the structured side of its AI join. */
export function buildDeterministicCrossLeftRows(args, query, sqlPath) {
  if (args.benchmark !== "ecomm" || String(query).toLowerCase() !== "q8") {
    throw new Error(`automatic filtered cross-table validation is not implemented `
      + `for ${args.benchmark}.${query}`);
  }
  const products = resolve(args.tableDir || args.dataDir, "ecomm_products.csv");
  if (!existsSync(products)) {
    throw new Error(`normalized EComm product view not found: ${products}`);
  }
  const dir = resolve(args.out, "_val", `${args.benchmark}-${query}`);
  const out = resolve(dir, "deterministic_left_rows.csv");
  const fbArgs = [
    resolve(__dirname, "frame_builder.py"),
    "--benchmark", args.benchmark, "--query", query,
    "--sql", sqlPath, "--products", products, "--cross-left", "--out", out,
  ];
  console.log(`\n[SemDB] [${query}] executing deterministic cross-join left prefix`);
  const proc = spawnSync("python3", fbArgs, { stdio: "inherit" });
  if (proc.status !== 0 || !existsSync(out)) {
    throw new Error(`frame_builder.py exited ${proc.status}; no deterministic `
      + `cross-join left frame was produced`);
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
    ...(spec.excludeEqualCol ? ["--exclude-equal-col", spec.excludeEqualCol] : []),
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

/** Add a stable source-record ordinal for collision-free per-row validation. */
export function buildPhysicalRowFrame(args, query, spec) {
  const dir = resolve(args.out, "_val", `${args.benchmark}-${query}`);
  const out = resolve(dir, "physical_rows.csv");
  const pfArgs = [
    resolve(__dirname, "physical_frame.py"),
    "--corpus", spec.corpusCsv, "--out", out,
    ...(spec.textCols || []).flatMap((column) => ["--text-col", column]),
    ...(spec.filterCol && spec.filterValues?.length
      ? ["--filter-col", spec.filterCol,
         ...spec.filterValues.flatMap((value) => ["--filter-value", value])]
      : []),
  ];
  const proc = spawnSync("python3", pfArgs, { stdio: "inherit" });
  if (proc.status !== 0 || !existsSync(out)) {
    throw new Error(`physical_frame.py exited ${proc.status}; no physical row frame`);
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

/** Every column contributed by one alias to an AI call, in prompt order. */
export function predicateCols(site, alias) {
  if (!site || !alias) return [];
  return (site.columns || [])
    .filter((c) => c.split(".")[0] === alias)
    .map((c) => c.split(".").slice(1).join("."));
}

/** The predicate alias backed by one planned physical table. */
export function predicateAliasForTable(site, table) {
  if (!site || !table) return null;
  const wanted = String(table).toLowerCase();
  const index = (site.bases || []).findIndex(
    (base) => String(base).toLowerCase() === wanted,
  );
  return index >= 0 ? (site.aliases || [])[index] || null : null;
}

/** Literal values from one deterministic SQL `column IN (...)` predicate. */
export function sqlInLiteralValues(sqlPath, column) {
  const sql = readFileSync(sqlPath, "utf-8");
  const escaped = String(column).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = sql.match(new RegExp(
    String.raw`(?:\b[A-Za-z_]\w*\.)?\b${escaped}\s+IN\s*\(([\s\S]*?)\)`,
    "i",
  ));
  if (!match) return [];
  const values = [];
  for (const literal of match[1].matchAll(/"((?:[^"]|"")*)"|'((?:[^']|'')*)'/g)) {
    values.push((literal[1] ?? literal[2] ?? "").replace(/""/g, '"').replace(/''/g, "'"));
  }
  return values;
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

/** How healthy was one solver execution, read out of its own diagnostic contract?
 *
 *  The prompt makes every generated program isolate per-row failures so that one
 *  unreadable file cannot kill a whole run. The cost is that a program which fails on
 *  EVERY row still exits 0, still writes a well-formed trace, and therefore used to be
 *  classified "ok" — indistinguishable from a program that ran fine and simply matched
 *  nothing. mmqa q2a lost 200/200 rows to one TypeError that way: the optimizer
 *  diagnosed it correctly three times, the generator fixed it in iter_1, and the loop
 *  still promoted the crashing iter_0 because both scored objective 0.0.
 *
 *  Sources, in order of trust:
 *    1. `[solve] rows_in=N rows_out=M errors=E` — the contract's own summary line.
 *    2. `[solve] WARN-TOTAL <reason> n=K` where the reason names an error, for programs
 *       written before `errors=` was part of the contract.
 *    3. `[solve] ERROR ...` / `type=<Exception>` per-row lines, which the contract caps
 *       at five per reason — a floor, never a count.
 *
 *  Returns null counts when the program said nothing; the caller must treat unknown as
 *  healthy rather than punishing a silent program twice.
 */
/** Branch/warning names that denote a raised EXCEPTION rather than a negative outcome.
 *
 *  Deliberately excludes "fail": the contract asks for branch names describing HOW a row
 *  was decided, and `equality_fail` / `predicate_fail` / `match_fail` are all ordinary
 *  "the predicate did not hold" branches. Counting those as errors made a healthy run
 *  look like it had lost 129 of its rows. */
const ERROR_COUNTER_NAME = /error|exception|crash|traceback|throw/i;

export function parseRunHealth(text) {
  const log = String(text || "");
  const num = (v) => (v === undefined ? null : Number(v));
  const summary = /\[solve\]\s+rows_in=(\d+)\s+rows_out=(\d+)(?:\s+errors=(\d+))?/i.exec(log);
  let rowsIn = summary ? num(summary[1]) : null;
  let rowsOut = summary ? num(summary[2]) : null;
  let errorRows = summary && summary[3] !== undefined ? num(summary[3]) : null;

  if (errorRows === null) {
    // WARN-TOTAL carries the true total; the per-row ERROR lines are capped and cannot.
    let total = 0;
    let sawErrorTotal = false;
    const totals = /\[solve\]\s+WARN-TOTAL\s+(\S+)\s+n=(\d+)/gi;
    for (let m = totals.exec(log); m; m = totals.exec(log)) {
      if (ERROR_COUNTER_NAME.test(m[1])) { total += Number(m[2]); sawErrorTotal = true; }
    }
    if (sawErrorTotal) errorRows = total;
  }
  if (errorRows === null) {
    const branches = /\[solve\]\s+branch=(\S+)\s+n=(\d+)/gi;
    let total = 0;
    let saw = false;
    for (let m = branches.exec(log); m; m = branches.exec(log)) {
      if (!ERROR_COUNTER_NAME.test(m[1])) continue;
      total += Number(m[2]);
      saw = true;
    }
    if (saw) errorRows = total;
  }
  // A program that emitted its contract summary and named no error branch reported ZERO
  // errors — that is a measurement, not a silence. Collapsing the two would make a clean
  // run incomparable to a failing one, which is exactly the tie the health check exists
  // to break.
  if (errorRows === null && summary) errorRows = 0;
  return { rowsIn, rowsOut, errorRows };
}

/** Fraction of a run's rows that failed, or null when the program reported nothing.
 *
 *  `rowsIn` is the DOMAIN size (a join's pair count), while the error counters are
 *  per unit of work actually attempted, so the two can legitimately differ by orders of
 *  magnitude. Cap at 1 rather than pretending the ratio is meaningful above it.
 */
export function runErrorRate(health) {
  if (!health || health.errorRows === null || health.errorRows === undefined) return null;
  const denom = [health.rowsIn, health.rowsOut].find((v) => Number.isFinite(v) && v > 0);
  if (!denom) return health.errorRows > 0 ? 1 : 0;
  return Math.min(1, health.errorRows / denom);
}

/** Did this execution fail on enough of its rows that its score is not worth comparing?
 *
 *  Half is deliberate: a program losing most of its rows to exceptions is not a weaker
 *  candidate, it is a broken one, and letting it tie on objective lets it win a
 *  tie-break against a candidate that actually runs.
 */
export const RUN_ERROR_RATE_FATAL = 0.5;

export function runIsBroken(health) {
  // Produced nothing AND lost rows to exceptions. The row-rate test cannot catch this
  // case: a join reports `rows_in` as its PAIR domain (mmqa q2a: 2600) while the errors
  // are counted per image (74), so a run that failed on every image it touched still
  // rates under 3%. An empty result is only a finding when nothing threw.
  if (health && health.rowsOut === 0 && health.errorRows > 0) return true;
  const rate = runErrorRate(health);
  return rate !== null && rate >= RUN_ERROR_RATE_FATAL;
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

/** Reduce benchmark metadata to an agent-safe natural-language intent.
 *
 * MMQA natural_language JSON files colocate `nl_question` with `ground_truth`.
 * Serializing the whole object into a prompt leaks final answers even though the
 * agent is told not to read them.  Use only explicit intent fields and never pass
 * answer/label metadata through as a fallback.
 */
export function sanitizeAgentQueryMetadata(raw) {
  if (typeof raw === "string") return raw;
  if (!raw || typeof raw !== "object") {
    return "(no natural_language/*.json found)";
  }
  for (const key of ["question", "nl_question", "nl", "description"]) {
    const value = raw[key];
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return "(natural-language intent unavailable; use the SQL)";
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
  const nl = sanitizeAgentQueryMetadata(nlRaw);
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

/** Usage counts for learned skills only, "" when none were loaded. */
async function renderLearnedSkillUsage(usagePath, skills) {
  if (!existsSync(usagePath)) return "";
  const log = await readJSON(usagePath);
  if (!log) return "";
  const learned = new Set(skills.filter((s) => !s.role).map((s) => s.name));
  const lines = [];
  for (const [queryId, entry] of Object.entries(log)) {
    for (const [name, count] of Object.entries(entry.skills || {})) {
      if (learned.has(name)) lines.push(`- ${name}: loaded ${count}× by ${queryId}`);
    }
  }
  return lines.join("\n");
}

/**
 * Post-run memory curation: the Memory Manager proposes, `applyMemoryUpdate` writes.
 *
 * The agent never touches `graph/`. It authors skill directories and one JSON
 * proposal; code recomputes every retrieval key from this run's own SQL and
 * re-checks every improvement claim. That split is why a mis-authored proposal can
 * only cost a curation pass, not corrupt retrieval.
 */
async function curateMemory(args, runPlans) {
  const queries = {};
  const evidenceLines = [];
  for (const plan of runPlans) {
    const runDir = resolve(args.out, `${args.benchmark}-${plan.query}`);
    const telemetry = await readJSON(resolve(runDir, "telemetry.json"));
    if (!telemetry) continue;
    const refine = telemetry.refine || {};
    const values = (refine.objective_history || [])
      .map((e) => Number(e?.value))
      .filter((v) => Number.isFinite(v));
    const direction = refine.objective_direction === "minimize" ? "minimize" : "maximize";
    const objective = {
      name: refine.objective || "objective",
      value: values.length
        ? (direction === "minimize" ? Math.min(...values) : Math.max(...values))
        : null,
      direction,
    };
    queries[plan.query] = {
      sql: plan.sql,
      nl: plan.nl,
      corpus: plan.corpus,
      tables: plan.tables,
      objective,
      iterations: refine.iterations ?? 0,
      replans: telemetry.replans ?? 0,
      actionCounts: telemetry.optimizer_actions ?? null,
      plan: await readJSON(resolve(runDir, "plan.json")),
      promoted: {
        plan_path: existsSync(resolve(runDir, "plan.json")) ? resolve(runDir, "plan.json") : null,
        helpers_path: existsSync(resolve(runDir, `_semantic_helpers_${plan.query}.py`))
          ? resolve(runDir, `_semantic_helpers_${plan.query}.py`) : null,
        solver_path: existsSync(resolve(runDir, `solve_${plan.query}.py`))
          ? resolve(runDir, `solve_${plan.query}.py`) : null,
        manifest_path: existsSync(resolve(runDir, "candidate_manifest.json"))
          ? resolve(runDir, "candidate_manifest.json") : null,
        candidate_id: telemetry.best_candidate_id ?? null,
      },
      metricFamily: telemetry.metrics?.metric_family ?? null,
      runDir,
    };

    const iterDirs = (await readdir(runDir, { withFileTypes: true }).catch(() => []))
      .filter((e) => e.isDirectory() && /^iter_\d+$/.test(e.name))
      .map((e) => e.name)
      .sort();
    const history = (refine.f1_history || [])
      .map((h) => `${h.iteration}:${h.objective?.value ?? "n/a"}${h.improved ? "+" : ""}`)
      .join(" → ");
    evidenceLines.push(
      `### ${plan.query}\n`
      + `- run dir: \`${runDir}\`\n`
      + `- objective: ${objective.name}=${objective.value ?? "n/a"} (${direction})\n`
      + `- trajectory: ${history || "(single shot)"}\n`
      + `- actions: ${JSON.stringify(telemetry.optimizer_actions || {})}, replans: ${telemetry.replans ?? 0}\n`
      + `- iterations on disk: ${iterDirs.join(", ") || "(none)"}\n`
      + `- memory match: \`${resolve(runDir, "memory_match.json")}\``,
    );
  }
  if (!Object.keys(queries).length) {
    console.log("[SemDB] memory curation: no telemetry to curate.");
    return;
  }

  const skills = await listSkills(args.skillRoot);
  const templates = await getNodesByLayer(1, args.memoryDir);
  const usagePath = resolve(args.out, "skill_usage.json");
  const updatePath = resolve(args.out, "memory_update.json");
  const summaryPath = resolve(args.out, "memory_update_summary.json");

  const template = readFileSync(memoryManagerConfig.userPromptPath, "utf-8");
  const userPrompt = renderTemplate(template, {
    run_id: args.runId,
    benchmark: args.benchmark,
    scale_factor: args.scaleFactor || "flat",
    out_dir: args.out,
    headroom_threshold: defaults.memory.differentialHeadroomThreshold,
    query_evidence: evidenceLines.join("\n\n"),
    skills_dir: skillsDirFor(args.skillRoot),
    existing_skills: skills.filter((s) => !s.role)
      .map((s) => `- **${s.name}**: ${s.description}`).join("\n"),
    existing_templates: templates
      .map((t) => `- ${t.id}: ${t.summary} (instances: ${t.content?.instance_count ?? 1})`)
      .join("\n"),
    // Only LEARNED skills. Generator still discovers its role procedure, while Planner
    // and Optimizer receive theirs directly as the system procedure. In either case role
    // procedures are not knowledge and must stay out of evidence curation.
    skill_usage: await renderLearnedSkillUsage(usagePath, skills),
    update_schema_path: resolve(__dirname, "contracts", "memory-update.schema.json"),
    update_path: updatePath,
  });

  const systemPrompt = await readFile(memoryManagerConfig.promptPath, "utf-8");
  const result = await runAgent(memoryManagerConfig.name, {
    systemPrompt,
    userPrompt,
    allowedTools: memoryManagerConfig.allowedTools,
    model: args.modelOverride || getAgentModel(memoryManagerConfig.configKey, args.agentProvider),
    effortLevel: getAgentEffort(memoryManagerConfig.configKey, args.agentProvider),
    configName: memoryManagerConfig.configKey,
    cwd: args.out,
    timeoutMs: defaults.agentTimeoutMs,
    useSkills: false,
  });
  if (result.error) {
    console.error(`[SemDB] memory curation: agent failed (non-fatal): ${result.error}`);
    return;
  }

  const update = await readJSON(updatePath);
  if (!update) {
    console.warn(`[SemDB] memory curation: no proposal at ${updatePath}.`);
    return;
  }
  const summary = await applyMemoryUpdate(update, {
    memoryDir: args.memoryDir,
    skillRoot: args.skillRoot,
    runId: args.runId,
    benchmark: args.benchmark,
    queries,
    config: defaults.memory,
    groundTruthDir: args.groundTruthDir,
  });
  await writeJsonAtomic(summaryPath, summary);
  const created = Object.entries(summary.nodes_created || {})
    .map(([layer, count]) => `${layer}:${count}`).join(", ") || "none";
  console.log(`[SemDB] memory curation: applied=${summary.applied} created=${created}`
    + ` updated=${summary.nodes_updated} edges=${summary.edges_created}`
    + ` skills=${summary.skills_accepted.length} rejected=${summary.rejected.length}`);
  for (const r of summary.rejected) {
    console.warn(`[SemDB]   rejected (${r.reason}): ${r.detail || r.field || ""}`);
  }
  console.log(`[SemDB] memory curation summary → ${summaryPath}`);
}

/** This query's skill loads from <out>/skill_usage.json ({} when nothing was loaded). */
async function readSkillUsage(outDir, query) {
  const log = await readJSON(resolve(outDir, "skill_usage.json"));
  return log?.[query]?.skills || {};
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
 * IMAGE table referenced in the predicate, then a TEXT one, then a PROXY one. AUDIO-only
 * queries have no supported corpus — the caller detects `modality === "audio"` and skips
 * them. A PROXY corpus (SUPG replay) has no unstructured content at all; it is ranked
 * last so a scenario that has real content never loses it to the proxy stand-in.
 */
async function chooseCorpus(sql, tables, args) {
  const used = new Set(tablesInPredicate(sql, args.benchmark));
  const inPred = tables.filter((t) => used.has(t.table));
  const pick = (cands) =>
    cands.find((t) => t.modality === "image") ||
    cands.find((t) => t.modality === "text") ||
    cands.find((t) => t.modality === "proxy") || null;
  return pick(inPred) || pick(tables) ||
    tables.find((t) => t.modality === "audio") ||   // audio-only → caller skips
    tables[tables.length - 1] || { table: "corpus", path: "", modality: "text", isImages: false };
}

/** List query ids (<name>.sql → <name>) in a query dir, sorted. */
async function listQueries(dir) {
  if (!dir) return [];
  const files = await readdir(dir);
  return files.filter((f) => f.toLowerCase().endsWith(".sql"))
    .map((f) => f.replace(/\.sql$/i, "")).sort();
}

function mergeAgentResults(results, extraProfile = {}) {
  const last = results.at(-1) || {};
  const tokens = {};
  const skillsUsed = {};
  for (const result of results) {
    for (const [key, value] of Object.entries(result.tokens || {})) {
      tokens[key] = (tokens[key] || 0) + (Number(value) || 0);
    }
    for (const [key, value] of Object.entries(result.skillsUsed || {})) {
      skillsUsed[key] = (skillsUsed[key] || 0) + (Number(value) || 0);
    }
  }
  const sumProfile = (key) => results.reduce(
    (total, result) => total + (Number(result.profile?.[key]) || 0), 0);
  return {
    ...last,
    durationMs: results.reduce((total, result) => total + (result.durationMs || 0), 0),
    tokens,
    costUsd: results.reduce((total, result) => total + (result.costUsd || 0), 0),
    numTurns: results.reduce((total, result) => total + (result.numTurns || 0), 0),
    skillsUsed,
    profile: {
      structured: results.some((result) => result.profile?.structured),
      structured_requests: sumProfile("structured_requests"),
      system_prompt_chars: sumProfile("system_prompt_chars"),
      user_prompt_chars: sumProfile("user_prompt_chars"),
      agent_messages: sumProfile("agent_messages"),
      agent_message_chars: sumProfile("agent_message_chars"),
      reasoning_items: sumProfile("reasoning_items"),
      reasoning_chars: sumProfile("reasoning_chars"),
      command_executions: sumProfile("command_executions"),
      command_output_chars: sumProfile("command_output_chars"),
      max_command_output_chars: Math.max(
        0,
        ...results.map((result) => Number(result.profile?.max_command_output_chars) || 0),
      ),
      file_changes: sumProfile("file_changes"),
      ...extraProfile,
    },
  };
}

async function validateStructuredArtifact(configKey, outputPath, vars) {
  if (configKey === "query_planner") {
    const previousPlan = vars.previous_plan_path
      ? await readJSON(vars.previous_plan_path)
      : null;
    const plan = await readAndValidatePlan(outputPath, {
      queryId: vars.query_id,
      ...(previousPlan ? { previousPlan } : {}),
    });
    const findings = [
      ...lintImagePlan(plan),
      ...lintPlanAgainstTableProfile(plan, vars.planner_table_profile),
    ];
    if (findings.length) {
      throw new Error(`Semantic plan lint failed: ${findings.join(" ")}`);
    }
    return plan;
  }
  if (configKey === "semantic_optimizer") {
    const manifest = await readJSON(vars.candidate_manifest_path);
    const action = await readAndValidateOptimizerAction(outputPath, {
      queryId: vars.query_id,
      candidateId: manifest?.candidate_id,
    });
    await validateStructuredOptimizerEvidence(action, vars);
    return action;
  }
  throw new Error(`No structured artifact validator for ${configKey}`);
}

/** A role sees its own procedure plus only retrieval-selected learned skills. Planner and
 * Optimizer put the procedure directly in their system prompt, so advertising it again
 * would recreate the skill-read loop this path is designed to remove. */
export function selectAgentSkills(agentConfig, availableSkills, relevantSkillNames, options = {}) {
  const relevant = new Set(Array.isArray(relevantSkillNames) ? relevantSkillNames : []);
  return (availableSkills || []).filter((skill) => {
    if (skill.role) {
      return !options.procedureInSystem && skill.name === agentConfig.skillName;
    }
    return relevant.has(skill.name);
  });
}

async function runPhase(agentConfig, vars, runDir, args, opts = {}) {
  const systemPromptPath = opts.systemPromptPath || agentConfig.promptPath;
  const userPromptPath = opts.userPromptPath || agentConfig.userPromptPath;
  const defaultRolePaths = !opts.systemPromptPath && !opts.userPromptPath;
  const canonicalAgentProcedure = defaultRolePaths
    && supportsStructuredRole(agentConfig.configKey)
    && Boolean(agentConfig.skillPath);
  const useStructured = args.agentProvider === "vllm"
    && args.agentExecution === "structured"
    && supportsStructuredRole(agentConfig.configKey)
    && defaultRolePaths;
  const model = args.modelOverride || getAgentModel(agentConfig.configKey, args.agentProvider);
  const effortLevel = getAgentEffort(agentConfig.configKey, args.agentProvider);

  let cachedAgentExecution = null;
  const prepareFullAgentExecution = async () => {
    if (cachedAgentExecution) return cachedAgentExecution;
    let systemPrompt;
    let userPrompt;
    let contextProfile = {};
    if (canonicalAgentProcedure) {
      const prepared = await prepareAgentRole(agentConfig, vars);
      const contextPath = resolve(runDir, `_agent_context_${agentConfig.configKey}.md`);
      if (!args.dryRun) {
        await writeTextAtomic(contextPath, `${prepared.contextText}\n`, { mode: 0o444 });
      }
      const template = await readFile(userPromptPath, "utf-8");
      systemPrompt = prepared.systemPrompt;
      userPrompt = renderTemplate(template, {
        ...vars,
        agent_context_path: contextPath,
        agent_context_sha256: prepared.contextSha256,
      });
      contextProfile = {
        canonical_role_procedure: agentConfig.skillName,
        agent_context_bundle_path: contextPath,
        agent_context_bundle_chars: prepared.contextText.length,
        agent_context_bundle_sha256: prepared.contextSha256,
      };
    } else {
      const [legacySystemPrompt, template] = await Promise.all([
        readFile(systemPromptPath, "utf-8"),
        readFile(userPromptPath, "utf-8"),
      ]);
      systemPrompt = legacySystemPrompt;
      userPrompt = renderTemplate(template, vars);
    }

    const skillCandidates = selectAgentSkills(
      agentConfig,
      args.availableSkills || [],
      vars.memory_inline_skills ? [] : vars.memory_relevant_skill_names,
      { procedureInSystem: canonicalAgentProcedure },
    );
    const skillsEnabled = args.enableAgentSkills !== false
      && Boolean(args.skillRoot)
      && (agentConfig.allowedTools || []).includes("Skill")
      && skillCandidates.length > 0;
    const skills = skillsEnabled ? skillCandidates : [];
    const allowedTools = skillsEnabled
      ? agentConfig.allowedTools
      : (agentConfig.allowedTools || []).filter((tool) => tool !== "Skill");
    cachedAgentExecution = {
      skillsEnabled,
      skills,
      profile: {
        ...contextProfile,
        discoverable_skill_count: skills.length,
        discoverable_skill_names: skills.map((skill) => skill.name),
      },
      options: {
        systemPrompt,
        userPrompt,
        allowedTools,
        model,
        effortLevel,
        configName: agentConfig.configKey,
        cwd: runDir,
        timeoutMs: defaults.agentTimeoutMs,
        useSkills: skillsEnabled,
        skillRoot: skillsEnabled ? args.skillRoot : undefined,
        skillsDir: skillsEnabled ? skillsDirFor(args.skillRoot) : undefined,
        skills,
        settingSources: skillsEnabled ? ["project"] : undefined,
      },
    };
    return cachedAgentExecution;
  };

  if (args.dryRun) {
    console.log(`\n[SemDB] --- ${agentConfig.name} (dry-run) ---`);
    if (useStructured) {
      const prepared = await prepareStructuredRole(agentConfig, vars, {
        effortLevel: args.structuredReasoningEffort,
        maxOutputTokens: agentConfig.configKey === "query_planner"
          ? args.plannerMaxOutputTokens
          : args.optimizerMaxOutputTokens,
      });
      console.log(`[SemDB] execution: structured, max_output_tokens=${prepared.maxOutputTokens}, effort=${prepared.effortLevel}`);
      console.log(prepared.userPrompt);
      return { dryRun: true };
    }
    const agentExecution = await prepareFullAgentExecution();
    console.log(`[SemDB] discoverable skills: ${
      agentExecution.skillsEnabled
        ? agentExecution.skills.map((skill) => skill.name).join(", ")
        : "(disabled)"
    }`);
    if (canonicalAgentProcedure) {
      console.log(`[SemDB] canonical procedure: ${agentConfig.skillName}`);
      console.log(`[SemDB] context bundle: ${agentExecution.profile.agent_context_bundle_path}`
        + ` (${agentExecution.profile.agent_context_bundle_chars} chars)`);
    }
    console.log(agentExecution.options.userPrompt);
    return { dryRun: true };
  }

  let result;
  if (useStructured) {
    const prepared = await prepareStructuredRole(agentConfig, vars, {
      effortLevel: args.structuredReasoningEffort,
      maxOutputTokens: agentConfig.configKey === "query_planner"
        ? args.plannerMaxOutputTokens
        : args.optimizerMaxOutputTokens,
    });
    const attempts = [];
    let correction = "";
    let validated = false;
    let requestMaxOutputTokens = prepared.maxOutputTokens;
    for (let attempt = 1; attempt <= 2; attempt++) {
      const structured = await runStructuredAgent(agentConfig.name, {
        systemPrompt: prepared.systemPrompt,
        userPrompt: prepared.userPrompt + correction,
        schema: prepared.schema,
        schemaName: prepared.schemaName,
        maxOutputTokens: requestMaxOutputTokens,
        model,
        timeoutMs: defaults.agentTimeoutMs,
        configName: agentConfig.configKey,
        effortLevel: prepared.effortLevel,
      });
      attempts.push(structured);
      if (structured.error) {
        if (attempt === 1 && structured.retryable) {
          correction = `\n\n## Retry correction\nThe previous structured response failed: ${structured.error.slice(0, 1200)}\nReturn one corrected JSON object.`;
          if (structured.profile?.incomplete_reason === "max_output_tokens") {
            requestMaxOutputTokens = prepared.retryMaxOutputTokens;
            correction += ` The retry output budget is ${requestMaxOutputTokens} tokens; finish the JSON before that limit.`;
          }
          continue;
        }
        break;
      }
      try {
        await writeJsonAtomic(prepared.outputPath, structured.structured);
        await validateStructuredArtifact(agentConfig.configKey, prepared.outputPath, vars);
        validated = true;
        break;
      } catch (error) {
        structured.error = `Structured artifact contract failed: ${error.message}`;
        structured.retryable = true;
        if (attempt === 1) {
          correction = `\n\n## Retry correction\nThe previous JSON failed the runtime contract: ${error.message.slice(0, 1200)}\nReturn the complete corrected object.`;
        }
      }
    }
    if (validated) {
      result = mergeAgentResults(attempts, {
        structured_attempts: attempts.length,
        fallback_used: false,
      });
      delete result.error;
    } else {
      const reason = attempts.at(-1)?.error || "unknown structured-output failure";
      console.warn(`[SemDB] ${agentConfig.name} structured path failed; falling back to the tool agent: ${reason}`);
      const agentExecution = await prepareFullAgentExecution();
      const fallback = await runAgent(agentConfig.name, agentExecution.options);
      result = mergeAgentResults([...attempts, fallback], {
        structured_attempts: attempts.length,
        fallback_used: true,
        fallback_reason: reason,
        ...agentExecution.profile,
      });
    }
  } else {
    const agentExecution = await prepareFullAgentExecution();
    result = await runAgent(agentConfig.name, agentExecution.options);
    result.profile = { ...(result.profile || {}), ...agentExecution.profile };
  }
  if (result.error) throw new Error(`${agentConfig.name} failed: ${result.error}`);
  const usageQueryId = opts.queryId || vars.query_id;
  if (usageQueryId && Object.keys(result.skillsUsed || {}).length) {
    await recordSkillUsage(args.out, usageQueryId, result.skillsUsed);
  }
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
      profile: r.profile || {},
      cost_usd: r.costUsd || 0,
      llm_calls: r.numTurns || 1,   // internal turns this agent made (min 1)
    });
    return r;
  };
}

function agentModelsLine(args) {
  const m = (k) => args.modelOverride || getAgentModel(k, args.agentProvider);
  return `planner=${m("query_planner")}, codegen=${m("semantic_code_generator")}, optimizer=${m("semantic_optimizer")}`;
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
           isImage: corpus.modality === "image", isAudio: corpus.modality === "audio",
           isProxy: corpus.modality === "proxy", plan };
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

  // Cross-run memory for this query (pre-computed in main(); absent = disabled).
  const mem = args.memoryClassifications?.[query] || null;
  if (mem) {
    console.log(`[SemDB] [${query}] memory: tier=${mem.tier} score=${mem.score.toFixed(2)}`
      + ` L1=${mem.matchedL1 || "none"}${mem.warmStart ? " warm-start" : ""}`);
    try {
      await writeJsonAtomic(resolve(runDir, "memory_match.json"), {
        tier: mem.tier,
        score: mem.score,
        matched_l1: mem.matchedL1,
        matched_l0: mem.matchedL0,
        warm_start: Boolean(mem.warmStart),
        skills_available: mem.skillsAvailable,
        injected_tokens: mem.injectedTokens,
        signature: mem.signature,
      });
    } catch (error) {
      console.warn(`[SemDB] [${query}] could not write memory_match.json (non-fatal): ${error.message}`);
    }
  }
  /** Memory prompt variables for one role; all empty when memory is off. */
  const memoryVars = (role) => ({
    memory_pre_injection: mem?.blocks?.[role] || "",
    memory_catalog: mem?.catalog || "",
    memory_inline_skills: mem?.inlineSkills || "",
    memory_relevant_skill_names: mem?.relevantSkillNames || [],
    memory_note: mem && mem.tier !== "novel" ? "true" : "",
  });

  // Describe every referenced table (header + path) for the solver.
  const tableLines = [];
  for (const t of tables) {
    const kind = t.isImages ? "image manifest" : (t.modality || "table");
    tableLines.push(`- ${t.table} (${kind}): path=${t.path}\n    columns: ${await headerOf(t.path)}`);
    if (t.modality === "proxy") {
      // The SQL header states the contract too, but the solver prompts are written for
      // content corpora; without this the agent reaches for CLIP or a text column that
      // does not exist. Say it where the corpus itself is described.
      const spec = supgSpec(args, query);
      tableLines.push(
        `    NO CONTENT: this corpus has no image, text or caption column — the raw\n`
        + `    ${t.table} data was never published. \`${t.col || "proxy_score"}\` is a `
        + `cheap proxy model's\n`
        + `    confidence in [0,1]; it is free and unlimited. The ground-truth label is\n`
        + `    reachable ONLY through the metered oracle:\n`
        + `        import supg_oracle\n`
        + `        labels = supg_oracle.oracle(ids)   # -> list[bool]\n`
        + `        supg_oracle.remaining()            # labels still affordable\n`
        + `    At most ${spec?.oracle_limit ?? 400} DISTINCT ids may be labelled; `
        + `re-reading an already-labelled\n`
        + `    id is cached and free. Exceeding the budget raises OracleBudgetExceeded\n`
        + `    and fails the run. Do NOT attempt to label the corpus row by row: spend\n`
        + `    the budget calibrating a threshold/sampling rule over `
        + `\`${t.col || "proxy_score"}\`,\n`
        + `    then emit every id the rule selects. Do NOT import torch, CLIP or any\n`
        + `    encoder — there is nothing for them to encode.`);
    }
  }
  if (args.benchmark === "ecomm") {
    const normalizedProducts = resolve(args.tableDir || args.dataDir, "ecomm_products.csv");
    if (existsSync(normalizedProducts)) {
      tableLines.push(
        `- ECOMM_PRODUCTS (normalized local adapter): path=${normalizedProducts}\n`
        + `    columns: ${await headerOf(normalizedProducts)}\n`
        + `    equivalence: one row per STYLES_DETAILS.id after the SQL `
        + `styleImages.default.imageURL = IMAGE_MAPPING.link = IMAGES mapping chain; `
        + `filename resolves locally under --image-dir; imageURL/link are ordinary `
        + `metadata and image ref is the decoded local file. This is the offline `
        + `EXTERNAL_OBJECT_TRANSFORM adapter, not validation data.`,
      );
    }
  }
  const plannerTableProfile = architecture === "pgo"
    ? await buildPlannerTableProfile(tables)
    : null;
  const isImage = corpus.isImage || corpus.modality === "image";
  // SUPG replay: the corpus is (id, proxy_score) and the label is reachable only via
  // the metered oracle. There is no content for a VLM/LLM to read, so this branch
  // replaces the extraction and oracle-labelling paths rather than degrading them.
  const isProxy = corpus.modality === "proxy";
  // EVERY supg query carries a PT/RT contract and is graded on precision or recall,
  // content mode included — only the metered ORACLE LIMIT is replay-specific. Gating
  // the whole spec on isProxy left the imagenet rows with no target_met verdict.
  const supg = supgSpec(args, query);
  if (isProxy && !supg) {
    throw new Error(`[SemDB] [${query}] corpus '${corpus.table}' is a SUPG replay table `
      + `but supg_targets.json has no entry for ${query}. Re-run `
      + `data/supg/build_supg_scenario.py.`);
  }
  if (isProxy && !supg.oracleLabelsPath) {
    throw new Error(`[SemDB] [${query}] SUPG replay needs withheld labels; `
      + `supg_targets.json records none for ${query}.`);
  }
  const supgLedger = isProxy ? resolve(runDir, "oracle_calls.json") : null;
  const directSqlPath = args.queryDir
    ? resolve(args.queryDir, `${query}.sql`)
    : null;
  const directSites = directSqlPath ? callSites(directSqlPath) : [];
  const directValidationPlan = args.valRate && directSqlPath
    ? validationPlan(directSqlPath, args.benchmark, query)
    : null;
  const validationCapability = directValidationPlan
    ? classifyValidationCapability(directValidationPlan, {
        benchmark: args.benchmark,
        query,
      })
    : null;
  if (args.valRate && args.valCallSite == null
      && validationCapability?.class === "not_compilable") {
    const capabilityPath = resolve(runDir, "validation_capability.json");
    await writeJsonAtomic(capabilityPath, validationCapability);
    console.log(`[SemDB] [${query}] validation capability NOT_COMPILABLE `
      + `(${validationCapability.reason_code}): ${validationCapability.reason}`);
    return { query, validation_capability: validationCapability, capabilityPath };
  }
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
  let valPairComposition = "";
  let valPairIncludeDiagonal = false;
  let valCorpusCsv = corpus.path;
  let valRowIdCol = "";
  let valLeftIdCol = "";        // structured-side key column of a pair id
  let valKeyExample = "";       // a real pair id, shown to the solver verbatim
  const buildsValidation = !valFile && !!args.valRate && !args.noRefine;
  const validationStarted = Date.now();
  if (!valFile && args.valRate && !args.noRefine) {
    if (isProxy && args.queryDir) {
      // SUPG replay draws its validation labels from the withheld label column, at a
      // FIXED size equal to the query's ORACLE LIMIT. No endpoint is involved, and the
      // spend is reported as val_oracle_calls rather than folded into accuracy.
      const { idCol } = await corpusCols(corpus, args);
      valRowIdCol = idCol;
      valFile = buildValSet(args, query, {
        corpusCsv: corpus.path, idCol, sqlPath: resolve(args.queryDir, `${query}.sql`),
        isImage: false,
        importanceBy: `column:${corpus.col || "proxy_score"}`,
        labelSource: "supg-oracle",
        oracleLabels: supg.oracleLabelsPath,
        valN: supg.oracle_limit,
      });
      if (!valFile) {
        throw new Error(
          `[SemDB] [${query}] SUPG replay validation set could not be built. Without it `
          + `the refinement loop would score against the FULL label column, which is the `
          + `whole quantity the oracle budget exists to ration.`);
      }
      console.log(`[SemDB] [${query}] SUPG replay: ${supg.oracle_limit} validation `
        + `labels from ${supg.oracleLabelsPath}`);
    } else if (!args.endpoint && !args.valPlanOnly) {
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
        if (vplan.candidate?.unit !== "pair") {
          console.log(`[SemDB] [${query}] typed/grouped multi-site plan: `
            + `${(vplan.candidate?.kinds || []).join(" + ")}. Candidate-frame `
            + `materialization is deferred to the typed-operator phase.`);
          return { query, validation_plan: vplan };
        }
        // A typed filter_then_extract composition has several AI sites but one
        // executable root pair frame. Continue through the ordinary pair builder.
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
        const moviePhysical = args.benchmark === "movie";
        const frame = buildSelfPairFrame(args, query, {
          corpusCsv: deterministic,
          idCol: moviePhysical ? "_semdb_row_id" : "id",
          ...(isImage ? { imageCol: "filename", imageDir }
            : { textCol: "semantic_text" }),
          clipModel, includeDiagonal: valPairIncludeDiagonal,
          ...(moviePhysical ? { excludeEqualCol: "reviewId" } : {}),
        });
        const pairMeta = await readJSON(frame + ".meta.json");
        const pairPopulation = pairMeta?.rows;
        if (!Number.isInteger(pairPopulation)) {
          throw new Error(`[SemDB] [${query}] pair frame metadata has no row count`);
        }
        const pairSample = Math.ceil(pairPopulation * args.valRate);
        console.log(`[SemDB] [${query}] self-join validation population: `
          + `${filteredRows} filtered physical rows -> ${pairPopulation} ordered `
          + `pairs after ordinary SQL inequalities; rate=${args.valRate} -> `
          + `${pairSample} validation rows.`);
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
        valLeftIdCol = moviePhysical ? "_semdb_row_id" : "id";
        const firstRow = (await readFile(frame, "utf-8")).split(/\r?\n/)[1] || "";
        valKeyExample = firstRow.split(",")[0] || "<left_id>-<right_id>";
      } else if (crossTable) {
        // Predicate argument order is not a physical-side contract. EComm q8 names
        // the image first and the structured description second, while mmqa names
        // the structured side first. Resolve the alias through predicate.py's aligned
        // aliases/bases arrays instead of assuming aliases[0] is the left row.
        const imageAlias = predicateAliasForTable(pairSite, corpus.table);
        const leftAlias = predicateAliasForTable(pairSite, structured.table)
          || (pairSite.aliases || []).find((alias) => alias !== imageAlias)
          || (pairSite.aliases || [])[0];
        // Pair/trace identity is a physical row identity, not necessarily a projected
        // value. q7 projects Airlines, but 200 rows contain only 135 distinct airline
        // names; using Airlines as the pair id makes keys collide. The configured key
        // (or first column, row_id for q7) is unique while the solver may still project
        // Airlines in its result CSV.
        const leftPhysical = await corpusCols(structured, args);
        const leftIdCol = leftPhysical.idCol;
        const leftPredicateCols = predicateCols(pairSite, leftAlias);
        // The text the predicate ASKS ABOUT, from the call site's own column list —
        // not a header heuristic. q2a's predicate reads t.Track (the racetrack name);
        // guessing from the header picks the last column, `Condition` ("Firm"/"Fast"),
        // and CLIP then scores logos against track surface conditions.
        let leftTextCol = leftPredicateCols[0]
          || (await corpusCols(structured, args)).textCol;
        let leftCsv = structured.path;
        let filteredLeftRows = null;
        let filteredFrameKey = "full-frame";
        if (args.benchmark === "ecomm" && String(query).toLowerCase() === "q8") {
          // q8's semantic predicate consumes BOTH the display name and the long
          // nested description, after CHAR_LENGTH(description) >= 3000. The
          // normalized product view lets frame_builder expose their faithful
          // concatenation as predicate_text and execute the CTE before pairing.
          leftCsv = buildDeterministicCrossLeftRows(args, query, sqlPath);
          const leftMeta = await readJSON(leftCsv + ".meta.json");
          filteredLeftRows = leftMeta?.output_rows;
          if (!Number.isInteger(filteredLeftRows)) {
            throw new Error(`[SemDB] [${query}] deterministic left frame metadata `
              + `has no logical output_rows count.`);
          }
          leftTextCol = "predicate_text";
          filteredFrameKey = "filtered-cross-q8";
        }
        console.log(`[SemDB] [${query}] pairwise call site (${pairSite.reason}) — `
          + `pair frame ${structured.table}.${leftTextCol} x ${corpus.table}`);
        // The validation population is the complete Cartesian product AFTER any
        // deterministic SQL prefix. Pruning to a global CLIP top-K changes the
        // estimand and made q7 look perfect on validation (40/40) while scoring
        // 0.1569 F1 over all 40,000 pairs.
        if (args.valPairTop) {
          console.warn(`[SemDB] [${query}] ignoring deprecated --val-pair-top `
            + `${args.valPairTop}: join validation samples the full pair population.`);
        }
        const leftRows = filteredLeftRows ?? await countRows(leftCsv);
        const rightRows = await countRows(corpus.path);
        const pairPopulation = (leftRows ?? 0) * (rightRows ?? 0);
        const pairSample = Math.ceil(pairPopulation * args.valRate);
        console.log(`[SemDB] [${query}] join validation population: ${leftRows} × `
          + `${rightRows} = ${pairPopulation} pairs; rate=${args.valRate} -> `
          + `${pairSample} validation rows.`);
        const frame = buildPairFrame(args, query, {
          leftCsv, leftIdCol, leftTextCol,
          rightCsv: corpus.path,
          rightIdCol: imgFilenameCol || imageCol,
          rightImageCol: imgFilepathCol || imgFilenameCol || imageCol,
          imageDir, clipModel, top: null,
        });
        if (frame) {
          if (args.valPlanOnly) {
            console.log(`[SemDB] [${query}] validation plan complete; `
              + `--val-plan-only skips Oracle labeling and agent execution.`);
            return {
              query,
              validation_plan: validationPlan(sqlPath, args.benchmark, query),
              deterministic_rows: filteredLeftRows,
              candidate_population: pairPopulation,
              sample_n: pairSample,
              frame,
            };
          }
          valCorpusCsv = frame;
          valFile = buildValSet(args, query, {
            corpusCsv: frame, idCol: "pair_id", sqlPath, isImage: true,
            pairwise: true, pairImageCols: ["file2"], textCols: ["text1"],
            imageDir, clipModel, endpoint: args.endpoint, oracleModel,
            importanceBy: "column:pair_score",
            frameKey: filteredFrameKey,
            ...(directValidationPlan?.candidate?.composition?.kind
                === "filter_then_extract" ? {
              oracleQuestion:
                "The text names a horse racetrack and the attached image is a candidate "
                + "logo. If the image is not that racetrack's logo, answer no_match. "
                + "If it is the logo, answer match:<primary logo color>. Choose exactly "
                + "one supplied label and provide no explanation.",
              oracleLabelType: "text",
              oracleChoices: [
                "no_match", "match:black", "match:white", "match:red",
                "match:orange", "match:yellow", "match:green", "match:blue",
                "match:purple", "match:pink", "match:brown", "match:gray",
                "match:silver", "match:gold",
              ],
            } : {}),
          });
          valPairwise = !!valFile;
          valPairKind = "cross";
          valPairComposition =
            directValidationPlan?.candidate?.composition?.kind || "";
          valLeftIdCol = leftIdCol;
          // A real key from the frame beats a described one: the solver copies it.
          const firstRow = (await readFile(frame, "utf-8")).split(/\r?\n/)[1] || "";
          valKeyExample = firstRow.split(",")[0] || `<${leftIdCol}>-<image filename>`;
        }
      } else {
        const textCols = semanticTextCols.length ? semanticTextCols : [textCol];
        const deterministicTitles =
          args.benchmark === "mmqa" && ["q4", "q5"].includes(String(query).toLowerCase())
            ? sqlInLiteralValues(sqlPath, "title")
            : [];
        const minimumValidationRate =
          args.benchmark === "mmqa" && String(query).toLowerCase() === "q5"
            ? Math.max(0, 1 - Number(args.valCertRate ?? 0))
            : 0;
        const rowCorpus = buildPhysicalRowFrame(args, query, {
          corpusCsv: corpus.path,
          textCols: isImage ? [] : textCols,
          ...(deterministicTitles.length
            ? { filterCol: "title", filterValues: deterministicTitles } : {}),
        });
        const rowIdCol = "_semdb_row_id";
        valCorpusCsv = rowCorpus;
        valRowIdCol = rowIdCol;
        if (args.valPlanOnly) {
          const population = (await readJSON(rowCorpus + ".meta.json"))?.output_rows;
          if (!Number.isInteger(population)) {
            throw new Error(`[SemDB] [${query}] row validation frame has no row count`);
          }
          const plannedRate = Math.max(args.valRate, minimumValidationRate);
          const sampleN = Math.ceil(population * plannedRate);
          console.log(`[SemDB] [${query}] per-row validation population: `
            + `${population}; rate=${plannedRate} -> ${sampleN} validation rows.`);
          console.log(`[SemDB] [${query}] validation plan complete; `
            + `--val-plan-only skips Oracle labeling and agent execution.`);
          return {
            query,
            validation_plan: validationPlan(sqlPath, args.benchmark, query),
            candidate_population: population,
            sample_n: sampleN,
            frame: rowCorpus,
          };
        }
        valFile = buildValSet(args, query, {
          corpusCsv: rowCorpus, idCol: rowIdCol,
          sqlPath, isImage,
          imageCol: imgFilenameCol || imageCol, imageDir, clipModel,
          textCols: isImage ? [] : ["semantic_text"],
          // Multi-field relational predicates need semantic relevance: TF-IDF cannot
          // connect Frankfurt with Germany/Europe, while the local CLIP encoder can.
          importanceBy: !isImage && textCols.length > 1
            ? "clip-text-similarity" : undefined,
          endpoint: args.endpoint, oracleModel,
          ...(
            args.benchmark === "mmqa"
            && ["q6b", "q6c"].includes(String(query).toLowerCase())
              ? {
                oracleQuestion:
                  `Use only the provided Destinations field. Answer true exactly when `
                  + `at least one listed destination city is geographically in `
                  + `${String(query).toLowerCase() === "q6b" ? "Germany" : "Europe"}. `
                  + `Ignore the airline name and any Airport field; a flight origin or `
                  + `airport name is not a destination.`,
                oracleLabelType: "boolean",
                oracleChoices: ["true", "false"],
              }
              : {}
          ),
          ...(
            args.benchmark === "mmqa" && String(query).toLowerCase() === "q4"
              ? {
                oracleQuestion:
                  "From the supplied movie title and description, return every "
                  + "applicable genre as a comma-separated list using only: action, "
                  + "biography, comedy, crime, drama, heist, horror, romance, satire, "
                  + "science fiction, thriller, war, western. Return genre names only, "
                  + "in alphabetical order, with no explanation.",
                oracleLabelType: "text",
              }
              : {}
          ),
          ...(
            args.benchmark === "mmqa" && String(query).toLowerCase() === "q5"
              ? {
                minimumRate: minimumValidationRate,
                oracleQuestion:
                  "Extract the full canonical names of every actor or actress "
                  + "explicitly named in the supplied movie description. Return a "
                  + "comma-separated alphabetical list of person names only, without "
                  + "roles or explanation.",
                oracleLabelType: "text",
              }
              : {}
          ),
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
    // The metered oracle must sit beside the script that imports it: python puts the
    // script's own directory on sys.path, and iterations execute out of iter_N/.
    if (isProxy) {
      installSupgOracle(dirname(iterCode), {
        labelsCsv: supg.oracleLabelsPath,
        budget: supg.oracle_limit,
        dataset: supg.dataset,
        query,
        ledger: resolve(dirname(iterCode), "oracle_calls.json"),
      });
    }
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
  const composedCrossPairTraceContract =
    [`This query composes a pairwise semantic filter with value extraction and is`,
     `scored once PER PAIR against a typed joint Oracle label.`,
     ``,
     `- \`<key>\` is \`"<physical_row_id>-<image_filename>"\`, using the structured`,
     `  row's \`${valLeftIdCol}\` and the image filename (e.g. \`"${valKeyExample}"\`).`,
     `- Form the complete structured-row × image domain. Apply \`--only-ids\` to`,
     `  this pair key before visual inference.`,
     `- Write exactly one trace value for every evaluated key: \`"no_match"\` when`,
     `  the image fails the SQL AI.IF logo predicate, otherwise`,
     `  \`"match:<canonical primary color>"\` (for example \`"match:blue"\`).`,
     `- The final CSV must still implement both SQL stages: emit only matching pairs`,
     `  and project the extracted color without the \`match:\` trace prefix.`,
     `- Do not emit a bare true/false trace: it discards the second semantic site's`,
     `  value and cannot validate the whole query.`].join("\n");
  const selfPairTraceContract =
    [`This query is a SEMANTIC SELF-JOIN and is scored PER ORDERED PAIR.`,
     ``,
     `- \`<key>\` is \`"<left_id>-<right_id>"\`, using \`${valLeftIdCol || "id"}\``,
     `  on both sides (e.g. \`"${valKeyExample}"\`). For \`_semdb_row_id\`, derive`,
     `  it as the zero-based CSV source-record ordinal before filtering. The final`,
     `  result still follows the SQL SELECT projection and preserves duplicates.`,
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
     `        key = f"{left['${valLeftIdCol || "id"}']}-{right['${valLeftIdCol || "id"}']}"`,
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
       `- \`<key>\` is \`${valRowIdCol || imgFilenameCol}\`. For`,
       `  \`_semdb_row_id\`, derive it as \`str(source_record_index)\` while`,
       `  enumerating the original manifest before any SQL filter. It exists only`,
       `  for trace identity and must not replace the SQL projection.`,
       `- \`--only-ids\` holds those row ids; apply it as a membership filter right`,
       `  after the manifest is loaded and enumerated:`,
       ``,
       "```python",
       `rows = [dict(r, _semdb_row_id=str(i)) for i, r in enumerate(rows)]`,
       `if only is not None:`,
       `    rows = [r for r in rows if r["_semdb_row_id"] in only]`,
       "```"].join("\n")
    : [`This query is scored PER ROW.`,
       ``,
       `- \`<key>\` is the \`${valRowIdCol || "primary-key"}\` value of the text`,
       `  corpus row, converted to a string. When this is \`_semdb_row_id\`, derive`,
       `  it as \`str(source_record_index)\` while enumerating the original CSV before`,
       `  any SQL filter; it exists only for trace identity and is never projected.`,
       `- Apply \`--only-ids\` immediately after loading rows, before semantic`,
       `  inference, and write one true/false or extracted-value trace entry for`,
       `  every listed row key.`].join("\n");
  const traceContract = tracePairKind
    ? (tracePairKind === "self"
        ? selfPairTraceContract
        : (valPairComposition === "filter_then_extract"
            ? composedCrossPairTraceContract : crossPairTraceContract))
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
    // A program that threw on most of its rows exits 0 and still writes a trace, so
    // without this it scores as "ok" and can win a tie against a candidate that runs.
    const health = parseRunHealth([run.stdout, run.stderr].filter(Boolean).join("\n"));
    const broken = runIsBroken(health);
    // A candidate that selects nothing scores F1 0 like any other bad candidate, but it
    // is strictly worse for the loop: with no selected rows the branch counters stop
    // moving and every later iteration sees the same empty output. It was only visible
    // by reading "retained 0" out of stderr_tail, so the optimizer kept treating it as an
    // ordinary precision problem. Count it and hand it over as a first-class signal.
    const selectedRows = countCsvDataRows(iterCsv);
    if (valMode) {
      // Pairwise: mistakes are looked up in the PAIR frame, not the image manifest —
      // its ids are what the val file and the solver's trace key on.
      const scored = await scoreInference(args, query, iterDir, valCorpusCsv, valFile, diffPath);
      // "ok" requires a trace to score; without one the agent must fix-first.
      const traceOk = existsSync(resolve(iterDir, `trace_${query}.json`));
      const status = (run.status === "crash" || broken) ? "crash" : (traceOk ? "ok" : "empty");
      if (broken) {
        console.warn(`[SemDB] [${query}] ${health.errorRows}/${health.rowsIn} rows failed `
          + `inside the per-row guard — treating this candidate as a crash, not a score.`);
      }
      return { ...scored, status, health, stage: run.stage ?? null, selectedRows,
               execMs: run.execMs ?? null,
               runLog: filterRunLog([run.stdout, run.stderr].filter(Boolean).join("\n")),
               stderrTail: (run.stderr || "").split("\n").slice(-40).join("\n") };
    }
    const scored = await scoreWithDiff(args, query, planObj, telePath, iterCsv, diffPath, csvPath, false);
    const status = (run.status === "crash" || broken) ? "crash" : (existsSync(iterCsv) ? "ok" : "empty");
    if (broken) {
      console.warn(`[SemDB] [${query}] ${health.errorRows}/${health.rowsIn} rows failed `
        + `inside the per-row guard — treating this candidate as a crash, not a score.`);
    }
    return { ...scored, status, health, stage: run.stage ?? null, selectedRows,
             execMs: run.execMs ?? null,
             runLog: filterRunLog([run.stdout, run.stderr].filter(Boolean).join("\n")),
             stderrTail: (run.stderr || "").split("\n").slice(-40).join("\n") };
  };

  // ONE operator library for both modalities: vadar/predefined.py holds the VISION
  // functions (over an ImagePatch) and the TEXT functions (over strings) together, so a
  // query that joins an image predicate against a text column reads a single file.
  const primitiveFiles = [resolve(__dirname, "vadar", "predefined.py")];
  const plannerVars = (planPath, previousPlanPath = "", optimizerActionPath = "",
                       plannerLint = "", referencePlanPath = "",
                       previousPlanVersion = null) => ({
    query_id: query,
    query_sql: sql,
    query_nl: nl || "(none)",
    modality: isImage ? "image" : "text",
    tables_doc: tableLines.join("\n"),
    planner_table_profile: plannerTableProfile,
    local_primitive_files: primitiveFiles.map((path) => `- ${path}`).join("\n"),
    trace_contract: agentTraceContract,
    plan_path: planPath,
    previous_plan_path: previousPlanPath,
    previous_plan_version: previousPlanVersion ?? "",
    required_plan_version: previousPlanVersion == null ? 1 : previousPlanVersion + 1,
    required_parent_plan_version: previousPlanVersion == null ? "null" : previousPlanVersion,
    optimizer_action_path: optimizerActionPath,
    planner_lint: plannerLint,
    plan_schema_path: resolve(__dirname, "contracts", "semantic-plan.schema.json"),
    memory_reference_plan_path: referencePlanPath,
    ...memoryVars("planner"),
  });
  const generatorVars = ({
    planPath,
    generationMode = "INITIAL",
    parentManifestPath = "",
    actionPath = "",
    helpersPath,
    solverPath,
    manifestPath,
  }) => ({
    query_id: query,
    generation_mode: generationMode,
    plan_path: planPath,
    plan_schema_path: resolve(__dirname, "contracts", "semantic-plan.schema.json"),
    parent_candidate_manifest_path: parentManifestPath,
    optimizer_action_path: actionPath,
    replan_action_path: generationMode === "REPLAN" ? actionPath : "",
    helpers_path: helpersPath,
    solve_path: solverPath,
    manifest_draft_path: manifestPath,
    tables_doc: tableLines.join("\n"),
    local_primitive_files: primitiveFiles.map((path) => `- ${path}`).join("\n"),
    semdb_dir: __dirname,
    runtime_args: isImage ? " --image-dir <dir> --clip-model <model>" : "",
    ...memoryVars("generator"),
  });

  // An exact memory match may warm-start iteration 0. The stored plan is passed as
  // a REFERENCE, not as replan context: the replan branch requires
  // plan_version = previous + 1, which would break the initial plan's
  // plan_version === 1 contract. `warm` is cleared if preflight later rejects the
  // seeded candidate, so a stale reference degrades to today's cold start.
  let warm = mem?.warmStart || null;

  // Lint findings owed to the NEXT planner call. The lint catches plan-owned defects the
  // optimizer cannot repair — the generator must implement the plan, and the optimizer can
  // only reword prompts — but re-planning immediately would mean two planner calls in one
  // iteration. Instead the findings ride along with the next replan, which is a planner
  // call the loop was going to make anyway, so every iteration issues exactly one.
  let pendingPlanLint = "";

  /** Lint a freshly written plan, report it, and queue it for the next planner call. */
  const carryPlanLint = (plan) => {
    const findings = lintImagePlan(plan);
    if (!findings.length) return;
    for (const finding of findings) {
      console.log(`[SemDB]   [${query}] plan lint: ${finding.split(".")[0]}.`);
    }
    pendingPlanLint = findings.map((f) => `- ${f}`).join("\n");
    console.log(`[SemDB]   [${query}] carried into the next replan's planner prompt `
      + `(advisory — this iteration proceeds with the plan as written).`);
  };

  /** Drain the queued lint so a finding is delivered once, not on every later replan. */
  const takePlanLint = () => {
    const text = pendingPlanLint;
    pendingPlanLint = "";
    return text;
  };

  const createInitialPlan = async ({ iterDir, planPath }) => {
    record("query_planner", await runPhase(
      queryPlannerConfig,
      plannerVars(planPath, "", "", "", warm?.planPath || ""),
      iterDir,
      args,
    ));
    const plan = await readAndValidatePlan(planPath, { queryId: query });
    if (plan.plan_version !== 1 || plan.parent_plan_version !== null) {
      throw new Error(
        `[SemDB] [${query}] initial plan must use plan_version=1 and parent_plan_version=null`,
      );
    }
    carryPlanLint(plan);
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
      plannerVars(
        planPath,
        previousPlanPath,
        actionPath,
        takePlanLint(),
        "",
        previousPlan.plan_version,
      ),
      iterDir,
      args,
    ));
    const plan = await readAndValidatePlan(planPath, {
      queryId: query,
      previousPlan,
    });
    carryPlanLint(plan);
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
    } else if (!action && warm?.helpersPath && warm?.solverPath) {
      // Warm start: seed iteration 0 with the previously promoted implementation.
      // generationMode stays INITIAL — the generator must reconcile these files
      // with the plan it was given, exactly as it does for a patch.
      try {
        await copyFile(warm.helpersPath, helperPath);
        await copyFile(warm.solverPath, solverPath);
        console.log(`[SemDB] [${query}] warm start: seeded iter_0 from ${warm.candidateId || "a past candidate"}`);
      } catch (error) {
        console.warn(`[SemDB] [${query}] warm start failed, falling back to a cold start`
          + ` (non-fatal): ${error.message}`);
        warm = null;
      }
    }
    record("semantic_code_generator", await runPhase(
      semanticCodeGeneratorConfig,
      generatorVars({
        planPath,
        generationMode: action?.action || "INITIAL",
        parentManifestPath:
          action?.action === "PATCH_CODE"
            ? (parentCandidate?.manifestPath || "")
            : "",
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
    const importRootViolations = localImportRootViolations([
      readFileSync(helperPath, "utf-8"),
      readFileSync(solverPath, "utf-8"),
    ]);
    if (importRootViolations.length) {
      throw new Error(
        `[SemDB] [${query}] generated local imports are not runnable: `
        + importRootViolations.join("; "),
      );
    }
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
        candidate_helpers_path: candidate.helperPath,
        candidate_solver_path: candidate.solverPath,
        candidate_diff_path: resolve(candidate.iterDir, "diff.json"),
        history_manifest_paths: historyManifestPaths,
        optimizer_action_path: actionPath,
        optimizer_action_schema_path: resolve(
          __dirname,
          "contracts",
          "optimizer-action.schema.json",
        ),
        local_primitive_files: primitiveFiles.map((path) => `- ${path}`).join("\n"),
        remaining_iteration_budget: remainingIterationBudget,
        remaining_replan_budget: remainingReplanBudget,
        ...memoryVars("optimizer"),
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
  const executePgoCandidate = async (candidate, context = {}) => {
    const withTraceSummary = async (run) => {
      const tracePath = resolve(candidate.iterDir, `trace_${query}.json`);
      const trace = existsSync(tracePath) ? await readJSON(tracePath) : null;
      return { ...run, traceSummary: summarizeTraceArtifact(trace) };
    };
    const run = await runSolver(
      candidate.iterDir,
      candidate.solverPath,
      candidate.csvPath,
      valMode ? valIdsPath : null,
      candidate.helperPath,
    );
    // A warm-started iteration 0 that will not even compile means the seeded
    // implementation no longer fits this corpus. Spend one extra generator call to
    // redo it cold rather than burning the whole iteration budget repairing
    // someone else's code.
    if (warm && candidate.iteration === 0 && run.status === "crash" && run.stage === "compile") {
      console.warn(`[SemDB] [${query}] warm-started iter_0 failed preflight —`
        + ` regenerating cold and discarding the memory reference.`);
      warm = null;
      const cold = await generateCandidate({
        query: context.query,
        iteration: 0,
        iterDir: candidate.iterDir,
        plan: candidate.plan,
        planPath: candidate.planPath,
        action: null,
        parentCandidate: null,
      });
      Object.assign(candidate, cold);
      return withTraceSummary(await runSolver(
        candidate.iterDir,
        candidate.solverPath,
        candidate.csvPath,
        valMode ? valIdsPath : null,
        candidate.helperPath,
      ));
    }
    return withTraceSummary(run);
  };
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
          candidate_helpers_path: dryHelperPath,
          candidate_solver_path: drySolverPath,
          candidate_diff_path: resolve(iter0Dir, "diff.json"),
          history_manifest_paths: `- ${dryManifestPath}`,
          optimizer_action_path: resolve(iter1Dir, "optimizer_action.json"),
          optimizer_action_schema_path: resolve(
            __dirname,
            "contracts",
            "optimizer-action.schema.json",
          ),
          local_primitive_files: primitiveFiles.map((path) => `- ${path}`).join("\n"),
          remaining_iteration_budget: args.maxIterations,
          remaining_replan_budget: args.maxReplans,
          ...memoryVars("optimizer"),
        },
        iter1Dir,
        args,
      ));
      return null;
    }
    if (args.semanticPlanOnly) {
      const iter0Dir = resolve(runDir, "iter_0");
      await mkdir(iter0Dir, { recursive: true });
      const planPath = resolve(iter0Dir, "plan.json");
      const planned = await createInitialPlan({ iterDir: iter0Dir, planPath });
      await copyFile(planPath, resolve(runDir, "plan.json"));
      console.log(`[SemDB] [${query}] semantic plan complete; `
        + `--semantic-plan-only skips Generator, Optimizer, and execution.`);
      return {
        query,
        semantic_plan: planned.plan,
        semantic_plan_path: planPath,
      };
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
  const agentMessages = phases.reduce(
    (s, p) => s + (Number(p.profile?.agent_messages) || 0), 0);
  const agentCommands = phases.reduce(
    (s, p) => s + (Number(p.profile?.command_executions) || 0), 0);
  const agentCost = phases.reduce((s, p) => s + p.cost_usd, 0);
  const wallMs = Date.now() - wallStart;
  const timingBreakdown = directTimingBreakdown(
    wallMs, agentMs, validationSamplingLlmMs, codeExecutionRuns);
  // Written by supg_oracle at interpreter exit; null when the final solver never ran.
  const supgLedgerData = supgLedger ? await readJSON(supgLedger) : null;
  const report = {
    query, corpus: corpus.table, provider: args.agentProvider, operator: "direct",
    mode: "direct", wall_clock_ms: wallMs,
    agent_architecture: architecture,
    plan_versions: planVersions,
    replans: replansUsed,
    optimizer_actions: actionCounts,
    best_candidate_id: bestCandidate?.candidate_id ?? null,
    // Memory's only observability. `skills_used` in particular is the sole signal
    // that free discovery is being exercised rather than ignored.
    memory: {
      enabled: Boolean(mem),
      tier: mem?.tier ?? null,
      score: mem?.score ?? null,
      matched_l1: mem?.matchedL1 ?? null,
      matched_l0: mem?.matchedL0 ?? null,
      warm_start: Boolean(mem?.warmStart) && Boolean(warm),
      skills_available: (args.availableSkills || []).filter((s) => !s.role).length,
      skills_used: await readSkillUsage(args.out, query),
      injected_tokens: mem?.injectedTokens ?? null,
    },
    direct: { agent_stage_ms: agentMs, agent_calls: agentCalls,
              agent_model_messages: agentMessages,
              agent_command_executions: agentCommands,
              agent_cost_usd: Number(agentCost.toFixed(4)),
              timing_breakdown_ms: timingBreakdown,
              code_execution_runs: codeExecutionRuns },
    naive_llm_calls: planObj.plan.naive ?? null,
    // SUPG accounting. `oracle_calls` is what the FINAL full-corpus program actually
    // spent, read from the ledger the oracle wrote rather than from anything the
    // program reported about itself; `val_oracle_calls` is the separate, never-charged
    // draw the refinement loop scored against. Both are surfaced so a target hit on a
    // large validation spend cannot read as a target hit on the budget alone.
    supg: supg ? {
      dataset: supg.dataset,
      kind: supg.kind,
      graded_metric: supg.metric,
      target: supg.target,
      probability: supg.probability,
      mode: supg.mode,
      // Budget accounting is REPLAY-ONLY. In content mode the oracle is a VLM the
      // program may call per row, with no limit enforced, so reporting `k` there
      // would claim a constraint that never applied.
      oracle_budget: isProxy ? supg.oracle_limit : null,
      oracle_calls: isProxy ? (supgLedgerData?.distinct_ids ?? null) : null,
      oracle_exceeded: isProxy ? (supgLedgerData?.exceeded ?? null) : null,
      val_oracle_calls: val?.provenance?.oracle?.cost?.labels
        ?? (val?.labels ? Object.keys(val.labels).length : null),
    } : null,
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
      const shown = formatQueryMetric(m);
      console.log(`[SemDB]   METRICS            ${m.metric || "unknown"}: ${shown.label}`);
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
  if (base.valRate && !base.valPlanOnly && base.endpoint) {
    const configuredOracle = base.oracleModel || defaults.extraction.strongImageModel;
    base.oracleModel = await validateEndpointModel(
      base.endpoint, configuredOracle, base.apiKey);
  }
  setAgentProvider(base.agentProvider);
  base.runId = new Date().toISOString().replace(/[:.]/g, "-");
  const csvPath = base.telemetryCsv || resolve(base.out, "results.csv");

  // --- Skills and cross-run memory ------------------------------------------
  // The skill root exists even without --memory-dir: Generator discovers its procedure
  // there, and all roles may discover retrieval-selected learned skills. Planner and
  // Optimizer receive their canonical procedure directly in the system prompt so they do
  // not spend a command loading it again.
  base.memoryReady = false;
  base.skillRoot = null;
  base.availableSkills = [];
  if (base.enableAgentSkills !== false) {
    try {
      base.skillRoot = resolveSkillRoot(base);
      await initSkillRoot(base.skillRoot);
      await publishRoleSkills(base.skillRoot);
      if (base.memoryDir) {
        const init = await initMemory(base.memoryDir, {
          ...defaults.memory,
          groundTruthDir: base.groundTruthDir,
        });
        base.memoryReady = init.ready;
        if (init.ready) console.log(`[SemDB] ${await getMemorySummary(base.memoryDir)}`);
      } else {
        await lintAllSkills(base.skillRoot, { skillNamePrefix: defaults.memory.skillNamePrefix });
      }
      base.availableSkills = await listSkills(base.skillRoot);
      const learned = base.availableSkills.filter((s) => !s.role).length;
      console.log(`[SemDB] skills: ${base.availableSkills.length} discoverable`
        + ` (${learned} learned) at ${base.skillRoot}`);
    } catch (error) {
      console.warn(`[SemDB] skill root unavailable (non-fatal): ${error.message}`);
      base.skillRoot = null;
      base.availableSkills = [];
    }
  }
  if (base.noMemorySkills) base.availableSkills = base.availableSkills.filter((s) => s.role);

  // --query takes ONE id or a list (comma/space separated): --query q2,q4  OR  --query "q2 q4".
  const queries = base.query
    ? base.query.split(/[,\s]+/).map((s) => s.trim()).filter(Boolean)
    : await listQueries(base.queryDir);
  if (queries.length === 0) {
    console.error("[SemDB] no query given and no *.sql found in --query-dir.");
    process.exit(1);
  }
  console.log(`[SemDB] provider=${base.agentProvider} models: ${agentModelsLine(base)}`);
  console.log(`[SemDB] agent execution=${base.agentExecution}`
    + (base.agentExecution === "structured"
      ? ` (Planner/Optimizer tool-free, fallback=agent)`
      : " (full tool agents)"));
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

  // Classify every query against memory up front: the tier decides whether a
  // query warm-starts from a past candidate, so it has to be known before the
  // per-query pipeline begins. Failures here are never fatal — a query that
  // cannot be classified simply runs as it does today.
  base.memoryClassifications = {};
  if (base.memoryReady) {
    for (const p of runPlans) {
      try {
        base.memoryClassifications[p.query] = await classifyQuery(
          p, base, base.memoryDir, { ...defaults.memory, skillRoot: base.skillRoot },
        );
      } catch (error) {
        console.warn(`[SemDB] [${p.query}] memory classification failed (non-fatal): ${error.message}`);
      }
    }
    const tiers = { exact: 0, structural: 0, novel: 0 };
    for (const c of Object.values(base.memoryClassifications)) tiers[c.tier] += 1;
    console.log(`[SemDB] memory tiers: ${tiers.exact} exact, ${tiers.structural} structural,`
      + ` ${tiers.novel} novel`);
  }

  // The VADAR agents (Signature → API → Solver) write ONE end-to-end solver per query.
  // There is no Schema Designer and no extract/attrs/compile split — that pipeline was
  // removed; `solve_<q>.py` composes the `vadar/` operator library and answers the whole
  // query itself.
  const summary = [];
  console.log(
    `[SemDB] ${base.agentArchitecture}: agent solver per query over the vadar operator library.`,
  );
  for (const p of runPlans) {
    console.log(`\n[SemDB] ==================== ${p.query} ====================`);
    try {
      const outcome = await runQueryDirect(base, p, csvPath);
      if (outcome?.validation_capability?.class === "not_compilable") {
        summary.push({ q: p.query, notCompilable: outcome.validation_capability });
        continue;
      }
      const tele = await readJSON(resolve(base.out, `${base.benchmark}-${p.query}`, "telemetry.json"));
      if (tele?.metrics) summary.push({ q: p.query, ...tele.metrics });
    } catch (e) {
      console.error(`[SemDB] [${p.query}] failed: ${e.message}`);
      summary.push({ q: p.query, error: e.message });
    }
  }

  // 3b) Memory curation. Runs once, after every query, so the Manager can see
  //     cross-query recurrences (which is what L2 and L5 need). Entirely non-fatal:
  //     a failed curation costs future speed, never this run's results.
  if (base.memoryReady && !base.memoryReadonly && !base.dryRun) {
    console.log(`\n[SemDB] ==================== MEMORY CURATION ====================`);
    try {
      await curateMemory(base, runPlans);
    } catch (error) {
      console.error(`[SemDB] memory curation failed (non-fatal): ${error.message}`);
    }
  }

  // 4) Workload summary.
  console.log(`\n[SemDB] ==================== SUMMARY ====================`);
  // Each query is printed with the metric SemBench actually scores it by: a grouping
  // query has an ARI and no F1, an aggregate has a relative error. Assuming P/R/F1 for
  // all of them printed `undefined` for every non-retrieval query.
  const families = new Map();
  for (const s of summary) {
    if (s.notCompilable) {
      console.log(`[SemDB]   ${s.q.padEnd(6)} NOT_COMPILABLE — `
        + `${s.notCompilable.reason_code}: ${s.notCompilable.reason}`);
      continue;
    }
    if (s.error) {
      console.log(`[SemDB]   ${s.q.padEnd(6)} FAILED — ${s.error}`);
      continue;
    }
    const shown = formatQueryMetric(s);
    console.log(`[SemDB]   ${s.q.padEnd(6)} ${shown.label}`);
    if (shown.value == null) continue;
    if (!families.has(shown.key)) families.set(shown.key, []);
    families.get(shown.key).push(shown.value);
  }
  // Mean WITHIN a metric family only — averaging an ARI against an F1 is meaningless.
  for (const [key, values] of families) {
    const mean = (values.reduce((a, v) => a + v, 0) / values.length).toFixed(4);
    console.log(`[SemDB]   ${"MEAN".padEnd(6)} ${key}=${mean}  (${values.length} scored)`);
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
