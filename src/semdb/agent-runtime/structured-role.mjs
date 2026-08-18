import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";

function fenced(label, content) {
  return `<${label}>\n${String(content || "").trim()}\n</${label}>`;
}

function numberedSource(content) {
  return String(content || "")
    .split("\n")
    .map((line, index) => `L${index + 1} | ${line}`)
    .join("\n");
}

function bulletPaths(value) {
  return String(value || "")
    .split("\n")
    .map((line) => line.trim().replace(/^-\s*/, ""))
    .filter((line) => line && line !== "(none)" && existsSync(line));
}

async function optionalFile(path) {
  return path && existsSync(path) ? readFile(path, "utf8") : "";
}

/** Mechanically extract public top-level signatures and their implementation docstrings
 * without importing the Python module (which may pull GPU dependencies or mutate state).
 * Stop before MODULES_SIGNATURES so the catalog is derived from executable definitions,
 * not the hand-written documentation string that follows them. */
export function extractPrimitiveSignatures(source, sourcePath = "predefined.py") {
  const marker = source.indexOf("MODULES_SIGNATURES =");
  const executable = marker >= 0 ? source.slice(0, marker) : source;
  const lines = executable.split("\n");
  const entries = [];
  for (let i = 0; i < lines.length; i++) {
    const match = lines[i].match(/^def\s+([A-Za-z_]\w*)\s*\(/);
    if (!match || match[1].startsWith("_")) continue;
    const signatureLines = [lines[i].trim()];
    while (!signatureLines.at(-1).trimEnd().endsWith(":") && i + 1 < lines.length) {
      signatureLines.push(lines[++i].trim());
    }
    let cursor = i + 1;
    while (cursor < lines.length && !lines[cursor].trim()) cursor++;
    const first = lines[cursor]?.trim() || "";
    const quote = first.startsWith('"""') ? '"""' : (first.startsWith("'''") ? "'''" : null);
    const doc = [];
    if (quote) {
      let part = first.slice(3);
      if (part.includes(quote)) {
        doc.push(part.slice(0, part.indexOf(quote)));
      } else {
        if (part) doc.push(part);
        cursor++;
        while (cursor < lines.length) {
          part = lines[cursor].trim();
          if (part.includes(quote)) {
            doc.push(part.slice(0, part.indexOf(quote)));
            break;
          }
          doc.push(part);
          cursor++;
        }
      }
    }
    const signature = signatureLines.join(" ").replace(/\s+/g, " ");
    const docstring = doc.join("\n").trim();
    entries.push(docstring ? `${signature}\n${docstring}` : signature);
  }
  if (!entries.length) throw new Error(`${sourcePath} has no public top-level functions`);
  return entries.join("\n\n");
}

/** Build a compact, source-hashed catalog from the authoritative primitive files. */
export async function buildPrimitiveCatalog(paths) {
  const sections = [];
  for (const path of paths) {
    const source = await readFile(path, "utf8");
    const hash = createHash("sha256").update(source).digest("hex");
    const signatures = extractPrimitiveSignatures(source, path);
    sections.push([
      `Source: ${path}`,
      `SHA256: ${hash}`,
      "The following catalog was mechanically extracted from public Python signatures and docstrings:",
      signatures,
    ].join("\n"));
  }
  return sections.join("\n\n");
}

function memoryContext(vars, options = {}) {
  return [
    vars.memory_pre_injection && fenced("prior_knowledge", vars.memory_pre_injection),
    options.includeCatalog !== false
      && vars.memory_catalog
      && fenced("memory_catalog", vars.memory_catalog),
    vars.memory_inline_skills && fenced("memory_skills", vars.memory_inline_skills),
  ].filter(Boolean).join("\n\n");
}

async function plannerInput(vars, primitiveCatalog, options = {}) {
  const toolFree = options.toolFree !== false;
  const previousPlan = await optionalFile(vars.previous_plan_path);
  const optimizerAction = await optionalFile(vars.optimizer_action_path);
  const referencePlan = await optionalFile(vars.memory_reference_plan_path);
  const replan = previousPlan ? [
    "## Replan lineage",
    `Previous version: ${vars.previous_plan_version}`,
    `Required plan_version: ${vars.required_plan_version}`,
    `Required parent_plan_version: ${vars.required_parent_plan_version}`,
    fenced("previous_plan_json", previousPlan),
    fenced("optimizer_action_json", optimizerAction),
  ].join("\n\n") : [
    "## Initial lineage",
    'Use exactly "plan_version": 1 and "parent_plan_version": null.',
  ].join("\n");
  return [
    `# ${toolFree ? "Structured" : "Full-agent"} semantic planning task: ${vars.query_id}`,
    toolFree
      ? "Return the complete plan JSON object. Do not request tools and do not describe it."
      : "This bundle is the authoritative snapshot for the planning task. Read it once, then use tools only when a concrete missing or inconsistent fact requires additional inspection.",
    "The root must contain exactly these contract fields: schema_version, query_id, "
      + "plan_version, parent_plan_version, modality, compilability, semantic_sites, "
      + "helper_dag, relational_plan, trace_contract, runtime_contract, "
      + "validation_contract, assumptions, invariants.",
    `Query id: ${vars.query_id}`,
    `Modality: ${vars.modality}`,
    fenced("query_sql", vars.query_sql),
    fenced("natural_language", vars.query_nl),
    fenced("tables_and_runtime_inputs", vars.tables_doc),
    vars.planner_table_profile && [
      "## Deterministic runtime-table profile",
      "These compact facts were aggregated across every runtime-input CSV row. They contain no validation labels, CERT, or ground truth. Treat a zero join overlap as authoritative only when both full_scan_complete and value_sets_complete are true; otherwise the distinct-memory cap makes the overlap inconclusive.",
      fenced("runtime_table_profile_json", JSON.stringify(vars.planner_table_profile, null, 2)),
    ].join("\n\n"),
    fenced("trace_contract", vars.trace_contract),
    fenced("authoritative_primitive_catalog", primitiveCatalog),
    replan,
    referencePlan && [
      "## Advisory reference plan from a past run",
      "This is not a specification. Keep initial lineage at 1/null.",
      fenced("reference_plan_json", referencePlan),
    ].join("\n\n"),
    vars.planner_lint && fenced("rejected_planning_attempt", vars.planner_lint),
    memoryContext(vars, { includeCatalog: toolFree }),
  ].filter(Boolean).join("\n\n");
}

async function optimizerInput(vars, primitiveCatalog, options = {}) {
  const toolFree = options.toolFree !== false;
  const plan = await optionalFile(vars.plan_path);
  const manifest = await optionalFile(vars.candidate_manifest_path);
  const feedback = await optionalFile(vars.iteration_feedback_path);
  const helpers = await optionalFile(vars.candidate_helpers_path);
  const solver = await optionalFile(vars.candidate_solver_path);
  const candidateDiff = await optionalFile(vars.candidate_diff_path);
  let candidateId = "<candidate_id>";
  try { candidateId = JSON.parse(manifest).candidate_id || candidateId; } catch { /* validated later */ }
  const histories = [];
  for (const path of bulletPaths(vars.history_manifest_paths)) {
    histories.push(`Path: ${path}\n${fenced("prior_manifest", await readFile(path, "utf8"))}`);
  }
  return [
    `# ${toolFree ? "Structured" : "Full-agent"} semantic optimization task: ${vars.query_id}`,
    toolFree
      ? "Return exactly one optimizer-action JSON object. Do not request tools or edit files."
      : "This bundle is the authoritative snapshot for the optimization task. Read it once, then use tools only when a concrete missing or inconsistent fact requires additional inspection. Do not edit candidate or repository artifacts.",
    "Use this exact root shape and nested field names (omit replan_reason unless action is REPLAN):",
    `{\"schema_version\":\"1.0\",\"query_id\":${JSON.stringify(vars.query_id)},`
      + `\"candidate_id\":${JSON.stringify(candidateId)},\"action\":\"PATCH_CODE | REPLAN | STOP\",`
      + `\"diagnosis\":{\"category\":\"...\",\"summary\":\"...\",\"evidence\":[\"...\"]},`
      + `\"targets\":[{\"artifact\":\"plan | helpers | solver\",\"symbol\":null,\"intent\":\"...\"}],`
      + `\"preserve\":[\"...\"],\"expected_effect\":{\"primary_metric\":\"...\",`
      + `\"direction\":\"increase | decrease | unchanged | unknown\",\"risk\":\"...\"},`
      + `\"replan_reason\":\"required only for REPLAN\"}`,
    `Remaining candidate iterations: ${vars.remaining_iteration_budget}`,
    `Remaining replans: ${vars.remaining_replan_budget}`,
    fenced("current_plan_json", plan),
    fenced("candidate_manifest_json", manifest),
    fenced("iteration_feedback_json", feedback),
    "## Candidate implementation evidence",
    "The exact current candidate sources are inline below. Before choosing PATCH_CODE, inspect them and cite at least one concrete line per targeted artifact in diagnosis.evidence using `helpers:L<number>` or `solver:L<number>`. A branch named `*_pass` does not imply a final true trace; the trace_summary is authoritative.",
    helpers
      ? fenced("current_helpers_source", numberedSource(helpers))
      : "<current_helpers_source>(unavailable)</current_helpers_source>",
    solver
      ? fenced("current_solver_source", numberedSource(solver))
      : "<current_solver_source>(unavailable)</current_solver_source>",
    candidateDiff && fenced("candidate_validation_diff_json", candidateDiff),
    histories.length ? `## Prior candidate manifests\n${histories.join("\n\n")}` : "",
    fenced("authoritative_primitive_catalog", primitiveCatalog),
    memoryContext(vars, { includeCatalog: toolFree }),
  ].filter(Boolean).join("\n\n");
}

const ROLE_SPECS = {
  query_planner: {
    schemaName: "semdb_semantic_plan",
    schemaPathVar: "plan_schema_path",
    outputPathVar: "plan_path",
    maxOutputTokens: 24_000,
    retryMaxOutputTokens: 32_000,
    effort: "medium",
    buildInput: plannerInput,
  },
  semantic_optimizer: {
    schemaName: "semdb_optimizer_action",
    schemaPathVar: "optimizer_action_schema_path",
    outputPathVar: "optimizer_action_path",
    maxOutputTokens: 12_000,
    retryMaxOutputTokens: 16_000,
    effort: "medium",
    buildInput: optimizerInput,
  },
};

export function supportsStructuredRole(configKey) {
  return Boolean(ROLE_SPECS[configKey]);
}

/** Prepare one tool-free role request. The role skill is the only procedure body; the
 * large legacy system prompt and filesystem skill protocol are intentionally omitted. */
export async function prepareStructuredRole(agentConfig, vars, overrides = {}) {
  const spec = ROLE_SPECS[agentConfig.configKey];
  if (!spec) throw new Error(`No structured role spec for ${agentConfig.configKey}`);
  const schemaPath = vars[spec.schemaPathVar];
  const outputPath = vars[spec.outputPathVar];
  if (!schemaPath || !outputPath) {
    throw new Error(`Structured ${agentConfig.configKey} is missing schema/output paths`);
  }
  const [procedure, schemaText] = await Promise.all([
    readFile(agentConfig.skillPath, "utf8"),
    readFile(schemaPath, "utf8"),
  ]);
  const primitiveCatalog = await buildPrimitiveCatalog(bulletPaths(vars.local_primitive_files));
  const adapter = [
    "## Structured execution adapter",
    "All authoritative inputs are inline in the task. You have no tools.",
    "Return the JSON object only; the orchestrator validates and writes it.",
    "Do not emit Markdown fences, commentary, a plan-of-work, or file operations.",
  ].join("\n");
  return {
    systemPrompt: `${procedure.trim()}\n\n${adapter}`,
    userPrompt: await spec.buildInput(vars, primitiveCatalog),
    schema: JSON.parse(schemaText),
    schemaName: spec.schemaName,
    outputPath,
    maxOutputTokens: overrides.maxOutputTokens || spec.maxOutputTokens,
    retryMaxOutputTokens: overrides.maxOutputTokens
      || spec.retryMaxOutputTokens
      || spec.maxOutputTokens,
    effortLevel: overrides.effortLevel || spec.effort,
  };
}

/** Prepare the same authoritative evidence as structured mode as one immutable-on-entry
 * context artifact for a full tool agent. The canonical role skill is already the system
 * procedure, so the agent must not spend a command loading it again. */
export async function prepareAgentRole(agentConfig, vars) {
  const spec = ROLE_SPECS[agentConfig.configKey];
  if (!spec) throw new Error(`No optimized agent role spec for ${agentConfig.configKey}`);
  const schemaPath = vars[spec.schemaPathVar];
  const outputPath = vars[spec.outputPathVar];
  if (!schemaPath || !outputPath) {
    throw new Error(`Agent ${agentConfig.configKey} is missing schema/output paths`);
  }
  const [procedure, schemaText] = await Promise.all([
    readFile(agentConfig.skillPath, "utf8"),
    readFile(schemaPath, "utf8"),
  ]);
  const primitiveCatalog = await buildPrimitiveCatalog(bulletPaths(vars.local_primitive_files));
  const evidence = await spec.buildInput(vars, primitiveCatalog, { toolFree: false });
  const contextText = [
    "# SemDB full-agent context bundle",
    "Generated by the orchestrator from the current runtime artifacts. The bundle contains no CERT or unauthorized final-ground-truth data. Its inline artifact contents are authoritative for this call; inspect their original paths only when this snapshot identifies a concrete omission or inconsistency.",
    evidence,
    "## Output artifact",
    `Schema path: ${schemaPath}`,
    fenced("output_json_schema", schemaText),
    `Write exactly one JSON artifact to: ${outputPath}`,
  ].join("\n\n");
  const adapter = [
    "## Full-agent execution adapter",
    "The role procedure above is already loaded. Do not read its SKILL.md again.",
    "Start with the single context bundle named in the task. It already contains the required plan, evidence, source, primitive catalog, memory injection, and JSON schema.",
    "You retain filesystem and shell tools for genuine missing evidence or diagnostics; avoid Glob/Grep/Read calls that merely reproduce bundle content.",
    "Write only the requested role output. Do not edit candidate or repository artifacts.",
  ].join("\n");
  return {
    systemPrompt: `${procedure.trim()}\n\n${adapter}`,
    contextText,
    contextSha256: createHash("sha256").update(contextText).digest("hex"),
    outputPath,
    schemaPath,
  };
}

/** Evidence guard for tool-free Optimizer actions. Schema validity only proves that the
 * JSON is shaped correctly; PATCH_CODE additionally has to demonstrate that the inline
 * implementation was inspected. */
export async function validateStructuredOptimizerEvidence(action, vars) {
  if (action?.action === "REPLAN" && Number(vars.remaining_replan_budget) <= 0) {
    throw new Error("REPLAN was selected with no remaining replan budget");
  }
  if (action?.action !== "PATCH_CODE") return action;

  const sources = {
    helpers: await optionalFile(vars.candidate_helpers_path),
    solver: await optionalFile(vars.candidate_solver_path),
  };
  const evidence = (action.diagnosis?.evidence || []).join("\n");
  for (const target of action.targets || []) {
    if (!target.symbol) {
      throw new Error(`PATCH_CODE target ${target.artifact} must name a concrete symbol or code location`);
    }
    const source = sources[target.artifact] || "";
    if (!source) {
      throw new Error(`PATCH_CODE target ${target.artifact} has no inline candidate source`);
    }
    const lineCount = source.split("\n").length;
    const citation = new RegExp(`\\b${target.artifact}:L(\\d+)\\b`, "gi");
    const citedLines = [...evidence.matchAll(citation)].map((match) => Number(match[1]));
    if (!citedLines.some((line) => line >= 1 && line <= lineCount)) {
      throw new Error(
        `PATCH_CODE target ${target.artifact} must cite a valid ${target.artifact}:L<number> `
        + `line in diagnosis.evidence (source has ${lineCount} lines)`,
      );
    }
  }

  const feedbackText = await optionalFile(vars.iteration_feedback_path);
  try {
    const feedback = JSON.parse(feedbackText);
    const trueCount = feedback?.trace_summary?.true_count;
    const claimsTrueTrace = /(?:trace|entr(?:y|ies)|rows?).{0,50}(?:as|are|=|:)\s*["']?true\b/is
      .test([action.diagnosis?.summary, evidence].filter(Boolean).join("\n"));
    if (trueCount === 0 && claimsTrueTrace) {
      throw new Error(
        "Optimizer evidence claims true trace entries, but trace_summary.true_count is 0",
      );
    }
  } catch (error) {
    if (error instanceof SyntaxError) {
      throw new Error(`Cannot validate Optimizer trace evidence: ${error.message}`);
    }
    throw error;
  }
  return action;
}
