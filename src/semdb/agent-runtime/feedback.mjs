import { assertSafeDataBoundary } from "./contracts.mjs";

export function assertSelectValidationPayload(payload, path = "<validation>") {
  if (!payload || payload.split !== "select") {
    throw new Error(
      `PGO optimizer requires a SELECT validation artifact at ${path}; `
      + `received split=${payload?.split ?? "<missing>"}`,
    );
  }
  if (!payload.labels || typeof payload.labels !== "object") {
    throw new Error(`SELECT validation artifact ${path} has no labels`);
  }
  return payload;
}

function finiteOrNull(value) {
  return value !== null && value !== undefined && Number.isFinite(Number(value))
    ? Number(value)
    : null;
}

function objectiveFrom(outcome = {}) {
  const explicit = outcome.objective || {};
  const fallbackF1 = finiteOrNull(outcome.f1);
  return {
    name: explicit.name || "f1",
    value: finiteOrNull(explicit.value) ?? fallbackF1,
    precision: finiteOrNull(outcome.metrics?.precision),
    recall: finiteOrNull(outcome.metrics?.recall),
    weighted: Boolean(
      explicit.weighted
      ?? outcome.metrics?.weighted
      ?? outcome.diff?.weighted,
    ),
    direction: explicit.direction === "minimize" ? "minimize" : "maximize",
  };
}

function boundedMistakes(scoreOutcome, cap) {
  const diff = scoreOutcome?.diff || {};
  if (Array.isArray(diff.mistakes)) return diff.mistakes.slice(0, cap);
  const rows = [];
  for (const row of diff.false_positives || []) {
    rows.push(
      typeof row === "object"
        ? { ...row, error_type: "false_positive" }
        : { id: String(row), error_type: "false_positive" },
    );
  }
  for (const row of diff.false_negatives || []) {
    rows.push(
      typeof row === "object"
        ? { ...row, error_type: "false_negative" }
        : { id: String(row), error_type: "false_negative" },
    );
  }
  return rows.slice(0, cap);
}

function safeHistory(history = []) {
  return history.map((entry) => ({
    iteration: entry.iteration ?? entry.iter ?? null,
    candidate_id: entry.candidate_id ?? null,
    action: entry.action ?? null,
    status: entry.status ?? null,
    objective: entry.objective
      ? {
          name: entry.objective.name ?? null,
          value: finiteOrNull(entry.objective.value),
          direction: entry.objective.direction ?? null,
        }
      : null,
    improved: Boolean(entry.improved),
  }));
}

export function buildIterationFeedback({
  query,
  candidate,
  runOutcome = {},
  scoreOutcome = {},
  history = [],
  dataBoundary = {
    source: "none",
    cert_accessed: false,
    full_ground_truth_accessed: false,
  },
  sampleCap = 15,
}) {
  assertSafeDataBoundary(dataBoundary);
  const diff = scoreOutcome.diff || {};
  const metrics = scoreOutcome.metrics || {};
  const feedback = {
    schema_version: "1.0",
    query_id: String(query?.query_id ?? query?.query ?? query?.id ?? query),
    candidate_id: String(
      candidate?.candidate_id
      ?? candidate?.manifest?.candidate_id
      ?? candidate?.id,
    ),
    iteration: Number(
      candidate?.iteration
      ?? candidate?.manifest?.iteration
      ?? 0,
    ),
    execution: {
      status: runOutcome.status ?? scoreOutcome.status ?? "empty",
      stage: runOutcome.stage ?? scoreOutcome.stage ?? null,
      preflight: runOutcome.preflight ?? scoreOutcome.preflight ?? null,
      stderr_tail:
        scoreOutcome.stderrTail
        ?? runOutcome.stderrTail
        ?? String(runOutcome.stderr || "").split("\n").slice(-40).join("\n"),
      runtime_ms: finiteOrNull(runOutcome.execMs ?? scoreOutcome.execMs),
    },
    objective: objectiveFrom(scoreOutcome),
    errors: {
      false_positive_total: Number(
        diff.fp_total ?? diff.false_positive_total ?? metrics.fp ?? 0,
      ),
      false_negative_total: Number(
        diff.fn_total ?? diff.false_negative_total ?? metrics.fn ?? 0,
      ),
      mistakes: boundedMistakes(scoreOutcome, sampleCap),
    },
    runtime_branches:
      scoreOutcome.runtimeBranches
      ?? runOutcome.runtimeBranches
      ?? scoreOutcome.runLog?.branches
      ?? runOutcome.runLog?.branches
      ?? (scoreOutcome.runLog
        ? { log: scoreOutcome.runLog }
        : (runOutcome.runLog ? { log: runOutcome.runLog } : {})),
    history: safeHistory(history),
    data_boundary: {
      source: dataBoundary.source,
      cert_accessed: false,
      full_ground_truth_accessed: false,
    },
  };
  return feedback;
}

export function hasMeasurableSelectSignal(feedback) {
  return feedback?.data_boundary?.source === "select_validation"
    && feedback.data_boundary.cert_accessed === false
    && feedback.data_boundary.full_ground_truth_accessed === false
    && (
      feedback.execution.status !== "ok"
      || finiteOrNull(feedback.objective?.value) !== null
    );
}
