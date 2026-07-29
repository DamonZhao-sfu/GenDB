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
  const details = explicit.details && typeof explicit.details === "object"
    ? explicit.details
    : {};
  const fidelity = outcome.metrics?.quality || outcome.metrics?.unweighted || {};
  const fallbackF1 = finiteOrNull(outcome.f1);
  const name = explicit.name || "f1";
  const f1Like = ["f1", "macro_f1", "predicate_fidelity_f1"].includes(name);
  const hasTypedObjective = outcome.objective
    && typeof outcome.objective === "object";
  return {
    name,
    value: hasTypedObjective ? finiteOrNull(explicit.value) : fallbackF1,
    precision:
      finiteOrNull(details.precision)
      ?? (f1Like
        ? (finiteOrNull(outcome.metrics?.precision)
          ?? finiteOrNull(fidelity.precision))
        : null),
    recall:
      finiteOrNull(details.recall)
      ?? (f1Like
        ? (finiteOrNull(outcome.metrics?.recall)
          ?? finiteOrNull(fidelity.recall))
        : null),
    weighted: Boolean(
      explicit.weighted
      ?? outcome.metrics?.weighted
      ?? outcome.diff?.weighted,
    ),
    direction: explicit.direction === "minimize" ? "minimize" : "maximize",
    scope: name === "query_metric_unavailable"
      ? "unavailable"
      : (name === "predicate_fidelity_f1"
          ? "operator_fidelity"
          : "query_metric"),
    details,
  };
}

function operatorFidelityFrom(outcome = {}) {
  const metrics = outcome.metrics || {};
  const quality = metrics.quality || metrics.unweighted || {};
  return {
    accuracy:
      finiteOrNull(quality.accuracy)
      ?? finiteOrNull(metrics.accuracy),
    precision:
      finiteOrNull(quality.precision)
      ?? finiteOrNull(metrics.precision),
    recall:
      finiteOrNull(quality.recall)
      ?? finiteOrNull(metrics.recall),
    f1:
      finiteOrNull(quality.f1)
      ?? finiteOrNull(metrics.f1),
    n: finiteOrNull(metrics.n),
    correct: finiteOrNull(metrics.correct),
    weighted: Boolean(metrics.weighted),
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
  const selectedObjective = objectiveFrom(scoreOutcome);
  const fidelity = operatorFidelityFrom(scoreOutcome);
  const binaryObjective = ["f1", "predicate_fidelity_f1"].includes(
    selectedObjective.name,
  );
  const effectiveStatus = runOutcome.status ?? scoreOutcome.status ?? "empty";
  const errorKind = effectiveStatus !== "ok"
    ? "execution"
    : (binaryObjective
        ? "classification"
        : (["adjusted_rand_index", "macro_f1"].includes(selectedObjective.name)
            ? "label_mismatch"
            : "metric_specific"));
  const fp = finiteOrNull(
    diff.fp_total
    ?? diff.false_positive_total
    ?? metrics.fp
    ?? metrics.quality?.fp
    ?? metrics.unweighted?.fp,
  );
  const fn = finiteOrNull(
    diff.fn_total
    ?? diff.false_negative_total
    ?? metrics.fn
    ?? metrics.quality?.fn
    ?? metrics.unweighted?.fn,
  );
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
    objective: selectedObjective,
    operator_fidelity: fidelity,
    errors: {
      kind: errorKind,
      false_positive_total: binaryObjective ? fp : null,
      false_negative_total: binaryObjective ? fn : null,
      mismatch_total: finiteOrNull(
        diff.n_mistakes ?? diff.mismatch_total,
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
