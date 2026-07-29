import { mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

import { buildIterationFeedback, hasMeasurableSelectSignal } from "./feedback.mjs";
import { readAndValidateFeedback, writeJsonAtomic } from "./contracts.mjs";

function queryId(query) {
  return String(query?.query_id ?? query?.query ?? query?.id ?? query);
}

function candidateId(candidate) {
  return candidate?.candidate_id ?? candidate?.manifest?.candidate_id ?? candidate?.id;
}

function candidateIteration(candidate) {
  return Number(candidate?.iteration ?? candidate?.manifest?.iteration ?? 0);
}

function objective(outcome = {}) {
  const explicit = outcome.objective;
  // A typed objective is authoritative even when its value is null.  In
  // particular, query_metric_unavailable must not fall back to a legacy fidelity
  // F1/accuracy field.
  if (explicit && typeof explicit === "object") {
    return {
      name: explicit.name || "objective",
      value:
        explicit.value !== null
        && explicit.value !== undefined
        && explicit.value !== ""
        && Number.isFinite(Number(explicit.value))
          ? Number(explicit.value)
          : null,
      direction: explicit.direction === "minimize" ? "minimize" : "maximize",
    };
  }
  if (Number.isFinite(Number(outcome.f1))) {
    return { name: "f1", value: Number(outcome.f1), direction: "maximize" };
  }
  return {
    name: explicit?.name || "objective",
    value: null,
    direction: explicit?.direction === "minimize" ? "minimize" : "maximize",
  };
}

function defaultImprovement(previous, next) {
  const previousOk = previous?.status === "ok";
  const nextOk = next?.status === "ok";
  if (previousOk && !nextOk) return false;
  if (!previousOk && nextOk) return true;
  if (!previousOk || !nextOk) return false;
  const before = objective(previous);
  const after = objective(next);
  if (
    before.value == null
    || after.value == null
    || before.name !== after.name
    || before.direction !== after.direction
  ) return before.value == null && after.value != null;
  return before.direction === "minimize"
    ? after.value < before.value
    : after.value > before.value;
}

export function routeOptimizerAction({
  action,
  currentCandidateId,
  replansUsed = 0,
  maxReplans = 0,
}) {
  if (!action || !["PATCH_CODE", "REPLAN", "STOP"].includes(action.action)) {
    throw new Error(`Unknown optimizer action: ${action?.action ?? "<missing>"}`);
  }
  if (action.candidate_id !== currentCandidateId) {
    throw new Error(
      `Optimizer action candidate id mismatch: expected ${currentCandidateId}, got ${action.candidate_id}`,
    );
  }
  if (action.action === "STOP") return { route: "stop", reason: "optimizer_stop" };
  if (action.action === "PATCH_CODE") return { route: "generator" };
  if (replansUsed >= maxReplans) {
    return { route: "stop", reason: "replan_budget_exhausted" };
  }
  return { route: "planner_then_generator" };
}

function historyEntry(candidate, outcome, action, improved) {
  return {
    iteration: candidateIteration(candidate),
    candidate_id: candidateId(candidate),
    action: action?.action ?? "INITIAL",
    status: outcome.status,
    objective: objective(outcome),
    improved,
    manifest_path: candidate.manifestPath ?? null,
    plan_version:
      candidate.plan?.plan_version
      ?? candidate.manifest?.plan_version
      ?? null,
  };
}

export async function runPgoLoop({
  args,
  query,
  runDir,
  createInitialPlan,
  replan,
  generateCandidate,
  optimize,
  executeCandidate,
  scoreCandidate,
  promoteCandidate,
  isImprovement = defaultImprovement,
  buildFeedback = buildIterationFeedback,
  hasValidationSignal,
}) {
  const id = queryId(query);
  const maxIterations = args.noRefine
    ? 0
    : Math.max(0, Number(args.maxIterations ?? 0));
  const maxReplans = Math.max(0, Number(args.maxReplans ?? 0));
  const actionCounts = { PATCH_CODE: 0, REPLAN: 0, STOP: 0 };
  let replansUsed = 0;

  const iter0Dir = resolve(runDir, "iter_0");
  await mkdir(iter0Dir, { recursive: true });
  let planResult = await createInitialPlan({
    query,
    iteration: 0,
    iterDir: iter0Dir,
    planPath: resolve(iter0Dir, "plan.json"),
  });
  let plan = planResult?.plan ?? planResult;
  let planPath = planResult?.planPath ?? resolve(iter0Dir, "plan.json");
  if (!existsSync(planPath) && plan && typeof plan === "object") {
    await writeJsonAtomic(planPath, plan);
  }

  let bestCandidate = await generateCandidate({
    query,
    iteration: 0,
    iterDir: iter0Dir,
    plan,
    planPath,
    action: null,
    parentCandidate: null,
  });
  let bestRun = await executeCandidate(bestCandidate, {
    query,
    iteration: 0,
    iterDir: iter0Dir,
  });
  let bestOutcome = await scoreCandidate(bestCandidate, bestRun, {
    query,
    iteration: 0,
    iterDir: iter0Dir,
  });
  const history = [historyEntry(bestCandidate, bestOutcome, null, true)];

  const boundarySource = hasValidationSignal === true
    ? "select_validation"
    : (hasValidationSignal === false ? "none" : null);
  const feedbackFor = async (candidate, runOutcome, scoreOutcome) => {
    const feedback = buildFeedback({
      query: { query_id: id },
      candidate,
      runOutcome,
      scoreOutcome,
      history,
      sampleCap: args.refineSampleCap,
      dataBoundary: {
        source: boundarySource
          ?? (objective(scoreOutcome).value == null ? "none" : "select_validation"),
        cert_accessed: false,
        full_ground_truth_accessed: false,
      },
    });
    const feedbackPath = resolve(candidate.iterDir, "iteration_feedback.json");
    await writeJsonAtomic(feedbackPath, feedback);
    const validated = await readAndValidateFeedback(feedbackPath, {
      queryId: id,
      candidateId: candidateId(candidate),
    });
    return { feedback: validated, feedbackPath };
  };

  let latestFeedback = await feedbackFor(bestCandidate, bestRun, bestOutcome);
  const canOptimize = hasValidationSignal === false
    ? false
    : hasMeasurableSelectSignal(latestFeedback.feedback);

  if (canOptimize) {
    for (let iteration = 1; iteration <= maxIterations; iteration++) {
      const iterDir = resolve(runDir, `iter_${iteration}`);
      await mkdir(iterDir, { recursive: true });
      const actionResult = await optimize({
        query,
        iteration,
        plan,
        planPath,
        candidate: bestCandidate,
        feedback: latestFeedback.feedback,
        feedbackPath: latestFeedback.feedbackPath,
        history,
        remainingIterationBudget: maxIterations - iteration + 1,
        remainingReplanBudget: maxReplans - replansUsed,
      });
      const action = actionResult?.actionObject ?? actionResult;
      const actionPath = actionResult?.actionPath
        ?? resolve(iterDir, "optimizer_action.json");
      if (!existsSync(actionPath)) await writeJsonAtomic(actionPath, action);
      actionCounts[action.action]++;
      const route = routeOptimizerAction({
        action,
        currentCandidateId: candidateId(bestCandidate),
        replansUsed,
        maxReplans,
      });
      if (route.route === "stop") break;

      if (route.route === "planner_then_generator") {
        const previousPlan = plan;
        const previousPlanPath = planPath;
        planResult = await replan({
          query,
          iteration,
          iterDir,
          previousPlan,
          previousPlanPath,
          action,
          actionPath,
          planPath: resolve(iterDir, "plan.json"),
        });
        plan = planResult?.plan ?? planResult;
        planPath = planResult?.planPath ?? resolve(iterDir, "plan.json");
        if (!existsSync(planPath) && plan && typeof plan === "object") {
          await writeJsonAtomic(planPath, plan);
        }
        replansUsed++;
        // A failed replan must not discard a runnable historical candidate. Keep the
        // rejected plan artifact for diagnosis, restore the last generatable plan,
        // stop refinement, and let promotion freeze the best measured candidate.
        if (plan?.compilability?.class === "not_compilable") {
          plan = previousPlan;
          planPath = previousPlanPath;
          break;
        }
      } else {
        planPath = resolve(iterDir, "plan.json");
        await writeJsonAtomic(planPath, plan);
      }

      const trial = await generateCandidate({
        query,
        iteration,
        iterDir,
        plan,
        planPath,
        action,
        actionPath,
        parentCandidate: bestCandidate,
      });
      const trialRun = await executeCandidate(trial, {
        query,
        iteration,
        iterDir,
      });
      const trialOutcome = await scoreCandidate(trial, trialRun, {
        query,
        iteration,
        iterDir,
      });
      const improved = isImprovement(bestOutcome, trialOutcome);
      history.push(historyEntry(trial, trialOutcome, action, improved));
      const trialFeedback = await feedbackFor(trial, trialRun, trialOutcome);
      if (improved) {
        bestCandidate = trial;
        bestRun = trialRun;
        bestOutcome = trialOutcome;
        plan = trial.plan ?? plan;
        planPath = trial.planPath ?? planPath;
        latestFeedback = trialFeedback;
      } else {
        latestFeedback = await feedbackFor(bestCandidate, bestRun, bestOutcome);
      }
    }
  }

  await promoteCandidate(bestCandidate, {
    query,
    runDir,
    outcome: bestOutcome,
    history,
  });
  return {
    bestCandidate,
    bestOutcome,
    bestIter: candidateIteration(bestCandidate),
    bestF1: bestOutcome.f1 ?? null,
    bestObjective: objective(bestOutcome),
    history,
    replansUsed,
    actionCounts,
    planVersions: new Set(
      history
        .map((entry) => entry.plan_version)
        .filter((version) => version != null),
    ).size || 1,
  };
}
