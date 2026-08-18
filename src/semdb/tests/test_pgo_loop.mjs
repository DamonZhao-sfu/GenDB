import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import { runPgoLoop } from "../agent-runtime/pgo-loop.mjs";

function plan(version = 1) {
  return { query_id: "q", plan_version: version };
}

async function scenario({
  actions,
  scores,
  hasValidationSignal = true,
  maxReplans = 1,
  replanCompilability = null,
}) {
  const runDir = await mkdtemp(resolve(tmpdir(), "semdb-pgo-"));
  const calls = {
    planner: 0,
    replan: 0,
    generator: [],
    optimizer: 0,
    promoted: null,
  };
  let actionIndex = 0;
  const result = await runPgoLoop({
    args: {
      maxIterations: actions.length,
      maxReplans,
      refineSampleCap: 2,
    },
    query: { query_id: "q" },
    runDir,
    hasValidationSignal,
    createInitialPlan: async () => {
      calls.planner++;
      return plan();
    },
    replan: async ({ previousPlan }) => {
      calls.replan++;
      const next = plan(previousPlan.plan_version + 1);
      if (replanCompilability) {
        next.compilability = { class: replanCompilability };
      }
      return next;
    },
    generateCandidate: async ({
      iteration, iterDir, plan: currentPlan, parentCandidate,
    }) => {
      calls.generator.push({
        iteration,
        parent: parentCandidate?.candidate_id ?? null,
        planVersion: currentPlan.plan_version,
      });
      return {
        candidate_id: `q-iter-${iteration}`,
        iteration,
        iterDir,
        plan: currentPlan,
        manifest: {
          candidate_id: `q-iter-${iteration}`,
          iteration,
          plan_version: currentPlan.plan_version,
        },
      };
    },
    optimize: async ({ candidate }) => {
      calls.optimizer++;
      return {
        schema_version: "1.0",
        query_id: "q",
        candidate_id: candidate.candidate_id,
        action: actions[actionIndex++],
      };
    },
    executeCandidate: async (candidate) => (
      scores[candidate.iteration]?.run
      ?? { status: "ok", execMs: 1 }
    ),
    scoreCandidate: async (candidate, run) => {
      const score = scores[candidate.iteration] || {};
      return {
        status: run.status,
        stage: run.stage,
        f1: score.f1 ?? null,
        objective: score.objective || {
          name: "f1",
          value: score.f1 ?? null,
          direction: "maximize",
        },
        metrics: {
          precision: score.f1 ?? null,
          recall: score.f1 ?? null,
        },
        diff: { fp_total: 0, fn_total: 0, mistakes: [] },
      };
    },
    promoteCandidate: async (candidate) => {
      calls.promoted = candidate.candidate_id;
    },
  });
  return { calls, result };
}

const replanned = await scenario({
  actions: ["REPLAN", "PATCH_CODE"],
  scores: [{ f1: 0.4 }, { f1: 0.6 }, { f1: 0.5 }],
});
assert.equal(replanned.calls.planner, 1);
assert.equal(replanned.calls.replan, 1);
assert.equal(replanned.calls.generator.length, 3);
assert.equal(replanned.calls.optimizer, 2);
assert.equal(replanned.result.bestIter, 1, "degraded candidate does not replace best");
assert.equal(replanned.calls.promoted, "q-iter-1");
assert.equal(
  replanned.calls.generator[2].parent,
  "q-iter-1",
  "Generator always anchors on current best",
);

const noSignal = await scenario({
  actions: ["PATCH_CODE"],
  scores: [{ f1: null }],
  hasValidationSignal: false,
});
assert.equal(noSignal.calls.generator.length, 1);
assert.equal(noSignal.calls.optimizer, 0);

const unavailableMetric = await scenario({
  actions: ["PATCH_CODE"],
  scores: [{
    f1: 0.99,
    objective: {
      name: "query_metric_unavailable",
      value: null,
      direction: "maximize",
      details: { reason: "multi-site query" },
    },
  }],
  hasValidationSignal: true,
});
assert.equal(unavailableMetric.calls.generator.length, 1);
assert.equal(unavailableMetric.calls.optimizer, 0,
  "a successful candidate without a query-specific metric remains single-shot");
assert.equal(unavailableMetric.result.bestObjective.name, "query_metric_unavailable");
assert.equal(unavailableMetric.result.bestObjective.value, null);

const compileRepair = await scenario({
  actions: ["PATCH_CODE"],
  scores: [
    { f1: null, run: { status: "crash", stage: "compile" } },
    { f1: 0.7, run: { status: "ok" } },
  ],
  hasValidationSignal: true,
});
assert.equal(compileRepair.result.bestIter, 1);
assert.equal(compileRepair.calls.optimizer, 0,
  "compile/preflight failures route directly to Generator without an Optimizer model call");

const budget = await scenario({
  actions: ["REPLAN", "REPLAN"],
  scores: [{ f1: 0.2 }, { f1: 0.3 }],
  maxReplans: 1,
});
assert.equal(budget.calls.replan, 1);
assert.equal(budget.calls.generator.length, 2);
assert.equal(budget.result.replansUsed, 1);

const rejectedReplan = await scenario({
  actions: ["REPLAN"],
  scores: [{ f1: 0.4 }],
  replanCompilability: "not_compilable",
});
assert.equal(rejectedReplan.calls.replan, 1);
assert.equal(rejectedReplan.calls.generator.length, 1,
  "a rejected replan does not enter the Generator");
assert.equal(rejectedReplan.calls.promoted, "q-iter-0",
  "the best runnable candidate survives a not-compilable replan");

const ariSelection = await scenario({
  actions: ["PATCH_CODE"],
  scores: [
    {
      f1: 0.9,
      objective: {
        name: "adjusted_rand_index",
        value: 0.2,
        direction: "maximize",
        details: { metric_type: "adjusted-rand-index" },
      },
    },
    {
      f1: 0.1,
      objective: {
        name: "adjusted_rand_index",
        value: 0.7,
        direction: "maximize",
        details: { metric_type: "adjusted-rand-index" },
      },
    },
  ],
});
assert.equal(ariSelection.result.bestIter, 1,
  "PGO candidate selection follows ARI rather than the legacy F1 field");
assert.equal(ariSelection.result.bestObjective.name, "adjusted_rand_index");
assert.equal(ariSelection.result.bestObjective.value, 0.7);

console.log("test_pgo_loop OK");
