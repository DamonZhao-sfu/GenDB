/** The planner is handed the bare query id (`q9`) but also sees the artifact path
 *  `runs/<out>/<benchmark>-<query>/iter_0/plan.json`. Twice in the recorded runs it wrote
 *  the DIRECTORY name into `query_id` (`ecomm-q9`, `movie-Q6`) and the contract check
 *  killed the whole query over it. A decoration is repaired and persisted; a genuinely
 *  different query's id still fails hard.
 *
 *  Fixtures are REAL artifacts from `runs/` — a hand-written plan would drift from the
 *  schema and test nothing. */
import assert from "node:assert";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  readAndValidatePlan, readAndValidateOptimizerAction,
} from "../agent-runtime/contracts.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const dir = await mkdtemp(resolve(tmpdir(), "semdb-contract-"));

const PLAN_FIXTURE = JSON.parse(await readFile(
  resolve(here, "..", "runs/ecomm_new/ecomm-q13/plan.json"), "utf8"));
const ACTION_FIXTURE = JSON.parse(await readFile(
  resolve(here, "..", "runs/ecomm_new/ecomm-q13/iter_1/optimizer_action.json"), "utf8"));

let seq = 0;
async function fixture(base, overrides) {
  const path = resolve(dir, `f${seq += 1}.json`);
  await writeFile(path, JSON.stringify({ ...base, ...overrides }, null, 2));
  return path;
}
const onDisk = async (path) => JSON.parse(await readFile(path, "utf8")).query_id;
const planOnDisk = async (path) => JSON.parse(await readFile(path, "utf8"));

// 1. `<benchmark>-<query>` is repaired, and the repair is PERSISTED — the manifest
//    cross-check and the next iteration both re-read this file.
const decorated = await fixture(PLAN_FIXTURE, { query_id: "ecomm-q13" });
assert.equal((await readAndValidatePlan(decorated, { queryId: "q13" })).query_id, "q13");
assert.equal(await onDisk(decorated), "q13",
  "the repair must be written back, not just applied in memory");

// 2. Case drift alone is repaired (movie ids are `Q6`; agents sometimes lowercase them).
const cased = await fixture(PLAN_FIXTURE, { query_id: "q13" });
assert.equal((await readAndValidatePlan(cased, { queryId: "Q13" })).query_id, "Q13");

// 3. An underscore-separated run directory is the same slip.
const under = await fixture(PLAN_FIXTURE, { query_id: "movie_Q13" });
assert.equal((await readAndValidatePlan(under, { queryId: "Q13" })).query_id, "Q13");

// 4. A DIFFERENT query's id must still fail — that is what a mis-seeded memory warm start
//    produces, and silently renaming it would hide a plan written for another query.
const wrong = await fixture(PLAN_FIXTURE, { query_id: "q7" });
await assert.rejects(
  () => readAndValidatePlan(wrong, { queryId: "q13" }),
  /query id mismatch at .*expected q13, got q7/,
);
assert.equal(await onDisk(wrong), "q7", "a hard mismatch must leave the file untouched");

// 5. Sharing a prefix is not sharing the tail: `ecomm-q13` is not a decoration of `q3`.
const tail = await fixture(PLAN_FIXTURE, { query_id: "ecomm-q13" });
await assert.rejects(() => readAndValidatePlan(tail, { queryId: "q3" }), /query id mismatch/);

// 6. The same reconciliation covers the optimizer action, and its candidate_id too.
const action = await fixture(ACTION_FIXTURE, {
  query_id: "ecomm-q13", candidate_id: "q13-iter-0",
});
const reconciled = await readAndValidateOptimizerAction(
  action, { queryId: "q13", candidateId: "q13-iter-0" });
assert.equal(reconciled.query_id, "q13");
assert.equal(reconciled.candidate_id, "q13-iter-0");

// 7. A wrong candidate_id still fails, even when the query id is fine.
const badCandidate = await fixture(ACTION_FIXTURE, {
  query_id: "q13", candidate_id: "q13-iter-9",
});
await assert.rejects(
  () => readAndValidateOptimizerAction(badCandidate, { queryId: "q13", candidateId: "q13-iter-0" }),
  /candidate id mismatch/,
);

// 8. Replan lineage is deterministic PGO metadata, not a semantic model decision. A
//    recorded Qwen run produced a correct replacement plan but copied the initial-plan
//    example's 1/null literals after 8m50s of work. Normalize and persist those two fields
//    from the authoritative previous plan instead of discarding the whole query.
const staleLineage = await fixture(PLAN_FIXTURE, {
  query_id: "q13",
  plan_version: 1,
  parent_plan_version: null,
});
const prior = { ...PLAN_FIXTURE, query_id: "q13", plan_version: 1 };
const revised = await readAndValidatePlan(staleLineage, {
  queryId: "q13",
  previousPlan: prior,
});
assert.equal(revised.plan_version, 2);
assert.equal(revised.parent_plan_version, 1);
const persisted = await planOnDisk(staleLineage);
assert.equal(persisted.plan_version, 2);
assert.equal(persisted.parent_plan_version, 1);

console.log("test_contract_id_reconcile OK");
