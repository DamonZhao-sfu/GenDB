import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

import {
  compileHybridCandidate,
  resolveHybridProfile,
} from "../agent-runtime/hybrid-compiler.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const semdbDir = resolve(here, "..");
const root = await mkdtemp(resolve(tmpdir(), "semdb-hybrid-"));
const data = resolve(root, "data");
await mkdir(data);
await writeFile(resolve(data, "styles_details.csv"), [
  "id,brandName,productDisplayName,articleType,productDescriptors",
  "100,Reebok,Classic Backpack,{},{}",
  "200,Other,Backpack,{},{}",
].join("\n"));
await writeFile(resolve(root, "ids.txt"), "0\n");

const plan = {
  query_id: "q1",
  plan_version: 1,
  helper_dag: [],
  relational_plan: [{ step: 1 }, { step: 2 }],
};
const planPath = resolve(root, "plan.json");
const helpersPath = resolve(root, "_semantic_helpers_q1.py");
const solverPath = resolve(root, "solve_q1.py");
const manifestPath = resolve(root, "candidate_manifest.json");
await writeFile(planPath, JSON.stringify(plan));
await writeFile(helpersPath, [
  "def execute(context):",
  "    context.output(['id'], 'is_backpack_from_reebok')",
  "    for index, row in enumerate(context.rows('styles_details')):",
  "        key = str(index)",
  "        if not context.admit(key):",
  "            continue",
  "        ok = row.get('brandName') == 'Reebok' and 'backpack' in row.get('productDisplayName', '').lower()",
  "        context.trace(key, ok)",
  "        if ok:",
  "            context.emit([row.get('id', '')])",
].join("\n"));
await compileHybridCandidate({
  benchmark: "ecomm",
  query: "q1",
  plan,
  helpersPath,
  solverPath,
  manifestPath,
  semdbDir,
});
const output = resolve(root, "out.csv");
const run = spawnSync("python3", [
  solverPath, output, "--data-dir", data, "--only-ids", resolve(root, "ids.txt"),
], { encoding: "utf8" });
assert.equal(run.status, 0, run.stderr);
assert.equal((await readFile(output, "utf8")).replace(/\r\n/g, "\n"), "id\n100\n");
const trace = JSON.parse(await readFile(resolve(root, "trace_q1.json"), "utf8"));
assert.deepEqual(trace.rows, { "0": "true" });
assert.equal(resolveHybridProfile("ecomm", "q1").profile, "generic_query");
assert.equal(resolveHybridProfile("mmqa", "q7").profile, "generic_query");
assert.equal(resolveHybridProfile("mmqa", "q7").max_llm_source_lines, 260);
assert.equal(resolveHybridProfile("mmqa", "q1").profile, "generic_query");
assert.equal(resolveHybridProfile("movie", "Q10").profile, "generic_query");
assert.equal(resolveHybridProfile("cars", "Q9").max_llm_source_lines, 260);
assert.equal(resolveHybridProfile("", "q1"), null);
const sembenchQueries = {
  mmqa: ["q1", "q2a", "q2b", "q3a", "q3b", "q3c", "q3d", "q3e", "q3f", "q3g",
    "q4", "q5", "q6a", "q6b", "q6c", "q7"],
  cars: Array.from({ length: 10 }, (_, index) => `Q${index + 1}`),
  medical: [...Array.from({ length: 11 }, (_, index) => `Q${index + 1}`), "Q7_filter"],
  animals: Array.from({ length: 10 }, (_, index) => `Q${index + 1}`),
  movie: Array.from({ length: 10 }, (_, index) => `Q${index + 1}`),
  ecomm: Array.from({ length: 14 }, (_, index) => `q${index + 1}`),
};
for (const [benchmark, queries] of Object.entries(sembenchQueries)) {
  for (const query of queries) {
    assert.equal(resolveHybridProfile(benchmark, query)?.profile, "generic_query",
      `${benchmark}:${query}`);
  }
}

const genericRoot = resolve(root, "generic");
const genericData = resolve(genericRoot, "data");
await mkdir(genericData, { recursive: true });
await writeFile(resolve(genericData, "movie_reviews_1000.csv"), [
  "row_id,reviewId,reviewText",
  "0,r0,Excellent film",
  "1,r1,Boring film",
].join("\n"));
await writeFile(resolve(genericRoot, "ids.txt"), "0\n");
const genericPlan = {
  query_id: "Q1", plan_version: 1, helper_dag: [],
  relational_plan: [{ step_id: "scan" }, { step_id: "filter" }],
};
const genericHelpers = resolve(genericRoot, "_semantic_helpers_Q1.py");
const genericSolver = resolve(genericRoot, "solve_Q1.py");
await writeFile(genericHelpers, [
  "def execute(context):",
  "    context.output(['reviewId'], 'positive')",
  "    for index, row in enumerate(context.rows('movie.reviews')):",
  "        key = context.row_key(row, index)",
  "        if not context.admit(key):",
  "            continue",
  "        holds = 'excellent' in row.get('reviewText', '').lower()",
  "        context.trace(key, holds)",
  "        context.branch('accepted' if holds else 'rejected')",
  "        if holds:",
  "            context.emit([row.get('reviewId', '')])",
].join("\n"));
await compileHybridCandidate({
  benchmark: "movie", query: "Q1", plan: genericPlan, helpersPath: genericHelpers,
  solverPath: genericSolver,
  manifestPath: resolve(genericRoot, "candidate_manifest.json"), semdbDir,
  tableBindings: { reviews: "movie_reviews_1000.csv" },
});
const genericOut = resolve(genericRoot, "out.csv");
const genericRun = spawnSync("python3", [
  genericSolver, genericOut, "--data-dir", genericData,
  "--only-ids", resolve(genericRoot, "ids.txt"),
], { encoding: "utf8" });
assert.equal(genericRun.status, 0, genericRun.stderr);
assert.equal(
  (await readFile(genericOut, "utf8")).replace(/\r\n/g, "\n"),
  "reviewId\nr0\n",
);
const genericTrace = JSON.parse(
  await readFile(resolve(genericRoot, "trace_Q1.json"), "utf8"),
);
assert.deepEqual(genericTrace, { attr: "positive", rows: { "0": "true" } });

const badGenericHelpers = resolve(genericRoot, "_bad_runtime_fragment.py");
await writeFile(badGenericHelpers, [
  "def execute(context):",
  "    with open('rows.csv') as handle:",
  "        return handle.read()",
].join("\n"));
await assert.rejects(
  compileHybridCandidate({
    benchmark: "movie", query: "Q2", plan: { ...genericPlan, query_id: "Q2" },
    helpersPath: badGenericHelpers,
    solverPath: resolve(genericRoot, "bad_solver.py"),
    manifestPath: resolve(genericRoot, "bad_manifest.json"), semdbDir,
  }),
  /runtime-owned code \(open\(\)\)/,
);

console.log("test_hybrid_codegen OK");
