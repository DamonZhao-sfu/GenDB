/** Exactly one Planner call per iteration.
 *
 *  The plan lint used to re-prompt the planner inside the same iteration, so a flagged
 *  iter_0 spawned the Planner twice back to back — ~4.5 min and $0.34 of extra agent time
 *  with nothing in the log to explain the second spawn. The findings now ride along with
 *  the NEXT replan, which is a planner call the loop was going to make anyway.
 *
 *  This is a source-level guard because `createInitialPlan` and `replan` are closures over
 *  the per-query state and are not exported. It asserts the shape that matters: one
 *  `runPhase(queryPlannerConfig, ...)` per function, and no loop wrapped around it.
 */
import assert from "node:assert";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(resolve(here, "..", "orchestrator.mjs"), "utf8");

/** Body of `const <name> = async (...) => { ... }`, matched by brace depth. */
function functionBody(name) {
  const start = src.indexOf(`const ${name} = async `);
  assert.notEqual(start, -1, `${name} not found in orchestrator.mjs`);
  const open = src.indexOf("{", src.indexOf("=>", start));
  let depth = 0;
  for (let i = open; i < src.length; i += 1) {
    if (src[i] === "{") depth += 1;
    else if (src[i] === "}") {
      depth -= 1;
      if (depth === 0) return src.slice(open, i + 1);
    }
  }
  throw new Error(`unbalanced braces in ${name}`);
}

const count = (text, needle) => text.split(needle).length - 1;

for (const name of ["createInitialPlan", "replan"]) {
  const body = functionBody(name);
  assert.equal(count(body, "queryPlannerConfig"), 1,
    `${name} must spawn the Planner exactly once per call`);
  // A loop around the call is how the double-spawn came back last time.
  for (const loop of ["for (", "while (", "do {"]) {
    assert.ok(!body.includes(loop),
      `${name} must not loop around its Planner call (found ${loop.trim()})`);
  }
}

// The lint must still reach a planner — dropped findings would make the rule dead weight.
const initial = functionBody("createInitialPlan");
const replanBody = functionBody("replan");
assert.ok(initial.includes("carryPlanLint"),
  "createInitialPlan must queue its lint findings for the next planner call");
assert.ok(replanBody.includes("takePlanLint"),
  "replan must consume the queued lint findings");
assert.ok(replanBody.includes("carryPlanLint"),
  "a replanned plan must be linted too, and its findings queued for the next replan");

// Draining on read: a finding is delivered once, not re-sent on every later replan.
const take = src.slice(src.indexOf("const takePlanLint"), src.indexOf("const createInitialPlan"));
assert.ok(/pendingPlanLint\s*=\s*""/.test(take),
  "takePlanLint must clear the queue so findings are delivered once");

console.log("test_one_planner_call_per_iteration OK");
