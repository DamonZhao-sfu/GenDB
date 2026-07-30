/** Agent-facing instructions must not name the benchmark they will be scored on.
 *
 *  The role skills are loaded on EVERY query, so a workload name, a query id, or a real
 *  data value written into them is benchmark knowledge injected into the planner before
 *  it has read anything. A rule that cannot be stated without naming a specific corpus
 *  value is fitted to that corpus and will not transfer.
 *
 *  Illustrations belong in the plan lint's error text and in tests, where they are read
 *  by a human debugging a finding — not in the standing instructions.
 */
import assert from "node:assert";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const semdb = resolve(here, "..");

/** SemBench scenario names, query-id shapes, and values that only exist in its corpora. */
const FORBIDDEN = [
  [/\bSemBench\b/i, "benchmark name"],
  [/\b(mmqa|ecomm|thalamusdb|ap_warrior)\b/i, "scenario/table name"],
  [/\bq\d+[a-g]?\b(?!\w)/i, "query id"],
  [/\b(santa anita|del mar|churchill downs|bay meadows|hollywood park|oak tree)\b/i,
    "corpus data value"],
  [/\b(british airways|delta air lines|discover airlines|edelweiss air|virgin atlantic)\b/i,
    "corpus data value"],
  [/\b(articleType|baseColour|articleAttributes|productDisplayName|styles_details)\b/,
    "corpus column name"],
];

/** Files an agent reads as standing instruction.
 *
 *  Includes the operator library: the prompts tell the model to read
 *  `MODULES_SIGNATURES` in `vadar/predefined.py` and the API spec in `vadar/API.md`, so
 *  those are agent-facing text just as much as a prompt is. */
function instructionFiles() {
  const out = [join(semdb, "vadar/API.md"), join(semdb, "vadar/predefined.py")];
  for (const kind of ["skills", "agents"]) {
    const root = join(semdb, kind);
    if (!existsSync(root)) continue;
    for (const entry of readdirSync(root, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const dir = join(root, entry.name);
      for (const f of readdirSync(dir)) {
        if (f.endsWith(".md")) out.push(join(dir, f));
      }
    }
  }
  return out.filter((f) => existsSync(f));
}

const offences = [];
for (const path of instructionFiles()) {
  const text = readFileSync(path, "utf8");
  text.split("\n").forEach((line, i) => {
    for (const [pattern, label] of FORBIDDEN) {
      const m = pattern.exec(line);
      if (m) offences.push(`${path.slice(semdb.length + 1)}:${i + 1} ${label} "${m[0]}"`);
    }
  });
}

assert.deepEqual(offences, [],
  `benchmark identifiers leaked into agent instructions:\n  ${offences.join("\n  ")}`);

// The rules themselves must survive the scrub — a generic statement, not a deleted one.
const planner = readFileSync(
  join(semdb, "skills/plan-semantic-query/SKILL.md"), "utf8");
for (const kept of [
  "function of BOTH sides",          // pair predicate
  "Veto first, identify second",
  "never require the target to win",
  "fallback on a DIFFERENT backend",
  "Compare within a primitive family",
  "Crop to the subject",
  "Never tune a branch against the validation sample",
]) {
  assert.ok(planner.includes(kept), `the scrub dropped the rule: ${kept}`);
}

console.log("test_no_benchmark_leakage OK");
