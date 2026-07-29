import assert from "node:assert/strict";

import { classifyValidationCapability } from "../validation_capability.mjs";
import { referencedTables } from "../validation_matrix.mjs";
import { parseArgs } from "../orchestrator.mjs";

const plan = (unit, sites = 1) => ({
  benchmark: "ecomm",
  query: "q",
  candidate: { unit },
  sites: Array.from({ length: sites }, (_, index) => ({ site_id: `s${index}` })),
});

assert.equal(classifyValidationCapability(plan("row")).class, "executable");
assert.equal(classifyValidationCapability(plan("pair")).class, "executable");

const tuple = classifyValidationCapability(plan("tuple", 5));
assert.equal(tuple.class, "not_compilable");
assert.equal(tuple.reason_code, "joint_multi_site_oracle_not_implemented");
assert.equal(tuple.site_count, 5);
assert.equal(tuple.obligations.length, 4);

const audio = classifyValidationCapability(plan("row"), { audioTables: ["audio_mm"] });
assert.equal(audio.class, "not_compilable");
assert.equal(audio.reason_code, "unsupported_audio_runtime");

assert.deepEqual(
  referencedTables(
    "SELECT * FROM cars_dataset.cars c JOIN cars_dataset.audio_mm a ON c.id=a.id",
    "cars"),
  ["cars", "audio_mm"],
);

assert.equal(parseArgs([
  "node", "orchestrator.mjs", "--direct", "--semantic-plan-only",
]).semanticPlanOnly, true);
assert.throws(
  () => parseArgs([
    "node", "orchestrator.mjs", "--agent-architecture", "legacy",
    "--semantic-plan-only",
  ]),
  /requires --agent-architecture pgo/,
);

console.log("test_validation_capability OK");
