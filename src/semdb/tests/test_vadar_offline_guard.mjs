import assert from "node:assert";
import { offlineVadarViolations, parseArgs } from "../orchestrator.mjs";

const localImage = `
import semvision, imagepatch
from vadar.predefined import classify
value = classify(patch, ["a", "b"])
`;
assert.deepEqual(offlineVadarViolations(localImage), []);

const localText = `
from vadar.predefined_text import contains_phrase
keep = contains_phrase(row["description"], "comedy")
`;
assert.deepEqual(offlineVadarViolations(localText), []);

for (const source of [
  "import semtext\nctx = semtext.get_ctx(model, url)",
  "from openai import OpenAI\nclient = OpenAI()",
  "import requests\nrequests.post(url)",
  "answer = judge(text, question)",
  "answer = semruntime.vlm_judge(prompt)",
  "raw = gen_endpoint(config, schema, prompt)",
  'ap.add_argument("--endpoint")',
  "api_key = 'secret'",
  "import subprocess\nsubprocess.run(['curl', url])",
]) {
  assert.ok(offlineVadarViolations(source).length > 0, source);
}

const production = parseArgs([
  "node", "orchestrator.mjs",
  "--query-dir", "/tmp/SemBench/files/mmqa/query/bigquery",
  "--data-dir", "/tmp/SemBench/files/mmqa/data/sf_200",
  "--ground-truth-dir", "/tmp/should-not-be-used",
  "--no-ground-truth",
  "--run",
]);
assert.equal(production.noGroundTruth, true);
assert.equal(production.groundTruthDir, null);
assert.equal(production.run, true);

console.log("test_vadar_offline_guard OK");
