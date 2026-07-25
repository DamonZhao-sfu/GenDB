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

// The region/pair primitives are pure-local and must stay allowed.
const localRegions = `
from vadar.predefined import detect, crop, regions_grid, regions_center, pair_score
cells = regions_grid(image, 2, 2, overlap=0.1)
hits = detect(cells[0], "zebra")
sim = pair_score(crop(image, 0.5, 0, 1, 1), regions_center(other, 0.6))
`;
assert.deepEqual(offlineVadarViolations(localRegions), []);

for (const source of [
  "import semtext\nctx = semtext.get_ctx(model, url)",
  "from openai import OpenAI\nclient = OpenAI()",
  "import requests\nrequests.post(url)",
  "answer = judge(text, question)",
  "answer = semruntime.vlm_judge(prompt)",
  "raw = gen_endpoint(config, schema, prompt)",
  // OpImgVQA / OpImgCap are VLM-backed: legitimate in the extraction layer,
  // never inside generated VADAR code.
  "import semvqa\nans, s = semvqa.img_vqa(path, 'damaged?')",
  "import semcaption\ncaps = semcaption.load(p)",
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
