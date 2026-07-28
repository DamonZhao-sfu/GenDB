import assert from "node:assert";
import { mkdtempSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { createServer } from "node:http";
import {
  validateDataDirectory, validateEndpointModel,
} from "../orchestrator.mjs";

const root = mkdtempSync(resolve(tmpdir(), "semdb-input-preflight-"));
const data = resolve(root, "data");
mkdirSync(resolve(data, "sf_100"), { recursive: true });
mkdirSync(resolve(data, "sf_250"), { recursive: true });

await validateDataDirectory({ dataDir: resolve(data, "sf_100") });
await assert.rejects(
  validateDataDirectory({ dataDir: resolve(data, "sf_200") }),
  /Available scale factors: 100, 250.*--sf 250/);

const server = createServer((req, res) => {
  res.setHeader("content-type", "application/json");
  res.end(JSON.stringify({ data: [{ id: "Qwen/Qwen3-VL-30B-A3B-Instruct" }] }));
});
await new Promise((done) => server.listen(0, "127.0.0.1", done));
const endpoint = `http://127.0.0.1:${server.address().port}/v1`;
try {
  await validateEndpointModel(endpoint, "Qwen/Qwen3-VL-30B-A3B-Instruct");
  await assert.rejects(
    validateEndpointModel(endpoint, "Qwen/Qwen3-32B"),
    /not served.*Available model.*Qwen\/Qwen3-VL-30B-A3B-Instruct/);
} finally {
  await new Promise((done) => server.close(done));
}

console.log("test_input_preflight OK");
