#!/usr/bin/env node
/** Inventory every configured SemBench SQL query and classify validation capability. */
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve, basename } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { BENCHMARKS, tableDesc } from "./benchmarks.mjs";
import { classifyValidationCapability } from "./validation_capability.mjs";

function naturalCompare(left, right) {
  return left.localeCompare(right, undefined, { numeric: true, sensitivity: "base" });
}

export function referencedTables(sql, benchmark) {
  const prefix = BENCHMARKS[benchmark]?.prefix || benchmark;
  const escaped = prefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return [...sql.matchAll(new RegExp(String.raw`\b${escaped}\.(\w+)`, "gi"))]
    .map((match) => match[1])
    .filter((value, index, all) =>
      all.findIndex((item) => item.toLowerCase() === value.toLowerCase()) === index);
}

export function buildValidationMatrix(sembenchDir, options = {}) {
  const python = options.python || "python3";
  const rows = [];
  for (const [benchmark, config] of Object.entries(BENCHMARKS)) {
    const queryDir = resolve(sembenchDir, "files", benchmark, config.queryDir);
    let names;
    try {
      names = readdirSync(queryDir)
        .filter((name) => name.toLowerCase().endsWith(".sql"))
        .sort(naturalCompare);
    } catch (error) {
      rows.push({
        benchmark,
        query: "*",
        class: "not_compilable",
        reason_code: "query_directory_unreadable",
        reason: String(error.message || error),
      });
      continue;
    }
    for (const name of names) {
      const query = basename(name, ".sql");
      const sqlPath = resolve(queryDir, name);
      const sql = readFileSync(sqlPath, "utf-8");
      const audioTables = referencedTables(sql, benchmark).filter(
        (table) => tableDesc(benchmark, table)?.modality === "audio",
      );
      const proc = spawnSync(python, [
        resolve(fileURLToPath(new URL(".", import.meta.url)), "validation_plan.py"),
        sqlPath, "--benchmark", benchmark, "--query", query,
      ], { encoding: "utf-8" });
      if (proc.status !== 0) {
        rows.push({
          benchmark, query,
          class: "not_compilable",
          reason_code: "validation_plan_parse_failed",
          reason: (proc.stderr || "").trim() || `validation_plan.py exited ${proc.status}`,
        });
        continue;
      }
      const plan = JSON.parse(proc.stdout);
      rows.push({
        ...classifyValidationCapability(plan, {
          benchmark, query, audioTables,
        }),
        sql_path: sqlPath,
      });
    }
  }
  return rows;
}

function main(argv) {
  const rootIndex = argv.indexOf("--sembench-dir");
  if (rootIndex < 0 || !argv[rootIndex + 1]) {
    throw new Error("usage: validation_matrix.mjs --sembench-dir <SemBench> [--out file]");
  }
  const rows = buildValidationMatrix(resolve(argv[rootIndex + 1]));
  const outIndex = argv.indexOf("--out");
  if (outIndex >= 0 && argv[outIndex + 1]) {
    writeFileSync(resolve(argv[outIndex + 1]), JSON.stringify(rows, null, 2) + "\n");
  }
  for (const row of rows) {
    console.log([
      row.benchmark.padEnd(8),
      row.query.padEnd(10),
      row.class.padEnd(14),
      row.reason_code,
    ].join(" "));
  }
  const nonAudio = rows.filter((row) => row.reason_code !== "unsupported_audio_runtime");
  const unknown = nonAudio.filter((row) =>
    !["executable", "not_compilable"].includes(row.class)
    || ["query_directory_unreadable", "validation_plan_parse_failed",
        "unknown_validation_unit"].includes(row.reason_code));
  const executable = nonAudio.filter((row) => row.class === "executable").length;
  const refused = nonAudio.filter((row) => row.class === "not_compilable").length;
  console.log(`[matrix] non-audio=${nonAudio.length} executable=${executable} `
    + `capability_not_compilable=${refused} unknown=${unknown.length}`);
  if (unknown.length) process.exitCode = 1;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(process.argv.slice(2)); }
  catch (error) {
    console.error(`[matrix] ${error.message}`);
    process.exitCode = 1;
  }
}
