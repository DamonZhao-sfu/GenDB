#!/usr/bin/env node
/**
 * Parquet QA oracle (GenDB port of Jailbreak's cpp_qa ground_truth + _diff_stats).
 *
 * Reads a Parquet file with DuckDB and emits deterministic per-column ground-truth
 * statistics (row_count + min/max/sum/distinct/null_count). The ingest QA step diffs
 * the ingested GenDB/Arrow storage against this oracle; a column that keeps
 * mismatching across ingest retries signals a spec-level error (revise parquet_spec.json
 * for that column) rather than a code bug.
 *
 * Usage:
 *   node tools/parquet-oracle.mjs <parquet_file> [--columns c1,c2,...] [--out oracle.json]
 *   node tools/parquet-oracle.mjs --self-test        # offline unit test (no duckdb)
 *
 * Output JSON shape (matches the KB §11 oracle + Jailbreak _diff_stats):
 *   { "row_count": 6001215,
 *     "columns": { "l_extendedprice": { "min":..., "max":..., "sum":..., "distinct":..., "null_count":0 }, ... } }
 *
 * Requires the `duckdb` CLI on PATH (conda install -c conda-forge python-duckdb / duckdb).
 */

import { execFileSync } from "child_process";
import { writeFileSync } from "fs";

const NUMERIC_DUCKDB_TYPES = /(^|_)(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|UTINYINT|USMALLINT|UINTEGER|UBIGINT|FLOAT|DOUBLE|REAL|DECIMAL)/i;

const q = (id) => `"${String(id).replace(/"/g, '""')}"`; // quote a SQL identifier/alias

/** Build the DuckDB aggregate SQL that returns one row of all column stats. */
export function buildStatsSql(parquetFile, schema) {
  const src = `'${parquetFile.replace(/'/g, "''")}'`;
  const parts = ["count(*) AS __row_count"];
  for (const { name, type } of schema) {
    const c = q(name);
    const numeric = NUMERIC_DUCKDB_TYPES.test(type);
    parts.push(`min(${c}) AS ${q(name + "__min")}`);
    parts.push(`max(${c}) AS ${q(name + "__max")}`);
    parts.push(numeric ? `sum(${c}) AS ${q(name + "__sum")}` : `NULL AS ${q(name + "__sum")}`);
    parts.push(`approx_count_distinct(${c}) AS ${q(name + "__distinct")}`);
    parts.push(`count(*) - count(${c}) AS ${q(name + "__nullcount")}`);
  }
  return `SELECT ${parts.join(", ")} FROM ${src}`;
}

/** Reshape a flat DuckDB result row ({col__min, col__max, ...}) into nested stats. */
export function reshapeRow(row, schema) {
  const out = { row_count: row.__row_count != null ? Number(row.__row_count) : null, columns: {} };
  for (const { name } of schema) {
    out.columns[name] = {
      min: row[`${name}__min`] ?? null,
      max: row[`${name}__max`] ?? null,
      sum: row[`${name}__sum`] ?? null,
      distinct: row[`${name}__distinct`] != null ? Number(row[`${name}__distinct`]) : null,
      null_count: row[`${name}__nullcount`] != null ? Number(row[`${name}__nullcount`]) : null,
    };
  }
  return out;
}

function duckdbJson(sql) {
  const out = execFileSync("duckdb", ["-json", "-c", sql], { encoding: "utf-8", maxBuffer: 64 * 1024 * 1024 });
  return JSON.parse(out);
}

function getSchema(parquetFile) {
  const src = `'${parquetFile.replace(/'/g, "''")}'`;
  const rows = duckdbJson(`DESCRIBE SELECT * FROM ${src}`);
  // DESCRIBE columns: column_name, column_type, ...
  return rows.map((r) => ({ name: r.column_name, type: r.column_type }));
}

function parseArgs(argv) {
  const a = { parquet: null, columns: null, out: null, selfTest: false };
  const rest = [];
  for (let i = 2; i < argv.length; i++) {
    const t = argv[i];
    if (t === "--self-test") a.selfTest = true;
    else if (t === "--columns" && argv[i + 1]) a.columns = argv[++i].split(",").map((s) => s.trim()).filter(Boolean);
    else if (t === "--out" && argv[i + 1]) a.out = argv[++i];
    else rest.push(t);
  }
  a.parquet = rest[0] || null;
  return a;
}

function selfTest() {
  const schema = [
    { name: "l_extendedprice", type: "DECIMAL(15,2)" },
    { name: "l_returnflag", type: "VARCHAR" },
  ];
  const sql = buildStatsSql("lineitem.parquet", schema);
  const okSql =
    sql.includes("count(*) AS __row_count") &&
    sql.includes('sum("l_extendedprice") AS "l_extendedprice__sum"') &&
    sql.includes('NULL AS "l_returnflag__sum"') &&           // non-numeric → NULL sum
    sql.includes('approx_count_distinct("l_returnflag") AS "l_returnflag__distinct"');
  const shaped = reshapeRow(
    { __row_count: 6001215, "l_extendedprice__min": 901, "l_extendedprice__max": 104949.5,
      "l_extendedprice__sum": 2.29e11, "l_extendedprice__distinct": 933900, "l_extendedprice__nullcount": 0,
      "l_returnflag__min": "A", "l_returnflag__max": "R", "l_returnflag__sum": null,
      "l_returnflag__distinct": 3, "l_returnflag__nullcount": 0 },
    schema
  );
  const okShape =
    shaped.row_count === 6001215 &&
    shaped.columns.l_extendedprice.sum === 2.29e11 &&
    shaped.columns.l_returnflag.distinct === 3 &&
    shaped.columns.l_returnflag.sum === null;
  console.log("SQL build:", okSql ? "PASS" : "FAIL");
  console.log("Reshape  :", okShape ? "PASS" : "FAIL");
  process.exit(okSql && okShape ? 0 : 1);
}

function main() {
  const args = parseArgs(process.argv);
  if (args.selfTest) return selfTest();
  if (!args.parquet) {
    console.error("Usage: node tools/parquet-oracle.mjs <parquet_file> [--columns c1,c2] [--out oracle.json]");
    console.error("       node tools/parquet-oracle.mjs --self-test");
    process.exit(1);
  }
  let schema;
  try {
    schema = getSchema(args.parquet);
  } catch (e) {
    console.error(`Failed to read schema via duckdb (is duckdb installed?): ${e.message}`);
    process.exit(1);
  }
  if (args.columns) schema = schema.filter((s) => args.columns.includes(s.name));
  const row = duckdbJson(buildStatsSql(args.parquet, schema))[0] || {};
  const oracle = reshapeRow(row, schema);
  const json = JSON.stringify(oracle, null, 2);
  if (args.out) { writeFileSync(args.out, json + "\n"); console.error(`Wrote oracle → ${args.out}`); }
  else console.log(json);
}

if (process.argv[1] && process.argv[1].endsWith("parquet-oracle.mjs")) {
  main();
}
