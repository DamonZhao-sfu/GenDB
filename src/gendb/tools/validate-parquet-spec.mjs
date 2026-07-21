#!/usr/bin/env node
/**
 * Mechanical validator for the Parquet Architect's `parquet_spec.json`
 * (GenDB port of Jailbreak's agents/spec_validator.py).
 *
 * It runs NO LLM — it deterministically catches the hallucination / knowledge-gap
 * patterns that silently corrupt storage before any C++ is generated:
 *   - missing required keys
 *   - unknown physical/logical/encoding/null-handling values
 *   - physical <-> logical type inconsistencies (e.g. STRING not on BYTE_ARRAY)
 *   - DECIMAL without precision/scale
 *   - dictionary-encoding flag vs. encodings vs. decode strategy mismatches
 *     (this is the l_returnflag "column row count mismatch" bug class)
 *   - nullable vs. null_handling inconsistencies
 *   - schema columns missing from the spec (when a schema is provided)
 *
 * Usage:
 *   node tools/validate-parquet-spec.mjs <parquet_spec.json> [schema.json]
 *
 * schema.json (optional) may be either
 *   { "tables": { "lineitem": { "columns": ["l_orderkey", ...] } } }
 * or { "columns": [ { "name": "l_orderkey" }, ... ] }   (single-table form)
 *
 * Exit code 0 = clean; 1 = issues found (printed one per line) or bad input.
 * Also exported as `validate(spec, schemaInfo) -> string[]` for programmatic use.
 */

import { readFileSync } from "fs";

const PHYSICAL_TYPES = new Set([
  "BOOLEAN", "INT32", "INT64", "INT96", "FLOAT", "DOUBLE",
  "BYTE_ARRAY", "FIXED_LEN_BYTE_ARRAY",
]);

const LOGICAL_TYPES = new Set([
  "STRING", "ENUM", "DECIMAL", "DATE", "TIME", "TIMESTAMP",
  "INT", "UINT", "FLOAT16", "JSON", "BSON", "UUID", "NONE",
]);

const ENCODINGS = new Set([
  "PLAIN", "PLAIN_DICTIONARY", "RLE_DICTIONARY", "RLE", "BIT_PACKED",
  "DELTA_BINARY_PACKED", "DELTA_LENGTH_BYTE_ARRAY", "DELTA_BYTE_ARRAY",
  "BYTE_STREAM_SPLIT",
]);

const DICT_ENCODINGS = new Set(["PLAIN_DICTIONARY", "RLE_DICTIONARY"]);

const NULL_HANDLING = new Set(["no_nulls", "validity_bitmap", "definition_levels"]);

const GENDB_CPP_TYPES = new Set([
  "char", "int8_t", "int16_t", "int32_t", "int64_t",
  "uint32_t", "uint64_t", "float", "double", "string",
]);

// Physical carriers allowed for each logical type. null = any physical is fine.
const LOGICAL_PHYSICAL = {
  STRING: ["BYTE_ARRAY"],
  ENUM: ["BYTE_ARRAY"],
  JSON: ["BYTE_ARRAY"],
  BSON: ["BYTE_ARRAY"],
  DATE: ["INT32"],
  TIME: ["INT32", "INT64"],
  TIMESTAMP: ["INT64", "INT96"],
  DECIMAL: ["INT32", "INT64", "FIXED_LEN_BYTE_ARRAY", "BYTE_ARRAY"],
  UUID: ["FIXED_LEN_BYTE_ARRAY"],
  FLOAT16: ["FIXED_LEN_BYTE_ARRAY"],
  INT: ["INT32", "INT64"],
  UINT: ["INT32", "INT64"],
  NONE: null,
};

const REQUIRED_COL_KEYS = [
  "parquet_name", "physical_type", "logical_type", "encoding",
  "is_dictionary_encoded", "arrow_array_type", "gendb_cpp_type",
  "nullable", "null_handling", "decode_strategy",
];

// Phrases in a decode_strategy that indicate the WRONG dictionary handling
// (writing indices / the dictionary size instead of one decoded value per row).
const BAD_DICT_PHRASES = [
  /write\s+(?:the\s+)?indices/i,
  /dictionary\s+(?:size|length|count)/i,
  /index\s+(?:buffer|count|length)/i,
  /number\s+of\s+distinct/i,
];
// Phrases that indicate CORRECT dictionary decoding.
const GOOD_DICT_PHRASES = [/decode/i, /dictionary/i, /\bcast\b/i, /values/i, /logical/i];

function isNonEmptyString(v) {
  return typeof v === "string" && v.trim().length > 0;
}

function validateColumn(table, colName, col) {
  const issues = [];
  const where = `tables['${table}'].columns['${colName}']`;

  if (typeof col !== "object" || col === null || Array.isArray(col)) {
    issues.push(`${where}: expected an object, got ${Array.isArray(col) ? "array" : typeof col}`);
    return issues;
  }

  for (const k of REQUIRED_COL_KEYS) {
    if (!(k in col)) issues.push(`${where}: missing required key '${k}'`);
  }

  const phys = col.physical_type;
  const logical = col.logical_type;
  const enc = col.encoding;

  if (phys !== undefined && !PHYSICAL_TYPES.has(phys)) {
    issues.push(`${where}: unknown physical_type '${phys}' — allowed: ${[...PHYSICAL_TYPES].join(", ")}`);
  }
  if (logical !== undefined && !LOGICAL_TYPES.has(logical)) {
    issues.push(`${where}: unknown logical_type '${logical}' — allowed: ${[...LOGICAL_TYPES].join(", ")}`);
  }

  if (enc !== undefined) {
    if (!Array.isArray(enc)) {
      issues.push(`${where}: 'encoding' must be an array of encoding names`);
    } else {
      for (const e of enc) {
        if (!ENCODINGS.has(e)) {
          issues.push(`${where}: unknown encoding '${e}' — allowed: ${[...ENCODINGS].join(", ")}`);
        }
      }
    }
  }

  if ("is_dictionary_encoded" in col && typeof col.is_dictionary_encoded !== "boolean") {
    issues.push(`${where}: 'is_dictionary_encoded' must be a boolean`);
  }
  if ("nullable" in col && typeof col.nullable !== "boolean") {
    issues.push(`${where}: 'nullable' must be a boolean`);
  }
  if (col.null_handling !== undefined && !NULL_HANDLING.has(col.null_handling)) {
    issues.push(`${where}: unknown null_handling '${col.null_handling}' — allowed: ${[...NULL_HANDLING].join(", ")}`);
  }
  if (col.gendb_cpp_type !== undefined && !GENDB_CPP_TYPES.has(col.gendb_cpp_type)) {
    issues.push(`${where}: unknown gendb_cpp_type '${col.gendb_cpp_type}' — allowed: ${[...GENDB_CPP_TYPES].join(", ")}`);
  }
  if ("decode_strategy" in col && !isNonEmptyString(col.decode_strategy)) {
    issues.push(`${where}: 'decode_strategy' must be a non-empty string describing how to materialize one value per row`);
  }

  // physical <-> logical consistency
  if (LOGICAL_TYPES.has(logical) && PHYSICAL_TYPES.has(phys)) {
    const allowed = LOGICAL_PHYSICAL[logical];
    if (allowed && !allowed.includes(phys)) {
      issues.push(`${where}: logical_type '${logical}' cannot sit on physical_type '${phys}' — expected one of: ${allowed.join(", ")}`);
    }
  }

  // DECIMAL requires precision + scale
  if (logical === "DECIMAL") {
    if (typeof col.decimal_precision !== "number") {
      issues.push(`${where}: DECIMAL column missing numeric 'decimal_precision'`);
    }
    if (typeof col.decimal_scale !== "number") {
      issues.push(`${where}: DECIMAL column missing numeric 'decimal_scale' — a wrong/absent scale corrupts every SUM/AVG`);
    }
  }

  // nullable <-> null_handling consistency
  if (col.nullable === false && col.null_handling && col.null_handling !== "no_nulls") {
    issues.push(`${where}: nullable=false but null_handling='${col.null_handling}' (expected 'no_nulls')`);
  }
  if (col.nullable === true && col.null_handling === "no_nulls") {
    issues.push(`${where}: nullable=true but null_handling='no_nulls' (expected 'validity_bitmap' or 'definition_levels')`);
  }

  // ---- Dictionary-encoding consistency (the l_returnflag bug class) ----
  const encHasDict = Array.isArray(enc) && enc.some((e) => DICT_ENCODINGS.has(e));
  if (encHasDict && col.is_dictionary_encoded === false) {
    issues.push(`${where}: encodings include a dictionary encoding (${enc.filter((e) => DICT_ENCODINGS.has(e)).join(", ")}) but is_dictionary_encoded=false — inconsistent`);
  }
  if (col.is_dictionary_encoded === true) {
    if (Array.isArray(enc) && !encHasDict) {
      issues.push(`${where}: is_dictionary_encoded=true but no dictionary encoding (PLAIN_DICTIONARY/RLE_DICTIONARY) listed in 'encoding'`);
    }
    if (isNonEmptyString(col.decode_strategy)) {
      const ds = col.decode_strategy;
      if (BAD_DICT_PHRASES.some((re) => re.test(ds))) {
        issues.push(`${where}: dictionary decode_strategy appears to write indices / dictionary size — it MUST decode to logical values (one per row). This is the column-row-count-mismatch bug.`);
      }
      if (!GOOD_DICT_PHRASES.some((re) => re.test(ds))) {
        issues.push(`${where}: is_dictionary_encoded=true but decode_strategy does not describe decoding to values (mention cast/decode/dictionary/values)`);
      }
    }
  }

  return issues;
}

/**
 * Validate a parquet_spec object. Returns a list of issue strings (empty = valid).
 * schemaInfo is optional; when provided, checks that every schema column appears.
 */
export function validate(spec, schemaInfo = null) {
  const issues = [];

  if (typeof spec !== "object" || spec === null) {
    return ["Spec is not a JSON object"];
  }
  if (spec.source_format !== "parquet") {
    issues.push(`Top level: source_format must be "parquet", got ${JSON.stringify(spec.source_format)}`);
  }
  if (!("file_layout" in spec)) {
    issues.push("Top level: missing 'file_layout'");
  }
  const tables = spec.tables;
  if (typeof tables !== "object" || tables === null || Object.keys(tables).length === 0) {
    issues.push("Top level: 'tables' must be a non-empty object");
    return issues;
  }

  // Optional schema coverage map: table -> Set(column names)
  const schemaCols = {};
  if (schemaInfo && typeof schemaInfo === "object") {
    if (schemaInfo.tables && typeof schemaInfo.tables === "object") {
      for (const [t, tdef] of Object.entries(schemaInfo.tables)) {
        const cols = (tdef.columns || []).map((c) => (typeof c === "string" ? c : c.name)).filter(Boolean);
        schemaCols[t] = new Set(cols);
      }
    } else if (Array.isArray(schemaInfo.columns)) {
      // single-table form — apply to every table in the spec
      const cols = schemaInfo.columns.map((c) => (typeof c === "string" ? c : c.name)).filter(Boolean);
      for (const t of Object.keys(tables)) schemaCols[t] = new Set(cols);
    }
  }

  for (const [table, tdef] of Object.entries(tables)) {
    const where = `tables['${table}']`;
    if (typeof tdef !== "object" || tdef === null) {
      issues.push(`${where}: expected an object`);
      continue;
    }
    if (!isNonEmptyString(tdef.parquet_file)) {
      issues.push(`${where}: missing 'parquet_file'`);
    }
    if (!isNonEmptyString(tdef.num_rows_source)) {
      issues.push(`${where}: missing 'num_rows_source' (where the authoritative row count comes from, e.g. "footer.num_rows")`);
    }
    const columns = tdef.columns;
    if (typeof columns !== "object" || columns === null || Object.keys(columns).length === 0) {
      issues.push(`${where}: 'columns' must be a non-empty object`);
      continue;
    }
    for (const [colName, col] of Object.entries(columns)) {
      issues.push(...validateColumn(table, colName, col));
    }
    // schema coverage
    if (schemaCols[table]) {
      const specColNames = new Set(Object.values(columns).map((c) => (c && c.parquet_name) || null).filter(Boolean));
      const specKeyNames = new Set(Object.keys(columns));
      for (const sc of schemaCols[table]) {
        if (!specColNames.has(sc) && !specKeyNames.has(sc)) {
          issues.push(`${where}: schema column '${sc}' is missing from the spec`);
        }
      }
    }
  }

  return issues;
}

function main() {
  const specPath = process.argv[2];
  const schemaPath = process.argv[3];
  if (!specPath) {
    console.error("Usage: node tools/validate-parquet-spec.mjs <parquet_spec.json> [schema.json]");
    process.exit(1);
  }
  let spec, schemaInfo = null;
  try {
    spec = JSON.parse(readFileSync(specPath, "utf-8"));
  } catch (e) {
    console.error(`Could not read/parse spec ${specPath}: ${e.message}`);
    process.exit(1);
  }
  if (schemaPath) {
    try { schemaInfo = JSON.parse(readFileSync(schemaPath, "utf-8")); } catch { /* optional */ }
  }

  const issues = validate(spec, schemaInfo);
  if (issues.length === 0) {
    console.log("parquet_spec.json: OK (no issues)");
    process.exit(0);
  }
  console.error(`parquet_spec.json: ${issues.length} issue(s):`);
  for (const i of issues) console.error(`  - ${i}`);
  process.exit(1);
}

// Run as CLI only when invoked directly (not when imported).
if (process.argv[1] && process.argv[1].endsWith("validate-parquet-spec.mjs")) {
  main();
}
