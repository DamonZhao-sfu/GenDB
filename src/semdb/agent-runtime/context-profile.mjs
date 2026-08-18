import { createReadStream } from "node:fs";
import { existsSync } from "node:fs";
import { stat } from "node:fs/promises";

const DEFAULT_MAX_DISTINCT_VALUES = 50_000;
const MAX_COLUMNS = 40;
const MAX_EXAMPLES = 3;
const MAX_VALUE_CHARS = 120;
const profileCache = new Map();

function compactValue(value) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  return text.length <= MAX_VALUE_CHARS
    ? text
    : `${text.slice(0, MAX_VALUE_CHARS - 1)}…`;
}

function normalizedJoinValue(value) {
  return String(value ?? "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLocaleLowerCase("en-US")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim()
    .replace(/\s+/g, " ");
}

/** Parse enough RFC-4180 records for focused unit tests and compatibility callers. */
export function parseCsvSample(text, maxDataRows = 256) {
  const records = [];
  let row = [];
  let field = "";
  let quoted = false;
  const limit = Math.max(1, Number(maxDataRows) || 256) + 2;
  const source = String(text || "").replace(/^\uFEFF/, "");

  const finishField = () => {
    row.push(field);
    field = "";
  };
  const finishRow = () => {
    finishField();
    if (row.some((value) => value.length > 0)) records.push(row);
    row = [];
  };

  for (let i = 0; i < source.length && records.length < limit; i++) {
    const char = source[i];
    if (quoted) {
      if (char === '"' && source[i + 1] === '"') {
        field += '"';
        i++;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
      continue;
    }
    if (char === '"' && field.length === 0) {
      quoted = true;
    } else if (char === ",") {
      finishField();
    } else if (char === "\n") {
      finishRow();
    } else if (char !== "\r") {
      field += char;
    }
  }
  if (records.length < limit && (field.length || row.length)) finishRow();

  const headers = (records[0] || []).map((value, index) => (
    String(value || "").trim() || `_column_${index}`
  ));
  const data = records.slice(1, maxDataRows + 1).map((values) => (
    Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]))
  ));
  return {
    headers,
    rows: data,
    complete: records.length <= maxDataRows + 1,
  };
}

/** Stream RFC-4180 records without loading the complete CSV into memory. The quote-pending
 * state handles an escaped quote whose second quote arrives in the next stream chunk. */
async function* streamCsvRecords(path) {
  const input = createReadStream(path, { encoding: "utf8" });
  let row = [];
  let field = "";
  let quoted = false;
  let quotePending = false;
  let firstCharacter = true;

  const finishField = () => {
    row.push(field);
    field = "";
  };

  for await (let chunk of input) {
    if (firstCharacter) {
      chunk = chunk.replace(/^\uFEFF/, "");
      firstCharacter = false;
    }
    for (const char of chunk) {
      if (quoted) {
        if (quotePending) {
          if (char === '"') {
            field += '"';
            quotePending = false;
            continue;
          }
          quoted = false;
          quotePending = false;
          // The current delimiter belongs to the unquoted state and is processed below.
        } else if (char === '"') {
          quotePending = true;
          continue;
        } else {
          field += char;
          continue;
        }
      }

      if (char === '"' && field.length === 0) {
        quoted = true;
      } else if (char === ",") {
        finishField();
      } else if (char === "\n") {
        finishField();
        if (row.some((value) => value.length > 0)) yield row;
        row = [];
      } else if (char !== "\r") {
        field += char;
      }
    }
  }
  if (field.length || row.length) {
    finishField();
    if (row.some((value) => value.length > 0)) yield row;
  }
}

function columnAccumulator() {
  return {
    nonEmptyRows: 0,
    values: new Set(),
    trackingComplete: true,
    examples: [],
  };
}

async function profileOneTable(table, maxDistinctValues) {
  if (!table.path || !existsSync(table.path)) {
    return {
      table: table.table,
      path: table.path || null,
      status: "missing",
      rows_scanned: 0,
      full_scan_complete: false,
      columns: {},
      _columnValues: {},
    };
  }

  const metadata = await stat(table.path);
  const key = `${table.path}\0${metadata.size}\0${metadata.mtimeMs}\0${maxDistinctValues}`;
  if (profileCache.has(key)) return profileCache.get(key);

  let headers = null;
  let selectedHeaders = [];
  let accumulators = [];
  let rowsScanned = 0;
  for await (const record of streamCsvRecords(table.path)) {
    if (!headers) {
      headers = record.map((value, index) => (
        String(value || "").trim() || `_column_${index}`
      ));
      selectedHeaders = headers.slice(0, MAX_COLUMNS);
      accumulators = selectedHeaders.map(columnAccumulator);
      continue;
    }
    rowsScanned++;
    for (let index = 0; index < selectedHeaders.length; index++) {
      const value = String(record[index] ?? "").trim();
      if (!value) continue;
      const acc = accumulators[index];
      acc.nonEmptyRows++;
      if (acc.examples.length < MAX_EXAMPLES && !acc.examples.includes(value)) {
        acc.examples.push(value);
      }
      if (!acc.trackingComplete) continue;
      acc.values.add(value);
      if (acc.values.size > maxDistinctValues) {
        acc.trackingComplete = false;
      }
    }
  }

  const columns = {};
  const columnValues = {};
  for (let index = 0; index < selectedHeaders.length; index++) {
    const header = selectedHeaders[index];
    const acc = accumulators[index];
    columns[header] = {
      non_empty_rows: acc.nonEmptyRows,
      distinct_values: acc.trackingComplete ? acc.values.size : null,
      distinct_values_lower_bound: acc.values.size,
      distinct_tracking_complete: acc.trackingComplete,
      examples: acc.examples.map(compactValue),
    };
    columnValues[header] = acc.values;
  }
  const result = {
    table: table.table,
    path: table.path,
    status: "ok",
    rows_scanned: rowsScanned,
    full_scan_complete: true,
    omitted_columns: Math.max(0, (headers || []).length - selectedHeaders.length),
    columns,
    _columnValues: columnValues,
  };
  profileCache.set(key, result);
  return result;
}

function containmentNearMatches(leftValues, rightValues) {
  const normalizedRight = [...rightValues]
    .map((raw) => ({ raw, normalized: normalizedJoinValue(raw) }))
    .filter((entry) => entry.normalized);
  const tokenIndex = new Map();
  for (let index = 0; index < normalizedRight.length; index++) {
    for (const token of new Set(normalizedRight[index].normalized.split(" ").filter(Boolean))) {
      if (!tokenIndex.has(token)) tokenIndex.set(token, []);
      tokenIndex.get(token).push(index);
    }
  }

  const near = [];
  for (const leftValue of leftValues) {
    const leftNormalized = normalizedJoinValue(leftValue);
    if (!leftNormalized) continue;
    const tokens = leftNormalized.split(" ").filter(Boolean).sort((a, b) => b.length - a.length);
    const candidates = new Set();
    for (const token of tokens) {
      for (const index of tokenIndex.get(token) || []) candidates.add(index);
      // One shared long token normally narrows qualified-name comparisons sufficiently.
      if (candidates.size && token.length >= 4) break;
    }
    for (const index of candidates) {
      const right = normalizedRight[index];
      if (leftNormalized === right.normalized) continue;
      if (leftNormalized.includes(right.normalized) || right.normalized.includes(leftNormalized)) {
        near.push([compactValue(leftValue), compactValue(right.raw)]);
        if (near.length >= MAX_EXAMPLES) return near;
      }
    }
  }
  return near;
}

function joinProfiles(left, right) {
  if (left.status !== "ok" || right.status !== "ok") return [];
  const rightHeaders = new Map(
    Object.keys(right.columns).map((header) => [header.toLocaleLowerCase("en-US"), header]),
  );
  const ignored = new Set([
    "id", "row_id", "_semdb_row_id", "uri", "url", "image_filename", "image_filepath",
  ]);
  const results = [];
  for (const leftHeader of Object.keys(left.columns)) {
    const folded = leftHeader.toLocaleLowerCase("en-US");
    if (ignored.has(folded) || !rightHeaders.has(folded)) continue;
    const rightHeader = rightHeaders.get(folded);
    const leftValues = left._columnValues[leftHeader] || new Set();
    const rightValues = right._columnValues[rightHeader] || new Set();
    const valueSetsComplete = left.columns[leftHeader].distinct_tracking_complete
      && right.columns[rightHeader].distinct_tracking_complete;
    const rightExact = new Set(rightValues);
    const exact = [...leftValues].filter((value) => rightExact.has(value));
    const rightNormalized = new Set([...rightValues].map(normalizedJoinValue).filter(Boolean));
    const normalized = [...leftValues]
      .filter((value) => rightNormalized.has(normalizedJoinValue(value)));
    results.push({
      left: `${left.table}.${leftHeader}`,
      right: `${right.table}.${rightHeader}`,
      full_scan_complete: left.full_scan_complete && right.full_scan_complete,
      value_sets_complete: valueSetsComplete,
      exact_overlap_count: valueSetsComplete ? exact.length : null,
      normalized_overlap_count: valueSetsComplete ? normalized.length : null,
      overlap_examples: exact.slice(0, MAX_EXAMPLES).map(compactValue),
      containment_near_match_examples: containmentNearMatches(leftValues, rightValues),
    });
  }
  return results;
}

/** Compact, validation-label-free facts aggregated across every runtime-input row. */
export async function buildPlannerTableProfile(tables, options = {}) {
  const maxDistinctValues = Math.max(
    1,
    Number(options.maxDistinctValues) || DEFAULT_MAX_DISTINCT_VALUES,
  );
  const profiled = await Promise.all(
    (tables || []).map((table) => profileOneTable(table, maxDistinctValues)),
  );
  const joins = [];
  for (let i = 0; i < profiled.length; i++) {
    for (let j = i + 1; j < profiled.length; j++) {
      joins.push(...joinProfiles(profiled[i], profiled[j]));
    }
  }
  return {
    profile_version: "2.0",
    scan_mode: "full_file_streaming",
    distinct_value_tracking_limit: maxDistinctValues,
    tables: profiled.map(({ _columnValues, ...profile }) => profile),
    same_name_join_candidates: joins.slice(0, 16),
    data_boundary: "all runtime input rows; no validation labels, CERT, or ground truth",
  };
}

/** Reject a strict equality join only when the full scan and tracked value sets prove that
 * it has no exact/normalized overlap while exposing a containment-shaped near match. */
export function lintPlanAgainstTableProfile(plan, profile) {
  if (!plan || !profile) return [];
  const relational = JSON.stringify(plan.relational_plan || []).toLocaleLowerCase("en-US");
  const findings = [];
  for (const candidate of profile.same_name_join_candidates || []) {
    if (!candidate.full_scan_complete
        || !candidate.value_sets_complete
        || candidate.exact_overlap_count !== 0
        || candidate.normalized_overlap_count !== 0
        || !candidate.containment_near_match_examples?.length) continue;
    const leftColumn = candidate.left.split(".").at(-1).toLocaleLowerCase("en-US");
    const rightColumn = candidate.right.split(".").at(-1).toLocaleLowerCase("en-US");
    const mentionsColumns = relational.includes(leftColumn) && relational.includes(rightColumn);
    const strictEquality = /==|\bequal(?:ity)?\b|exact match|inner join/.test(relational);
    const robustMatch = /contain|subset|fuzzy|near.match|strip.*parenthe|canonical.*title/.test(relational);
    if (mentionsColumns && strictEquality && !robustMatch) {
      findings.push(
        `Full runtime-input profiles prove that ${candidate.left} and ${candidate.right} `
        + `have zero exact and zero normalized overlap, while containment-shaped near matches `
        + `exist (${JSON.stringify(candidate.containment_near_match_examples[0])}). A strict `
        + `equality join is therefore empty; plan an explicit canonicalization/containment rule `
        + `or mark the requirement unresolved.`,
      );
    }
  }
  return findings;
}
