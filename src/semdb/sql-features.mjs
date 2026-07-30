/**
 * Lightweight SQL scanners shared by the orchestrator and the memory system.
 *
 * These were private to orchestrator.mjs. They moved here so memory/signature.mjs
 * can derive its retrieval key from the SAME scanning the pipeline uses — a
 * retrieval key computed differently from the pipeline's own view of a query
 * silently matches the wrong template. orchestrator.mjs re-exports aliasMap for
 * its existing importers.
 */

import { BENCHMARKS } from "./benchmarks.mjs";

/** SQL table-qualifier prefix for a benchmark (e.g. mmqa, cars_dataset). */
export function benchPrefix(bench) {
  return (BENCHMARKS[bench] && BENCHMARKS[bench].prefix) || bench;
}

export function prefixRe(bench, tail) {
  return new RegExp(String.raw`\b${benchPrefix(bench)}\.(\w+)` + (tail || ""), "gi");
}

/** alias → table map from FROM/JOIN clauses (<prefix>.table [AS] alias).
 *
 * BigQuery external-object tables hide the ordinary table reference inside
 * `EXTERNAL_OBJECT_TRANSFORM(TABLE `<prefix>.IMAGES`, ...) AS images`. Keep that
 * wrapper in this lightweight scanner: otherwise EComm q8 looks text-only and its
 * pairwise image join is incorrectly sent to per-row validation.
 */
export function aliasMap(sql, bench) {
  const m = {};
  for (const x of sql.matchAll(prefixRe(bench, String.raw`\s+(?:AS\s+)?(\w+)`))) m[x[2]] = x[1];
  const escapedPrefix = benchPrefix(bench).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const transformed = new RegExp(
    String.raw`EXTERNAL_OBJECT_TRANSFORM\s*\(\s*TABLE\s+\`?`
      + escapedPrefix
      + String.raw`\.(\w+)\`?[\s\S]*?\)\s+(?:AS\s+)?(\w+)`,
    "gi",
  );
  for (const x of sql.matchAll(transformed)) m[x[2]] = x[1];
  return m;
}

/** The argument text of the AI.IF / AI.GENERATE call (up to connection_id). */
export function semanticArgs(sql) {
  const i = sql.search(/AI\.(IF|GENERATE)\s*\(/i);
  if (i < 0) return "";
  const j = sql.toLowerCase().indexOf("connection_id", i);
  return sql.slice(i, j < 0 ? Math.min(i + 500, sql.length) : j);
}

/**
 * Argument text of EVERY AI.IF / AI.GENERATE call, concatenated.
 *
 * `semanticArgs` returns only the first call because the validation path needs a
 * single call site. A retrieval key must not: ecomm q11 carries seven AI calls and
 * keying on the first alone would make it look like a single-predicate filter.
 */
export function allSemanticArgs(sql) {
  const out = [];
  const re = /AI\.(?:IF|GENERATE)\s*\(/gi;
  let match;
  while ((match = re.exec(sql)) !== null) {
    const start = match.index;
    const stop = sql.toLowerCase().indexOf("connection_id", start);
    out.push(sql.slice(start, stop < 0 ? Math.min(start + 500, sql.length) : stop));
  }
  return out.join("\n");
}

/** Tables whose alias is referenced inside the semantic predicate. */
export function tablesInPredicate(sql, bench) {
  const amap = aliasMap(sql, bench);
  const arg = semanticArgs(sql);
  const used = Object.keys(amap)
    .filter((al) => new RegExp(`\\b${al}\\.`).test(arg))
    .map((al) => amap[al]);
  return [...new Set(used)];
}
