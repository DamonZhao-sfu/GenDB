/**
 * SemDB memory retrieval key.
 *
 * GenDB keys L0/L1 on SQL structure (join count, table set, boolean features).
 * That is the wrong key here: two SemBench queries can have identical relational
 * shape and completely unrelated semantics, because the discriminative part of a
 * SemDB query lives in the AI predicate and the corpus modality, not the join
 * graph.
 *
 * Stage A (this key) is plan-independent: it is computable at Phase 0, before the
 * Planner has run, from exactly the same SQL scanning the pipeline itself uses.
 * Stage B (planFeatures) is plan-derived and is stored as payload, never as a key.
 *
 * Two HARD GATES — modality and operator kind — precede all weighting. GenDB's
 * purely additive score is what let one template (`L1_tpch_Q9`) absorb 8 of 17
 * queries in their own memory_report.json; a false structural match costs a whole
 * iteration budget of misleading context, so a disagreement on either axis is a
 * veto rather than a penalty.
 */

import { createHash } from "node:crypto";

import { allSemanticArgs, tablesInPredicate } from "../sql-features.mjs";

export const SIGNATURE_VERSION = 1;

/** Hard ceiling for the weighted (non-identical) similarity path. */
export const WEIGHTED_SIMILARITY_CEILING = 0.95;

/** Words that carry no discriminative signal in an AI predicate or NL description. */
const STOPWORDS = new Set([
  "the", "and", "for", "with", "that", "this", "from", "into", "are", "was", "were",
  "has", "have", "had", "not", "any", "all", "out", "its", "his", "her", "you", "your",
  "which", "what", "when", "where", "who", "whom", "how", "why", "does", "did", "can",
  "could", "would", "should", "will", "shall", "may", "might", "must", "true", "false",
  "null", "select", "from", "where", "join", "left", "right", "inner", "outer", "group",
  "order", "limit", "having", "case", "then", "else", "end", "prompt", "connection",
  "string", "text", "value", "values", "column", "table", "row", "rows", "return",
  "returns", "answer", "output", "input", "given", "based", "using", "please",
]);

const AGGREGATE_RE = /\b(count|sum|avg|min|max|stddev|variance|approx_count_distinct)\s*\(/i;

// ---------------------------------------------------------------------------
// Stage A — plan-independent features
// ---------------------------------------------------------------------------

/**
 * Coarse operator kind. Order matters: a semantic predicate spanning two tables
 * is a join no matter what else the query does, because the sampling unit (and
 * therefore the whole validation and trace contract) follows from that alone.
 *
 * @returns {"join"|"group"|"agg"|"topk"|"filter"|"map"}
 */
export function operatorKind(sql, benchmark) {
  if (tablesInPredicate(sql, benchmark).length >= 2) return "join";
  if (/\bGROUP\s+BY\b/i.test(sql)) return "group";
  if (AGGREGATE_RE.test(selectList(sql))) return "agg";
  if (/\bORDER\s+BY\b/i.test(sql) && /\bLIMIT\b/i.test(sql)) return "topk";
  if (/\bAI\.IF\s*\(/i.test(sql)) return "filter";
  return "map";
}

/** Physical validation unit implied by the operator kind. */
export function samplingUnitFor(kind) {
  if (kind === "join") return "pair";
  if (kind === "group") return "group";
  return "row";
}

/**
 * Projection list of the OUTER query.
 *
 * Every non-trivial SemBench query opens with `WITH <cte> AS (SELECT ...)`, so the
 * first SELECT in the text belongs to a CTE and describes nothing about what the
 * query returns. Take the last SELECT at paren depth 0 instead, and read to its
 * matching FROM.
 */
function selectList(sql) {
  let depth = 0;
  let start = -1;
  for (let i = 0; i < sql.length; i += 1) {
    const ch = sql[i];
    if (ch === "(") depth += 1;
    else if (ch === ")") depth -= 1;
    else if (depth === 0 && /select/i.test(sql.slice(i, i + 6))
             && !/\w/.test(sql[i - 1] || " ") && !/\w/.test(sql[i + 6] || " ")) {
      start = i;
    }
  }
  if (start < 0) return "";
  depth = 0;
  for (let i = start + 6; i < sql.length; i += 1) {
    const ch = sql[i];
    if (ch === "(") depth += 1;
    else if (ch === ")") depth -= 1;
    else if (depth === 0 && /from/i.test(sql.slice(i, i + 4))
             && !/\w/.test(sql[i - 1] || " ") && !/\w/.test(sql[i + 4] || " ")) {
      return sql.slice(start + 6, i);
    }
  }
  return sql.slice(start + 6);
}

/** Number of top-level projected expressions. `SELECT *` counts as 1. */
export function projectionArity(sql) {
  const list = selectList(sql).trim();
  if (!list) return 0;
  let depth = 0;
  let count = 1;
  for (const ch of list) {
    if (ch === "(" || ch === "[") depth += 1;
    else if (ch === ")" || ch === "]") depth -= 1;
    else if (ch === "," && depth === 0) count += 1;
  }
  return count;
}

/**
 * Literal-elided SQL. Two runs of the same query with different parameters must
 * produce one template, or every parameter sweep looks like a novel query.
 */
export function sqlTemplate(sql) {
  return sql
    .replace(/'(?:[^']|'')*'/g, ":p")
    .replace(/"(?:[^"]|"")*"/g, ":p")
    .replace(/\b\d+(?:\.\d+)?\b/g, ":p")
    .replace(/`/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Token set of the query's RELATIONAL shape, with the AI calls blanked out.
 *
 * Without this, movie Q1/Q2 and Q3/Q4 score a perfect 1.0 against each other:
 * same tables, same AI predicate, same arity, same flags. They are not the same
 * query — Q2 adds `r.id = 'taken_3'`, and Q4 computes SUM(CASE WHEN ...)/COUNT(*)
 * where Q3 counts rows. Reusing one's solver for the other silently answers a
 * different question, so the relational skeleton has to be part of the key.
 */
export function relationalSkeleton(sql) {
  let blanked = sqlTemplate(sql);
  // Replace each AI call, including its balanced argument list, with one token.
  for (;;) {
    const match = blanked.match(/AI\.(?:IF|GENERATE)\s*\(/i);
    if (!match) break;
    const start = match.index;
    let depth = 0;
    let end = start;
    for (let i = start + match[0].length - 1; i < blanked.length; i += 1) {
      if (blanked[i] === "(") depth += 1;
      else if (blanked[i] === ")") {
        depth -= 1;
        if (depth === 0) { end = i + 1; break; }
      }
      end = i + 1;
    }
    blanked = `${blanked.slice(0, start)}AI_CALL${blanked.slice(end)}`;
  }
  const tokens = blanked
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, " ")
    .split(" ")
    .filter((t) => t.length > 1 && t !== "p");
  return [...new Set(tokens)].sort();
}

/** Discriminative token set of ALL AI predicates plus the NL description. */
export function predicateTokens(sql, nl) {
  const text = `${allSemanticArgs(sql)} ${nl || ""}`.toLowerCase();
  const tokens = text
    .replace(/[^a-z0-9]+/g, " ")
    .split(" ")
    .filter((t) => t.length > 2 && !STOPWORDS.has(t) && !/^\d+$/.test(t));
  return [...new Set(tokens)].sort();
}

function sqlFlags(sql) {
  return {
    group_by: /\bGROUP\s+BY\b/i.test(sql),
    order_by: /\bORDER\s+BY\b/i.test(sql),
    limit: /\bLIMIT\b/i.test(sql),
    aggregate: AGGREGATE_RE.test(sql),
    distinct: /\bDISTINCT\b/i.test(sql),
  };
}

/**
 * Build the stage-A retrieval key from a planQuery() result.
 *
 * @param {{query:string, sql:string, nl:string|null, benchmark:string,
 *          corpus:{table:string, modality:string},
 *          tables:Array<{table:string}>}} planObj
 * @returns {object} query signature
 */
export function buildQuerySignature(planObj) {
  const { sql, nl, corpus } = planObj;
  const benchmark = planObj.benchmark;
  const kind = operatorKind(sql, benchmark);
  const template = sqlTemplate(sql);
  const modality = corpus?.modality || "text";
  const tokens = predicateTokens(sql, nl);
  return {
    signature_version: SIGNATURE_VERSION,
    modality,
    operator_kind: kind,
    sampling_unit: samplingUnitFor(kind),
    corpus_table: corpus?.table || null,
    tables: [...new Set((planObj.tables || []).map((t) => t.table))].sort(),
    predicate_tokens: tokens,
    relational_skeleton: relationalSkeleton(sql),
    projection_arity: projectionArity(sql),
    flags: sqlFlags(sql),
    sql_template: template,
    // The predicate fingerprint is PART of the template identity, not merely an
    // extra scoring channel. In SemBench a query's semantics live inside the AI
    // call's string literals — exactly what sqlTemplate() elides — so hashing the
    // elided SQL alone makes "does the image show the racetrack's logo" and "does
    // the image depict a sailing vessel" the same template, and would hand the
    // second query the first one's solver as an exact match. Connection id and
    // model params sit after `connection_id` and are excluded from the tokens, so
    // ordinary parameter variants still collapse to a single template.
    template_hash: createHash("sha256")
      .update(`${template} ${modality} ${tokens.join(" ")}`)
      .digest("hex")
      .slice(0, 16),
  };
}

// ---------------------------------------------------------------------------
// Similarity
// ---------------------------------------------------------------------------

/** Jaccard over arrays. Both empty → 1 (they agree); one empty → 0. */
export function jaccard(a = [], b = []) {
  if (a.length === 0 && b.length === 0) return 1;
  if (a.length === 0 || b.length === 0) return 0;
  const setB = new Set(b);
  let inter = 0;
  const setA = new Set(a);
  for (const x of setA) if (setB.has(x)) inter += 1;
  return inter / (setA.size + setB.size - inter);
}

function flagAgreement(a = {}, b = {}) {
  const keys = ["group_by", "order_by", "limit", "aggregate", "distinct"];
  let same = 0;
  for (const k of keys) if (Boolean(a[k]) === Boolean(b[k])) same += 1;
  return same / keys.length;
}

/**
 * @returns {number} 0..1. 0 means "not comparable", not merely "dissimilar".
 */
export function signatureSimilarity(a, b) {
  if (!a || !b) return 0;
  if (a.signature_version !== b.signature_version) return 0;
  if (a.modality !== b.modality) return 0;              // hard gate
  if (a.operator_kind !== b.operator_kind) return 0;    // hard gate
  if (a.template_hash && a.template_hash === b.template_hash) return 1;
  const weighted = 0.15 * (a.sampling_unit === b.sampling_unit ? 1 : 0)
    + 0.15 * jaccard(a.tables, b.tables)
    + 0.30 * jaccard(a.predicate_tokens, b.predicate_tokens)
    + 0.25 * jaccard(a.relational_skeleton, b.relational_skeleton)
    + 0.05 * (a.projection_arity === b.projection_arity ? 1 : 0)
    + 0.10 * flagAgreement(a.flags, b.flags);
  // The weighted score is a similarity heuristic; it is NEVER proof of identity, so
  // it is capped below any sane exact threshold. Only an identical template hash
  // means "the same query with different parameters" — and only that may license
  // reusing another query's promoted solver. Uncapped, movie Q1/Q2 reached exactly
  // 1.0 while differing by a WHERE conjunct that changes the answer set.
  return Math.min(WEIGHTED_SIMILARITY_CEILING, weighted);
}

/** Stable id fragment for a signature (used to name L1 nodes). */
export function signatureHash(sig) {
  return createHash("sha256")
    .update(JSON.stringify({
      v: sig.signature_version,
      m: sig.modality,
      k: sig.operator_kind,
      t: sig.template_hash,
    }))
    .digest("hex")
    .slice(0, 12);
}

// ---------------------------------------------------------------------------
// Stage B — plan-derived features (payload, never a key)
// ---------------------------------------------------------------------------

/** @param {object} plan a validated semantic plan */
export function planFeatures(plan) {
  const sites = Array.isArray(plan?.semantic_sites) ? plan.semantic_sites : [];
  const helpers = Array.isArray(plan?.helper_dag) ? plan.helper_dag : [];
  const primitives = new Set();
  for (const h of helpers) {
    for (const step of h.primitive_steps || []) {
      if (step?.primitive) primitives.add(String(step.primitive));
    }
  }
  const relational = (plan?.relational_plan || []).map((entry) => {
    const text = typeof entry === "string" ? entry : (entry?.op || entry?.type || "");
    return String(text).trim().split(/\s+/)[0].toLowerCase();
  }).filter(Boolean);
  return {
    site_operators: [...new Set(sites.map((s) => s.operator).filter(Boolean))].sort(),
    sampling_units: [...new Set(sites.map((s) => s.sampling_unit).filter(Boolean))].sort(),
    output_types: [...new Set(sites.map((s) => s.output_type).filter(Boolean))].sort(),
    value_space_kinds: [...new Set(sites.map((s) => valueSpaceKind(s.value_space)))].sort(),
    helper_names: [...new Set(helpers.map((h) => h.name).filter(Boolean))].sort(),
    primitives: [...primitives].sort(),
    relational_ops: [...new Set(relational)].sort(),
    compilability: plan?.compilability?.class || null,
  };
}

function valueSpaceKind(space) {
  if (space === null || space === undefined) return "open";
  if (Array.isArray(space)) return "enum";
  if (typeof space === "object") return "structured";
  return "string";
}

/** How close two plans are structurally. Used to decide whether to cite a stored helper DAG. */
export function planAffinity(a, b) {
  if (!a || !b) return 0;
  return 0.5 * jaccard(a.primitives, b.primitives)
    + 0.3 * jaccard(a.site_operators, b.site_operators)
    + 0.2 * (jaccard(a.sampling_units, b.sampling_units) === 1 ? 1 : 0);
}
