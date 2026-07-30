/**
 * Ground-truth leak guard.
 *
 * The Memory Manager runs after final evaluation, so unlike the generating
 * agents it *can* see ground truth. Everything it writes is later injected into,
 * or discoverable by, the Planner / Generator / Optimizer — which must never see
 * a label. The write boundary is the only place those two facts meet, so the
 * guard lives here and is applied to both graph nodes and skill files.
 *
 * Aggregate metrics and technique prose are fine. Per-row expected values,
 * ground-truth file paths, and CERT references are not.
 */

/**
 * Unconditional: a literal path or an id dump is data-bearing whatever the prose
 * around it says.
 */
const HARD_TRIPWIRES = [
  { re: /raw_results/i, reason: "references the SemBench raw_results tree" },
  { re: /\/val\.json\b/i, reason: "references a validation label file" },
  { re: /\b\d{20,}\b/, reason: "looks like a row-id dump" },
];

/**
 * Conditional: naming ground truth or CERT is only a problem when the text
 * *directs* an agent at them. Our own role skills and the Manager's own
 * instructions must be able to say "never read ground truth" — a guard that
 * cannot tell a prohibition from an instruction would forbid writing down the
 * very rule it exists to enforce.
 */
const DIRECTED_TRIPWIRES = [
  { re: /ground[_\s-]?truth/gi, reason: "directs the reader at ground truth" },
  { re: /\bCERT\b/g, reason: "directs the reader at the sealed CERT split" },
];

const NEGATION_RE = /\b(never|not|no|without|don'?t|cannot|can't|must not|excluded?|forbidden|prohibit\w*|avoid)\b/i;
const ACCESS_VERB_RE = /\b(read|open|load|fetch|consult|inspect|see|check|copy|import|parse|compare\s+against|refer\s+to|look\s+at|use)\b/i;

/**
 * A bare mention is only a leak when the text POINTS the reader at the data.
 *
 *   "Read the ground truth file for the expected answers"  → leak (access verb)
 *   "Never access CERT or the final ground truth"          → fine (negated)
 *   "Stop if feedback includes CERT information"           → fine (a condition,
 *                                                             no access verb)
 *
 * Getting this wrong in the strict direction is not safe-by-default: it would
 * reject the role skills and the Manager's own boundary rules, i.e. forbid
 * writing down the very policy the guard exists to enforce.
 */
function isDirectedAccess(text, index) {
  const before = text.slice(Math.max(0, index - 80), index);
  if (NEGATION_RE.test(before)) return false;
  return ACCESS_VERB_RE.test(text.slice(Math.max(0, index - 40), index));
}

/** Keys whose array-of-objects payloads are row-level evidence, not knowledge. */
const ROW_PAYLOAD_KEYS = new Set(["rows", "mistakes", "expected", "labels", "fp_rows", "fn_rows"]);

/**
 * @param {unknown} value  any JSON-ish value, or a string (e.g. a file's contents)
 * @param {{groundTruthDir?: string|null, label?: string}} [options]
 * @returns {{ok: boolean, violations: Array<{field: string, reason: string}>}}
 */
export function checkLeak(value, options = {}) {
  const violations = [];
  const gtDir = options.groundTruthDir || null;

  const visitString = (text, field) => {
    for (const { re, reason } of HARD_TRIPWIRES) {
      if (re.test(text)) violations.push({ field, reason });
    }
    for (const { re, reason } of DIRECTED_TRIPWIRES) {
      for (const match of text.matchAll(re)) {
        if (isDirectedAccess(text, match.index)) {
          violations.push({ field, reason });
          break;
        }
      }
    }
    if (gtDir && text.includes(gtDir)) {
      violations.push({ field, reason: "contains an absolute path under the ground-truth dir" });
    }
  };

  const walk = (node, field) => {
    if (node === null || node === undefined) return;
    if (typeof node === "string") return visitString(node, field);
    if (typeof node !== "object") return;
    if (Array.isArray(node)) {
      node.forEach((item, i) => walk(item, `${field}[${i}]`));
      return;
    }
    for (const [key, child] of Object.entries(node)) {
      const childField = field ? `${field}.${key}` : key;
      if (ROW_PAYLOAD_KEYS.has(key) && Array.isArray(child)
          && child.some((x) => x && typeof x === "object")) {
        violations.push({ field: childField, reason: `per-row payload under "${key}"` });
      }
      walk(child, childField);
    }
  };

  walk(value, options.label || "");
  // Dedupe (field, reason) pairs so one repeated word does not produce 40 rows.
  const seen = new Set();
  const unique = violations.filter((v) => {
    const key = `${v.field}::${v.reason}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  return { ok: unique.length === 0, violations: unique };
}
