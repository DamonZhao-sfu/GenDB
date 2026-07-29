/**
 * Cross-table (pairwise) validation-set wiring.
 *
 * mmqa q2a/q7 join a structured table to the image table with the AI predicate itself
 * as the join condition, so the sampling unit is a PAIR and the val set keys on
 * "<structured_id>-<image_filename>" -- the composite key SemBench's own join ground
 * truth lists. These tests pin the two pure pieces of that wiring plus the invariant
 * that matters most: the agent must never be shown ground-truth rows.
 */
import assert from "node:assert";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import {
  aliasMap, predicateAliasForTable, predicateCol, predicateCols, predicateTextCols,
  renderFeedback, selectedKeyCol, validationPlan,
} from "../orchestrator.mjs";

// --- predicateCol: the text side comes from the PREDICATE, not the header ---
//
// ap_warrior.csv is ID,Finish,Race,Distance,Track,Condition. A header heuristic falls
// through to the last column, `Condition` ("Firm"/"Fast"), and CLIP then scores logos
// against track surface conditions. predicate.py already records `t.Track`.

const q2aSite = { aliases: ["t", "i"], columns: ["t.Track", "i.uri"], shape: "pairwise" };
const q7Site = { aliases: ["t", "i"], columns: ["t.Airlines", "i.uri"], shape: "pairwise" };

assert.strictEqual(predicateCol(q2aSite, "t"), "Track",
  "q2a's predicate reads the racetrack NAME, not Condition");
assert.strictEqual(predicateCol(q2aSite, "i"), "uri");
assert.strictEqual(predicateCol(q7Site, "t"), "Airlines");
assert.strictEqual(predicateCol(q2aSite, "zz"), null, "an unreferenced alias yields null");
assert.strictEqual(predicateCol({ columns: [] }, "t"), null, "no columns yields null");
assert.strictEqual(predicateCol(null, "t"), null, "a missing site yields null");

assert.deepStrictEqual(
  predicateTextCols({ columns: ["Destinations", "Airlines"] },
                    "row_id,Airlines,Destinations,Airport"),
  ["Airlines", "Destinations"],
  "q6 validation uses every predicate column in stable corpus-header order");

// EComm q8 hides its image table inside EXTERNAL_OBJECT_TRANSFORM. The orchestrator
// must still plan IMAGES as the image corpus and resolve the structured operand even
// though the prompt names the image first.
const q8Wrapped = `WITH product_selection AS (
  SELECT * FROM fashion_product_images.STYLES_DETAILS styles_details
)
SELECT * FROM product_selection styles_details
JOIN EXTERNAL_OBJECT_TRANSFORM(
  TABLE \`fashion_product_images.IMAGES\`, ['SIGNED_URL']) AS images
ON AI.IF(('fits?', images.ref, styles_details.productDisplayName, ' ',
          styles_details.productDescriptors.description.value),
         connection_id => 'x')`;
assert.deepStrictEqual(aliasMap(q8Wrapped, "ecomm"), {
  styles_details: "STYLES_DETAILS",
  images: "IMAGES",
});
const q8Site = {
  aliases: ["images", "styles_details"],
  bases: ["images", "styles_details"],
  columns: [
    "images.ref",
    "styles_details.productDisplayName",
    "styles_details.productDescriptors.description.value",
  ],
};
assert.strictEqual(predicateAliasForTable(q8Site, "STYLES_DETAILS"), "styles_details");
assert.deepStrictEqual(predicateCols(q8Site, "styles_details"), [
  "productDisplayName",
  "productDescriptors.description.value",
]);

// --- selectedKeyCol: the structured side's key is what the query SELECTs ----

const q2a = `SELECT t.ID, i.uri
FROM mmqa.ap_warrior t, mmqa.images i
WHERE AI.IF(STRUCT("...", t.Track, ", Image: ", i.uri));`;

const q7 = `SELECT t.Airlines, i.uri
FROM mmqa.tampa_international_airport t, mmqa.images i
WHERE AI.IF(STRUCT("...", t.Airlines, ", Image: ", i.uri));`;

assert.strictEqual(selectedKeyCol(q2a, "t"), "ID",
  "q2a projects t.ID -- Q2a.json's ground truth lists that value");
assert.strictEqual(selectedKeyCol(q7, "t"), "Airlines",
  "q7 projects t.Airlines -- Q7.json lists airline names, not row ids");
assert.strictEqual(selectedKeyCol(q2a, "i"), "uri",
  "the image side resolves too");

// Columns from the WHERE clause must not be mistaken for the SELECT key: q2a's
// predicate mentions t.Track, but the ground truth keys on t.ID.
assert.notStrictEqual(selectedKeyCol(q2a, "t"), "Track",
  "only the SELECT list is scanned, not the predicate");

assert.strictEqual(selectedKeyCol(q2a, "zz"), null, "an absent alias yields null");
assert.strictEqual(selectedKeyCol(q2a, null), null, "a missing alias yields null");
assert.strictEqual(selectedKeyCol("not sql at all", "t"), null,
  "unparseable SQL yields null rather than throwing");

// A CTE query still resolves against its outer SELECT.
const withCte = `WITH x AS (SELECT a.b FROM t.u a)
SELECT p.Name, i.uri FROM mmqa.people p, mmqa.images i WHERE AI.IF(...);`;
assert.strictEqual(selectedKeyCol(withCte, "p"), "Name");

// --- self-join candidate domain comes from SQL semantics -------------------

const planDir = mkdtempSync(resolve(tmpdir(), "semdb-self-plan-"));
const q7PlanSql = resolve(planDir, "q7.sql");
writeFileSync(q7PlanSql, `WITH product_selection AS (
  SELECT * FROM products styles_details WHERE price <= 500
)
SELECT p1.id, p2.id FROM product_selection p1 JOIN product_selection p2
ON AI.IF(('same?', p1.text, p2.text), connection_id => 'x')`);
const q7Plan = validationPlan(q7PlanSql, "ecomm", "q7");
assert.strictEqual(q7Plan.candidate.ordered, true);
assert.strictEqual(q7Plan.candidate.include_diagonal, true,
  "q7 has no ordinary inequality, so p-p is in the SQL pair domain");

const q9PlanSql = resolve(planDir, "q9.sql");
writeFileSync(q9PlanSql, `WITH product_selection AS (
  SELECT * FROM products styles_details WHERE price < 800
)
SELECT p1.id, p2.id FROM product_selection p1 JOIN product_selection p2
ON p1.id != p2.id
AND AI.IF(('same?', p1.text, p2.text), connection_id => 'x')`);
const q9Plan = validationPlan(q9PlanSql, "ecomm", "q9");
assert.strictEqual(q9Plan.candidate.include_diagonal, false,
  "q9's p1 != p2 condition excludes diagonal pairs");

// --- the invariant: pairwise feedback carries val labels, never ground truth ---

const pairFeedback = renderFeedback({
  status: "ok", f1: 0.5,
  metrics: { accuracy: 0.5, n: 2, correct: 1,
             quality: { precision: 1, recall: 0.5, f1: 0.667, tp: 1, fp: 0, fn: 1 } },
  diff: { n_mistakes: 1, mistakes: [
    { id: "5-117d500aaa630023c4038b8268b309c0.png", text: "Del Mar Racetrack",
      predicted: "false", expected: "true" },
  ] },
  history: [{ iter: 0, f1: 0.5, status: "ok", improved: true }],
});
assert.ok(pairFeedback.includes("PER-ROW INFERENCE"),
  "a pair-keyed diff takes the validation branch");
assert.ok(pairFeedback.includes("5-117d500aaa630023c4038b8268b309c0.png"),
  "the pair id reaches the agent so it can locate the mistake");
assert.ok(!pairFeedback.includes("FALSE POSITIVES")
  && !pairFeedback.includes("FALSE NEGATIVES"),
  "GROUND-TRUTH LEAK: the F1 block is computed from the ground truth and must never "
  + "appear while refining against a validation set");

console.log("test_pair_valset OK");
