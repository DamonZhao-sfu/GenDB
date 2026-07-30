/**
 * Retrieval-key tests. The fixtures are the real SemBench query shapes (CTE +
 * EXTERNAL_OBJECT_TRANSFORM + multiple AI calls), because those are exactly the
 * forms a naive SQL scanner gets wrong.
 */

import assert from "node:assert/strict";

import {
  SIGNATURE_VERSION,
  WEIGHTED_SIMILARITY_CEILING,
  buildQuerySignature,
  jaccard,
  operatorKind,
  planAffinity,
  planFeatures,
  predicateTokens,
  projectionArity,
  relationalSkeleton,
  samplingUnitFor,
  signatureHash,
  signatureSimilarity,
  sqlTemplate,
} from "../memory/signature.mjs";

// --- fixtures ---------------------------------------------------------------

const MMQA_Q2A = `SELECT t.ID, i.uri
FROM mmqa.ap_warrior t, mmqa.images i
WHERE AI.IF(
  STRUCT(
    "You will be provided with a horse racetrack name and an image. ",
    "Determine if the image shows the logo of the racetrack. ",
    "Racetrack: ", t.Track, ", Image: ", i.uri
  ),
  connection_id => '<<connection>>'
);`;

// Same query, different literal parameters — must collapse to one template.
const MMQA_Q2A_VARIANT = MMQA_Q2A.replace("horse racetrack", "horse racetrack")
  .replace("'<<connection>>'", "'other-connection'");

const ECOMM_Q4 = `WITH product_selection AS (
  SELECT images.*
  FROM fashion_product_images.STYLES_DETAILS styles_details
  JOIN EXTERNAL_OBJECT_TRANSFORM(TABLE \`fashion_product_images.IMAGES\`, ['SIGNED_URL']) as images
    ON ARRAY_LAST(SPLIT(images.uri, '/')) = mapping.filename
  WHERE baseColour IN ('Black', 'Blue')
)
SELECT
  ARRAY_FIRST(SPLIT(ARRAY_LAST(SPLIT(images.uri, '/')), '.')) as id,
  AI.GENERATE(
    ('Extract the primary color of the product in the image: ', images.ref),
    connection_id => '<<connection>>'
  ).result AS category
FROM product_selection as images
;`;

const ECOMM_Q8 = `WITH product_selection AS (
  SELECT * FROM fashion_product_images.STYLES_DETAILS styles_details
)
SELECT ARRAY_FIRST(SPLIT(images.uri, '.')) AS id
FROM product_selection as styles_details
JOIN EXTERNAL_OBJECT_TRANSFORM(TABLE \`fashion_product_images.IMAGES\`, ['SIGNED_URL']) as images
  ON AI.IF(
    ('The image ', images.ref, ' fits the description: ', styles_details.productDisplayName),
    connection_id => '<<connection>>'
  )
;`;

const sig = (over = {}) => buildQuerySignature({
  query: "q", sql: MMQA_Q2A, nl: "match racetrack logos to images",
  benchmark: "mmqa",
  corpus: { table: "images", modality: "image" },
  tables: [{ table: "ap_warrior" }, { table: "images" }],
  ...over,
});

// --- determinism ------------------------------------------------------------

assert.deepEqual(sig(), sig(), "signature must be deterministic");
assert.equal(sig().signature_version, SIGNATURE_VERSION);

// --- literal elision --------------------------------------------------------

assert.equal(
  sqlTemplate("SELECT a FROM t WHERE x = 'abc' AND y = 42"),
  "SELECT a FROM t WHERE x = :p AND y = :p",
);
assert.equal(
  sig().template_hash,
  sig({ sql: MMQA_Q2A_VARIANT }).template_hash,
  "parameter variants of one query must share a template hash",
);
// An underscore-suffixed identifier is not a literal.
assert.ok(sqlTemplate("SELECT * FROM styles_2").includes("styles_2"));

// --- outer projection, not the CTE's --------------------------------------

assert.equal(projectionArity("SELECT a, b, f(c, d) FROM t"), 3);
assert.equal(projectionArity("SELECT * FROM t"), 1);
assert.equal(
  projectionArity(ECOMM_Q4), 2,
  "must read the outer SELECT (2 items), not the CTE's `images.*`",
);
assert.equal(projectionArity(ECOMM_Q8), 1);

// --- operator kind ----------------------------------------------------------

assert.equal(operatorKind(MMQA_Q2A, "mmqa"), "join",
  "AI predicate spans ap_warrior and images");
assert.equal(operatorKind(ECOMM_Q4, "ecomm"), "map",
  "single-table AI.GENERATE extraction is a map, not a filter");
assert.equal(operatorKind(ECOMM_Q8, "ecomm"), "join");
assert.equal(operatorKind("SELECT count(*) FROM mmqa.images i WHERE AI.IF(('x', i.uri), connection_id => 'c')", "mmqa"), "agg");
assert.equal(operatorKind("SELECT c, count(*) FROM mmqa.images i WHERE AI.IF(('x', i.uri), connection_id => 'c') GROUP BY c", "mmqa"), "group");
assert.equal(operatorKind("SELECT a FROM mmqa.images i WHERE AI.IF(('x', i.uri), connection_id => 'c')", "mmqa"), "filter");

assert.equal(samplingUnitFor("join"), "pair");
assert.equal(samplingUnitFor("group"), "group");
assert.equal(samplingUnitFor("filter"), "row");

// --- predicate tokens cover every AI call ----------------------------------

const multiCall = `SELECT a FROM mmqa.images i
WHERE AI.IF(('is the photo blurry', i.uri), connection_id => 'c')
  AND AI.IF(('does it depict a lighthouse', i.uri), connection_id => 'c')`;
const toks = predicateTokens(multiCall, null);
assert.ok(toks.includes("blurry"), "tokens from the first AI call");
assert.ok(toks.includes("lighthouse"), "tokens from the second AI call too");
assert.ok(!toks.includes("the"), "stopwords dropped");
assert.deepEqual(toks, [...toks].sort(), "tokens must be sorted for stable hashing");

// --- hard gates -------------------------------------------------------------

const imageJoin = sig();
const textJoin = sig({ corpus: { table: "docs", modality: "text" } });
assert.equal(signatureSimilarity(imageJoin, textJoin), 0,
  "modality disagreement is a veto, not a penalty");

const imageMap = buildQuerySignature({
  query: "q4", sql: ECOMM_Q4, nl: "extract the primary color",
  benchmark: "ecomm",
  corpus: { table: "IMAGES", modality: "image" },
  tables: [{ table: "IMAGES" }, { table: "STYLES_DETAILS" }],
});
assert.equal(signatureSimilarity(imageJoin, imageMap), 0,
  "operator-kind disagreement is a veto");

// identical → 1 via template hash
assert.equal(signatureSimilarity(imageJoin, sig()), 1);

// same shape, unrelated predicate → comparable but well below the exact bar
const otherImageJoin = buildQuerySignature({
  query: "qX",
  sql: MMQA_Q2A.replace("shows the logo of the racetrack", "depicts a wooden sailing vessel")
    .replace("horse racetrack name", "vessel registry name"),
  nl: "match vessels",
  benchmark: "mmqa",
  corpus: { table: "images", modality: "image" },
  tables: [{ table: "vessels" }, { table: "images" }],
});
const s = signatureSimilarity(imageJoin, otherImageJoin);
assert.ok(s > 0 && s < 0.98, `unrelated same-shape queries must not be exact (got ${s})`);

// version mismatch is not comparable
assert.equal(signatureSimilarity(imageJoin, { ...imageJoin, signature_version: 99 }), 0);

// --- the weighted path can never claim identity -----------------------------
// Regression: movie Q1/Q2 and Q3/Q4 share tables, AI predicate, arity and flags,
// so the weighted sum hit exactly 1.0 and they were classified `exact` — which
// would seed one query with the other's solver. Q2 adds a WHERE conjunct and Q4
// computes a ratio where Q3 counts; neither is the same query.

const MOVIE_PREDICATE = "AI.IF(('Determine if the following movie review is clearly positive, review: ', r.reviewText), connection_id => 'c')";
const movie = (sql) => buildQuerySignature({
  query: "Q", sql, nl: null, benchmark: "movie",
  corpus: { table: "reviews", modality: "text" },
  tables: [{ table: "reviews" }],
});

const movieQ1 = movie(`SELECT r.reviewId FROM movie.reviews r WHERE ${MOVIE_PREDICATE}`);
const movieQ2 = movie(`SELECT r.reviewId FROM movie.reviews r WHERE r.id = 'taken_3' AND ${MOVIE_PREDICATE}`);
const q1q2 = signatureSimilarity(movieQ1, movieQ2);
assert.ok(q1q2 < 0.98, `an extra WHERE conjunct must not read as the same query (got ${q1q2})`);
assert.ok(q1q2 > 0.55, `...but they are genuinely similar, so still structural (got ${q1q2})`);
assert.notEqual(movieQ1.template_hash, movieQ2.template_hash);

const movieQ3 = movie(`SELECT COUNT(*) AS n FROM movie.reviews r WHERE r.id = 'taken_3' AND ${MOVIE_PREDICATE}`);
const movieQ4 = movie(`SELECT CAST(SUM(CASE WHEN ${MOVIE_PREDICATE} THEN 1 ELSE 0 END) AS FLOAT64) / COUNT(*) AS ratio FROM movie.reviews r WHERE r.id = 'taken_3'`);
assert.ok(
  signatureSimilarity(movieQ3, movieQ4) < 0.98,
  "counting rows and computing a ratio are different questions",
);

assert.equal(WEIGHTED_SIMILARITY_CEILING, 0.95);
assert.ok(
  signatureSimilarity(movieQ1, { ...movieQ1, template_hash: "different0000000" }) <= WEIGHTED_SIMILARITY_CEILING,
  "only an identical template hash may score 1.0",
);

// The skeleton ignores AI-call text (that is the predicate channel's job) but
// keeps the relational structure.
const skelA = relationalSkeleton(`SELECT a FROM t WHERE ${MOVIE_PREDICATE}`);
const skelB = relationalSkeleton("SELECT a FROM t WHERE AI.IF(('something else entirely', x.y), connection_id => 'c')");
assert.deepEqual(skelA, skelB, "predicate wording must not perturb the relational skeleton");
assert.ok(relationalSkeleton("SELECT a FROM t WHERE b = 1").includes("ai_call") === false);

// --- jaccard edge cases -----------------------------------------------------

assert.equal(jaccard([], []), 1);
assert.equal(jaccard([], ["a"]), 0);
assert.equal(jaccard(["a", "b"], ["b", "c"]), 1 / 3);

// --- hash stability ---------------------------------------------------------

assert.equal(signatureHash(imageJoin), signatureHash(sig()));
assert.notEqual(signatureHash(imageJoin), signatureHash(imageMap));

// --- plan features ----------------------------------------------------------

const plan = {
  semantic_sites: [
    { site_id: "s1", operator: "sem_join", sampling_unit: "pair",
      output_type: "boolean", value_space: ["true", "false"] },
  ],
  helper_dag: [
    { helper_id: "h1", name: "logo_name", args: [], return_type: "str",
      depends_on: [], confidence_signal: null,
      primitive_steps: [{ primitive: "best_ocr_match" }, { primitive: "classify" }] },
  ],
  relational_plan: ["scan images", "join on predicted name"],
  compilability: { class: "exact", obligations: [], unresolved: [] },
};
const pf = planFeatures(plan);
assert.deepEqual(pf.primitives, ["best_ocr_match", "classify"]);
assert.deepEqual(pf.site_operators, ["sem_join"]);
assert.deepEqual(pf.value_space_kinds, ["enum"]);
assert.equal(pf.compilability, "exact");
assert.equal(planAffinity(pf, pf), 1);
assert.ok(planAffinity(pf, planFeatures({ semantic_sites: [], helper_dag: [] })) < 1);

// a plan with no sites/helpers must not throw
assert.doesNotThrow(() => planFeatures({}));

console.log("test_memory_signature: PASS");
