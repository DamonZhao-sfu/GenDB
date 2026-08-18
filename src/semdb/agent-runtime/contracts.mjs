import { createHash, randomUUID } from "node:crypto";
import { chmod, readFile, rename, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { basename, dirname, isAbsolute, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import Ajv2020 from "ajv/dist/2020.js";

const here = dirname(fileURLToPath(import.meta.url));
const contractsDir = resolve(here, "..", "contracts");

const schemaFiles = {
  plan: "semantic-plan.schema.json",
  manifest: "candidate-manifest.schema.json",
  feedback: "iteration-feedback.schema.json",
  action: "optimizer-action.schema.json",
};

const ajv = new Ajv2020({ allErrors: true, strict: true });
const validators = {};
for (const [kind, filename] of Object.entries(schemaFiles)) {
  const schema = JSON.parse(await readFile(resolve(contractsDir, filename), "utf8"));
  validators[kind] = ajv.compile(schema);
}

function validationError(kind, path, errors) {
  const detail = (errors || [])
    .map((error) => `${error.instancePath || "/"} ${error.message}`)
    .join("; ");
  return new Error(
    `Invalid SemDB ${kind} at ${path}: ${detail || "schema validation failed"}`,
  );
}

function assertValidated(kind, path, value) {
  const validate = validators[kind];
  if (!validate(value)) throw validationError(kind, path, validate.errors);
  return value;
}

async function readJson(path, kind) {
  let value;
  try {
    value = JSON.parse(await readFile(path, "utf8"));
  } catch (error) {
    throw new Error(`Cannot read SemDB ${kind} at ${path}: ${error.message}`);
  }
  return assertValidated(kind, path, value);
}

function expectedValue(options, camel, snake) {
  return options?.[camel] ?? options?.[snake] ?? null;
}

/** Is `actual` the expected id wearing the run directory's decoration?
 *
 *  The agents are handed the bare id (`q9`) but also see the artifact path
 *  `runs/<out>/<benchmark>-<query>/iter_0/plan.json`, and occasionally write the
 *  DIRECTORY name (`ecomm-q9`, `movie-Q6`) into `query_id` instead. That is a
 *  transcription slip about a field the orchestrator itself owns — there is exactly one
 *  correct value and it was supplied in the prompt — so it must not cost the query its
 *  whole iteration budget.
 *
 *  Deliberately narrow: only a case difference, or a prefix ending at the LAST `-`/`_`/`/`
 *  whose tail is exactly the expected id. A plan carrying a DIFFERENT query's id (which is
 *  what a mis-seeded memory warm start would produce) still fails hard, because that
 *  artifact really does belong to another query.
 */
function idIsDecorated(actual, expected) {
  if (typeof actual !== "string" || typeof expected !== "string") return false;
  const a = actual.trim();
  const e = expected.trim();
  if (!a || !e) return false;
  if (a.toLowerCase() === e.toLowerCase()) return true;
  const cut = Math.max(a.lastIndexOf("-"), a.lastIndexOf("_"), a.lastIndexOf("/"));
  return cut > 0 && a.slice(cut + 1).toLowerCase() === e.toLowerCase();
}

/** Reconcile the id fields against what the caller expects, in place.
 *  Returns the list of repairs made; throws on any mismatch that is not a decoration. */
function reconcileExpectedIds(value, path, options = {}) {
  const repaired = [];
  const fields = [
    ["query_id", "queryId", "query_id"],
    ["candidate_id", "candidateId", "candidate_id"],
  ];
  for (const [field, camel, snake] of fields) {
    const expected = expectedValue(options, camel, snake);
    if (expected == null || value[field] === expected) continue;
    if (idIsDecorated(value[field], expected)) {
      repaired.push({ field, was: value[field], now: expected });
      value[field] = expected;
      continue;
    }
    throw new Error(
      `SemDB contract ${field.replace("_", " ")} mismatch at ${path}: `
      + `expected ${expected}, got ${value[field]}`,
    );
  }
  return repaired;
}

function assertExpectedIds(value, path, options = {}) {
  reconcileExpectedIds(value, path, options);
  return value;
}

/** Read a contract file, repair a decorated id, and persist the repair.
 *
 *  Writing back matters: `readAndValidateManifest` cross-checks the manifest's
 *  `query_id` against the plan FILE, later iterations re-read the plan as
 *  `previousPlanPath`, and memory backfill keys on it. An in-memory-only fix would
 *  surface as a manifest mismatch one phase later.
 */
async function readReconciled(path, kind, options) {
  const value = await readJson(path, kind);
  const repaired = reconcileExpectedIds(value, path, options);
  if (repaired.length) {
    await writeJsonAtomic(path, value);
    for (const { field, was, now } of repaired) {
      console.warn(
        `[SemDB] ${kind} at ${path} declared ${field}="${was}"; `
        + `normalized to "${now}" (the run directory name is not the query id).`,
      );
    }
  }
  return value;
}

/** Persist the deterministic lineage owned by the PGO loop.
 *
 *  A replan is a complete replacement document, so the model sees the initial-plan JSON
 *  example as well as the previous artifact. Qwen can correctly revise all semantic
 *  fields yet copy the example's `plan_version: 1, parent_plan_version: null`. Spending a
 *  nine-minute Planner call and then discarding the query over those two bookkeeping
 *  literals is both expensive and unnecessary: the orchestrator already has the sole
 *  authoritative previous version.
 *
 *  This does NOT repair query identity or semantic content. `readReconciled` has already
 *  rejected a plan belonging to another query, and the schema has already validated the
 *  document. We only stamp lineage to the one value the PGO state machine permits.
 */
async function reconcilePlanLineage(plan, path, previousPlan) {
  if (!previousPlan) return [];
  const expectedVersion = previousPlan.plan_version + 1;
  const expectedParent = previousPlan.plan_version;
  const repaired = [];
  if (plan.plan_version !== expectedVersion) {
    repaired.push({ field: "plan_version", was: plan.plan_version, now: expectedVersion });
    plan.plan_version = expectedVersion;
  }
  if (plan.parent_plan_version !== expectedParent) {
    repaired.push({
      field: "parent_plan_version",
      was: plan.parent_plan_version,
      now: expectedParent,
    });
    plan.parent_plan_version = expectedParent;
  }
  if (repaired.length) {
    await writeJsonAtomic(path, plan);
    const changes = repaired
      .map(({ field, was, now }) => `${field}=${JSON.stringify(was)} -> ${now}`)
      .join(", ");
    console.warn(
      `[SemDB] plan at ${path} emitted stale replan lineage; normalized ${changes}.`,
    );
  }
  return repaired;
}

export async function readAndValidatePlan(path, options = {}) {
  const plan = await readReconciled(path, "plan", options);
  if (options.requireCompilable) assertPlanGeneratable(plan);
  if (options.previousPlan) {
    await reconcilePlanLineage(plan, path, options.previousPlan);
    if (plan.plan_version !== options.previousPlan.plan_version + 1) {
      throw new Error(
        `Replanned SemDB plan_version must increment by one at ${path}`,
      );
    }
    if (plan.parent_plan_version !== options.previousPlan.plan_version) {
      throw new Error(
        `Replanned SemDB parent_plan_version mismatch at ${path}`,
      );
    }
  }
  return plan;
}

/** Primitives that read a SPECIFIC identity out of an image, as opposed to scoring a
 *  generic property. At least one of these must appear in an image plan that filters
 *  rows, otherwise the predicate cannot tell one named entity from another. */
const DISCRIMINATIVE_IMAGE_PRIMITIVES = new Set([
  "best_ocr_match", "best_ocr_match_detail", "read_text", "ocr_detail",
  "classify", "classify_detail", "classify_multi", "domain_classify",
  "topk_text", "topk_similar", "detect", "detect_detail", "detect_open",
  "dominant_colors", "pair_score", "score", "embed",
]);

/** Does a `verify_property` step build its property string from a DATA value?
 *
 *  This is the line between the two uses of the primitive. A constant property —
 *  `"a damaged car"`, `"a sports shoe"` — is a real generic visual question and CLIP
 *  answers it usefully. A property assembled from a column value —
 *  `"the logo of " + track_name` — is an identity question, and `verify_property`
 *  compares `prop` against `not prop`, which is positively biased: it returns true for
 *  nearly every image, scoring recall 1.0 at precision near zero.
 */
function verifyPropertyIsParameterized(step, helper) {
  const argNames = (Array.isArray(helper?.args) ? helper.args : [])
    .map((a) => String(a?.name || ""))
    .filter((n) => n && !/^(image|image_ref|resolved_image|img|patch)$/i.test(n));
  const inputs = (Array.isArray(step?.inputs) ? step.inputs : []).map(String);
  const template = String(step?.property_template || "");
  if (/\{[^}]+\}/.test(template)) return true;
  return inputs.some((raw) => {
    // Drop the image argument; only the property expression matters here.
    if (/^\s*(image|image_ref|resolved_image|img|patch)\b\s*[:=]?\s*\w*\s*$/i.test(raw)) return false;
    if (raw.includes("+") || /\{[^}]+\}/.test(raw)) return true;
    return argNames.some((n) => new RegExp(`\\b${n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`).test(raw));
  });
}

/** Lint an image plan for the failure mode that made mmqa q2a/q7 score P≈0.02.
 *
 *  Fires only on a data-parameterized `verify_property` with no discriminative sibling
 *  primitive anywhere in the DAG — an identity predicate that no amount of rewording can
 *  fix, so the optimizer would burn its whole budget on prompt tweaks. A constant-property
 *  `verify_property` is left alone; that is the primitive's legitimate use.
 *
 *  Returns human-readable findings (empty when the plan is fine). Advisory by design: the
 *  caller re-prompts the planner once rather than failing the query.
 */
/** Primitives that ALWAYS return one of `options` — argmax with no abstention.
 *
 *  `best_ocr_match*` is deliberately absent: it really does return "none" when the OCR
 *  text matches nothing, so a `!= "none"` guard on it is meaningful. */
const ARGMAX_ALWAYS_LABELS = new Set([
  "classify", "classify_detail", "best_lexical_match",
  "text_classify_detail",
]);

/** Does a filter site decide membership with an argmax primitive and no threshold?
 *
 *  `classify`/`classify_detail` cannot return "none", so on a FILTER — where most rows
 *  match none of the options — they label every row and the predicate degenerates to
 *  "true". mmqa q7 shipped exactly this: `if classify_detail(...)[0] != "none"` is a
 *  guard that is always true, the confidence it computed was discarded, and all 200
 *  images were assigned an airline (tp=0, fp=16). The plan even declared a
 *  `confidence_signal` — it just never named a threshold for the generator to apply.
 */
function argmaxFilterFindings(plan, helpers, filtersRows) {
  if (!filtersRows) return [];
  const offenders = [];
  for (const helper of helpers) {
    const steps = Array.isArray(helper.primitive_steps) ? helper.primitive_steps : [];
    const argmax = steps.map((s) => String(s?.primitive || ""))
      .filter((name) => ARGMAX_ALWAYS_LABELS.has(name));
    if (!argmax.length) continue;
    const signal = helper.confidence_signal;
    const threshold = signal && [signal.threshold, signal.min_confidence, signal.cutoff]
      .find((v) => typeof v === "number" && Number.isFinite(v));
    if (threshold === undefined) {
      offenders.push({ helper: helper.name || helper.helper_id || "<helper>", argmax });
    }
  }
  if (!offenders.length) return [];
  const detail = offenders
    .map((o) => `\`${o.helper}\` binds ${[...new Set(o.argmax)].map((a) => `\`${a}\``).join(", ")}`)
    .join("; ");
  return [
    `This plan decides row membership with an argmax primitive and no abstention `
    + `threshold (${detail}). \`classify\`/\`classify_detail\` ALWAYS return one of `
    + `\`options\` — they can never return "none" — so on a filter they label every row `
    + `and the predicate is true for the whole corpus (recall 1.0, precision ≈ the base `
    + `rate). A \`!= "none"\` guard on their result is always true and does nothing. `
    + `Either bind \`classify_or_none(image, options, min_conf)\`, or keep `
    + `\`classify_detail\` and record the numeric cutoff as \`confidence_signal.threshold\` `
    + `so the generator applies it and the optimizer can tune it.`,
  ];
}

/** A pair/join site whose helpers never see the OTHER side of the pair.
 *
 *  `AI.IF(track, image)` is a predicate over BOTH arguments. Collapsing the image to one
 *  label first — `infer(image, all_tracks) -> str` then `label == row.Track` in the
 *  solver — turns a join predicate into a classification task, and the damage is
 *  structural rather than a matter of model quality: mmqa q2a's ground truth is one
 *  Santa Anita wordmark that must match BOTH "Santa Anita Park" and "Oak Tree at Santa
 *  Anita Park", so a single argmax label can never satisfy more than one of them. That
 *  plan's ceiling was recall 3/5 before a single image was read. The version that kept
 *  the predicate binary (`matches(evidence, track)`, with containment rather than
 *  equality) scored recall 1.0.
 *
 *  Fires only when NO helper both returns a boolean and accepts a scalar from the other
 *  side. An evidence-then-decide split is fine and is what the good solver did: extract
 *  per-image evidence once, then decide per (evidence, value).
 */
function pairPredicateFindings(plan, helpers) {
  const sites = Array.isArray(plan.semantic_sites) ? plan.semantic_sites : [];
  const pairSites = sites.filter((s) => /^(ordered_)?pair$/i.test(String(s?.sampling_unit || ""))
    && /^bool(ean)?$/i.test(String(s?.output_type || "")));
  if (!pairSites.length || !helpers.length) return [];

  const isCollection = (t) => /\b(list|array|set|dict|map|sequence|\[\])/i.test(String(t || ""));
  const isImage = (t) => /imagepatch|image\b/i.test(String(t || ""));
  const returnsBool = (h) => /^bool(ean)?$/i.test(String(h?.return_type || "").trim());
  // The scalar carrying the other side: not the image, not the whole value space.
  const takesOtherSide = (h) => (Array.isArray(h?.args) ? h.args : [])
    .some((a) => !isImage(a?.type) && !isCollection(a?.type) && String(a?.type || "").trim());

  if (helpers.some((h) => returnsBool(h) && takesOtherSide(h))) return [];

  const shapes = helpers
    .map((h) => `\`${h.name || h.helper_id}(${(h.args || []).map((a) => a.name).join(", ")}) -> ${h.return_type}\``)
    .join(", ");
  return [
    `This is a ${pairSites[0].sampling_unit} site — the predicate is a function of BOTH `
    + `sides — but no helper both returns a boolean and takes the other side's value as a `
    + `scalar argument (${shapes}). A helper that takes the whole value space and returns `
    + `ONE label, compared with \`==\` in the solver, is a classification task, not a join `
    + `predicate: one input can then satisfy at most one value, so any ground-truth row `
    + `where a single image matches TWO values of the paired column is unreachable no `
    + `matter how accurate the vision is. Declare a helper that decides one (image, value) `
    + `pair and returns bool — extracting per-image evidence once and then deciding per `
    + `(evidence, value) is the cheap way to do it — and prefer containment/normalized `
    + `matching over string equality when one value can be a qualified form of another.`,
  ];
}

/** A filter/pair site that decides membership with `classify_multi`.
 *
 *  `classify_multi` scores each label INDEPENDENTLY — it is for attributes that can hold
 *  at once (damage types on one car). A value space where exactly one value applies (six
 *  racetrack names) has one answer, and an independent-label rule must return roughly
 *  half of them by construction. mmqa q2a bound it over the Track value space and every
 *  admitted image matched every track: 65 rows out, 0 correct. The same query answered
 *  with argmax + a high confidence threshold scores F1 0.83.
 */
function multilabelFilterFindings(plan, helpers, decidesRows) {
  if (!decidesRows) return [];
  const offenders = helpers.filter((h) => (Array.isArray(h.primitive_steps) ? h.primitive_steps : [])
    .some((s) => String(s?.primitive || "") === "classify_multi"));
  if (!offenders.length) return [];
  const names = offenders.map((h) => `\`${h.name || h.helper_id}\``).join(", ");
  return [
    `${names} decides row membership with \`classify_multi\`, which scores every option `
    + `INDEPENDENTLY. That is right for an attribute that holds several values at once, `
    + `and wrong for a value space where exactly one value applies: an independent-label `
    + `rule returns roughly half the value space for every input, so each accepted row `
    + `matches nearly every candidate value. Use \`classify_detail\` (argmax) with a HIGH `
    + `confidence threshold, or \`classify_or_none\`, and let the vetoes upstream do the `
    + `rejecting.`,
  ];
}

export function lintImagePlan(plan) {
  if (!plan || plan.modality !== "image") return [];
  if (plan.compilability?.class === "not_compilable") return [];
  const helpers = Array.isArray(plan.helper_dag) ? plan.helper_dag : [];
  const primitives = helpers
    .flatMap((h) => (Array.isArray(h.primitive_steps) ? h.primitive_steps : []))
    .map((s) => String(s?.primitive || ""));
  // A site that decides row membership is the one that has to discriminate; a plan that
  // only extracts an attribute for projection is judged by a different metric.
  const decidesRows = (Array.isArray(plan.semantic_sites) ? plan.semantic_sites : [])
    .some((s) => /^bool(ean)?$/i.test(String(s?.output_type || "")));
  // Runs independently of the verify_property rule below: a plan can bind a perfectly
  // discriminative primitive and still forget to let it abstain.
  const argmaxFindings = [
    ...pairPredicateFindings(plan, helpers),
    ...multilabelFilterFindings(plan, helpers, decidesRows),
    ...argmaxFilterFindings(plan, helpers, decidesRows),
  ];
  if (primitives.some((p) => DISCRIMINATIVE_IMAGE_PRIMITIVES.has(p))) return argmaxFindings;
  // A site that decides row membership is the one that has to discriminate; a plan that
  // only extracts an attribute for projection is judged by a different metric.
  if (!decidesRows) return argmaxFindings;
  const parameterized = helpers.some((h) => (Array.isArray(h.primitive_steps) ? h.primitive_steps : [])
    .some((s) => s?.primitive === "verify_property" && verifyPropertyIsParameterized(s, h)));
  if (!parameterized) return argmaxFindings;
  const findings = [...argmaxFindings, 
    "This plan decides row membership with a `verify_property` whose property string is "
    + "built from a data value, and no other visual primitive. `verify_property` compares "
    + "`prop` against `not prop` and is positively biased, so an identity property such as "
    + "`'the logo of ' + X` returns true for nearly every image — recall 1.0 at precision "
    + "near zero, and no rewording fixes it. Add a discriminative step that reads the "
    + "identity out of the image: `best_ocr_match_detail` against the value space read from "
    + "the runtime column, or a closed-set `classify_detail` over that value space. Keep "
    + "`verify_property` only as a cheap constant-property gate beside it.",
  ];
  if (helpers.length
      && helpers.every((h) => h.confidence_signal === null || h.confidence_signal === undefined)) {
    findings.push(
      "Every helper also declares `confidence_signal: null`, so the plan exposes no "
      + "threshold the optimizer can tune. Bind the `*_detail` variant of the discriminative "
      + "primitive and record its gating threshold in `confidence_signal`.",
    );
  }
  return findings;
}

export function assertPlanGeneratable(plan) {
  if (plan?.compilability?.class === "not_compilable") {
    throw new Error(
      `SemDB plan ${plan.query_id || "<unknown>"} is not_compilable and cannot enter the Generator`,
    );
  }
  return plan;
}

export async function readAndValidateManifest(path, options = {}) {
  const manifest = await readReconciled(path, "manifest", options);
  if (options.verifyHashes !== false) {
    const root = dirname(path);
    const artifactHashFields = {
      plan: "plan_sha256",
      helpers: "helpers_sha256",
      solver: "solver_sha256",
    };
    for (const [artifact, hashField] of Object.entries(artifactHashFields)) {
      const declaredPath = manifest.artifacts[artifact];
      const artifactPath = resolve(root, declaredPath);
      const relativePath = relative(root, artifactPath);
      if (
        isAbsolute(declaredPath)
        || relativePath === ".."
        || relativePath.startsWith("../")
      ) {
        throw new Error(
          `SemDB candidate ${manifest.candidate_id} ${artifact} artifact escapes its candidate directory`,
        );
      }
      if (!existsSync(artifactPath)) {
        throw new Error(
          `SemDB candidate ${manifest.candidate_id} is missing ${artifact} artifact ${artifactPath}`,
        );
      }
      const actual = await sha256File(artifactPath);
      if (actual !== manifest.hashes[hashField]) {
        throw new Error(
          `SemDB candidate ${manifest.candidate_id} ${artifact} hash mismatch: `
          + `expected ${manifest.hashes[hashField]}, got ${actual}`,
        );
      }
    }
    const artifactPlan = await readJson(
      resolve(root, manifest.artifacts.plan),
      "plan",
    );
    if (
      artifactPlan.query_id !== manifest.query_id
      || artifactPlan.plan_version !== manifest.plan_version
    ) {
      throw new Error(
        `SemDB candidate ${manifest.candidate_id} manifest does not match its plan query/version`,
      );
    }
  }
  return manifest;
}

export async function readAndValidateFeedback(path, options = {}) {
  const feedback = await readReconciled(path, "feedback", options);
  assertSafeDataBoundary(feedback.data_boundary);
  return feedback;
}

export async function readAndValidateOptimizerAction(path, options = {}) {
  return readReconciled(path, "action", options);
}

export function assertSafeDataBoundary(boundary) {
  if (!boundary
      || boundary.cert_accessed !== false
      || boundary.full_ground_truth_accessed !== false
      || !["select_validation", "none"].includes(boundary.source)) {
    throw new Error(
      "Optimizer feedback must be isolated from CERT and final ground truth",
    );
  }
  return boundary;
}

export async function writeJsonAtomic(path, value) {
  const tempPath = resolve(
    dirname(path),
    `.${basename(path)}.${process.pid}.${randomUUID()}.tmp`,
  );
  const text = `${JSON.stringify(value, null, 2)}\n`;
  await writeFile(tempPath, text, { encoding: "utf8", flag: "wx" });
  await rename(tempPath, path);
}

export async function writeTextAtomic(path, text, options = {}) {
  const tempPath = resolve(
    dirname(path),
    `.${basename(path)}.${process.pid}.${randomUUID()}.tmp`,
  );
  await writeFile(tempPath, String(text), { encoding: "utf8", flag: "wx" });
  if (options.mode != null) await chmod(tempPath, options.mode);
  await rename(tempPath, path);
}

export async function sha256File(path) {
  const data = await readFile(path);
  return createHash("sha256").update(data).digest("hex");
}

export async function finalizeCandidateManifest(path, draft, options = {}) {
  const root = dirname(path);
  const artifacts = {
    plan: draft?.artifacts?.plan || "plan.json",
    helpers: draft?.artifacts?.helpers,
    solver: draft?.artifacts?.solver,
  };
  for (const [kind, relativePath] of Object.entries(artifacts)) {
    if (!relativePath) {
      throw new Error(`Candidate manifest draft is missing artifacts.${kind}`);
    }
    const artifactPath = resolve(root, relativePath);
    if (!existsSync(artifactPath)) {
      throw new Error(`Generator did not create ${kind} artifact ${artifactPath}`);
    }
  }
  const manifest = {
    schema_version: "1.0",
    candidate_id: options.candidateId ?? draft.candidate_id,
    query_id: options.queryId ?? draft.query_id,
    iteration: options.iteration ?? draft.iteration,
    plan_version: options.planVersion ?? draft.plan_version,
    parent_candidate_id:
      options.parentCandidateId !== undefined
        ? options.parentCandidateId
        : (draft.parent_candidate_id ?? null),
    trigger_action: options.triggerAction ?? draft.trigger_action,
    artifacts,
    hashes: {
      plan_sha256: await sha256File(resolve(root, artifacts.plan)),
      helpers_sha256: await sha256File(resolve(root, artifacts.helpers)),
      solver_sha256: await sha256File(resolve(root, artifacts.solver)),
    },
  };
  assertExpectedIds(assertValidated("manifest", path, manifest), path, options);
  await writeJsonAtomic(path, manifest);
  return readAndValidateManifest(path, options);
}
