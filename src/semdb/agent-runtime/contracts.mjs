import { createHash, randomUUID } from "node:crypto";
import { readFile, rename, writeFile } from "node:fs/promises";
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

function assertExpectedIds(value, path, options = {}) {
  const queryId = expectedValue(options, "queryId", "query_id");
  const candidateId = expectedValue(options, "candidateId", "candidate_id");
  if (queryId != null && value.query_id !== queryId) {
    throw new Error(
      `SemDB contract query id mismatch at ${path}: expected ${queryId}, got ${value.query_id}`,
    );
  }
  if (candidateId != null && value.candidate_id !== candidateId) {
    throw new Error(
      `SemDB contract candidate id mismatch at ${path}: expected ${candidateId}, got ${value.candidate_id}`,
    );
  }
  return value;
}

export async function readAndValidatePlan(path, options = {}) {
  const plan = assertExpectedIds(await readJson(path, "plan"), path, options);
  if (options.requireCompilable) assertPlanGeneratable(plan);
  if (options.previousPlan) {
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
export function lintImagePlan(plan) {
  if (!plan || plan.modality !== "image") return [];
  if (plan.compilability?.class === "not_compilable") return [];
  const helpers = Array.isArray(plan.helper_dag) ? plan.helper_dag : [];
  const primitives = helpers
    .flatMap((h) => (Array.isArray(h.primitive_steps) ? h.primitive_steps : []))
    .map((s) => String(s?.primitive || ""));
  if (primitives.some((p) => DISCRIMINATIVE_IMAGE_PRIMITIVES.has(p))) return [];
  // A site that decides row membership is the one that has to discriminate; a plan that
  // only extracts an attribute for projection is judged by a different metric.
  const filtersRows = (Array.isArray(plan.semantic_sites) ? plan.semantic_sites : [])
    .some((s) => /^bool(ean)?$/i.test(String(s?.output_type || "")));
  if (!filtersRows) return [];
  const parameterized = helpers.some((h) => (Array.isArray(h.primitive_steps) ? h.primitive_steps : [])
    .some((s) => s?.primitive === "verify_property" && verifyPropertyIsParameterized(s, h)));
  if (!parameterized) return [];
  const findings = [
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
  const manifest = assertExpectedIds(await readJson(path, "manifest"), path, options);
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
  const feedback = assertExpectedIds(await readJson(path, "feedback"), path, options);
  assertSafeDataBoundary(feedback.data_boundary);
  return feedback;
}

export async function readAndValidateOptimizerAction(path, options = {}) {
  return assertExpectedIds(await readJson(path, "action"), path, options);
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
