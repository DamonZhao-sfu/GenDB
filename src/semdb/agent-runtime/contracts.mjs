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
