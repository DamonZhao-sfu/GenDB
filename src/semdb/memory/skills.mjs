/**
 * The pull channel: an isolated Agent-SDK skill root that PGO agents discover
 * freely.
 *
 * Layout: <memoryDir>/skill-root/.claude/skills/<skill-name>/
 * The SDK resolves project settings from its cwd, so pointing an agent's cwd at
 * <memoryDir>/skill-root gives it exactly these skills and nothing else — not
 * GenDB's C++ optimization skills in the repo root, and not the operator's
 * personal global skills. Free discovery inside a curated namespace.
 *
 * Two kinds of skill live here:
 *   - role skills    (plan-semantic-query, generate-semantic-program,
 *                     optimize-semantic-program) — the procedures that used to be
 *                     statically bound into the system prompt;
 *   - memory skills  (semdb-*) — L2-L5 knowledge authored by the Memory Manager.
 *
 * lintSkill is the only gate between what the Manager writes and what every
 * future agent reads, so it fails closed: a skill that does not pass is moved to
 * quarantine/ and simply stops existing as far as discovery is concerned.
 */

import { cp, mkdir, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { checkLeak } from "./leak-guard.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const ROLE_SKILL_SRC = resolve(here, "..", "skills");

export const ROLE_SKILLS = [
  "plan-semantic-query",
  "generate-semantic-program",
  "optimize-semantic-program",
];

const SKILL_NAME_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const MAX_SKILL_BYTES = 64 * 1024;

/** cwd handed to the agent SDK so `.claude/skills` resolves to our namespace. */
export function skillRootFor(memoryDir) {
  return resolve(memoryDir, "skill-root");
}

/**
 * Where this run's skills live.
 *
 * The root cannot be conditional on --memory-dir: Generator still discovers its role
 * procedure here, and every tool agent may discover retrieval-selected learned skills.
 * Planner/Optimizer use their repo skill directly as the canonical system procedure.
 */
export function resolveSkillRoot({ memoryDir, out }) {
  return memoryDir ? skillRootFor(memoryDir) : resolve(out, "_skills");
}

export function skillsDirFor(skillRoot) {
  return resolve(skillRoot, ".claude", "skills");
}

export function quarantineDirFor(skillRoot) {
  return resolve(skillRoot, "_quarantine");
}

export async function initSkillRoot(skillRoot) {
  await mkdir(skillsDirFor(skillRoot), { recursive: true });
  await mkdir(quarantineDirFor(skillRoot), { recursive: true });
  return skillRoot;
}

/**
 * Publish the three role skills into the discoverable root.
 *
 * They used to be read from src/semdb/skills/ and injected verbatim into the
 * system prompt. Now they are discovered like any other skill, so they must
 * physically exist here. Copied (not symlinked) on every init so an edit to the
 * repo copy propagates and a stale copy cannot silently win.
 */
export async function publishRoleSkills(skillRoot) {
  const dest = skillsDirFor(skillRoot);
  await mkdir(dest, { recursive: true });
  const published = [];
  for (const name of ROLE_SKILLS) {
    const from = resolve(ROLE_SKILL_SRC, name);
    if (!existsSync(resolve(from, "SKILL.md"))) continue;
    await cp(from, resolve(dest, name), { recursive: true, force: true });
    published.push(name);
  }
  return published;
}

// ---------------------------------------------------------------------------
// Frontmatter
// ---------------------------------------------------------------------------

/** Minimal YAML frontmatter reader: `name` and `description` only. */
export function parseSkillFrontmatter(source) {
  const normalized = source.replace(/\r\n/g, "\n");
  if (!normalized.startsWith("---\n")) return { error: "missing YAML frontmatter" };
  const close = normalized.indexOf("\n---\n", 4);
  if (close < 0) return { error: "unterminated YAML frontmatter" };
  const header = normalized.slice(4, close);
  const body = normalized.slice(close + 5).trim();
  const values = {};
  let lastKey = null;
  for (const line of header.split("\n")) {
    if (!line.trim()) continue;
    const match = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (match) {
      lastKey = match[1];
      values[lastKey] = match[2].trim().replace(/^(['"])(.*)\1$/, "$2");
    } else if (lastKey && /^\s+\S/.test(line)) {
      // folded scalar continuation (`description: >` then an indented block)
      values[lastKey] = `${values[lastKey]} ${line.trim()}`.trim();
    } else {
      return { error: `unsupported frontmatter syntax: ${line.trim()}` };
    }
  }
  return { values, body };
}

// ---------------------------------------------------------------------------
// Lint
// ---------------------------------------------------------------------------

async function dirBytes(dir) {
  let total = 0;
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const p = resolve(dir, entry.name);
    if (entry.isDirectory()) total += await dirBytes(p);
    else total += (await stat(p)).size;
  }
  return total;
}

async function readAllText(dir) {
  const chunks = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const p = resolve(dir, entry.name);
    if (entry.isDirectory()) chunks.push(await readAllText(p));
    else if (/\.(md|json|py|txt)$/.test(entry.name)) {
      chunks.push(await readFile(p, "utf8"));
    }
  }
  return chunks.join("\n");
}

/**
 * @param {string} dir absolute path to one skill directory
 * @param {{expectedName?: string, requirePrefix?: string|null,
 *          requireEvidence?: boolean, groundTruthDir?: string|null}} [options]
 * @returns {Promise<{ok: boolean, errors: string[], name: string|null, description: string|null}>}
 */
export async function lintSkill(dir, options = {}) {
  const errors = [];
  const expectedName = options.expectedName || dir.split("/").filter(Boolean).pop();
  const skillFile = resolve(dir, "SKILL.md");
  if (!existsSync(skillFile)) {
    return { ok: false, errors: ["missing SKILL.md"], name: null, description: null };
  }

  const source = await readFile(skillFile, "utf8");
  const { values, body, error } = parseSkillFrontmatter(source);
  if (error) return { ok: false, errors: [error], name: null, description: null };

  const name = values.name || null;
  const description = values.description || null;
  const extraKeys = Object.keys(values).filter((k) => k !== "name" && k !== "description");
  if (extraKeys.length) errors.push(`unsupported frontmatter fields: ${extraKeys.join(", ")}`);
  if (!name) errors.push("missing frontmatter name");
  else {
    if (!SKILL_NAME_RE.test(name) || name.length >= 64) {
      errors.push(`name must be lowercase kebab-case under 64 chars (got "${name}")`);
    }
    if (name !== expectedName) errors.push(`name "${name}" does not match directory "${expectedName}"`);
    if (options.requirePrefix && !name.startsWith(options.requirePrefix)) {
      errors.push(`learned skills must be prefixed "${options.requirePrefix}" (got "${name}")`);
    }
  }
  if (!description || !description.trim()) errors.push("empty frontmatter description");
  else if (options.requireTriggerPhrasing !== false
           && !/^(use when|load when)/i.test(description.trim())) {
    // Semantic matching happens on the description alone; a learned skill whose
    // description does not state its trigger condition is never selected at the
    // right moment. Role skills are exempt: their prompts name them explicitly.
    errors.push('description must start with "Use when" or "Load when"');
  }
  if (!body) errors.push("empty skill body");

  if (options.requireEvidence) {
    const evidencePath = resolve(dir, "evidence.json");
    if (!existsSync(evidencePath)) errors.push("missing evidence.json");
    else {
      try {
        const evidence = JSON.parse(await readFile(evidencePath, "utf8"));
        const entries = Array.isArray(evidence) ? evidence : evidence.queries;
        if (!Array.isArray(entries) || entries.length === 0) {
          errors.push("evidence.json has no entries");
        } else if (!entries.every((e) => e && e.run_id && e.query_id)) {
          errors.push("every evidence entry needs run_id and query_id");
        }
      } catch (err) {
        errors.push(`evidence.json does not parse: ${err.message}`);
      }
    }
  }

  const bytes = await dirBytes(dir);
  if (bytes > MAX_SKILL_BYTES) {
    errors.push(`skill directory is ${bytes} bytes (max ${MAX_SKILL_BYTES})`);
  }

  const leak = checkLeak(await readAllText(dir), {
    groundTruthDir: options.groundTruthDir,
    label: "skill",
  });
  for (const v of leak.violations) errors.push(`ground-truth leak: ${v.reason}`);

  return { ok: errors.length === 0, errors, name, description };
}

export async function quarantineSkill(skillRoot, name, errors) {
  const from = resolve(skillsDirFor(skillRoot), name);
  const to = resolve(quarantineDirFor(skillRoot), name);
  await mkdir(quarantineDirFor(skillRoot), { recursive: true });
  await rm(to, { recursive: true, force: true });
  await rename(from, to);
  await writeFile(
    resolve(to, "lint_errors.json"),
    JSON.stringify({ name, errors, quarantined_at: new Date().toISOString() }, null, 2),
  );
  return to;
}

/**
 * Lint everything in the root, quarantining failures.
 * Role skills are exempt from the evidence and prefix requirements.
 */
export async function lintAllSkills(skillRoot, options = {}) {
  const dir = skillsDirFor(skillRoot);
  if (!existsSync(dir)) return { kept: [], quarantined: [] };
  const kept = [];
  const quarantined = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name.startsWith(".")) continue;
    const isRole = ROLE_SKILLS.includes(entry.name);
    const result = await lintSkill(resolve(dir, entry.name), {
      expectedName: entry.name,
      requirePrefix: isRole ? null : (options.skillNamePrefix || null),
      requireEvidence: !isRole,
      requireTriggerPhrasing: !isRole,
      groundTruthDir: options.groundTruthDir,
    });
    if (result.ok) kept.push({ name: entry.name, description: result.description, role: isRole });
    else {
      await quarantineSkill(skillRoot, entry.name, result.errors);
      quarantined.push({ name: entry.name, errors: result.errors });
    }
  }
  return { kept, quarantined };
}

/** Discoverable skills, without linting. */
export async function listSkills(skillRoot) {
  const dir = skillsDirFor(skillRoot);
  if (!existsSync(dir)) return [];
  const out = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name.startsWith(".")) continue;
    const skillFile = resolve(dir, entry.name, "SKILL.md");
    if (!existsSync(skillFile)) continue;
    const { values } = parseSkillFrontmatter(await readFile(skillFile, "utf8"));
    out.push({
      name: values?.name || entry.name,
      description: values?.description || "",
      dir: resolve(dir, entry.name),
      role: ROLE_SKILLS.includes(entry.name),
    });
  }
  return out.sort((a, b) => a.name.localeCompare(b.name));
}

// ---------------------------------------------------------------------------
// Catalog + usage
// ---------------------------------------------------------------------------

const approxTokens = (text) => Math.ceil(text.length / 4);

/**
 * Render the discovery hint. Every discoverable skill's description already
 * enters the agent's context whether or not it is loaded, so this stays a list
 * of one-liners; the bodies are what the agent pulls.
 */
export function renderCatalog(skills, maxTokens = 700) {
  const usable = skills.filter((s) => !s.role);
  if (!usable.length) return "";
  const head = [
    "## Available Memory Skills",
    "",
    "Learned from past runs. Load one with the Skill tool when its description matches your",
    "situation. They are advisory prior knowledge — the plan schema, the table metadata and",
    "the authoritative primitive API remain the source of truth, and they never contain",
    "ground-truth answers.",
    "",
  ];
  const lines = [];
  for (const s of usable) {
    const line = `- **${s.name}**: ${String(s.description).replace(/\s+/g, " ").slice(0, 180)}`;
    if (approxTokens([...head, ...lines, line].join("\n")) > maxTokens) {
      lines.push(`- _(${usable.length - lines.length} more skills not listed — catalog truncated)_`);
      break;
    }
    lines.push(line);
  }
  return [...head, ...lines, ""].join("\n");
}

/** Merge one agent call's skill usage into <outDir>/skill_usage.json. */
export async function recordSkillUsage(outDir, queryId, skillsUsed) {
  if (!outDir || !skillsUsed || Object.keys(skillsUsed).length === 0) return;
  const path = resolve(outDir, "skill_usage.json");
  let log = {};
  try {
    log = JSON.parse(await readFile(path, "utf8"));
  } catch {
    log = {};
  }
  if (!log[queryId]) log[queryId] = { skills: {} };
  for (const [skill, count] of Object.entries(skillsUsed)) {
    log[queryId].skills[skill] = (log[queryId].skills[skill] || 0) + count;
  }
  await mkdir(outDir, { recursive: true });
  await writeFile(path, JSON.stringify(log, null, 2));
}
