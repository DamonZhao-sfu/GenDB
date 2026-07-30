/**
 * SemDB cross-run agent memory — public API.
 *
 * Layers and their storage follow GenDB exactly:
 *   L0 Query Instances        HAG   ┐ structural retrieval, pushed into the prompt
 *   L1 Query Templates        HAG   ┘
 *   L2 Sub-Structure Patterns Skill ┐
 *   L3 Operator Techniques    Skill │ semantic retrieval, discovered by the agent
 *   L4 Optimization Strategies Skill│
 *   L5 Performance Principles Skill ┘
 *
 * The skill root is isolated per memory directory, so "free discovery" means
 * free within SemDB's own learned namespace — never the repo root's GenDB skills
 * or the operator's global collection.
 */

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

import { initGraphDirs, readIndex } from "./graph.mjs";
import { initSkillRoot, lintAllSkills, publishRoleSkills, skillRootFor } from "./skills.mjs";

export { classifyQuery } from "./retrieval.mjs";
export { buildQuerySignature, planFeatures, signatureSimilarity } from "./signature.mjs";
export {
  listSkills,
  lintAllSkills,
  publishRoleSkills,
  recordSkillUsage,
  resolveSkillRoot,
  initSkillRoot,
  skillRootFor,
  skillsDirFor,
} from "./skills.mjs";

export const MEMORY_SCHEMA_VERSION = 1;

/**
 * Create/validate the memory directory and its skill root.
 *
 * Also re-lints every skill on the way in: a run that crashed mid-curation, or a
 * hand-edited skill, must not become part of three agents' context just because
 * it happens to sit in the directory.
 *
 * @returns {Promise<{ready: boolean, skillRoot: string|null, lint: object|null}>}
 */
export async function initMemory(memoryDir, config = {}) {
  if (!memoryDir) return { ready: false, skillRoot: null, lint: null };
  try {
    await mkdir(memoryDir, { recursive: true });
    await initGraphDirs(memoryDir);
    const skillRoot = skillRootFor(memoryDir);
    await initSkillRoot(skillRoot);
    await publishRoleSkills(skillRoot);

    const lint = await lintAllSkills(skillRoot, {
      skillNamePrefix: config.skillNamePrefix ?? "semdb-",
      groundTruthDir: config.groundTruthDir ?? null,
    });
    if (lint.quarantined.length) {
      for (const q of lint.quarantined) {
        console.warn(`[SemDB] memory: quarantined skill "${q.name}" — ${q.errors.join("; ")}`);
      }
    }

    const configPath = resolve(memoryDir, "config.json");
    const now = new Date().toISOString();
    let createdAt = now;
    if (existsSync(configPath)) {
      try {
        createdAt = JSON.parse(await readFile(configPath, "utf8")).createdAt || now;
      } catch {
        // a corrupt config.json is not worth failing a run over
      }
    }
    const index = await readIndex(memoryDir);
    await writeFile(configPath, JSON.stringify({
      schemaVersion: MEMORY_SCHEMA_VERSION,
      createdAt,
      updatedAt: now,
      skillRoot,
      stats: index.stats,
    }, null, 2));

    return { ready: true, skillRoot, lint };
  } catch (error) {
    console.error(`[SemDB] memory: failed to initialize (non-fatal): ${error.message}`);
    return { ready: false, skillRoot: null, lint: null };
  }
}

/** One-line status for the run header. */
export async function getMemorySummary(memoryDir) {
  if (!memoryDir || !existsSync(memoryDir)) return "memory: disabled";
  const index = await readIndex(memoryDir);
  const byLayer = Object.entries(index.stats.by_layer || {})
    .filter(([, count]) => count > 0)
    .map(([layer, count]) => `${layer}:${count}`)
    .join(", ");
  return `memory: ${index.stats.total} nodes (${byLayer || "empty"})`;
}
