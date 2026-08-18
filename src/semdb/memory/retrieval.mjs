/**
 * Tiered retrieval over L0/L1, plus the discovery hint for the skill layers.
 *
 * Mirrors GenDB's three-tier split (exact / structural / novel) because the tier
 * drives control flow, not just prompt text: an exact match seeds iteration 0
 * with a previously promoted candidate, a structural match only injects advice,
 * and a novel query runs exactly as it does today.
 */

import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { getConnectedNodes, getNodesByLayer } from "./graph.mjs";
import { renderInlineSkills, renderPreInjection } from "./render.mjs";
import { buildQuerySignature, signatureSimilarity } from "./signature.mjs";
import { listSkills, renderCatalog, skillRootFor, skillsDirFor } from "./skills.mjs";

const ROLES = ["planner", "generator", "optimizer"];

/**
 * Is this objective good enough to imitate?
 *
 * A backfilled run that scored 0 is still data, but it is not a reference: warm
 * starting from it seeds a known-failed solver, and describing its plan shape as
 * one that "succeeded" is simply false. Below the floor a reference is inverted
 * into a warning instead of being dropped, since knowing which shape failed is
 * itself useful.
 */
export function isUsefulObjective(objective, config = {}) {
  const value = Number(objective?.value);
  if (!Number.isFinite(value)) return false;
  const floor = config.minReferenceObjective ?? 0;
  return objective?.direction === "minimize" ? true : value > floor;
}

/** Direction-aware "is a better than b". */
function betterObjective(a, b) {
  const av = a?.value;
  const bv = b?.value;
  if (av === null || av === undefined || !Number.isFinite(Number(av))) return false;
  if (bv === null || bv === undefined || !Number.isFinite(Number(bv))) return true;
  return a.direction === "minimize" ? Number(av) < Number(bv) : Number(av) > Number(bv);
}

/**
 * A stored warm start is only usable if its files still exist. Runs get deleted,
 * moved and rewritten; a dangling reference must degrade to a cold start rather
 * than fail the query.
 */
function resolveWarmStart(l0) {
  const promoted = l0?.content?.promoted;
  if (!promoted) return null;
  const planPath = promoted.plan_path;
  const helpersPath = promoted.helpers_path;
  const solverPath = promoted.solver_path;
  if (!planPath || !existsSync(planPath)) return null;
  return {
    planPath,
    helpersPath: helpersPath && existsSync(helpersPath) ? helpersPath : null,
    solverPath: solverPath && existsSync(solverPath) ? solverPath : null,
    candidateId: promoted.candidate_id || null,
  };
}

/**
 * Classify one query against memory.
 *
 * @param {object} planObj  a planQuery() result ({query, sql, nl, tables, corpus, ...})
 * @param {{benchmark:string, agentProvider:string}} args
 * @param {string} memoryDir
 * @param {object} config   defaults.memory
 */
export async function classifyQuery(planObj, args, memoryDir, config = {}) {
  const exactMin = config.exactMatchMinScore ?? 0.98;
  const structuralMin = config.structuralMatchMinScore ?? 0.55;
  const preCap = config.maxPreInjectionTokens ?? 3000;
  const catalogCap = config.maxCatalogTokens ?? 700;

  const signature = buildQuerySignature({ ...planObj, benchmark: args.benchmark });

  // --- match L1 templates, same benchmark only ------------------------------
  // Instances and templates never transfer across benchmarks: the corpus, the
  // tables and the label semantics all change. Only the skill layers generalize.
  const candidates = (await getNodesByLayer(1, memoryDir))
    .filter((node) => node.benchmark === args.benchmark && node.signature)
    .map((node) => ({ node, score: signatureSimilarity(signature, node.signature) }))
    .sort((a, b) => b.score - a.score);

  const best = candidates[0] || null;
  const score = best?.score ?? 0;
  let tier = "novel";
  let l1 = null;
  if (score >= exactMin) {
    tier = "exact";
    l1 = best.node;
  } else if (score >= structuralMin) {
    tier = "structural";
    l1 = best.node;
  }

  // --- best instance of that template ---------------------------------------
  let l0 = null;
  if (l1) {
    const instances = await getConnectedNodes(l1.id, "instance_of", memoryDir, "incoming");
    for (const node of instances) {
      if (!l0 || betterObjective(node.content?.objective, l0.content?.objective)) l0 = node;
    }
  }

  const referenceIsGood = isUsefulObjective(l0?.content?.objective, config);
  // Never seed iteration 0 from a candidate that did not work.
  const warmStart = tier === "exact" && config.warmStart !== false && referenceIsGood
    ? resolveWarmStart(l0)
    : null;

  // --- push channel ---------------------------------------------------------
  const blocks = {};
  for (const role of ROLES) {
    blocks[role] = renderPreInjection(
      { tier, score, l1, l0, role, warmStart, referenceIsGood }, preCap,
    );
  }

  // --- pull channel ---------------------------------------------------------
  const skillRoot = config.skillRoot || skillRootFor(memoryDir);
  const skills = await listSkills(skillRoot);
  const catalog = renderCatalog(skills, catalogCap);

  // Resolve the edge-linked learned skills once. Agent runtimes use this list to expose
  // only relevant pull-channel knowledge instead of advertising every role and memory
  // skill on every turn.
  const relevantSkillNames = [];
  if (l1) {
    const linked = [];
    for (const type of ["uses_operator", "exhibits_pattern"]) {
      linked.push(...(await getConnectedNodes(l1.id, type, memoryDir, "outgoing")));
    }
    for (const name of linked.map((n) => n.skill_name).filter(Boolean)) {
      if (!relevantSkillNames.includes(name)) relevantSkillNames.push(name);
      if (relevantSkillNames.length >= (config.maxInlineSkills ?? 3)) break;
    }
  }

  // Codex has no Skill tool; inline the edge-linked skills instead so the two
  // providers see the same knowledge.
  let inlineSkills = "";
  if (args.agentProvider === "codex" && relevantSkillNames.length) {
    const bodies = [];
    for (const name of relevantSkillNames) {
      const path = resolve(skillsDirFor(skillRoot), name, "SKILL.md");
      if (!existsSync(path)) continue;
      bodies.push({ name, body: await readFile(path, "utf8") });
    }
    inlineSkills = renderInlineSkills(bodies, config.maxInlineSkillTokens ?? 2000);
  }

  return {
    tier,
    score,
    signature,
    matchedL1: l1?.id ?? null,
    matchedL0: l0?.id ?? null,
    warmStart,
    referenceIsGood,
    blocks,
    catalog,
    inlineSkills,
    relevantSkillNames,
    skillsAvailable: skills.filter((s) => !s.role).length,
    injectedTokens: {
      pre: Math.ceil((blocks.planner || "").length / 4),
      catalog: Math.ceil(catalog.length / 4),
      inline: Math.ceil(inlineSkills.length / 4),
    },
  };
}
