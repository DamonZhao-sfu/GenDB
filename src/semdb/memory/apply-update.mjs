/**
 * Deterministic writer for the Memory Manager's proposal.
 *
 * GenDB lets its Memory Manager write graph JSON directly and repairs the damage
 * afterwards (patchL1Signatures exists precisely because an LLM authored the
 * retrieval key). Here the agent proposes and code writes, so there is exactly one
 * enforcement point for each rule:
 *
 *   leak guard          — no ground truth reaches a node, ever
 *   evidence guard      — no skill layer without a real, in-run measurement
 *   signature ownership — recomputed from the SQL, never taken from the agent
 *   headroom check      — an improvement claim must survive the same arithmetic
 *                         the loop used, so a mis-labeled run cannot inflate memory
 *   lint gate           — a skill that fails lint never becomes discoverable
 *   prune               — the discoverable namespace stays bounded
 */

import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import Ajv2020 from "ajv/dist/2020.js";

import { checkLeak } from "./leak-guard.mjs";
import {
  addEdge,
  deleteNode,
  getNodesByLayer,
  updateIndex,
  writeNode,
} from "./graph.mjs";
import { buildQuerySignature, planFeatures, signatureHash, signatureSimilarity } from "./signature.mjs";
import { lintSkill, quarantineSkill, skillsDirFor } from "./skills.mjs";

const here = dirname(fileURLToPath(import.meta.url));

const DATA_BOUNDARY = {
  source: "select_validation",
  cert_accessed: false,
  full_ground_truth_accessed: false,
};

const LAYER_EDGE = {
  2: "exhibits_pattern",       // L1 → L2
  3: "uses_operator",          // L1 → L3
  4: "uses_operator",          // L1 → L4 (an L4 is still reached from the template)
  5: "exemplifies_principle",  // L4 → L5; from a template it is informational only
};

const ajv = new Ajv2020({ allErrors: true, strict: true });
const updateSchema = JSON.parse(
  await readFile(resolve(here, "..", "contracts", "memory-update.schema.json"), "utf8"),
);
const validateUpdateSchema = ajv.compile(updateSchema);

export function validateUpdate(update) {
  if (!validateUpdateSchema(update)) {
    const detail = (validateUpdateSchema.errors || [])
      .map((e) => `${e.instancePath || "/"} ${e.message}`)
      .join("; ");
    throw new Error(`Invalid memory update: ${detail}`);
  }
  return update;
}

/**
 * Improvement as a fraction of the REMAINING headroom, not a raw delta.
 *
 * For a bounded maximize objective, 0.90 → 0.95 is a far larger achievement than
 * 0.20 → 0.25, and raw deltas rank them the other way round. The Manager is told
 * to use this rule; recomputing it here is what makes the instruction binding.
 */
export function headroomGain(before, after, direction = "maximize", ceiling = 1) {
  const a = Number(after);
  if (!Number.isFinite(a)) return 0;
  // `Number(null)` is 0, not NaN — treating a missing baseline as a real 0 would
  // turn "no previous measurement" into "improved from zero" and inflate the gain.
  if (before === null || before === undefined || before === "") return 1;
  const b = Number(before);
  if (!Number.isFinite(b)) return 1;            // first measurement is all upside
  const epsilon = 1e-9;
  return direction === "minimize"
    ? (b - a) / Math.max(epsilon, b)
    : (a - b) / Math.max(epsilon, ceiling - b);
}

function idFragment(text) {
  return String(text).replace(/[^A-Za-z0-9]/g, "_");
}

/**
 * @param {object} update  the agent's proposal (schema-validated here)
 * @param {{memoryDir:string, skillRoot:string, runId:string, benchmark:string,
 *          queries:Record<string,{sql:string,nl:string|null,corpus:object,tables:Array,
 *                                 objective:object,iterations:number,replans:number,
 *                                 actionCounts:object,plan:object|null,promoted:object,
 *                                 metricFamily:string|null,runDir:string}>,
 *          config?:object, groundTruthDir?:string|null}} ctx
 */
export async function applyMemoryUpdate(update, ctx) {
  const config = ctx.config || {};
  const summary = {
    run_id: ctx.runId,
    benchmark: ctx.benchmark,
    classifications: { NOVEL_SUCCESS: 0, SIGNIFICANT_IMPROVEMENT: 0, FAMILIAR: 0 },
    nodes_created: {},
    nodes_updated: 0,
    edges_created: 0,
    skills_accepted: [],
    rejected: [],
  };

  validateUpdate(update);

  // 1. Leak guard — one violation rejects the whole proposal. Partially applying an
  //    update that tried to smuggle labels in is not a safe middle ground.
  const leak = checkLeak(update, { groundTruthDir: ctx.groundTruthDir, label: "update" });
  if (!leak.ok) {
    for (const v of leak.violations) {
      summary.rejected.push({ reason: `gt_leak: ${v.reason}`, field: v.field });
    }
    summary.applied = false;
    return summary;
  }

  const exactMin = config.exactMatchMinScore ?? 0.98;
  const headroomMin = config.differentialHeadroomThreshold ?? 0.3;
  const templateIdByQuery = new Map();

  // 2. Queries → L0 (+ L1 by merge)
  for (const proposal of update.queries || []) {
    const meta = ctx.queries?.[proposal.query_id];
    if (!meta) {
      summary.rejected.push({
        reason: "unknown_query",
        detail: `${proposal.query_id} was not part of this run`,
      });
      continue;
    }
    summary.classifications[proposal.classification] += 1;
    if (proposal.classification === "FAMILIAR") continue;

    // 3. Signature ownership: recomputed from the run's own SQL.
    const signature = buildQuerySignature({
      query: proposal.query_id,
      sql: meta.sql,
      nl: meta.nl,
      benchmark: ctx.benchmark,
      corpus: meta.corpus,
      tables: meta.tables,
    });

    // 4. Improvement claims must survive the loop's own arithmetic.
    const verifiedBreakthroughs = [];
    for (const b of proposal.breakthroughs || []) {
      const gain = headroomGain(b.before, b.after, meta.objective?.direction);
      if (gain + 1e-9 >= headroomMin) verifiedBreakthroughs.push({ ...b, headroom_gain: gain });
      else {
        summary.rejected.push({
          reason: "breakthrough_below_threshold",
          detail: `${proposal.query_id} iter ${b.iteration}: headroom gain ${gain.toFixed(3)}`
            + ` < ${headroomMin}`,
        });
      }
    }

    const l0Id = `L0_${ctx.benchmark}_${idFragment(proposal.query_id)}_${signatureHash(signature)}`;
    await writeNode({
      id: l0Id,
      layer: 0,
      signature_version: signature.signature_version,
      benchmark: ctx.benchmark,
      signature,
      summary: `${ctx.benchmark} ${proposal.query_id}`
        + ` (${meta.objective?.name ?? "objective"}=${meta.objective?.value ?? "n/a"})`,
      content: {
        query_id: proposal.query_id,
        sql: meta.sql,
        nl: meta.nl,
        modality: signature.modality,
        corpus_table: meta.corpus?.table ?? null,
        objective: meta.objective ?? null,
        metric_family: meta.metricFamily ?? null,
        iterations: meta.iterations ?? 0,
        replans: meta.replans ?? 0,
        action_counts: meta.actionCounts ?? null,
        plan_features: meta.plan ? planFeatures(meta.plan) : null,
        promoted: meta.promoted ?? null,
        breakthroughs: verifiedBreakthroughs,
        run_id: ctx.runId,
        source_run_dir: meta.runDir ?? null,
      },
      data_boundary: DATA_BOUNDARY,
      tags: [signature.modality, signature.operator_kind, ctx.benchmark],
    }, ctx.memoryDir);
    summary.nodes_created.L0 = (summary.nodes_created.L0 || 0) + 1;

    // 5. Merge into a template, creating one only if nothing matches.
    const existingTemplates = await getNodesByLayer(1, ctx.memoryDir);
    let template = existingTemplates.find(
      (node) => node.benchmark === ctx.benchmark
        && signatureSimilarity(signature, node.signature) >= exactMin,
    );
    const features = meta.plan ? planFeatures(meta.plan) : null;
    const skeleton = features
      ? {
        sampling_unit: features.sampling_units?.[0] ?? signature.sampling_unit,
        helper_names: features.helper_names ?? [],
        primitives: features.primitives ?? [],
        relational_ops: features.relational_ops ?? [],
      }
      : null;
    const mergeList = (existing = [], incoming = [], cap = 12) => [
      ...new Set([...existing, ...incoming.filter(Boolean)]),
    ].slice(0, cap);

    if (template) {
      template.content.proven_strategies = mergeList(
        template.content.proven_strategies, proposal.template?.proven_strategies,
      );
      template.content.anti_patterns = mergeList(
        template.content.anti_patterns, proposal.template?.anti_patterns,
      );
      if (skeleton) template.content.plan_skeleton = skeleton;
      template.content.instance_count = (template.content.instance_count || 0) + 1;
      template.content.instance_queries = mergeList(
        template.content.instance_queries, [proposal.query_id], 40,
      );
      await writeNode(template, ctx.memoryDir);
      summary.nodes_updated += 1;
    } else {
      const templateId = `L1_${ctx.benchmark}_${signature.operator_kind}${signature.modality}`
        + `_${signatureHash(signature)}`;
      template = {
        id: templateId,
        layer: 1,
        signature_version: signature.signature_version,
        benchmark: ctx.benchmark,
        signature,
        summary: `${ctx.benchmark} ${signature.modality} ${signature.operator_kind} template`,
        content: {
          template_name: `${signature.modality}-${signature.operator_kind}`,
          proven_strategies: mergeList([], proposal.template?.proven_strategies),
          anti_patterns: mergeList([], proposal.template?.anti_patterns),
          plan_skeleton: skeleton,
          best_instance_id: l0Id,
          instance_count: 1,
          instance_queries: [proposal.query_id],
          metric_families: meta.metricFamily ? [meta.metricFamily] : [],
          origin: "memory_manager",
        },
        data_boundary: DATA_BOUNDARY,
        tags: [signature.modality, signature.operator_kind, ctx.benchmark],
      };
      await writeNode(template, ctx.memoryDir);
      summary.nodes_created.L1 = (summary.nodes_created.L1 || 0) + 1;
    }
    templateIdByQuery.set(proposal.query_id, template.id);
    if (await addEdge({ source: l0Id, target: template.id, type: "instance_of" }, ctx.memoryDir)) {
      summary.edges_created += 1;
    }
  }

  // 6. Skills → L2-L5 reference nodes, but only if the directory passes lint.
  for (const skill of update.skills || []) {
    const dir = resolve(skillsDirFor(ctx.skillRoot), skill.name);
    if (!existsSync(dir)) {
      summary.rejected.push({ reason: "skill_missing", detail: skill.name });
      continue;
    }
    const lint = await lintSkill(dir, {
      expectedName: skill.name,
      requirePrefix: config.skillNamePrefix ?? "semdb-",
      requireEvidence: true,
      requireTriggerPhrasing: true,
      groundTruthDir: ctx.groundTruthDir,
    });
    if (!lint.ok) {
      await quarantineSkill(ctx.skillRoot, skill.name, lint.errors);
      summary.rejected.push({ reason: "skill_lint", detail: `${skill.name}: ${lint.errors.join("; ")}` });
      continue;
    }

    // Evidence must name queries that really ran in this run.
    const evidence = [];
    for (const e of skill.evidence) {
      const meta = ctx.queries?.[e.query_id];
      if (!meta) continue;
      evidence.push({
        run_id: ctx.runId,
        benchmark: ctx.benchmark,
        query_id: e.query_id,
        objective: meta.objective ?? { name: "objective", value: null, direction: "maximize" },
        before: e.before ?? null,
        after: e.after ?? null,
        iteration: e.iteration ?? null,
        action: e.action ?? null,
      });
    }
    if (!evidence.length) {
      await quarantineSkill(ctx.skillRoot, skill.name, ["no evidence from queries in this run"]);
      summary.rejected.push({ reason: "skill_evidence", detail: skill.name });
      continue;
    }

    const nodeId = `L${skill.layer}_${idFragment(skill.name)}`;
    await writeNode({
      id: nodeId,
      layer: skill.layer,
      benchmark: null,
      signature: null,
      skill_name: skill.name,
      skill_path: dir,
      summary: skill.summary,
      content: { description: skill.description ?? "", source_queries: evidence.map((e) => e.query_id) },
      evidence,
      evidence_count: evidence.length,
      data_boundary: DATA_BOUNDARY,
      tags: [`L${skill.layer}`],
    }, ctx.memoryDir);
    summary.nodes_created[`L${skill.layer}`] = (summary.nodes_created[`L${skill.layer}`] || 0) + 1;
    summary.skills_accepted.push(skill.name);

    for (const queryId of skill.links_queries || evidence.map((e) => e.query_id)) {
      const templateId = templateIdByQuery.get(queryId);
      if (!templateId) continue;
      if (await addEdge(
        { source: templateId, target: nodeId, type: LAYER_EDGE[skill.layer] },
        ctx.memoryDir,
      )) summary.edges_created += 1;
    }
  }

  // 7. Prune, then rebuild the index once.
  summary.pruned = await pruneLayers(ctx, config);
  const index = await updateIndex(ctx.memoryDir);
  summary.applied = true;
  summary.index = index.stats;
  return summary;
}

/**
 * Keep each (layer, benchmark) bounded, evicting the least-evidenced first.
 *
 * This matters most for the skill layers: every discoverable skill's description
 * enters all three agents' context whether or not it is loaded, so an unbounded
 * namespace is a permanent tax rather than a growing asset.
 */
async function pruneLayers(ctx, config) {
  const maxNodes = config.maxNodesPerLayer ?? 40;
  const maxSkills = config.maxSkills ?? 30;
  const pruned = [];

  for (const layer of [0, 1]) {
    const nodes = await getNodesByLayer(layer, ctx.memoryDir);
    const byBenchmark = new Map();
    for (const node of nodes) {
      const key = node.benchmark || "_";
      if (!byBenchmark.has(key)) byBenchmark.set(key, []);
      byBenchmark.get(key).push(node);
    }
    for (const group of byBenchmark.values()) {
      if (group.length <= maxNodes) continue;
      group.sort((a, b) => String(a.updated_at).localeCompare(String(b.updated_at)));
      for (const node of group.slice(0, group.length - maxNodes)) {
        await deleteNode(node.id, ctx.memoryDir);
        pruned.push(node.id);
      }
    }
  }

  const skillNodes = [];
  for (const layer of [2, 3, 4, 5]) {
    skillNodes.push(...(await getNodesByLayer(layer, ctx.memoryDir)));
  }
  if (skillNodes.length > maxSkills) {
    skillNodes.sort((a, b) => (a.evidence_count || 0) - (b.evidence_count || 0)
      || String(a.updated_at).localeCompare(String(b.updated_at)));
    for (const node of skillNodes.slice(0, skillNodes.length - maxSkills)) {
      await deleteNode(node.id, ctx.memoryDir);
      if (node.skill_name) {
        try {
          await quarantineSkill(ctx.skillRoot, node.skill_name, ["evicted: skill namespace cap reached"]);
        } catch {
          // already gone
        }
      }
      pruned.push(node.id);
    }
  }
  return pruned;
}
