/**
 * Memory Manager contract: its prompts must state every rule that the writer
 * actually enforces, or the agent gets rejected for reasons it was never told.
 */

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { renderTemplate } from "../../gendb/shared.mjs";
import { config } from "../agents/memory-manager/index.mjs";
import { defaults } from "../semdb.config.mjs";
import { LAYER_NAMES } from "../memory/graph.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const systemPrompt = await readFile(config.promptPath, "utf8");
const userTemplate = await readFile(config.userPromptPath, "utf8");

// --- config -----------------------------------------------------------------

assert.equal(config.configKey, "memory_manager");
assert.ok(defaults.providers.claude.agentModels.memory_manager, "claude model configured");
assert.ok(defaults.providers.codex.agentModels.memory_manager, "codex model configured");
assert.ok(defaults.providers.claude.agentEffortLevels.memory_manager, "claude effort configured");
assert.ok(
  !config.allowedTools.includes("Skill"),
  "the Manager authors skills; giving it the Skill tool would let it consume its own output",
);
assert.ok(!config.skillPath, "the Manager has no bound procedure — binding is gone");

// --- layer names match GenDB ------------------------------------------------

for (const name of LAYER_NAMES) {
  assert.ok(
    systemPrompt.includes(name),
    `the Manager prompt must name layer "${name}" exactly as GenDB does`,
  );
}
assert.match(systemPrompt, /L0 \| Query Instances/);
assert.match(systemPrompt, /L5 \| Performance Principles/);

// --- every enforced rule is stated ------------------------------------------

const statedRules = [
  [/headroom/i, "the headroom rule (raw deltas rank quality metrics backwards)"],
  [/recomputed when your proposal is applied/i, "that improvement claims are re-checked"],
  [/evidence\.json/, "the evidence requirement"],
  [/Use when.*Load when|"Use when" or "Load when"/, "the description trigger-phrase rule"],
  [/semdb-/, "the skill name prefix"],
  [/64 ?KB/i, "the size cap"],
  [/quarantin/i, "that failures are quarantined"],
  [/rejected \*\*in full\*\*|rejected in full/i, "that a leak rejects the whole proposal"],
  [/never write graph|do not write graph/i, "that it must not write graph nodes"],
  [/REPLAN/, "the plan-level breakthrough priority"],
  [/first-iteration successes are low priority/i, "the non-obvious principle"],
  [/placeholder/i, "distill-not-copy for code patterns"],
  [/NOVEL_SUCCESS/, "the classification vocabulary"],
  [/FAMILIAR/, "the skip classification"],
];
for (const [re, what] of statedRules) {
  assert.ok(re.test(systemPrompt), `the Manager prompt must state ${what}`);
}

// Stating the boundary is explicitly permitted — the guard allows prohibitions, and
// the prompt has to say so or the agent will avoid writing the rule it must convey.
assert.match(systemPrompt, /Stating the boundary itself/i);

// --- the user prompt renders completely -------------------------------------

const rendered = renderTemplate(userTemplate, {
  run_id: "2026-07-29T12-00-00",
  benchmark: "mmqa",
  scale_factor: "200",
  out_dir: "/out",
  headroom_threshold: 0.3,
  query_evidence: "### q2a\n- run dir: `/out/mmqa-q2a`",
  skills_dir: "/mem/skill-root/.claude/skills",
  existing_skills: "- **semdb-ocr-name-join**: Use when …",
  existing_templates: "- L1_mmqa_joinimage_x: image join template (instances: 2)",
  skill_usage: '{"q2a":{"skills":{"semdb-ocr-name-join":1}}}',
  update_schema_path: "/contracts/memory-update.schema.json",
  update_path: "/out/memory_update.json",
});
assert.ok(!rendered.includes("{{"), "unresolved placeholder in the Manager task prompt");
assert.match(rendered, /Write the proposal to: `\/out\/memory_update\.json`/);
assert.match(rendered, /extend, do not duplicate/i);
assert.match(rendered, /actually loaded during this run/i);

// The optional sections must vanish on a first run, when nothing exists yet.
const firstRun = renderTemplate(userTemplate, {
  run_id: "r", benchmark: "mmqa", scale_factor: "200", out_dir: "/out",
  headroom_threshold: 0.3, query_evidence: "### q1", skills_dir: "/s",
  existing_skills: "", existing_templates: "", skill_usage: "",
  update_schema_path: "/schema.json", update_path: "/out/memory_update.json",
});
assert.ok(!firstRun.includes("{{"));
assert.ok(!firstRun.includes("extend, do not duplicate"),
  "no skills exist yet, so the de-duplication section must not appear");
assert.ok(!firstRun.includes("actually loaded during this run"));

console.log("test_memory_manager_prompt: PASS");
