/**
 * The additive invariant: with memory off, the rendered prompts contain no trace
 * of it, and with memory on they differ ONLY by the memory blocks.
 *
 * This is the guarantee that makes `--no-memory` a usable baseline. If memory
 * leaked into a prompt some other way, an A/B comparison would be measuring two
 * different prompts rather than the presence of memory.
 */

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { renderTemplate } from "../../gendb/shared.mjs";
import { renderPreInjection, renderInlineSkills, capTokens } from "../memory/render.mjs";
import { renderCatalog } from "../memory/skills.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const agents = resolve(here, "..", "agents");

const plannerUser = await readFile(resolve(agents, "query-planner/user-prompt.md"), "utf8");
const generatorUser = await readFile(resolve(agents, "semantic-code-generator/user-prompt.md"), "utf8");
const optimizerUser = await readFile(resolve(agents, "semantic-optimizer/user-prompt.md"), "utf8");

const BASE = {
  planner: {
    query_id: "q2a", query_sql: "SELECT 1", query_nl: "nl", modality: "image",
    tables_doc: "- images.csv", local_primitive_files: "- predefined.py",
    trace_contract: "row key: id", previous_plan_path: "", optimizer_action_path: "",
    planner_lint: "", plan_schema_path: "/schema.json", plan_path: "/run/plan.json",
  },
  generator: {
    query_id: "q2a", generation_mode: "INITIAL", plan_path: "/run/plan.json",
    plan_schema_path: "/schema.json", parent_candidate_manifest_path: "",
    optimizer_action_path: "", replan_action_path: "", tables_doc: "- images.csv",
    local_primitive_files: "- predefined.py", semdb_dir: "/semdb",
    solve_path: "/run/solve.py", helpers_path: "/run/helpers.py",
    manifest_draft_path: "/run/manifest.json", runtime_args: "",
  },
  optimizer: {
    query_id: "q2a", plan_path: "/run/plan.json",
    candidate_manifest_path: "/run/candidate_manifest.json",
    iteration_feedback_path: "/run/iteration_feedback.json",
    history_manifest_paths: "- /run/candidate_manifest.json",
    local_primitive_files: "- predefined.py", remaining_iteration_budget: 2,
    remaining_replan_budget: 1, optimizer_action_schema_path: "/action.json",
    optimizer_action_path: "/run/optimizer_action.json",
  },
};

// What the orchestrator's memoryVars() produces when memory is disabled.
const MEMORY_OFF = {
  memory_pre_injection: "", memory_catalog: "", memory_inline_skills: "", memory_note: "",
  memory_reference_plan_path: "",
};

const templates = {
  planner: plannerUser, generator: generatorUser, optimizer: optimizerUser,
};

// --- memory off: nothing leaks, and absent === empty ------------------------

for (const [role, template] of Object.entries(templates)) {
  const withoutKeys = renderTemplate(template, BASE[role]);
  const withEmpty = renderTemplate(template, { ...BASE[role], ...MEMORY_OFF });
  assert.equal(withEmpty, withoutKeys,
    `${role}: an empty memory var must render identically to an absent one`);
  assert.ok(!withEmpty.includes("{{"), `${role}: unresolved placeholder`);
  for (const marker of ["Prior Knowledge", "Available Memory Skills", "Reference Plan From a Past Run", "Memory Skills (inlined)"]) {
    assert.ok(!withEmpty.includes(marker),
      `${role}: "${marker}" must not appear when memory is off`);
  }
}

// --- memory on: the ONLY difference is the memory blocks --------------------

const l1 = {
  id: "L1_mmqa_joinimage_aaa",
  content: {
    proven_strategies: ["bind the join through best_ocr_match over a closed name set"],
    anti_patterns: ["free-form captioning then string equality"],
    plan_skeleton: { sampling_unit: "pair", helper_names: ["logo_name"], primitives: ["best_ocr_match"] },
  },
};
const l0 = {
  id: "L0_mmqa_q2a_1",
  content: { objective: { name: "f1", value: 0.82, direction: "maximize" }, iterations: 2 },
};

const preInjection = renderPreInjection(
  { tier: "structural", score: 0.71, l1, l0, role: "planner", warmStart: null }, 3000,
);
const catalog = renderCatalog([{
  name: "semdb-ocr-name-join", role: false,
  description: "Use when joining an image corpus to a name column.",
}], 700);
const inline = renderInlineSkills([{ name: "semdb-ocr-name-join", body: "# X\n\nBind through best_ocr_match." }], 2000);

assert.ok(preInjection.includes("Prior Knowledge"));
assert.ok(preInjection.includes("advisory"));
assert.ok(catalog.includes("Available Memory Skills"));
assert.ok(inline.includes("Memory Skills (inlined)"));

const on = renderTemplate(plannerUser, {
  ...BASE.planner,
  memory_pre_injection: preInjection,
  memory_catalog: catalog,
  memory_inline_skills: inline,
  memory_note: "true",
  memory_reference_plan_path: "/past/plan.json",
});
const off = renderTemplate(plannerUser, { ...BASE.planner, ...MEMORY_OFF });

assert.ok(on.includes("Prior Knowledge"));
assert.ok(on.includes("Reference Plan From a Past Run"));
assert.ok(on.includes("keep\n`plan_version` at `1`") || on.includes("plan_version` at `1`"),
  "the reference branch must preserve the initial-plan contract");

// Everything the memory-off rendering says must still be said with memory on: the
// blocks are additive, they never replace part of the task.
for (const line of off.split("\n").map((l) => l.trim()).filter((l) => l.length > 12)) {
  assert.ok(on.includes(line), `memory must be additive; missing line: ${line}`);
}

// --- caps hold on the rendered artifacts ------------------------------------

const fatL1 = {
  id: "L1_x",
  content: { proven_strategies: Array.from({ length: 300 }, (_, i) => `strategy ${i} padded out with text`) },
};
const capped = renderPreInjection(
  { tier: "structural", score: 0.7, l1: fatL1, l0: null, role: "planner", warmStart: null }, 120,
);
assert.ok(Math.ceil(capped.length / 4) <= 130, "pre-injection cap holds");
assert.match(capped, /truncated/);

const fatCatalog = renderCatalog(
  Array.from({ length: 80 }, (_, i) => ({
    name: `semdb-s${i}`, role: false,
    description: `Use when case ${i} happens and the description is long enough to matter.`,
  })), 150,
);
assert.ok(Math.ceil(fatCatalog.length / 4) <= 210, "catalog cap holds");

const fatInline = renderInlineSkills(
  Array.from({ length: 20 }, (_, i) => ({ name: `semdb-s${i}`, body: "body ".repeat(400) })), 300,
);
assert.ok(Math.ceil(fatInline.length / 4) <= 400, "inline-skill cap holds");

assert.equal(capTokens("", 10), "");
assert.equal(renderPreInjection({ tier: "novel", score: 0, l1: null, l0: null, role: "planner" }), "");

console.log("test_memory_injection: PASS");
