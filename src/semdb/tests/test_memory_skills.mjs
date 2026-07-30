/**
 * Skill-root tests: isolation, the lint gate, quarantine, catalog and usage.
 *
 * With free discovery the lint gate is the only thing standing between what the
 * Memory Manager writes and what every future agent reads, so each rejection
 * reason gets its own case.
 */

import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import {
  ROLE_SKILLS,
  initSkillRoot,
  lintAllSkills,
  lintSkill,
  listSkills,
  parseSkillFrontmatter,
  publishRoleSkills,
  recordSkillUsage,
  renderCatalog,
  skillRootFor,
  skillsDirFor,
} from "../memory/skills.mjs";

const memoryDir = await mkdtemp(resolve(tmpdir(), "semdb-mem-skills-"));
const root = skillRootFor(memoryDir);
await initSkillRoot(root);

// --- isolation --------------------------------------------------------------

assert.equal(skillRootFor(memoryDir), resolve(memoryDir, "skill-root"));
assert.equal(skillsDirFor(root), resolve(memoryDir, "skill-root", ".claude", "skills"));
assert.ok(existsSync(skillsDirFor(root)));
assert.ok(
  !skillRootFor(memoryDir).includes("/GenDB/.claude"),
  "the skill root must not resolve into the repo root, or agents would discover GenDB's C++ skills",
);
assert.deepEqual(await listSkills(root), [], "a fresh root discovers nothing");

// --- role skills are published, not bound -----------------------------------

const published = await publishRoleSkills(root);
assert.deepEqual(published.sort(), [...ROLE_SKILLS].sort(),
  "all three role skills must be discoverable now that static binding is gone");
for (const name of ROLE_SKILLS) {
  assert.ok(existsSync(resolve(skillsDirFor(root), name, "SKILL.md")));
}

// --- helpers ----------------------------------------------------------------

const GOOD_BODY = `# OCR name join

## When to Use
Joining an image corpus to a name column.

## Technique
Bind the join through best_ocr_match over the closed candidate set.
`;

async function makeSkill(name, { frontmatter, body = GOOD_BODY, evidence, extraFile } = {}) {
  const dir = resolve(skillsDirFor(root), name);
  await mkdir(dir, { recursive: true });
  const fm = frontmatter ?? `---\nname: ${name}\ndescription: Use when joining an image corpus to a name column.\n---\n\n`;
  await writeFile(resolve(dir, "SKILL.md"), `${fm}${body}`);
  if (evidence !== null) {
    await writeFile(
      resolve(dir, "evidence.json"),
      JSON.stringify(evidence ?? [{
        run_id: "2026-07-29T10-11-12", benchmark: "mmqa", query_id: "q2a",
        before: 0.41, after: 0.82,
      }], null, 2),
    );
  }
  if (extraFile) await writeFile(resolve(dir, extraFile.name), extraFile.content);
  return dir;
}

const lintLearned = (dir) => lintSkill(dir, {
  requirePrefix: "semdb-", requireEvidence: true, requireTriggerPhrasing: true,
});

// --- frontmatter parsing ----------------------------------------------------

const folded = parseSkillFrontmatter(
  "---\nname: semdb-x\ndescription: >\n  Use when the predicate\n  spans two tables.\n---\n\n# X\nbody\n",
);
assert.equal(folded.values.name, "semdb-x");
assert.match(folded.values.description, /Use when the predicate spans two tables\./);
assert.equal(parseSkillFrontmatter("no frontmatter").error, "missing YAML frontmatter");
assert.match(parseSkillFrontmatter("---\nname: x\n").error, /unterminated/);

// --- lint: accept -----------------------------------------------------------

const ok = await lintLearned(await makeSkill("semdb-ocr-name-join"));
assert.deepEqual(ok.errors, []);
assert.equal(ok.ok, true);
assert.equal(ok.name, "semdb-ocr-name-join");

// --- lint: each rejection reason -------------------------------------------

const cases = [
  ["missing SKILL.md", async () => {
    const dir = resolve(skillsDirFor(root), "semdb-empty");
    await mkdir(dir, { recursive: true });
    return dir;
  }, /missing SKILL.md/],

  ["name does not match directory", () => makeSkill("semdb-mismatch", {
    frontmatter: "---\nname: semdb-other\ndescription: Use when x.\n---\n\n",
  }), /does not match directory/],

  ["missing prefix", () => makeSkill("ocr-unprefixed", {
    frontmatter: "---\nname: ocr-unprefixed\ndescription: Use when x.\n---\n\n",
  }), /must be prefixed/],

  ["description without a trigger phrase", () => makeSkill("semdb-no-trigger", {
    frontmatter: "---\nname: semdb-no-trigger\ndescription: A technique for joins.\n---\n\n",
  }), /must start with "Use when"/],

  ["unsupported frontmatter field", () => makeSkill("semdb-extra-field", {
    frontmatter: "---\nname: semdb-extra-field\ndescription: Use when x.\nuser-invocable: false\n---\n\n",
  }), /unsupported frontmatter fields/],

  ["no evidence", () => makeSkill("semdb-no-evidence", { evidence: null }), /missing evidence.json/],

  ["evidence without run_id", () => makeSkill("semdb-thin-evidence", {
    evidence: [{ query_id: "q2a" }],
  }), /needs run_id and query_id/],

  ["oversized directory", () => makeSkill("semdb-huge", {
    extraFile: { name: "code-patterns.py", content: "x".repeat(70 * 1024) },
  }), /bytes \(max/],

  ["ground-truth leak", () => makeSkill("semdb-leaky", {
    body: `${GOOD_BODY}\nRead the ground truth file for the expected answers.\n`,
  }), /ground-truth leak/],

  ["CERT reference", () => makeSkill("semdb-cert", {
    body: `${GOOD_BODY}\nCompare against the CERT split.\n`,
  }), /ground-truth leak/],

  ["ground-truth path", () => makeSkill("semdb-path", {
    body: `${GOOD_BODY}\nSee /home/u/SemBench/files/mmqa/raw_results/gt/Q2.csv\n`,
  }), /ground-truth leak/],
];

for (const [label, build, expected] of cases) {
  const dir = await build();
  const result = await lintLearned(dir);
  assert.equal(result.ok, false, `${label}: expected lint failure`);
  assert.ok(
    result.errors.some((e) => expected.test(e)),
    `${label}: expected an error matching ${expected}, got ${JSON.stringify(result.errors)}`,
  );
}

// A PROHIBITION is not a leak. The role skills and the Manager's own rules must
// be able to say "never read ground truth"; a guard that cannot distinguish that
// from "read the ground truth" would forbid writing down the rule it enforces.
const prohibition = await lintLearned(await makeSkill("semdb-prohibition", {
  body: `${GOOD_BODY}\nNever access CERT or the final ground truth while iterating.\n`,
}));
assert.deepEqual(prohibition.errors, [], "prohibition language must pass the leak guard");

// --- quarantine: failures leave the discoverable namespace ------------------

const before = (await listSkills(root)).map((s) => s.name);
assert.ok(before.includes("semdb-leaky"), "the bad skill is discoverable before linting");

const { kept, quarantined } = await lintAllSkills(root, { skillNamePrefix: "semdb-" });
const keptNames = kept.map((k) => k.name);
const quarantinedNames = quarantined.map((q) => q.name);

assert.ok(keptNames.includes("semdb-ocr-name-join"), "the good skill survives");
for (const role of ROLE_SKILLS) {
  assert.ok(
    keptNames.includes(role),
    `role skill ${role} must survive: it is exempt from the prefix, evidence and trigger rules`,
  );
}
assert.ok(quarantinedNames.includes("semdb-leaky"));
assert.ok(quarantinedNames.includes("semdb-no-evidence"));

const after = (await listSkills(root)).map((s) => s.name);
assert.ok(!after.includes("semdb-leaky"), "a quarantined skill is no longer discoverable");
assert.ok(existsSync(resolve(root, "_quarantine", "semdb-leaky", "lint_errors.json")));
const reasons = JSON.parse(
  await readFile(resolve(root, "_quarantine", "semdb-leaky", "lint_errors.json"), "utf8"),
);
assert.ok(reasons.errors.length > 0 && reasons.quarantined_at);

// --- catalog ----------------------------------------------------------------

const catalog = renderCatalog(await listSkills(root));
assert.match(catalog, /## Available Memory Skills/);
assert.match(catalog, /semdb-ocr-name-join/);
assert.ok(
  !catalog.includes("plan-semantic-query"),
  "role skills are procedures, not memory — they must not pad the memory catalog",
);
assert.match(catalog, /never contain\s+ground-truth answers/, "catalog states the boundary");

const many = Array.from({ length: 60 }, (_, i) => ({
  name: `semdb-skill-${i}`,
  description: `Use when situation number ${i} arises and a long description pads the catalog out.`,
  role: false,
}));
const capped = renderCatalog(many, 200);
assert.ok(Math.ceil(capped.length / 4) <= 260, "catalog respects its token cap");
assert.match(capped, /catalog truncated/);
assert.equal(renderCatalog([]), "", "no learned skills → no catalog block at all");

// --- usage log --------------------------------------------------------------

const outDir = await mkdtemp(resolve(tmpdir(), "semdb-out-"));
await recordSkillUsage(outDir, "q2a", { "semdb-ocr-name-join": 1 });
await recordSkillUsage(outDir, "q2a", { "semdb-ocr-name-join": 2, "semdb-other": 1 });
await recordSkillUsage(outDir, "q4", {});
const usage = JSON.parse(await readFile(resolve(outDir, "skill_usage.json"), "utf8"));
assert.equal(usage.q2a.skills["semdb-ocr-name-join"], 3, "usage accumulates across calls");
assert.equal(usage.q2a.skills["semdb-other"], 1);
assert.ok(!usage.q4, "an agent that loaded nothing adds no entry");

console.log("test_memory_skills: PASS");
