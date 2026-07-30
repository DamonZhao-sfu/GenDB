/**
 * Provider-level skill capability.
 *
 * Claude discovers skills natively (cwd + settingSources). Codex has no Skill
 * tool, so it gets the same capability through an explicit filesystem protocol —
 * without it, a codex run would silently have no access to learned memory while a
 * claude run did, and any provider comparison would be measuring the wrong thing.
 */

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { buildSkillProtocol, extractSkillUsage } from "../../gendb/providers/codex.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const providersDir = resolve(here, "..", "..", "gendb", "providers");

// --- claude: cwd and settingSources are overridable, defaults unchanged -----

const claudeSrc = await readFile(resolve(providersDir, "claude.mjs"), "utf8");
assert.match(claudeSrc, /cwd:\s*skillRoot \|\| REPO_ROOT/,
  "claude must honor an isolated skill root but keep REPO_ROOT for GenDB");
assert.match(claudeSrc, /settingSources:\s*settingSources \|\| \['user', 'project'\]/,
  "settingSources must be overridable so SemDB can scope discovery to 'project'");
assert.match(claudeSrc, /skillRoot, settingSources/, "both params are accepted");

// --- codex: the filesystem skill protocol ----------------------------------

const skills = [
  { name: "semdb-ocr-name-join", description: "Use when joining an image corpus to a name column." },
  { name: "semdb-closed-value-space", description: "Use when a helper returns an open string." },
];
const skillsDir = "/tmp/semdb-memory/skill-root/.claude/skills";
const protocol = buildSkillProtocol(skillsDir, skills);

assert.match(protocol, /## Skill tool/, "codex is told it has a skill capability");
assert.match(protocol, /semdb-ocr-name-join/);
assert.match(protocol, /semdb-closed-value-space/);
assert.ok(
  protocol.includes(`${skillsDir}/semdb-ocr-name-join/SKILL.md`),
  "each skill is addressable by absolute path — that is how codex loads one",
);
assert.match(protocol, /SKILL_LOADED/, "usage is self-reported as a fallback signal");
assert.match(protocol, /you may load several, or none/, "discovery is free, not forced");

assert.equal(buildSkillProtocol(skillsDir, []), "", "no skills → no protocol section");
assert.equal(buildSkillProtocol(null, skills), "", "no skill root → no protocol section");

// --- codex: usage extraction ------------------------------------------------

assert.deepEqual(
  extractSkillUsage(skills, [`cat ${skillsDir}/semdb-ocr-name-join/SKILL.md`], ""),
  { "semdb-ocr-name-join": 1 },
  "reading the file counts as loading the skill",
);
assert.deepEqual(
  extractSkillUsage(skills, [], "Done.\nSKILL_LOADED: semdb-closed-value-space"),
  { "semdb-closed-value-space": 1 },
  "a self-reported load is also counted",
);
assert.deepEqual(
  extractSkillUsage(skills, ["ls /tmp"], "nothing relevant"), {},
  "an agent that ignored the skills records no usage",
);
assert.deepEqual(extractSkillUsage(undefined, [], ""), {}, "no skills configured → no usage");

// Both signals for one skill must not double count into a wrong shape.
const both = extractSkillUsage(
  skills,
  [`sed -n 1,50p ${skillsDir}/semdb-ocr-name-join/SKILL.md`],
  "SKILL_LOADED: semdb-ocr-name-join",
);
assert.deepEqual(Object.keys(both), ["semdb-ocr-name-join"]);

console.log("test_provider_skills: PASS");
