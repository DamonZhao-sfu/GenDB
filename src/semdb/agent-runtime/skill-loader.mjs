import { readFile } from "node:fs/promises";

const ALLOWED_FRONTMATTER = new Set(["name", "description"]);
const SKILL_NAME = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

function parseFrontmatter(source, skillPath) {
  const normalized = source.replace(/\r\n/g, "\n");
  if (!normalized.startsWith("---\n")) {
    throw new Error(`Agent skill ${skillPath} is missing YAML frontmatter`);
  }
  const close = normalized.indexOf("\n---\n", 4);
  if (close < 0) {
    throw new Error(`Agent skill ${skillPath} has unterminated YAML frontmatter`);
  }
  const header = normalized.slice(4, close);
  const body = normalized.slice(close + 5).trim();
  const values = {};
  for (const [index, line] of header.split("\n").entries()) {
    if (!line.trim()) continue;
    const match = line.match(/^([A-Za-z0-9_-]+):\s*(.+)$/);
    if (!match) {
      throw new Error(
        `Agent skill ${skillPath} has unsupported frontmatter syntax on line ${index + 2}`,
      );
    }
    const [, key, raw] = match;
    if (!ALLOWED_FRONTMATTER.has(key)) {
      throw new Error(`Agent skill ${skillPath} has unknown frontmatter field "${key}"`);
    }
    if (Object.hasOwn(values, key)) {
      throw new Error(`Agent skill ${skillPath} repeats frontmatter field "${key}"`);
    }
    values[key] = raw.trim().replace(/^(['"])(.*)\1$/, "$2");
  }
  return { values, body };
}

export async function loadAgentSkill(skillPath, expectedName = null) {
  let source;
  try {
    source = await readFile(skillPath, "utf8");
  } catch (error) {
    throw new Error(`Cannot load bound agent skill ${skillPath}: ${error.message}`);
  }
  const { values, body } = parseFrontmatter(source, skillPath);
  const keys = Object.keys(values);
  if (keys.length !== 2 || !values.name || !values.description) {
    throw new Error(
      `Agent skill ${skillPath} frontmatter must contain only name and description`,
    );
  }
  if (values.name.length >= 64 || !SKILL_NAME.test(values.name)) {
    throw new Error(
      `Agent skill ${skillPath} name must be lowercase hyphen-case and shorter than 64 characters`,
    );
  }
  if (!values.description.trim()) {
    throw new Error(`Agent skill ${skillPath} has an empty description`);
  }
  if (!body) throw new Error(`Agent skill ${skillPath} has an empty body`);
  if (expectedName && values.name !== expectedName) {
    throw new Error(
      `Agent skill mismatch for ${skillPath}: expected ${expectedName}, got ${values.name}`,
    );
  }
  const prompt = [
    "## Bound procedural skill",
    `This procedural skill is statically bound to the current agent role: \`${values.name}\`.`,
    "Follow it for this invocation. Do not discover or load any other skill.",
    "",
    body,
  ].join("\n");
  return {
    name: values.name,
    description: values.description,
    body,
    prompt,
  };
}

export async function loadBoundAgentSkill(agentConfig, enabled = true) {
  if (!enabled) return null;
  if (!agentConfig?.skillPath || !agentConfig?.skillName) {
    throw new Error(
      `Agent ${agentConfig?.name || "<unknown>"} has no fixed skillPath/skillName binding`,
    );
  }
  return loadAgentSkill(agentConfig.skillPath, agentConfig.skillName);
}
