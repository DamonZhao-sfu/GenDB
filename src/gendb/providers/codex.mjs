/**
 * OpenAI Codex SDK provider for GenDB.
 * Wraps @openai/codex-sdk to provide the same runAgent interface as the Claude provider.
 *
 * Key mappings from GenDB concepts to Codex SDK:
 *   systemPrompt    → developer_instructions config
 *   userPrompt      → thread.runStreamed(prompt)
 *   allowedTools    → N/A (Codex uses sandbox_mode)
 *   model           → config.model
 *   cwd             → startThread({ workingDirectory })
 *   effort          → config.model_reasoning_effort
 *   permissions     → approval_policy: "never", sandbox_mode: "danger-full-access"
 */

import { Codex } from "@openai/codex-sdk";
import { defaults, getProviderConfig } from "../gendb.config.mjs";
import { formatDuration } from "../shared.mjs";

/**
 * Codex exposes no Skill tool, but it does have full filesystem access, and a
 * skill is just a directory of Markdown. This gives it the same capability
 * through an explicit protocol: the skills are enumerated with absolute paths,
 * and "loading" one means reading its SKILL.md. Without this section Codex agents
 * would silently have no access to learned memory at all while Claude agents did,
 * which would make any provider comparison meaningless.
 */
export function buildSkillProtocol(skillsDir, skills) {
  if (!skillsDir || !skills || skills.length === 0) return "";
  const lines = [
    "## Skill tool (filesystem protocol)",
    "",
    `Skills available to you live under \`${skillsDir}\`. Each is a directory containing`,
    "`SKILL.md` (instructions) and optionally `code-patterns/`, `evidence.json` and `gotchas.md`.",
    "",
    "To LOAD a skill, read its `SKILL.md` with your file tools, then follow it. Load a skill",
    "whenever its description matches your situation — you may load several, or none.",
    "After loading one, write a line `SKILL_LOADED: <name>` in your final message so the run",
    "can record which knowledge was used.",
    "",
    "Available skills:",
  ];
  for (const s of skills) {
    lines.push(`- **${s.name}** — ${String(s.description || "").replace(/\s+/g, " ").slice(0, 200)}`);
    lines.push(`  path: \`${skillsDir}/${s.name}/SKILL.md\``);
  }
  return lines.join("\n");
}

/** Which skills the transcript shows were actually read. */
export function extractSkillUsage(skills, commands, finalText) {
  const used = {};
  for (const s of skills || []) {
    const readFromShell = commands.some((c) => c.includes(`${s.name}/SKILL.md`));
    const declared = new RegExp(`SKILL_LOADED:\\s*${s.name}\\b`).test(finalText || "");
    if (readFromShell || declared) used[s.name] = (used[s.name] || 0) + 1;
  }
  return used;
}

export async function runAgent(name, { systemPrompt, userPrompt, allowedTools, model, cwd, timeoutMs, configName, useSkills, domainSkillsPrompt, skillRoot, skillsDir, skills, effortLevel: effortOverride, verbose = false }) {
  // Build the effective system prompt (same logic as Claude provider)
  const resolvedSkillsDir = skillsDir || (skillRoot ? `${skillRoot}/.claude/skills` : null);
  const skillProtocol = useSkills !== false ? buildSkillProtocol(resolvedSkillsDir, skills) : "";
  const effectivePrompt = [
    systemPrompt,
    (useSkills !== false && domainSkillsPrompt) ? domainSkillsPrompt : "",
    skillProtocol,
  ].filter(Boolean).join("\n\n");

  const timeout = timeoutMs || defaults.agentTimeoutMs;
  const providerCfg = getProviderConfig("codex");
  const codexEffort = effortOverride || (configName && providerCfg.agentEffortLevels[configName]) || "medium";
  const effectiveModel = model || providerCfg.model;

  console.log(`\n[${"=".repeat(60)}]`);
  console.log(`[Orchestrator] Spawning agent: ${name} (provider: codex, model: ${effectiveModel}, timeout: ${formatDuration(timeout)}, effort: ${codexEffort})`);
  console.log(`[${"=".repeat(60)}]\n`);

  const startTime = Date.now();
  let timedOut = false;
  let resultText = "";
  let tokens = { input: 0, output: 0, cache_read: 0, cache_creation: 0 };
  let costUsd = 0;
  let agentError = null;
  let numTurns = 0;
  const commandsRun = [];

  let timer;
  const abortController = new AbortController();
  try {
    // developer_instructions goes in the Codex constructor's config
    const codex = new Codex({
      config: { developer_instructions: effectivePrompt },
    });

    // model/sandbox/approval/effort are direct ThreadOptions properties
    const thread = codex.startThread({
      workingDirectory: cwd || process.cwd(),
      skipGitRepoCheck: true,
      model: effectiveModel,
      sandboxMode: "danger-full-access",
      approvalPolicy: "never",
      modelReasoningEffort: codexEffort,
    });

    // Set up timeout with AbortController to actually kill the Codex process
    timer = setTimeout(() => {
      timedOut = true;
      console.error(`\n[Orchestrator] Agent "${name}" timed out after ${formatDuration(timeout)}, aborting...`);
      abortController.abort();
    }, timeout);

    // Use runStreamed to capture progress and usage
    const { events } = await thread.runStreamed(userPrompt, { signal: abortController.signal });

    for await (const event of events) {
      // Collect agent message text from completed items
      if (event.type === "item.completed" && event.item?.type === "agent_message") {
        resultText += event.item.text;
      }

      // Track shell commands so skill reads can be attributed (Codex loads a
      // skill by reading its file, so the command log is the usage log).
      if (event.type === "item.completed" && event.item?.type === "command_execution") {
        commandsRun.push(String(event.item.command || ""));
      }

      if (verbose && event.type === "item.completed" && event.item) {
        const item = event.item;
        if (item.type === "agent_message") {
          console.log(`[${name}] ${item.text.slice(0, 200)}`);
        }
        if (item.type === "command_execution") {
          console.log(`[${name}] Command: ${item.command}`);
        }
      }

      // Capture usage from turn.completed events (includes cached_input_tokens)
      if (event.type === "turn.completed") {
        numTurns++;
      }
      if (event.type === "turn.completed" && event.usage) {
        tokens = {
          input: (tokens.input || 0) + (event.usage.input_tokens || 0),
          output: (tokens.output || 0) + (event.usage.output_tokens || 0),
          cache_read: (tokens.cache_read || 0) + (event.usage.cached_input_tokens || 0),
          cache_creation: 0,
        };
      }

      // Surface turn failures with the actual error message
      if (event.type === "turn.failed" && event.error) {
        throw new Error(event.error.message);
      }
    }

  } catch (err) {
    if (!timedOut && err.name === "AbortError") {
      timedOut = true;
    }
    agentError = timedOut
      ? `Agent "${name}" timed out after ${formatDuration(timeout)}`
      : `Agent "${name}" failed: ${err.message}`;
  } finally {
    clearTimeout(timer);
  }

  const durationMs = Date.now() - startTime;

  // Estimate cost for Codex (pricing may differ; use a rough estimate)
  // Codex pricing is not provided via SDK, so we track tokens only
  if (!costUsd && (tokens.input || tokens.output)) {
    costUsd = estimateCodexCost(effectiveModel, tokens);
  }

  const skillsUsed = extractSkillUsage(skills, commandsRun, resultText);

  if (agentError) {
    console.error(`\n[Orchestrator] Agent "${name}" failed (${formatDuration(durationMs)}, ${tokens.input + tokens.output} tokens, $${costUsd.toFixed(2)}): ${agentError}`);
    return { result: resultText, durationMs, tokens, costUsd, numTurns, error: agentError, skillsUsed };
  }

  console.log(`\n[Orchestrator] Agent "${name}" completed (${formatDuration(durationMs)}, ${tokens.input + tokens.output} tokens, $${costUsd.toFixed(2)})`);
  return { result: resultText, durationMs, tokens, costUsd, numTurns, skillsUsed };
}

/**
 * Rough cost estimation for Codex models.
 * Update these rates as OpenAI publishes official pricing.
 */
function estimateCodexCost(model, tokens) {
  // Pricing per million tokens (from https://developers.openai.com/api/docs/pricing)
  const CODEX_PRICING = {
    "gpt-5.4":           { input: 2.50, cached: 0.25, output: 15 },
    "gpt-5.3-codex":     { input: 1.75, cached: 0.175, output: 14 },
    "gpt-5.2-codex":     { input: 1.75, cached: 0.175, output: 14 },
    "gpt-5.1-codex-max": { input: 1.25, cached: 0.125, output: 10 },
    "gpt-5.1-codex":     { input: 1.25, cached: 0.125, output: 10 },
    "gpt-5-codex":       { input: 1.25, cached: 0.125, output: 10 },
  };
  const pricing = CODEX_PRICING[model] || { input: 1.75, cached: 0.175, output: 14 };
  const perM = 1_000_000;
  const nonCachedInput = tokens.input - (tokens.cache_read || 0);
  return (nonCachedInput * pricing.input) / perM
    + ((tokens.cache_read || 0) * pricing.cached) / perM
    + (tokens.output * pricing.output) / perM;
}
