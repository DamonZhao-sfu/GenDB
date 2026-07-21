/**
 * Local vLLM provider for GenDB (open-weight models, e.g. Qwen / DeepSeek).
 *
 * Reuses the OpenAI Codex agent runtime (@openai/codex-sdk) but points it at a
 * locally-served, OpenAI-compatible vLLM endpoint instead of OpenAI's backend.
 * Same runAgent(name, options) contract as the claude/codex providers.
 *
 * Key differences vs. the codex provider:
 *   - Injects a custom `model_provider` (base_url + wire_api="responses") so the
 *     Codex runtime talks to vLLM.
 *   - Disables Codex's grouped-tool features (apps / web_search / multi_agent).
 *     Those are emitted as `type:"namespace"` tools which vLLM's Responses API
 *     rejects (see openai/codex#23186). GenDB does its own multi-agent
 *     orchestration in JS and needs only the native shell + apply_patch tools.
 *   - Cost is reported as 0 (local inference is not metered).
 *
 * vLLM must be launched with tool + reasoning parsing enabled, e.g.:
 *   vllm serve <model> --enable-auto-tool-choice \
 *     --tool-call-parser qwen3_coder --reasoning-parser qwen3
 *
 * base_url resolution order: VLLM_BASE_URL env > providers.vllm.baseUrl config.
 */

import { Codex } from "@openai/codex-sdk";
import { defaults, getProviderConfig } from "../gendb.config.mjs";
import { formatDuration } from "../shared.mjs";

export async function runAgent(name, { systemPrompt, userPrompt, allowedTools, model, cwd, timeoutMs, configName, useSkills, domainSkillsPrompt, effortLevel: effortOverride, verbose = false }) {
  // Build the effective system prompt (same logic as the codex provider).
  const effectivePrompt = (useSkills !== false && domainSkillsPrompt)
    ? systemPrompt + "\n\n" + domainSkillsPrompt
    : systemPrompt;

  const timeout = timeoutMs || defaults.agentTimeoutMs;
  const providerCfg = getProviderConfig("vllm");
  const reasoningEffort = effortOverride || (configName && providerCfg.agentEffortLevels[configName]) || "medium";
  const effectiveModel = model || providerCfg.model;
  const baseUrl = process.env.VLLM_BASE_URL || providerCfg.baseUrl;

  console.log(`\n[${"=".repeat(60)}]`);
  console.log(`[Orchestrator] Spawning agent: ${name} (provider: vllm, model: ${effectiveModel}, base: ${baseUrl}, timeout: ${formatDuration(timeout)}, effort: ${reasoningEffort})`);
  console.log(`[${"=".repeat(60)}]\n`);

  const startTime = Date.now();
  let timedOut = false;
  let resultText = "";
  let tokens = { input: 0, output: 0, cache_read: 0, cache_creation: 0 };
  const costUsd = 0; // local inference is not metered
  let agentError = null;

  let timer;
  const abortController = new AbortController();
  try {
    // Point the Codex runtime at the local vLLM endpoint and disable every
    // Codex feature that emits `type:"namespace"` tools (vLLM rejects them).
    const codex = new Codex({
      config: {
        developer_instructions: effectivePrompt,
        model_provider: "vllm",
        "model_providers.vllm.name": "vLLM",
        "model_providers.vllm.base_url": baseUrl,
        "model_providers.vllm.wire_api": "responses",
        "features.apps": false,
        "features.web_search": false,
        "features.multi_agent": false,
      },
    });

    const thread = codex.startThread({
      workingDirectory: cwd || process.cwd(),
      skipGitRepoCheck: true,
      model: effectiveModel,
      sandboxMode: "danger-full-access",
      approvalPolicy: "never",
      modelReasoningEffort: reasoningEffort,
    });

    timer = setTimeout(() => {
      timedOut = true;
      console.error(`\n[Orchestrator] Agent "${name}" timed out after ${formatDuration(timeout)}, aborting...`);
      abortController.abort();
    }, timeout);

    const { events } = await thread.runStreamed(userPrompt, { signal: abortController.signal });

    for await (const event of events) {
      if (event.type === "item.completed" && event.item?.type === "agent_message") {
        resultText += event.item.text;
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

      if (event.type === "turn.completed" && event.usage) {
        tokens = {
          input: (tokens.input || 0) + (event.usage.input_tokens || 0),
          output: (tokens.output || 0) + (event.usage.output_tokens || 0),
          cache_read: (tokens.cache_read || 0) + (event.usage.cached_input_tokens || 0),
          cache_creation: 0,
        };
      }

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

  if (agentError) {
    console.error(`\n[Orchestrator] Agent "${name}" failed (${formatDuration(durationMs)}, ${tokens.input + tokens.output} tokens): ${agentError}`);
    return { result: resultText, durationMs, tokens, costUsd, error: agentError, skillsUsed: {} };
  }

  console.log(`\n[Orchestrator] Agent "${name}" completed (${formatDuration(durationMs)}, ${tokens.input + tokens.output} tokens)`);
  return { result: resultText, durationMs, tokens, costUsd, skillsUsed: {} };
}
