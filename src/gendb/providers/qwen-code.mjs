/**
 * Qwen Code provider for GenDB.
 *
 * Drives the Qwen Code CLI (QwenLM/qwen-code) in headless mode against a local
 * vLLM endpoint over the mature OpenAI /v1/chat/completions API — avoiding the
 * Codex + vLLM Responses-API path (namespace tools / _postprocess_messages
 * crashes). Qwen Code brings its own tool loop (read/write/edit/shell), so
 * GenDB's agents keep working the same way (write JSON to exact paths, run g++).
 *
 * vLLM must be launched with tool calling + reasoning parsing, e.g.:
 *   vllm serve <model> --enable-auto-tool-choice \
 *     --tool-call-parser hermes --reasoning-parser qwen3
 * (Use the tool-call parser that yields valid JSON tool_calls for your model.)
 *
 * Requires the `qwen` CLI on PATH:  npm i -g @qwen-code/qwen-code
 * Endpoint: VLLM_BASE_URL env > providers.qwen-code.baseUrl config.
 */

import { spawn } from "child_process";
import { defaults, getProviderConfig } from "../gendb.config.mjs";
import { formatDuration } from "../shared.mjs";

export async function runAgent(name, { systemPrompt, userPrompt, allowedTools, model, cwd, timeoutMs, configName, useSkills, domainSkillsPrompt, effortLevel: effortOverride, verbose = false }) {
  const effectivePrompt = (useSkills !== false && domainSkillsPrompt)
    ? systemPrompt + "\n\n" + domainSkillsPrompt
    : systemPrompt;

  const timeout = timeoutMs || defaults.agentTimeoutMs;
  const providerCfg = getProviderConfig("qwen-code");
  const effectiveModel = model || providerCfg.model;
  const baseUrl = process.env.VLLM_BASE_URL || providerCfg.baseUrl;

  console.log(`\n[${"=".repeat(60)}]`);
  console.log(`[Orchestrator] Spawning agent: ${name} (provider: qwen-code, model: ${effectiveModel}, base: ${baseUrl}, timeout: ${formatDuration(timeout)})`);
  console.log(`[${"=".repeat(60)}]\n`);

  const startTime = Date.now();
  let resultText = "";
  let tokens = { input: 0, output: 0, cache_read: 0, cache_creation: 0 };
  const costUsd = 0; // local inference is not metered
  let agentError = null;
  let timedOut = false;

  try {
    resultText = await new Promise((resolvePromise, rejectPromise) => {
      const child = spawn("qwen", [
        "--system-prompt", effectivePrompt,   // per-agent role (replaces built-in prompt)
        "--approval-mode", "yolo",            // auto-approve all tool calls (write/edit/shell)
        "--output-format", "json",            // structured output on stdout
      ], {
        cwd: cwd || process.cwd(),
        env: {
          ...process.env,
          OPENAI_BASE_URL: baseUrl,
          OPENAI_API_KEY: process.env.OPENAI_API_KEY || "vllm",  // vLLM ignores the value
          OPENAI_MODEL: effectiveModel,
        },
        stdio: ["pipe", "pipe", "pipe"],
      });

      let stdout = "", stderr = "";
      const timer = setTimeout(() => {
        timedOut = true;
        child.kill("SIGKILL");
      }, timeout);

      child.stdout.on("data", (d) => {
        stdout += d;
        if (verbose) process.stdout.write(`[${name}] ${d}`);
      });
      child.stderr.on("data", (d) => { stderr += d; });

      child.on("error", (err) => {
        clearTimeout(timer);
        rejectPromise(err.code === "ENOENT"
          ? new Error("`qwen` CLI not found on PATH — install with: npm i -g @qwen-code/qwen-code")
          : err);
      });

      child.on("close", (code) => {
        clearTimeout(timer);
        if (timedOut) return rejectPromise(new Error(`timed out after ${formatDuration(timeout)}`));
        if (code !== 0) return rejectPromise(new Error(`qwen exited ${code}: ${stderr.slice(-500) || stdout.slice(-500)}`));
        // Parse --output-format json: prefer a structured response/stats, fall back to raw.
        let text = stdout;
        try {
          const j = JSON.parse(stdout);
          text = j.response ?? j.result ?? j.output ?? stdout;
          const u = j.stats?.tokens || j.usage || {};
          tokens = {
            input: u.input_tokens ?? u.prompt_tokens ?? u.input ?? 0,
            output: u.output_tokens ?? u.completion_tokens ?? u.output ?? 0,
            cache_read: u.cached_input_tokens ?? u.cache_read ?? 0,
            cache_creation: 0,
          };
        } catch { /* not JSON — keep raw stdout as the result */ }
        resolvePromise(typeof text === "string" ? text : JSON.stringify(text));
      });

      // Feed the task via stdin to avoid ARG_MAX limits on large user prompts.
      child.stdin.write(userPrompt);
      child.stdin.end();
    });
  } catch (err) {
    agentError = timedOut
      ? `Agent "${name}" timed out after ${formatDuration(timeout)}`
      : `Agent "${name}" failed: ${err.message}`;
  }

  const durationMs = Date.now() - startTime;

  if (agentError) {
    console.error(`\n[Orchestrator] Agent "${name}" failed (${formatDuration(durationMs)}): ${agentError}`);
    return { result: resultText, durationMs, tokens, costUsd, error: agentError, skillsUsed: {} };
  }

  console.log(`\n[Orchestrator] Agent "${name}" completed (${formatDuration(durationMs)}, ${tokens.input + tokens.output} tokens)`);
  return { result: resultText, durationMs, tokens, costUsd, skillsUsed: {} };
}
