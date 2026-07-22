/**
 * SemDB runtime configuration — a thin, GenDB-style config for the semantic
 * operator compilation pipeline (Schema Designer → Extractor → Code Generator).
 *
 * Provider/model selection mirrors gendb.config.mjs so the same Claude Agent SDK
 * plumbing (src/gendb/providers) is reused unchanged.
 */

export const defaults = {
  agentProvider: "claude",

  // Corpus / benchmark selection (SemBench).
  benchmark: "mmqa",          // mmqa | cars | ecomm | animals | movie
  querySource: "bigquery",    // which query dialect folder under files/<b>/query/

  // Extraction models — small first, escalate rarely.
  extraction: {
    smallImageModel: "HuggingFaceTB/SmolVLM-256M-Instruct",
    strongImageModel: "Qwen/Qwen3-VL-2B-Instruct",
    smallTextModel: "Qwen/Qwen3-0.6B",
    escalationImageModel: "Qwen/Qwen3-VL-2B-Instruct",
    theta: 0.5,               // residual confidence floor
  },

  // Per-agent LLM (the *compiler* agents, not the extraction models).
  agentModels: {
    schema_designer: "opus",  // decomposition is the hard judgement call
    extractor: "sonnet",      // mostly driver plumbing
    code_generator: "opus",   // must be result-equivalent to the oracle
  },
  agentEffortLevels: {
    schema_designer: "high",
    extractor: "low",
    code_generator: "medium",
  },

  agentTimeoutMs: 20 * 60 * 1000,
};

export function getExtractionConfig() {
  return defaults.extraction;
}
