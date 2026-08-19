/**
 * SemDB runtime configuration — a GenDB-style, provider-structured config for the
 * semantic operator compilation pipeline (Schema Designer → Extractor → Code
 * Generator).
 *
 * Three agent providers are supported, reusing src/gendb/providers unchanged:
 *   - "claude" → @anthropic-ai/claude-agent-sdk   (providers/claude.mjs)
 *   - "codex"  → @openai/codex-sdk                (providers/codex.mjs)
 *   - "vllm"   → local Responses-compatible vLLM  (providers/vllm.mjs)
 *
 * Pick one with `defaults.agentProvider` or the orchestrator's `--agent-provider`
 * flag. Each provider block sets the model per agent and the reasoning effort.
 */

export const defaults = {
  // Use the locally served Qwen model by default.
  agentProvider: "vllm",

  // Corpus / benchmark selection (SemBench).
  benchmark: "mmqa",          // mmqa | cars | ecomm | animals | movie
  querySource: "bigquery",    // which query dialect folder under files/<b>/query/

  // Extraction models — the SMALL models that read each row/image (NOT the
  // compiler agents). Independent of the agent provider above.
  extraction: {
    clipModel: "openai/clip-vit-base-patch32",   // tier-② CLIP for non-VLM image extraction (semvision)
    smallImageModel: "Qwen/Qwen3.8-27B-FP8",
    strongImageModel: "Qwen/Qwen3.8-27B-FP8",
    smallTextModel: "Qwen/Qwen3.8-27B-FP8",
    escalationImageModel: "Qwen/Qwen3.8-27B-FP8",
    captionModel: "Qwen/Qwen3.8-27B-FP8",   // OpImgCap — one caption per corpus image
    theta: 0.5,               // residual confidence floor, cut on OpImgVQA's logprob score
  },

  agentTimeoutMs: 20 * 60 * 1000,

  // --- Iterative refinement loop (GenDB-style) ---
  maxRefineIterations: 5,     // per-query optimize→run→score iterations (0 = single-shot)
  refineStallThreshold: 2,    // stop after this many consecutive non-improving iterations
  refineSampleCap: 15,        // max FP and FN rows shown to the agent per iteration
  directAgentArchitecture: "pgo", // pgo | legacy; only affects --direct
  agentExecution: "agent",        // agent | structured; vLLM Planner/Optimizer only
  codegenMode: "full",            // full | hybrid; hybrid uses one generic ABI
  enableAgentSkills: true,    // publish role + learned skills into a discoverable root
  maxReplans: 1,              // evidence-backed Planner revisions per query

  // --- Cross-run agent memory (opt-in: --memory-dir enables it) ---
  // Layers follow GenDB exactly:
  //   L0 Query Instances / L1 Query Templates      → HAG, pushed into the prompt
  //   L2 Sub-Structure Patterns / L3 Operator Techniques /
  //   L4 Optimization Strategies / L5 Performance Principles
  //                                                → skills, discovered by the agent
  memoryDir: null,
  memory: {
    exactMatchMinScore: 0.98,       // seed iter_0 from a past candidate at/above this
    structuralMatchMinScore: 0.55,  // inject advice at/above this
    maxPreInjectionTokens: 3000,    // L0/L1 push, per role
    maxCatalogTokens: 700,          // L2-L5 discovery hint, per role
    maxInlineSkills: 3,             // codex fallback: skills inlined as text
    maxInlineSkillTokens: 2000,
    differentialHeadroomThreshold: 0.30,
    maxNodesPerLayer: 40,
    // Every discoverable skill's description enters context whether or not it is
    // loaded, so an uncapped namespace is a per-call tax on all three roles.
    maxSkills: 30,
    crossBenchmarkLayers: [2, 3, 4, 5],
    skillNamePrefix: "semdb-",
    warmStart: true,
  },

  // --- Provider-specific settings for the three compiler agents ---
  providers: {
    claude: {
      model: "opus",          // default when an agent has no explicit entry
      agentModels: {
        schema_designer: "opus",   // decomposition is the hard judgement call
        extractor: "sonnet",       // mostly driver plumbing
        code_generator: "opus",    // must be result-equivalent to the oracle
        query_planner: "opus",
        semantic_code_generator: "opus",
        semantic_optimizer: "opus",
        memory_manager: "opus",
      },
      // Claude effort: "low" | "medium" | "high" | "max"
      agentEffortLevels: {
        schema_designer: "high",
        extractor: "low",
        code_generator: "medium",
        query_planner: "high",
        semantic_code_generator: "medium",
        semantic_optimizer: "high",
        memory_manager: "high",
      },
      escalationModel: "opus",
    },

    codex: {
      // "codex 5.6 sol" → set this to the exact model id your Codex deployment
      // exposes. Common ids: "gpt-5.6-sol", "gpt-5.6-sol-max", "gpt-5.6".
      // Change this one line to re-point every agent, or override per-agent below.
      model: "gpt-5.6-luna",
      agentModels: {
        schema_designer: "gpt-5.6-luna",
        extractor: "gpt-5.6-luna",
        code_generator: "gpt-5.6-luna",
        query_planner: "gpt-5.6-luna",
        semantic_code_generator: "gpt-5.6-luna",
        semantic_optimizer: "gpt-5.6-luna",
        memory_manager: "gpt-5.6-luna",
      },
      // Codex effort: "minimal" | "low" | "medium" | "high" | "xhigh"
      agentEffortLevels: {
        schema_designer: "medium",
        extractor: "low",
        code_generator: "medium",
        query_planner: "high",
        semantic_code_generator: "medium",
        semantic_optimizer: "high",
        memory_manager: "high",
      },
      escalationModel: "gpt-5.6-luna",
    },

    vllm: {
      model: "Qwen/Qwen3.8-27B-FP8",
      agentModels: {
        schema_designer: "Qwen/Qwen3.8-27B-FP8",
        extractor: "Qwen/Qwen3.8-27B-FP8",
        code_generator: "Qwen/Qwen3.8-27B-FP8",
        query_planner: "Qwen/Qwen3.8-27B-FP8",
        semantic_code_generator: "Qwen/Qwen3.8-27B-FP8",
        semantic_optimizer: "Qwen/Qwen3.8-27B-FP8",
        memory_manager: "Qwen/Qwen3.8-27B-FP8",
      },
      agentEffortLevels: {
        schema_designer: "medium",
        extractor: "medium",
        code_generator: "medium",
        query_planner: "medium",
        semantic_code_generator: "medium",
        semantic_optimizer: "medium",
        memory_manager: "medium",
      },
      escalationModel: "Qwen/Qwen3.8-27B-FP8",
    },
  },
};

/** Provider block for the active (or given) provider. */
export function getProviderConfig(providerName) {
  const name = providerName || defaults.agentProvider;
  const cfg = defaults.providers[name];
  if (!cfg) {
    const avail = Object.keys(defaults.providers).join(", ");
    throw new Error(`No SemDB provider config for "${name}". Available: ${avail}`);
  }
  return cfg;
}

/** Model id for one agent under the active (or given) provider. */
export function getAgentModel(configKey, providerName) {
  const cfg = getProviderConfig(providerName);
  return cfg.agentModels[configKey] || cfg.model;
}

/** Reasoning-effort level for one agent under the active (or given) provider. */
export function getAgentEffort(configKey, providerName) {
  return getProviderConfig(providerName).agentEffortLevels[configKey];
}

export function getExtractionConfig() {
  return defaults.extraction;
}
