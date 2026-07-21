/**
 * Live capture for the orchestrator.
 *
 * createCapture(args) returns a capture handle whose methods NEVER throw into
 * the pipeline (all errors are swallowed and logged), so wiring it into the
 * optimize loop is non-breaking. When no Postgres connection is configured it
 * returns a no-op handle, so the pipeline is byte-for-byte unchanged with the
 * feature off (the default).
 *
 * Enabled by --experience-pg <conn> (args.experiencePg) or env
 * GENDB_EXPERIENCE_PG. PostgreSQL + pgvector only.
 */

import { openExperienceStore } from "./index.mjs";
import { ingestHistory } from "./ingest.mjs";

const NOOP = {
  enabled: false,
  async captureQuery() {},
  async close() {},
};

/**
 * @param args orchestrator args: { experiencePg?, targetBenchmark, scaleFactor,
 *   runId, optimizationTarget, model?, agentProvider? }
 */
export async function createCapture(args) {
  const conn = args.experiencePg || process.env.GENDB_EXPERIENCE_PG || null;
  if (!conn) return NOOP; // opt-in only; never auto-enable on generic libpq env vars

  let store;
  try {
    store = await openExperienceStore({ connectionString: conn });
  } catch (e) {
    console.warn(`[experience] capture disabled — could not open store: ${e.message}`);
    return NOOP;
  }
  console.log(`[experience] live capture ON (postgres, pgvector HNSW)`);

  return {
    enabled: true,

    /**
     * Ingest one query's completed optimization history into the graph.
     * Called once per query, right before runQueryFullPipeline returns.
     */
    async captureQuery(queryId, sqlText, optimizationHistory, queryDir) {
      try {
        const res = await ingestHistory(store, {
          benchmark: args.targetBenchmark,
          scaleFactor: args.scaleFactor,
          runId: args.runId,
          queryId,
          target: args.optimizationTarget || "hot",
          sqlText,
          iterations: optimizationHistory?.iterations || [],
          queryDir,
          session: {
            model: args.model, agent_provider: args.agentProvider,
            status: "captured", started_at: null, completed_at: new Date().toISOString(),
          },
        });
        if (res.sessionId) {
          console.log(`[experience] captured ${queryId}: ${res.nodeCount} nodes, best=${res.bestNodeId?.split("__").pop() ?? "n/a"}`);
        }
      } catch (e) {
        console.warn(`[experience] capture failed for ${queryId} (non-fatal): ${e.message}`);
      }
    },

    async close() {
      try { await store.close(); } catch {}
    },
  };
}
