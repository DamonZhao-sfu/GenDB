/**
 * Experience Graph — unified entry point.
 *
 * Opens a self-evolving memory store backed by either SQLite (embedded, zero
 * setup; sqlite-vec exhaustive KNN) or PostgreSQL (unified engine; pgvector
 * HNSW ANN). Both backends expose the SAME async interface, so callers and the
 * orchestrator are backend-agnostic.
 *
 *   const store = await openExperienceStore({ backend: "sqlite", dir });
 *   const store = await openExperienceStore({ backend: "postgres", connectionString });
 *
 * Backend selection precedence: explicit opts.backend, else GENDB_EXPERIENCE_PG
 * (a Postgres connection string) if set, else "sqlite".
 */

import { createSqliteBackend } from "./backends/sqlite.mjs";

// Re-export identity helpers + embedding so callers have one import surface.
export { taskId, sessionId, nodeId } from "./store.mjs";
export { embed, EMBED_DIM } from "./embed.mjs";

export async function openExperienceStore(opts = {}) {
  const pgUrl = opts.connectionString || process.env.GENDB_EXPERIENCE_PG || null;
  const backend = opts.backend || (pgUrl ? "postgres" : "sqlite");

  if (backend === "postgres") {
    // Lazy import so SQLite-only users never need `pg` loaded.
    const { createPostgresBackend } = await import("./backends/postgres.mjs");
    return createPostgresBackend({ connectionString: pgUrl, pg: opts.pg });
  }
  if (backend === "sqlite") {
    return createSqliteBackend({ dir: opts.dir });
  }
  throw new Error(`Unknown experience backend: ${backend}`);
}
