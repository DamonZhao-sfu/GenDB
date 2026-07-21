/**
 * Experience Graph — unified entry point (PostgreSQL backend).
 *
 * A self-evolving memory store backed by PostgreSQL + pgvector, exposing the
 * paper's three access patterns as real database operations:
 *   graph traverse — recursive CTE over nodes.parent_node_id
 *   relation join   — SQL joins across tasks/sessions/nodes/prompts
 *   vector search   — pgvector cosine (<=>) over an HNSW ANN index
 *
 *   const store = await openExperienceStore({ connectionString: "postgres://…" });
 *
 * Connection precedence: opts.connectionString → opts.pg → env GENDB_EXPERIENCE_PG
 * → standard libpq env vars (PGHOST/PGPORT/PGUSER/PGDATABASE). Independent of
 * GenDB's HAG memory.
 */

// Re-export identity helpers + embedding so callers have one import surface.
export { taskId, sessionId, nodeId } from "./ids.mjs";
export { embed, EMBED_DIM } from "./embed.mjs";

export async function openExperienceStore(opts = {}) {
  const connectionString = opts.connectionString || process.env.GENDB_EXPERIENCE_PG || null;
  const { createPostgresBackend } = await import("./backends/postgres.mjs");
  return createPostgresBackend({ connectionString, pg: opts.pg });
}
