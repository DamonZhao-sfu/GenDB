-- Experience Graph — PostgreSQL schema (backend: postgres).
--
-- Same four-table model as schema.sql (SQLite), but backed by a real unified
-- engine for all three access patterns:
--   * graph traverse — recursive CTE over nodes.parent_node_id
--   * relation join   — foreign keys across the four tables
--   * vector search   — pgvector with an HNSW ANN index (real approximate NN)
--
-- Large artifacts stored BY REFERENCE (paths into output/**).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tasks (
  task_id             text PRIMARY KEY,
  benchmark           text NOT NULL,
  query_id            text NOT NULL,
  scale_factor        integer NOT NULL,
  sql_text            text,
  template_signature  jsonb,
  embedding           vector(256),
  global_best_node_id text,
  created_at          timestamptz,
  updated_at          timestamptz
);

CREATE TABLE IF NOT EXISTS sessions (
  session_id                text PRIMARY KEY,
  task_id                   text NOT NULL REFERENCES tasks(task_id),
  run_id                    text,
  query_id                  text,
  model                     text,
  agent_provider            text,
  optimization_target       text,
  hardware_fingerprint      text,
  root_node_id              text,
  best_node_id              text,
  warm_started_from_node_id text,
  status                    text,
  started_at                timestamptz,
  completed_at              timestamptz
);

CREATE TABLE IF NOT EXISTS nodes (
  node_id             text PRIMARY KEY,
  session_id          text NOT NULL REFERENCES sessions(session_id),
  task_id             text NOT NULL REFERENCES tasks(task_id),
  parent_node_id      text REFERENCES nodes(node_id),
  iteration           integer,
  reward_ms           double precision,
  hot_ms              double precision,
  cold_ms             double precision,
  validation          text,
  improved            integer,
  is_best_in_session  integer,
  delta_vs_parent_pct double precision,
  delta_vs_root_pct   double precision,
  visit_count         integer DEFAULT 0,
  best_descendant_ms  double precision,
  categories          jsonb,
  strategy            text,
  embedding           vector(256),
  operation_timings   jsonb,
  cpp_path            text,
  plan_path           text,
  exec_results_path   text,
  created_at          timestamptz
);

CREATE TABLE IF NOT EXISTS prompts (
  prompt_id     text PRIMARY KEY,
  node_id       text NOT NULL REFERENCES nodes(node_id),
  agent         text,
  request_path  text,
  response_path text,
  tokens        integer,
  cost_usd      double precision,
  duration_ms   double precision,
  created_at    timestamptz
);

CREATE INDEX IF NOT EXISTS idx_nodes_session ON nodes(session_id);
CREATE INDEX IF NOT EXISTS idx_nodes_parent  ON nodes(parent_node_id);
CREATE INDEX IF NOT EXISTS idx_nodes_task    ON nodes(task_id);
CREATE INDEX IF NOT EXISTS idx_sessions_task ON sessions(task_id);
CREATE INDEX IF NOT EXISTS idx_prompts_node  ON prompts(node_id);

-- HNSW ANN indexes for vector search (cosine) — real approximate nearest
-- neighbor, used by the searchSimilar* queries (ORDER BY embedding <=> $1).
CREATE INDEX IF NOT EXISTS idx_tasks_hnsw ON tasks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_nodes_hnsw ON nodes USING hnsw (embedding vector_cosine_ops);
