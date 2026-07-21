-- Experience Graph — four-table schema (Tasks / Sessions / Nodes / Prompts).
--
-- Independent self-evolving memory store. Backs the paper's three access
-- patterns as first-class database operations:
--   * vector search  — embedding BLOBs on tasks/nodes (+ cosine in JS)
--   * relation join   — foreign keys across the four tables
--   * graph traverse  — recursive CTE over nodes.parent_node_id (the search-tree edge)
--
-- Large artifacts (C++, plans, execution results, prompt logs) are stored BY
-- REFERENCE (paths into output/**), never inlined.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- A task = one optimization problem: (benchmark, query template, scale factor).
CREATE TABLE IF NOT EXISTS tasks (
  task_id             TEXT PRIMARY KEY,     -- <benchmark>__<queryId|template>__sf<N>
  benchmark           TEXT NOT NULL,
  query_id            TEXT NOT NULL,
  scale_factor        INTEGER NOT NULL,
  sql_text            TEXT,
  template_signature  TEXT,                 -- JSON structural features
  embedding           BLOB,                 -- vector(sql_text) for vector search
  global_best_node_id TEXT,                 -- best node across all sessions
  created_at          TEXT,
  updated_at          TEXT
);

-- A session = one run's attempt at one task (the per-query optimize loop).
CREATE TABLE IF NOT EXISTS sessions (
  session_id                TEXT PRIMARY KEY,   -- <run_id>__<query_id>
  task_id                   TEXT NOT NULL REFERENCES tasks(task_id),
  run_id                    TEXT,
  query_id                  TEXT,
  model                     TEXT,
  agent_provider            TEXT,
  optimization_target       TEXT,               -- 'hot' | 'cold'
  hardware_fingerprint      TEXT,
  root_node_id              TEXT,
  best_node_id              TEXT,
  warm_started_from_node_id TEXT,               -- cross-session reuse edge (Phase c)
  status                    TEXT,
  started_at                TEXT,
  completed_at              TEXT
);

-- A node = one iteration = one search-tree node (attempt + artifact + reward).
CREATE TABLE IF NOT EXISTS nodes (
  node_id             TEXT PRIMARY KEY,         -- <session_id>__iter_<N>
  session_id          TEXT NOT NULL REFERENCES sessions(session_id),
  task_id             TEXT NOT NULL REFERENCES tasks(task_id),      -- denormalized for task-scoped joins
  parent_node_id      TEXT REFERENCES nodes(node_id),               -- THE search-tree edge (graph traverse)
  iteration           INTEGER,
  reward_ms           REAL,                     -- selected timing for the run's optimization_target
  hot_ms              REAL,
  cold_ms             REAL,
  validation          TEXT,                     -- 'pass' | 'fail' | ...
  improved            INTEGER,                  -- 0/1: became the new best-so-far
  is_best_in_session  INTEGER,                  -- 0/1
  delta_vs_parent_pct REAL,
  delta_vs_root_pct   REAL,
  visit_count         INTEGER DEFAULT 0,        -- MCTS backprop rollup
  best_descendant_ms  REAL,                     -- min reward over this node's subtree
  categories          TEXT,                     -- JSON array
  strategy            TEXT,                     -- optimizer root-cause / strategy text
  embedding           BLOB,                     -- vector(strategy) for vector search
  operation_timings   TEXT,                     -- JSON
  cpp_path            TEXT,                     -- artifacts BY REFERENCE
  plan_path           TEXT,
  exec_results_path   TEXT,
  created_at          TEXT
);

-- A prompt = the agent I/O that produced a node.
CREATE TABLE IF NOT EXISTS prompts (
  prompt_id      TEXT PRIMARY KEY,              -- <node_id>__<agent>
  node_id        TEXT NOT NULL REFERENCES nodes(node_id),
  agent          TEXT,                          -- query_planner | query_optimizer | code_generator
  request_path   TEXT,                          -- BY REFERENCE to on-disk logs
  response_path  TEXT,
  tokens         INTEGER,
  cost_usd       REAL,
  duration_ms    REAL,
  created_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_nodes_session ON nodes(session_id);
CREATE INDEX IF NOT EXISTS idx_nodes_parent  ON nodes(parent_node_id);
CREATE INDEX IF NOT EXISTS idx_nodes_task    ON nodes(task_id);
CREATE INDEX IF NOT EXISTS idx_sessions_task ON sessions(task_id);
CREATE INDEX IF NOT EXISTS idx_prompts_node  ON prompts(node_id);
