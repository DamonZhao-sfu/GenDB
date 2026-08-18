# Task: Plan semantic query `{{query_id}}`

Your canonical `plan-semantic-query` procedure is already active as the system
instruction. Do not load its `SKILL.md` again.

Read this complete context bundle once:

`{{agent_context_path}}`

Expected SHA-256: `{{agent_context_sha256}}`

The bundle already contains the query, full-file runtime profile, table metadata,
primitive catalog, trace/lineage evidence, relevant memory, output schema, and output
path. Use additional filesystem or shell tools only if you identify a concrete missing
or inconsistent fact; do not repeat reads already satisfied by the bundle.

Write exactly the complete plan JSON requested by the bundle. Do not write Python or
explanatory prose.
