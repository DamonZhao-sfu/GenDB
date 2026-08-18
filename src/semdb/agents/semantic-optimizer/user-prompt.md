# Task: Optimize semantic query `{{query_id}}`

Your canonical `optimize-semantic-program` procedure is already active as the system
instruction. Do not load its `SKILL.md` again.

Read this complete context bundle once:

`{{agent_context_path}}`

Expected SHA-256: `{{agent_context_sha256}}`

The bundle already contains the current plan, candidate manifest, feedback, bounded
history, helper/solver source, validation diff, primitive catalog, remaining budgets,
relevant memory, output schema, and output path. Use additional filesystem or shell tools
only if you identify a concrete missing or inconsistent fact; do not repeat reads already
satisfied by the bundle.

Write exactly one `PATCH_CODE`, `REPLAN`, or `STOP` action JSON as requested by the bundle.
Do not edit candidate or repository artifacts.
