# Task: Curate memory from run `{{run_id}}`

## Run

- Benchmark: `{{benchmark}}`
- Scale factor: `{{scale_factor}}`
- Output directory: `{{out_dir}}`
- Headroom threshold for a breakthrough: `{{headroom_threshold}}`

## Per-query evidence

Each query below lists the artifacts to read. For every query: read `telemetry.json` for the
objective history and action counts, then read each `iter_*/iteration_feedback.json` and
`iter_*/optimizer_action.json` to see what was tried and what happened, then diff the plans
and the solvers between the iterations where the objective moved.

{{query_evidence}}

## Skills directory

Write skills to: `{{skills_dir}}`

{{#if existing_skills}}
### Skills that already exist — extend, do not duplicate

{{existing_skills}}
{{/if}}

{{#if existing_templates}}
### Query templates already in memory

{{existing_templates}}
{{/if}}

{{#if skill_usage}}
### Skills that were actually loaded during this run

Add a usage row to each of these skills' `evidence.json`: being loaded is itself evidence
that the description matched a real situation.

{{skill_usage}}
{{/if}}

## Output

Memory-update schema: `{{update_schema_path}}`

Write the proposal to: `{{update_path}}`

Reminders that decide whether your work survives:

- the proposal is applied by code, which recomputes every retrieval key and re-checks every
  improvement claim against the headroom rule;
- a skill with no `evidence.json` row naming a query from THIS run is quarantined;
- a description that does not begin "Use when" or "Load when" is quarantined;
- any ground-truth path, per-row expected value, or instruction to read ground truth or CERT
  rejects the entire proposal.
