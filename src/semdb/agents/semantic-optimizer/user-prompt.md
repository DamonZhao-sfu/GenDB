# Task: Optimize semantic query `{{query_id}}`

## First: load your procedure

Load the `optimize-semantic-program` skill before doing anything else — it is the procedure
for this role. Then load any other available skill whose description matches this failure.

## Current Best Candidate

Current validated plan: `{{plan_path}}`

Current best-candidate manifest: `{{candidate_manifest_path}}`

Diagnose this candidate only. The action's `candidate_id` must match the manifest.

## Structured Execution and Validation Evidence

Iteration feedback: `{{iteration_feedback_path}}`

Read the feedback's data boundary, execution/preflight status, query-specific objective,
operator fidelity, errors, runtime branches, and bounded history. Do not access CERT or
final ground truth.

## Prior Candidate Manifests

{{history_manifest_paths}}

Use history to avoid repeating actions that regressed or had no observable effect.

{{#if memory_pre_injection}}
{{memory_pre_injection}}
{{/if}}

{{#if memory_catalog}}
{{memory_catalog}}
{{/if}}

{{#if memory_inline_skills}}
{{memory_inline_skills}}
{{/if}}

## Authoritative Local Primitive API

{{local_primitive_files}}

Read these files only when diagnosing primitive capability, arguments, value-space
behavior, or confidence semantics. Never invent a replacement primitive.

## Remaining Budget

- candidate iterations: `{{remaining_iteration_budget}}`
- replans: `{{remaining_replan_budget}}`

Select exactly one action:

- `PATCH_CODE` for a concrete helpers/solver implementation mismatch;
- `REPLAN` for a plan-owned semantic or relational defect with remaining budget;
- `STOP` when no safe evidence-supported action remains.

## Output Contract

Optimizer-action schema: `{{optimizer_action_schema_path}}`

Write the single validated action to: `{{optimizer_action_path}}`

Write only that JSON file. Do not edit any candidate or repository artifact.
