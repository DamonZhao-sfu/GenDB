# Task: Generate candidate for semantic query `{{query_id}}`

Generation mode: `{{generation_mode}}`

## First: load your procedure

Load the `generate-semantic-program` skill before doing anything else — it is the procedure
for this role. Then load any other available skill whose description matches this plan.

## Authoritative Plan

Validated plan: `{{plan_path}}`

Plan schema: `{{plan_schema_path}}`

Implement the current plan literally. It owns all primitive choices, prompt phrases,
thresholds, value spaces, helper semantics, acceptance paths, and relational placement.

{{#if parent_candidate_manifest_path}}
## PATCH_CODE Context

Parent best-candidate manifest: `{{parent_candidate_manifest_path}}`

Structured optimizer action: `{{optimizer_action_path}}`

Read the parent helper and solver through the manifest. Patch only the named implementation
artifact/symbol, preserve the current plan version and all listed invariants, then write a
complete helper and solver to the new output paths.
{{/if}}

{{#if replan_action_path}}
## REPLAN Context

Evidence that triggered the revised plan: `{{replan_action_path}}`

Use it only as diagnostic context. Generate fresh code from the revised plan and do not
reuse a parent helper or solver.
{{/if}}

{{#if memory_pre_injection}}
{{memory_pre_injection}}
{{/if}}

{{#if memory_catalog}}
{{memory_catalog}}
{{/if}}

{{#if memory_inline_skills}}
{{memory_inline_skills}}
{{/if}}

## Tables and Runtime Inputs

{{tables_doc}}

## Authoritative Local Primitive API

{{local_primitive_files}}

Read these files before importing or calling any primitive.

SemDB runtime directory: `{{semdb_dir}}`

Import contract:

- the operator library is ONE package: import it as `from vadar import ...` /
  `from vadar.predefined import ...`, and prepend exactly `{{semdb_dir}}` to `sys.path`;
- for `from semdb...`, prepend the parent directory of `{{semdb_dir}}`;
- generated helper imports must resolve when the solver starts from any working directory.

## Runtime Invocation

`python3 {{solve_path}} <out.csv> --data-dir <dir>{{runtime_args}}`

The solver must also support optional `--only-ids <path>` exactly as specified by the
plan and write `trace_{{query_id}}.json` beside the output CSV.

## Output Artifacts

- helper module: `{{helpers_path}}`
- end-to-end solver: `{{solve_path}}`
- manifest draft: `{{manifest_draft_path}}`

Write the minimal manifest draft structure from the system prompt, not a finalized
candidate manifest. Write only these three paths. The orchestrator owns schema finalization,
execution, scoring, hashes, candidate identity, and promotion.
