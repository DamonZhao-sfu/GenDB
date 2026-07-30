# Task: Plan semantic query `{{query_id}}`

Query id (copy verbatim into the plan's `query_id`): `{{query_id}}`

## First: load your procedure

Load the `plan-semantic-query` skill before doing anything else — it is the procedure for
this role. Then load any other available skill whose description matches this query.

## Query

```sql
{{query_sql}}
```

Natural-language description: {{query_nl}}

Modality: `{{modality}}`

## Tables and Runtime Inputs

{{tables_doc}}

## Authoritative Local Primitive API

Read these files before selecting or binding any primitive:

{{local_primitive_files}}

Design the minimum reusable helper interfaces and specify their complete typed primitive
implementations in `helper_dag`.

{{#if memory_pre_injection}}
{{memory_pre_injection}}
{{/if}}

{{#if memory_catalog}}
{{memory_catalog}}
{{/if}}

{{#if memory_inline_skills}}
{{memory_inline_skills}}
{{/if}}

{{#if memory_reference_plan_path}}
## Reference Plan From a Past Run

A structurally identical query was planned before: `{{memory_reference_plan_path}}`

Read it as a reference, not a specification. This is still the initial plan: keep
`plan_version` at `1` and `parent_plan_version` at `null`. Adopt only the parts the current
tables and the authoritative primitive API actually support, and drop anything the current
query does not need.
{{/if}}

## Trace and Sampling Contract

{{trace_contract}}

The plan must preserve this physical validation unit, trace key, and `--only-ids` boundary
independently of the final SQL projection.

{{#if previous_plan_path}}
## Replan Context

Previous validated plan: `{{previous_plan_path}}`

Evidence-backed optimizer action: `{{optimizer_action_path}}`

Produce a complete revised plan. Increment `plan_version` by exactly one, set
`parent_plan_version` to the previous version, preserve unaffected sections, and change
only plan-owned fields supported by the action's evidence.
{{/if}}

{{#if planner_lint}}
## Rejected Planning Attempt

The previous attempt failed the physical-operator check:

{{planner_lint}}

Replace the invalid binding with a discriminative local primitive supported by the
authoritative API. This is still the initial plan: keep `plan_version` at `1` and
`parent_plan_version` at `null`.
{{/if}}

## Output Contract

Plan schema: `{{plan_schema_path}}`

Write the complete validated plan to: `{{plan_path}}`

Write only that JSON file. Do not write Python or explanatory prose.
