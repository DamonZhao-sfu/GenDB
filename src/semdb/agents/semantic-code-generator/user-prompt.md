# Generate candidate for semantic query `{{query_id}}`

Validated plan: `{{plan_path}}`

{{#if parent_candidate_manifest_path}}
Parent best-candidate manifest: `{{parent_candidate_manifest_path}}`
Use its helper and solver as the starting point only when the action is `PATCH_CODE`.
{{/if}}
{{#if optimizer_action_path}}
Structured optimizer action: `{{optimizer_action_path}}`
{{/if}}

Tables and data paths:

{{tables_doc}}

SemDB runtime directory: `{{semdb_dir}}`

Import contract:

- for `import semvision`, `import imagepatch`, or `from vadar...`, prepend exactly
  `{{semdb_dir}}` to `sys.path`;
- for `from semdb...`, prepend the parent directory of `{{semdb_dir}}`;
- generated helper imports must resolve when the solver starts from any working
  directory.

Write:

- helper module: `{{helpers_path}}`
- end-to-end solver: `{{solve_path}}`
- manifest draft: `{{manifest_draft_path}}`

The solver is invoked as:

`python3 {{solve_path}} <out.csv> --data-dir <dir>{{runtime_args}}`

It must write `trace_{{query_id}}.json` beside its output CSV and support optional
`--only-ids <path>` exactly as specified by the plan.
