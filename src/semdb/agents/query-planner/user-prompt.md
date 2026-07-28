# Plan semantic query `{{query_id}}`

Query SQL:

```sql
{{query_sql}}
```

Natural-language description: {{query_nl}}

Modality: `{{modality}}`

Tables:

{{tables_doc}}

Local primitive files (read them as the authoritative API):

{{local_primitive_files}}

Trace contract:

{{trace_contract}}

Write the validated plan to: `{{plan_path}}`

{{#if previous_plan_path}}
This is a replan. Read the previous plan at `{{previous_plan_path}}`.
{{/if}}
{{#if optimizer_action_path}}
Read the evidence-backed optimizer action at `{{optimizer_action_path}}`.
{{/if}}
