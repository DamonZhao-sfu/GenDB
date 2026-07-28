---
name: generate-semantic-program
description: Generate or revise a GenDB SemDB offline semantic program from a validated typed plan. Use when creating helper functions and an end-to-end solver, applying a structured optimizer patch, regenerating after a replan, preserving trace and result contracts, or repairing compile/runtime defects without changing the query's planned semantics.
---

# Generate Semantic Program

Implement the validated plan as a complete candidate. Do not redesign the query.

## Read required inputs

Read the validated `plan.json`, exact local primitive source files, table metadata, required output paths, and any parent candidate manifest. When revising code, also read the structured optimizer action and the parent helper and solver files.

If the plan is `not_compilable`, stop without generating a misleading program.

## Generate a complete candidate

1. Map every helper DAG node to one Python function with matching argument and return types.
2. Generate the helper module at the requested path.
3. Generate the end-to-end solver at the requested path.
4. Implement deterministic relational operations in the plan's stated order.
5. Implement the exact SQL result projection and CSV shape.
6. Implement `--only-ids` at the plan-specified sampling-unit boundary.
7. Write `trace_<q>.json` with one entry for every evaluated row, pair, or tuple.
8. Add concise stderr branch diagnostics without printing secrets or complete validation data.
9. Handle individual row/pair failures without corrupting unrelated results.
10. Leave candidate metadata for the orchestrator to hash and finalize.

## Apply optimizer actions

For `PATCH_CODE`, edit only the named artifact or symbol and preserve every listed invariant. Use the parent candidate as the starting point.

For `REPLAN`, regenerate the complete helper and solver from the new plan. Do not mix helper code from an older plan version.

Do not implement requests unsupported by the plan. If an action conflicts with the plan, fail with a clear `NEEDS_REPLAN` diagnostic.

## Enforce runtime constraints

- Do not call a network service, remote LLM, endpoint, shell download, or external API.
- Do not read CERT or final ground truth.
- Do not hardcode validation ids, labels, expected outputs, or mistake rows.
- Do not invent repository APIs.
- Do not silently weaken the semantic predicate.
- Do not change pair direction, diagonal inclusion, trace key format, or result projection.
- Keep confidence values aligned with the primitive's documented score semantics.

## Verify before finishing

Run syntax and static preflight checks that do not execute the full corpus. Confirm imports and referenced names resolve. Confirm helper and solver plan versions match.

Write only the requested helper, solver, and manifest draft paths. The orchestrator owns execution, scoring, hashes, best-candidate selection, and promotion.
