---
name: plan-semantic-query
description: Plan GenDB SemDB semantic SQL operators into a typed, offline-executable physical plan. Use for initial planning or evidence-driven replanning when identifying AI call sites, determining row/pair/tuple sampling units, selecting local VADAR primitives, defining helper DAGs, preserving relational semantics, specifying trace contracts, or deciding that a query is not compilable with the available local runtime.
---

# Plan Semantic Query

Produce a typed semantic plan. Do not write implementation code.

## Read required inputs

Read the query SQL and natural-language description, table metadata, local primitive implementations, trace contract, and output schema. On a replan, also read the previous plan and the optimizer action.

Treat the primitive implementation files as authoritative. Never invent a function, parameter, return type, score meaning, or model capability.

## Build the plan

1. Identify every semantic call site in the SQL.
2. Normalize each call site into its input columns, predicate, output type, and sampling unit.
3. Distinguish row, cross-table pair, ordered self-pair, grouped, and tuple domains.
4. Preserve deterministic relational operations before and after semantic evaluation.
5. Classify compilability as `exact`, `bounded_approximation`, or `not_compilable`.
6. List every compilability obligation and unresolved semantic requirement.
7. Build a typed helper DAG. Give every helper explicit arguments, return type, dependencies, primitive bindings, and confidence signal.
8. Define the relational plan in execution order.
9. Define runtime inputs, result projection, trace key/value semantics, and `--only-ids` filtering location.
10. State assumptions and invariants explicitly.

## Apply planning rules

- Use only local, offline primitives available in the repository.
- Keep value spaces explicit. Read closed value spaces from database columns when allowed; do not infer them from validation labels.
- Preserve ordered-pair direction and diagonal rules for self-joins.
- Apply `--only-ids` after forming the correct validation unit and before semantic inference.
- Require a trace entry for every evaluated validation unit, including negative decisions.
- Keep physical trace identity separate from the SQL result projection.
- Reject silent fallback to a different semantic predicate.
- Mark a plan `not_compilable` when required information or capability is unavailable.

## Replan from evidence

Preserve all unaffected plan sections. Increment `plan_version`, set `parent_plan_version`, and change only what the optimizer evidence supports.

Do not replan merely because one sample is difficult. Require evidence of a plan-level problem such as a wrong sampling unit, missing input, invalid primitive capability, wrong value space, or impossible trace contract.

## Validate before finishing

Confirm that:

- every semantic site is represented;
- every helper dependency resolves;
- every primitive exists;
- every type edge is compatible;
- the relational plan preserves SQL projection and ordering semantics;
- trace keys match the validation unit;
- the runtime remains offline;
- no CERT, final ground truth, or validation label was used for planning.

Write only the requested `plan.json`. Make it conform to `src/semdb/contracts/semantic-plan.schema.json`.
