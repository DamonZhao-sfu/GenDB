---
name: optimize-semantic-program
description: Diagnose a GenDB SemDB candidate from structured preflight, runtime, SELECT-validation, metric, trace, and history evidence, then choose a bounded code patch, plan revision, or stop action. Use during iterative optimization to separate implementation defects from plan defects, target precision/recall/F1 failures, protect validation integrity, and produce an optimizer action without editing program files.
---

# Optimize Semantic Program

Diagnose the latest candidate and write one structured action. Do not edit Python or rewrite the plan.

## Read required inputs

Read the current plan, candidate manifest, iteration feedback, bounded history, remaining iteration budget, and remaining replan budget.

Treat `data_boundary` as a hard policy. Stop if feedback includes CERT or unauthorized final-ground-truth information.

## Diagnose in order

1. Check contract and preflight failures.
2. Check runtime failures and missing outputs.
3. Check trace coverage, key format, and sampling-unit errors.
4. Check relational projection and join/filter placement.
5. Check primitive arguments, value-space mapping, prompts, thresholds, and confidence semantics.
6. Compare precision and recall to distinguish over-broad from under-broad behavior.
7. Use capped mistakes as supporting evidence, not as cases to memorize.
8. Compare history to avoid repeating a failed action.

## Choose one action

Choose `PATCH_CODE` when the plan is sound and the defect is in implementation, mapping, threshold, prompt wording, control flow, trace handling, or error handling.

Choose `REPLAN` only when evidence shows a plan-level defect such as a wrong sampling unit, missing semantic input, invalid primitive choice, wrong value space, impossible helper type, or incorrect relational placement. Require remaining replan budget.

Choose `STOP` when no evidence-supported safe change remains, the candidate satisfies the configured goal, the budget is exhausted, or the query is not compilable with available primitives.

## Bound the proposed change

Name the target artifact and symbol. State the evidence path, intended change, preserved invariants, expected metric direction, and regression risk.

Prefer one falsifiable change per iteration. Do not request an unrelated rewrite.

## Protect validation integrity

- Never encode row ids, expected labels, false-positive rows, or false-negative rows into code.
- Never infer a hidden label rule from a small validation sample.
- Never request access to CERT or final ground truth.
- Never optimize accuracy alone when class imbalance makes precision/recall/F1 available.
- Never claim a statistical guarantee from point estimates.
- Never change the validation unit or metric definition to make the score look better.

## Validate before finishing

Confirm that the action references the current query and candidate, conforms to the remaining budgets, cites available evidence, and preserves offline runtime, trace identity, and SQL result semantics.

Write only the requested `optimizer_action.json`. Make it conform to `src/semdb/contracts/optimizer-action.schema.json`.
