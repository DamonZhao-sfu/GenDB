---
name: optimize-semantic-program
description: Diagnose a GenDB SemDB candidate from structured preflight, runtime, SELECT-validation, query-specific metric, trace, and history evidence, then choose a bounded code patch, plan revision, or stop action. Use during iterative optimization to separate implementation defects from plan defects, target the official metric family, protect validation integrity, and produce an optimizer action without editing program files.
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
6. Diagnose according to `objective.name`, `objective.direction`, and
   `objective.details`; do not assume the objective is F1.
7. Use capped mistakes as supporting evidence, not as cases to memorize.
8. Compare history to avoid repeating a failed action.

## Use metric-specific evidence

- For `f1` or `predicate_fidelity_f1`, compare precision and recall and use binary
  false-positive/false-negative totals when present.
- For `macro_f1`, look for class collapse, missing labels, and minority-class recall.
  Row label mismatches are diagnostic; binary FP/FN terminology is not.
- For `adjusted_rand_index`, optimize partition agreement. Category names may be
  permuted without hurting ARI, so focus on incorrect merges, splits, missing rows,
  and inconsistent assignments. Do not reinterpret operator accuracy or binary F1
  as ARI.
- For `relative_error` or `mape`, read expected/predicted aggregate details and
  remember that the direction is `minimize`. A row-level fidelity increase matters
  only if it improves the aggregate.
- For `spearman_correlation`, diagnose ordering and score ties; exact row-label
  accuracy is not the ranking objective.
- For `query_metric_unavailable`, do not invent a surrogate query score. Repair
  compile/runtime/trace-contract failures if present; otherwise choose `STOP`.

`operator_fidelity` is a separate diagnostic. It can explain the query metric but
must never replace a measurable query-specific objective during candidate selection.

## Choose one action

Choose `PATCH_CODE` when the plan is sound and the defect is in implementation, mapping, threshold, prompt wording, control flow, trace handling, or error handling.

Choose `REPLAN` only when evidence shows a plan-level defect such as a wrong sampling unit, missing semantic input, invalid primitive choice, wrong value space, impossible helper type, or incorrect relational placement. Require remaining replan budget.

Full recall with near-zero precision on an image predicate is an **invalid primitive
choice**, not an imprecise prompt. When the site is bound to a single `verify_property`
over a named entity, choose `REPLAN` while budget remains; the fix is a discriminative
step (OCR or closed-set classify over the runtime value space), a cheap gate, a
`*_detail` threshold, or an assignment/dedup rule.

Never ask the Generator to lengthen a CLIP phrase, add exclusions, or add "reject X"
wording: CLIP reads ~77 tokens, compares phrases rather than sentences, and does not
process negation, so a longer prompt lowers precision. Treat a repeated "make the prompt
stricter" request as already shown to regress once precision has failed to rise.

`execution.selected_rows == 0` is the emergency case, and it outranks every other
diagnosis. An empty result scores F1 0 and freezes the branch counters, so every later
iteration sees identical evidence and the loop learns nothing. Remove or loosen the most
arbitrary rejection rule immediately — an absolute CLIP cutoff such as `>= 0.5` is
almost always the cause, because CLIP scores are not calibrated across images or prompts.
Restoring output outranks preserving strictness: a permissive candidate can be tightened
from its false positives, an empty one cannot be tightened from anything. Identical
`runtime_branches` across two iterations likewise mean the previous action had no effect;
change a different factor and name the counter you expect to move.

Choose `STOP` when no evidence-supported safe change remains, the candidate satisfies the configured goal, the budget is exhausted, or the query is not compilable with available primitives.

## Bound the proposed change

Name the target artifact and symbol. State the evidence path, intended change, preserved invariants, expected metric direction, and regression risk.

Prefer one falsifiable change per iteration. Do not request an unrelated rewrite.

## Protect validation integrity

- Never encode row ids, expected labels, false-positive rows, or false-negative rows into code.
- Never infer a hidden label rule from a small validation sample.
- Never request access to CERT or final ground truth.
- Never optimize accuracy alone when class imbalance makes precision/recall/F1 available.
- Never report binary FP/FN analysis for ARI, ranking, aggregation, or multiclass
  macro-F1 feedback.
- Never claim a statistical guarantee from point estimates.
- Never change the validation unit or metric definition to make the score look better.

## Validate before finishing

Confirm that the action references the current query and candidate, conforms to the remaining budgets, cites available evidence, and preserves offline runtime, trace identity, and SQL result semantics.

Write only the requested `optimizer_action.json`. Make it conform to `src/semdb/contracts/optimizer-action.schema.json`.
