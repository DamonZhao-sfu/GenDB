/**
 * Parent-pointer reconstruction for the Experience Graph.
 *
 * The optimize loop records a LINEAR optimization_history.json, but it is a
 * projection of a branching tree: each iteration warm-starts from best-so-far
 * (orchestrator.mjs:1523/1612 — "Copy best code as starting point"). This
 * module recovers the parent edges using the loop's own best-so-far rule, so
 * backfilled trees match what live capture would have produced.
 */

/** Pick the reward for the run's optimization target (falls back to timing_ms). */
export function rewardOf(it, target) {
  if (target === "cold") return it.cold_timing_ms ?? it.timing_ms ?? null;
  return it.hot_timing_ms ?? it.timing_ms ?? null;
}

/**
 * Reconstruct parent pointers for a linear iterations[] list.
 * @returns {Map<number, number|null>} iteration -> parentIteration (null = root)
 */
export function reconstructParents(iterations, target = "hot") {
  const parents = new Map();
  let bestIter = null;
  let bestReward = Infinity;
  for (const it of iterations) {
    const passing = (it.validation ?? "pass") === "pass";
    const reward = rewardOf(it, target);
    // Parent = best-so-far BEFORE this iteration was attempted.
    parents.set(it.iteration, bestIter);
    // Advance best-so-far using the loop's own signal (improved + passing),
    // or seed it with the first passing iteration.
    if (it.improved && passing && reward != null && reward < bestReward) {
      bestIter = it.iteration;
      bestReward = reward;
    } else if (bestIter === null && passing && reward != null) {
      bestIter = it.iteration;
      bestReward = reward;
    }
  }
  return parents;
}
