/**
 * Rendering for the push channel (L0/L1 only).
 *
 * L2-L5 are skills and are pulled by the agent, so nothing here renders them —
 * the discovery hint comes from skills.renderCatalog(). What is pushed is the
 * part an agent cannot be trusted to go looking for: the specific template this
 * exact query matched, and what previously worked or failed on it.
 *
 * GenDB's equivalent (loadMemoryNode) crashes on layers 2-5 because it reads an
 * undefined `n`. Keeping the push channel to L0/L1 removes that class of bug
 * along with the reason for it.
 */

const ADVISORY_HEADER = [
  "## Prior Knowledge (advisory — from past runs, NOT a specification)",
  "",
  "This may be stale or wrong for the query in front of you. The plan schema, the table",
  "metadata and the authoritative primitive API remain the source of truth. Verify before",
  "you rely on any of it. It never contains ground-truth answers, so no value here is a label.",
  "",
];

const approxTokens = (text) => Math.ceil(text.length / 4);

/** Trim to a token budget on a paragraph boundary, marking the cut. */
export function capTokens(text, maxTokens) {
  if (!text) return "";
  if (approxTokens(text) <= maxTokens) return text;
  const maxChars = Math.max(0, maxTokens * 4 - 40);
  const cut = text.slice(0, maxChars);
  const boundary = cut.lastIndexOf("\n");
  return `${boundary > maxChars * 0.6 ? cut.slice(0, boundary) : cut}\n\n_[prior knowledge truncated]_\n`;
}

function objectiveLine(objective) {
  if (!objective || objective.value === null || objective.value === undefined) return "n/a";
  return `${objective.name}=${objective.value} (${objective.direction})`;
}

function bullets(title, items, limit = 6) {
  const list = (items || []).filter(Boolean).slice(0, limit);
  if (!list.length) return [];
  return [`**${title}**`, ...list.map((s) => `- ${s}`), ""];
}

/**
 * @param {{tier:string, score:number, l1:object|null, l0:object|null,
 *          role:"planner"|"generator"|"optimizer", warmStart:object|null}} ctx
 * @param {number} maxTokens
 * @returns {string} markdown, or "" when there is nothing worth pushing
 */
export function renderPreInjection(ctx, maxTokens = 3000) {
  const { tier, score, l1, l0, role, warmStart, referenceIsGood = true } = ctx;
  if (tier === "novel" || !l1) return "";

  const lines = [...ADVISORY_HEADER];
  const c1 = l1.content || {};

  if (referenceIsGood) {
    lines.push(
      tier === "exact"
        ? `A past run solved a query with this exact template (\`${l1.id}\`).`
        : `A past run solved a structurally similar query (\`${l1.id}\`, similarity ${score.toFixed(2)}).`,
      "",
    );
  } else {
    lines.push(
      `A past run attempted this template (\`${l1.id}\`) and FAILED to score on it.`,
      "Treat everything below as a record of what did NOT work. Do not reuse this approach"
      + " unless you can identify and fix the reason it failed.",
      "",
    );
  }

  lines.push(...bullets(
    referenceIsGood ? "Strategies that worked" : "Strategies recorded (unverified — the reference run failed)",
    c1.proven_strategies,
  ));
  lines.push(...bullets("Anti-patterns — these made it worse", c1.anti_patterns));

  const skeleton = c1.plan_skeleton;
  if (skeleton && (role === "planner" || role === "generator")) {
    const parts = [];
    if (skeleton.sampling_unit) parts.push(`sampling unit \`${skeleton.sampling_unit}\``);
    if (skeleton.helper_names?.length) parts.push(`helpers ${skeleton.helper_names.map((h) => `\`${h}\``).join(", ")}`);
    if (skeleton.primitives?.length) parts.push(`primitives ${skeleton.primitives.map((p) => `\`${p}\``).join(", ")}`);
    if (parts.length) {
      lines.push(
        referenceIsGood ? "**Plan shape that succeeded**" : "**Plan shape that FAILED**",
        `- ${parts.join("; ")}`,
        "",
      );
    }
  }

  if (l0) {
    const c0 = l0.content || {};
    lines.push(`**Reference result** (\`${l0.id}\`)`);
    lines.push(`- objective: ${objectiveLine(c0.objective)}`);
    if (c0.iterations !== undefined) {
      lines.push(`- reached in ${c0.iterations} refinement iteration(s)`
        + (c0.replans ? `, ${c0.replans} replan(s)` : ""));
    }
    if (role === "generator" && warmStart?.solverPath) {
      lines.push(`- the promoted helpers and solver from that run were copied into your working`
        + ` directory as a starting point; reconcile them with the current plan before trusting them`);
    }
    lines.push("");
  }

  if (role === "optimizer" && c1.anti_patterns?.length) {
    lines.push("Do not re-try an action listed above as an anti-pattern unless the feedback shows"
      + " the earlier failure cause is gone.", "");
  }

  return capTokens(lines.join("\n"), maxTokens);
}

/**
 * Codex fallback: inline whole skill bodies as text.
 *
 * The Codex SDK exposes no Skill tool, so the pull channel degrades to a push.
 * Information parity, different delivery.
 */
export function renderInlineSkills(skills, maxTokens = 2000) {
  if (!skills.length) return "";
  const head = [
    "## Memory Skills (inlined)",
    "",
    "Your runtime cannot discover skills, so the most relevant learned skills are included",
    "here in full. Same status as prior knowledge: advisory, possibly stale, never labels.",
    "",
  ];
  const parts = [];
  for (const s of skills) {
    const block = `### ${s.name}\n\n${s.body}\n`;
    if (approxTokens([...head, ...parts, block].join("\n")) > maxTokens) break;
    parts.push(block);
  }
  if (!parts.length) return "";
  return [...head, ...parts].join("\n");
}
