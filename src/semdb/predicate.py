"""Extract the AI call sites from a SemBench BigQuery-dialect query.

The oracle question a validation set needs is already written in the SQL -- there is
no reason to retype it as ``--query-nl`` and risk paraphrasing the thing being
measured. Every SemBench AI operator has the same shape::

    AI.IF( (<prompt-and-column tuple>), connection_id => '...', model_params => ... )

so one scanner handles ``AI.IF`` / ``AI.GENERATE`` / ``AI.CLASSIFY`` / ``AI.SCORE``.

The other thing this module decides is the **label shape**, which determines whether
a per-row validation set is even defined for a call site:

    per_row    the predicate is a function of ONE row. A val set maps id -> label.
    pairwise   the predicate is a function of TWO rows. A per-row label does not
               exist; the val set must key on a pair, "<id1>-<id2>" (the format
               SemBench's own join ground truth already uses).

Two tests produce that classification:

  1. **Self-join.** Two aliases in the same call resolving to the same base table or
     CTE means the call compares a row to another row of the same relation --
     ``FROM images AS images1, images AS images2`` (q10/q11), ``FROM product_selection
     p1 LEFT OUTER JOIN product_selection p2`` (q9).
  2. **Deterministic-join connectivity.** Otherwise, the call is per-row only if all
     its aliases are already tied together 1:1 by ordinary (non-AI) equi-joins. The
     equality conditions in the query form a graph over aliases; aliases in one
     connected component denote the same entity, so a predicate over them is a
     function of one row.

Counting distinct aliases is NOT sufficient: q13 references two aliases
(``images`` and ``styles_details``) but a chain of equi-joins through IMAGE_MAPPING
puts them in one component, so the predicate is a function of one product -- per_row.
q8 and q14 reference the same two aliases with *no* deterministic edge between them,
because the only thing joining STYLES_DETAILS to IMAGES is the AI predicate itself --
pairwise. Using "does the call sit in a JOIN ... ON clause" instead would get q14's
``AI.SCORE`` wrong, since that one lives in an ORDER BY while still ranging over the
same unjoined cross product.

A single query commonly contains BOTH shapes. q10 has three per-row filters
("is it a shoe?", "is it bottomwear?", "is it topwear?") alongside two pairwise
"same colour and brand" predicates, so a join query still yields per-row validation
sets for its base-table filters.

CLI::

    python3 predicate.py q10.sql            # table of call sites
    python3 predicate.py q10.sql --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass

AI_CALL = re.compile(r"\bAI\.(IF|GENERATE|CLASSIFY|SCORE)\s*\(", re.IGNORECASE)

# alias.column, alias.a.b.c -- the leading identifier is the alias.
COLUMN_REF = re.compile(r"\b([A-Za-z_]\w*)((?:\.[A-Za-z_]\w*)+)\b")
BARE_IDENTIFIER = re.compile(r"\b([A-Za-z_]\w*)\b")

# Identifiers that look like a column reference but are SQL/BigQuery constructs.
NOT_AN_ALIAS = frozenset({
    "json", "ai", "safe", "array", "struct", "unnest", "cast", "timestamp",
    "date", "datetime", "interval", "numeric", "bignumeric",
})

# Bare identifiers inside the first AI argument are unqualified column references in
# queries such as mmqa q6 (`Destinations || Airlines`). Keep SQL syntax and named
# arguments out of CallSite.columns; qualified alias.column references are handled by
# COLUMN_REF before this pass.
NOT_A_BARE_COLUMN = frozenset({
    *NOT_AN_ALIAS,
    "prompt", "connection_id", "model_params", "output_schema", "endpoint",
    "true", "false", "null", "struct", "json", "array", "string",
    "if", "generate", "classify", "score", "as", "and", "or", "not",
})

# `FROM x AS y`, `JOIN x y`, `, x AS y` -- BigQuery allows the AS to be omitted.
# EXTERNAL_OBJECT_TRANSFORM(...) is handled separately since it wraps the table.
_ALIAS_TAIL = r"(?:\s+AS)?\s+([A-Za-z_]\w*)"
FROM_SIMPLE = re.compile(
    r"(?:\bFROM\b|\bJOIN\b|,)\s+(`?[A-Za-z_][\w.]*`?)" + _ALIAS_TAIL, re.IGNORECASE)
# Scan explicit FROM/JOIN sources in a separate pass.  In ``SELECT a.id, b.id FROM
# products a`` the broad comma pattern above can otherwise consume `, b.id FROM` as
# a fake table occurrence and advance past the real FROM token.
FROM_JOIN_SIMPLE = re.compile(
    r"(?:\bFROM\b|\bJOIN\b)\s+(`?[A-Za-z_][\w.]*`?)" + _ALIAS_TAIL, re.IGNORECASE)
FROM_TRANSFORM = re.compile(
    r"EXTERNAL_OBJECT_TRANSFORM\s*\(\s*TABLE\s+`?([\w.]+)`?.*?\)" + _ALIAS_TAIL,
    re.IGNORECASE | re.DOTALL)
CTE_NAME = re.compile(r"(?:\bWITH\b|,)\s*([A-Za-z_]\w*)\s+AS\s*\(", re.IGNORECASE)

# Keywords that must never be read as a table alias (`FROM x WHERE`, `JOIN y ON`).
ALIAS_STOPWORDS = frozenset({
    "on", "where", "using", "join", "inner", "left", "right", "full", "outer",
    "cross", "group", "order", "limit", "having", "window", "qualify", "union",
    "select", "as", "and", "or", "when", "then", "else", "end",
    "from",
})


@dataclass(frozen=True)
class CallSite:
    """One AI.* invocation, with everything an oracle labeler needs to reproduce it."""
    kind: str                       # if | generate | classify | score
    prompt: str                     # the string literals, concatenated in order
    aliases: tuple[str, ...]        # distinct table aliases referenced, in order
    bases: tuple[str, ...]          # base table/CTE each alias resolves to
    columns: tuple[str, ...]        # full alias.col references, in order
    choices: tuple[str, ...]        # AI.CLASSIFY categories; () otherwise
    shape: str                      # per_row | pairwise
    reason: str                     # why it got that shape
    line: int                       # 1-based line of the call in the SQL
    start: int                      # source offset of `AI.<kind>` (inclusive)
    end: int                        # source offset just after the closing `)`

    @property
    def is_pairwise(self) -> bool:
        return self.shape == "pairwise"


# --------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------

def _skip_string(sql: str, i: int) -> int:
    """Index just past the string literal starting at `i`. Handles ''' and \"\"\".

    Returns `i` unchanged when `sql[i]` does not open a string.
    """
    quote = sql[i]
    if quote not in "'\"":
        return i
    triple = sql[i:i + 3] in ("'''", '"""')
    delim = quote * 3 if triple else quote
    j = i + len(delim)
    while j < len(sql):
        if sql[j] == "\\":
            j += 2
            continue
        if sql.startswith(delim, j):
            return j + len(delim)
        j += 1
    return len(sql)          # unterminated: consume the rest rather than loop


def _match_paren(sql: str, open_idx: int) -> int:
    """Index of the `)` matching the `(` at `open_idx`, skipping string literals."""
    depth = 0
    i = open_idx
    while i < len(sql):
        ch = sql[i]
        if ch in "'\"":
            i = _skip_string(sql, i)
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _split_top_level(body: str) -> list[str]:
    """Split on commas that are not inside parens/brackets/strings."""
    parts, depth, start, i = [], 0, 0, 0
    while i < len(body):
        ch = body[i]
        if ch in "'\"":
            i = _skip_string(body, i)
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(body[start:i])
            start = i + 1
        i += 1
    parts.append(body[start:])
    return [p.strip() for p in parts if p.strip()]


def _string_literals(text: str) -> list[str]:
    """Every string literal in `text`, in order, unquoted."""
    out, i = [], 0
    while i < len(text):
        if text[i] in "'\"":
            end = _skip_string(text, i)
            quote = text[i]
            delim = quote * 3 if text[i:i + 3] in ("'''", '"""') else quote
            out.append(text[i + len(delim):end - len(delim)])
            i = end
            continue
        i += 1
    return out


def _strip_strings(text: str) -> str:
    """`text` with string literals blanked, so column scanning cannot read prose."""
    out, i = [], 0
    while i < len(text):
        if text[i] in "'\"":
            end = _skip_string(text, i)
            out.append(" " * (end - i))
            i = end
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _prompt_text(expr: str, columns: list[str]) -> str:
    """Recover a readable prompt from a SQL concatenation expression.

    Concatenating only string literals turns mmqa q6 into
    ``Given destinations '' of , ...``. For ``||`` expressions, retain column
    references as named placeholders so the oracle can connect the question to the
    separately supplied row fields. Ordinary tuple/STRUCT prompts keep their
    historical literal-only rendering.
    """
    if "||" not in _strip_strings(expr):
        return "".join(_string_literals(expr)).strip()

    parts: list[str] = []
    i = 0
    while i < len(expr):
        if expr[i] in "'\"":
            end = _skip_string(expr, i)
            quote = expr[i]
            delim = quote * 3 if expr[i:i + 3] in ("'''", '"""') else quote
            parts.append(expr[i + len(delim):end - len(delim)])
            i = end
            continue
        qualified = COLUMN_REF.match(expr, i)
        if qualified:
            name = qualified.group(0)
            if name in columns:
                parts.append("{" + name + "}")
            i = qualified.end()
            continue
        bare = BARE_IDENTIFIER.match(expr, i)
        if bare:
            name = bare.group(1)
            if name in columns:
                parts.append("{" + name + "}")
            i = bare.end()
            continue
        i += 1
    return "".join(parts).strip()


# --------------------------------------------------------------------------
# alias resolution
# --------------------------------------------------------------------------

def alias_map(sql: str) -> dict[str, str]:
    """alias -> base table/CTE name, lowercased.

    Later definitions win, which is what shadowing in nested scopes implies for the
    only thing this map is used for (deciding whether two aliases are the same
    relation).
    """
    scan = _strip_strings(sql)
    out: dict[str, str] = {}
    for pattern in (FROM_JOIN_SIMPLE, FROM_SIMPLE, FROM_TRANSFORM):
        for match in pattern.finditer(scan):
            table, alias = match.group(1).strip("`"), match.group(2)
            if alias.lower() in ALIAS_STOPWORDS:
                continue
            out[alias] = table.split(".")[-1].lower()
    return out


def cte_names(sql: str) -> set[str]:
    return {m.group(1).lower() for m in CTE_NAME.finditer(_strip_strings(sql))}


# --------------------------------------------------------------------------
# shape
# --------------------------------------------------------------------------

def _blank_ai_calls(sql: str) -> str:
    """`sql` with every AI.*(...) span blanked.

    Join connectivity must be derived from DETERMINISTIC conditions only. An AI
    predicate used as a join condition is exactly the thing being validated, so
    letting it contribute an edge would declare its own operands 'already joined'
    and classify the call as per-row -- the error this module exists to avoid.
    """
    scan = list(_strip_strings(sql))
    for match in AI_CALL.finditer("".join(scan)):
        close = _match_paren(sql, match.end() - 1)
        if close < 0:
            continue
        for i in range(match.start(), close + 1):
            scan[i] = " "
    return "".join(scan)


def join_components(sql: str) -> list[set[str]]:
    """Connected components of aliases under ordinary equi-join conditions.

    Every `a.x = b.y` (however wrapped in functions) ties `a` and `b` to the same
    entity. Union-find over those edges; aliases sharing a component denote one row.
    """
    scan = _blank_ai_calls(sql)
    parent: dict[str, str] = {}

    def find(a: str) -> str:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def aliases_in(text: str) -> set[str]:
        return {m.group(1) for m in COLUMN_REF.finditer(text)
                if m.group(1).lower() not in NOT_AN_ALIAS}

    # Conjunct-level: split on AND/OR/ON/WHERE so one `=` is examined at a time.
    for conjunct in re.split(r"\b(?:AND|OR|ON|WHERE)\b", scan, flags=re.IGNORECASE):
        # `=` but not `>=`, `<=`, `!=`, `=>` (BigQuery named arguments).
        eq = re.search(r"(?<![<>!=])=(?![=>])", conjunct)
        if not eq:
            continue
        left, right = aliases_in(conjunct[:eq.start()]), aliases_in(conjunct[eq.end():])
        for a in left:
            for b in right:
                union(a, b)

    groups: dict[str, set[str]] = {}
    for alias in parent:
        groups.setdefault(find(alias), set()).add(alias)
    return list(groups.values())


def classify(aliases: tuple[str, ...], bases: tuple[str, ...],
             components: list[set[str]]) -> tuple[str, str]:
    """(shape, reason) for one call site."""
    known = [b for b in bases if b]
    if len(known) >= 2 and len(set(known)) < len(known):
        dup = next(b for b in known if known.count(b) > 1)
        return "pairwise", f"self-join: aliases {', '.join(aliases)} all resolve to {dup!r}"
    if len(aliases) >= 2:
        joined = any(set(aliases) <= comp for comp in components)
        if joined:
            return "per_row", (f"aliases {', '.join(aliases)} are tied 1:1 by ordinary "
                               f"equi-joins, so the predicate is a function of one row")
        return "pairwise", (f"aliases {', '.join(aliases)} are not connected by any "
                            f"non-AI join condition -- the AI predicate itself is what "
                            f"pairs them")
    return "per_row", "single row alias"


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def parse_sql(sql: str) -> list[CallSite]:
    """Every AI.* call site in `sql`, in source order."""
    amap = alias_map(sql)
    ctes = cte_names(sql)
    components = join_components(sql)
    sites: list[CallSite] = []

    for match in AI_CALL.finditer(_strip_strings(sql)):
        open_idx = match.end() - 1
        close_idx = _match_paren(sql, open_idx)
        if close_idx < 0:
            continue
        args = _split_top_level(sql[open_idx + 1:close_idx])
        if not args:
            continue

        # The prompt tuple is the first positional argument. It is normally
        # parenthesized; a lone string literal is accepted too.
        tup = args[0]
        if tup.startswith("(") and tup.endswith(")"):
            tup = tup[1:-1]

        aliases: list[str] = []
        columns: list[str] = []
        stripped_tup = _strip_strings(tup)
        qualified_spans: list[tuple[int, int]] = []
        for ref in COLUMN_REF.finditer(stripped_tup):
            alias, tail = ref.group(1), ref.group(2)
            if alias.lower() in NOT_AN_ALIAS:
                continue
            columns.append(alias + tail)
            qualified_spans.append(ref.span())
            if alias not in aliases:
                aliases.append(alias)

        choices: list[str] = []
        kind = match.group(1).lower()
        if kind == "if":
            # AI.IF is a typed boolean operator even though its SQL syntax does not
            # spell out categories. Passing this value space to guided decoding keeps
            # an oracle from answering with an entity copied from the input.
            choices.extend(("true", "false"))
        elif kind == "classify":
            for arg in args[1:]:
                if arg.lower().startswith("categories"):
                    # categories => [('Dress', 'desc'), ('Socks', 'desc'), ...]
                    inner = arg[arg.index("[") + 1:arg.rindex("]")] if "[" in arg else ""
                    for cat in _split_top_level(inner):
                        lits = _string_literals(cat)
                        if lits:
                            choices.append(lits[0])

        # Capture unqualified columns after blanking every qualified reference. This
        # is required for mmqa q6, whose AI.IF expression names `Destinations` and
        # `Airlines` without a table alias.
        bare_scan = list(stripped_tup)
        for start, end in qualified_spans:
            bare_scan[start:end] = " " * (end - start)
        bare_text = "".join(bare_scan)
        for ref in BARE_IDENTIFIER.finditer(bare_text):
            name = ref.group(1)
            if name.lower() in NOT_A_BARE_COLUMN:
                continue
            # Function calls (`CONCAT(...)`) and named arguments (`prompt =>`) are
            # syntax, not columns.
            tail = bare_text[ref.end():]
            if re.match(r"\s*(?:\(|=>)", tail):
                continue
            if name not in columns:
                columns.append(name)

        prompt = _prompt_text(tup, columns)
        bases = tuple(amap.get(a, a.lower() if a.lower() in ctes else "") for a in aliases)
        shape, reason = classify(tuple(aliases), bases, components)
        sites.append(CallSite(
            kind=kind, prompt=prompt,
            aliases=tuple(aliases), bases=bases, columns=tuple(columns),
            choices=tuple(choices), shape=shape, reason=reason,
            line=sql.count("\n", 0, match.start()) + 1,
            start=match.start(), end=close_idx + 1,
        ))
    return sites


def parse_file(path: str) -> list[CallSite]:
    with open(path, encoding="utf-8") as handle:
        return parse_sql(handle.read())


def per_row_sites(sites: list[CallSite]) -> list[CallSite]:
    return [s for s in sites if s.shape == "per_row"]


def pick_site(sites: list[CallSite], index: int | None, shape: str | None) -> CallSite:
    """Select one call site, failing loudly rather than guessing.

    A query with several predicates has no single "the" predicate, so an ambiguous
    selection must be an error: silently taking the first one would build a
    validation set for a different question than the caller believes.
    """
    if index is not None:
        if not 0 <= index < len(sites):
            raise ValueError(f"call-site index {index} out of range (0..{len(sites) - 1})")
        return sites[index]
    # Index by position, never by value: two call sites can compare equal as frozen
    # dataclasses (q11 repeats the same predicate over different alias pairs).
    pool = [(i, s) for i, s in enumerate(sites) if not shape or s.shape == shape]
    if not pool:
        if not sites:
            raise ValueError("this query has no AI.IF/AI.GENERATE/AI.CLASSIFY/AI.SCORE call")
        # Saying only "no per_row call site" reads as "no AI call found". Name what IS
        # there, so the caller can tell a parse failure from a shape mismatch.
        have = ", ".join(f"[{i}] {s.kind} {s.shape} (line {s.line}: {s.reason})"
                         for i, s in enumerate(sites))
        raise ValueError(
            f"no {shape} call site in this query; its {len(sites)} AI call site(s) are: {have}")
    if len(pool) > 1:
        listing = "\n".join(
            f"  [{i}] line {s.line} {s.kind} {s.shape}: {s.prompt[:70]}..." for i, s in pool)
        raise ValueError(
            f"{len(pool)} {shape or 'AI'} call sites -- pass --call-site <index>:\n{listing}")
    return pool[0][1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="List the AI call sites in a SemBench query.")
    ap.add_argument("sql", help="Path to the .sql file.")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    ap.add_argument("--shape", choices=["per_row", "pairwise"], help="Filter by shape.")
    args = ap.parse_args(argv)

    sites = parse_file(args.sql)
    if args.shape:
        sites = [s for s in sites if s.shape == args.shape]
    if args.json:
        json.dump([asdict(s) for s in sites], sys.stdout, indent=2)
        print()
        return 0
    if not sites:
        print("(no AI call sites)")
        return 0
    for i, s in enumerate(sites):
        print(f"[{i}] line {s.line}  {s.kind.upper():8s} {s.shape:8s} "
              f"aliases={','.join(s.aliases) or '-'}  ({s.reason})")
        print(f"     prompt: {s.prompt[:110]}{'...' if len(s.prompt) > 110 else ''}")
        if s.choices:
            print(f"     choices: {', '.join(s.choices)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
