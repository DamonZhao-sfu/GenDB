#!/usr/bin/env python3
"""Build the source-aware validation plan shared by frame construction and scoring.

This module deliberately plans only facts that can be proved from the SQL.  In
particular, a self-join is an *ordered* domain because the two aliases have distinct
roles.  Whether diagonal pairs exist is derived from an ordinary ``a.col != b.col``
condition, never guessed from the natural-language prompt.

The first implementation milestone consumes the self-join portion of this plan for
EComm q7/q9.  The versioned JSON shape is intentionally broader: later multi-site
validation can add candidate composition without changing call-site identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict
from typing import Any

import predicate


def _fingerprint(site: predicate.CallSite) -> str:
    """Semantic call-site identity; ordered aliases preserve input roles."""
    payload = {
        "kind": site.kind,
        "prompt": " ".join(site.prompt.split()),
        "choices": list(site.choices),
        "shape": site.shape,
        "columns": list(site.columns),
        "roles": list(site.aliases),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _has_cross_alias_inequality(sql: str, left: str, right: str) -> bool:
    """Whether ordinary SQL explicitly excludes a left/right self pair.

    EComm q9 spells this as ``p1.uri != p2.uri``.  Strings are blanked first so an
    example embedded in a prompt cannot accidentally alter the candidate domain.
    The comparison may be written in either direction and may use ``<>``.
    """
    clean = predicate._strip_strings(sql)  # shared SQL-aware string blanking
    ident = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
    lref = rf"\b{re.escape(left)}\.{ident}"
    rref = rf"\b{re.escape(right)}\.{ident}"
    return bool(re.search(rf"(?:{lref}\s*(?:!=|<>)\s*{rref}|"
                          rf"{rref}\s*(?:!=|<>)\s*{lref})", clean,
                          flags=re.IGNORECASE))


def build_plan(sql: str, *, benchmark: str = "", query: str = "",
               sql_path: str | None = None) -> dict[str, Any]:
    sites = predicate.parse_sql(sql)
    rendered_sites = []
    for index, site in enumerate(sites):
        item = asdict(site)
        item["site_id"] = f"s{index}"
        item["source_index"] = index
        item["fingerprint"] = _fingerprint(site)
        rendered_sites.append(item)

    candidate: dict[str, Any] = {"unit": "row"}
    self_sites = [
        (index, site) for index, site in enumerate(sites)
        if site.shape == "pairwise"
        and len(site.aliases) == 2
        and len(set(site.bases)) == 1
    ]
    if len(self_sites) == 1:
        index, site = self_sites[0]
        left, right = site.aliases
        candidate = {
            "unit": "pair",
            "site_id": f"s{index}",
            "aliases": [left, right],
            "base": site.bases[0] if site.bases else "",
            # SQL aliases are role-bearing, even when the predicate happens to be
            # symmetric.  Label reuse may canonicalize later; the candidate domain may
            # not.
            "ordered": True,
            "include_diagonal": not _has_cross_alias_inequality(
                sql, left, right),
        }
    elif len(sites) > 1:
        # A common two-stage SQL shape first keeps a cross-table pair with AI.IF,
        # then extracts a value from that same matched image with AI.GENERATE.  The
        # root validation unit is still the pair; its typed label jointly represents
        # both sites as either "no_match" or "match:<generated value>".  Keeping this
        # composition explicit lets one Oracle call validate the whole query without
        # leaking either intermediate label into the optimizer.
        first, second = sites[0], sites[1]
        if (len(sites) == 2 and first.kind == "if"
                and first.shape == "pairwise"
                and second.kind in {"generate", "classify"}
                and second.shape == "per_row"):
            candidate = {
                "unit": "pair",
                "site_id": "s0",
                "aliases": list(first.aliases),
                "bases": list(first.bases),
                "ordered": True,
                "include_diagonal": False,
                "composition": {
                    "kind": "filter_then_extract",
                    "gate_site": "s0",
                    "value_site": "s1",
                    "negative_value": "no_match",
                    "positive_prefix": "match:",
                },
            }
            return {
                "version": 2,
                "benchmark": benchmark,
                "query": query,
                "sql_path": os.path.abspath(sql_path) if sql_path else None,
                "sql_sha256": hashlib.sha256(sql.encode()).hexdigest(),
                "candidate": candidate,
                "sites": rendered_sites,
            }
        aliases: list[str] = []
        for site in sites:
            for alias in site.aliases:
                if alias not in aliases:
                    aliases.append(alias)
        nonempty_bases = [base for site in sites for base in site.bases if base]
        common_bases = set(nonempty_bases)
        # q10/q11 are conjunctions over several aliases of one deterministic CTE.
        # Their root validation unit is the ordered alias tuple, not an independent
        # sample for each predicate.
        if (aliases and len(aliases) >= 2 and len(common_bases) == 1
                and all(site.kind == "if" for site in sites)):
            candidate = {
                "unit": "tuple",
                "aliases": aliases,
                "base": next(iter(common_bases)),
                "arity": len(aliases),
                "ordered": True,
                "composition": {
                    "kind": "boolean_and",
                    "sites": [f"s{i}" for i in range(len(sites))],
                },
            }
        else:
            candidate = {
                "unit": "multi_site",
                "aliases": aliases,
                "kinds": [site.kind for site in sites],
                "reason": "typed/grouped multi-site composition planning required",
            }
    elif sites and sites[0].shape == "pairwise":
        candidate = {
            "unit": "pair",
            "site_id": "s0",
            "aliases": list(sites[0].aliases),
            "bases": list(sites[0].bases),
            "ordered": True,
            "include_diagonal": False,
        }

    return {
        "version": 2,
        "benchmark": benchmark,
        "query": query,
        "sql_path": os.path.abspath(sql_path) if sql_path else None,
        "sql_sha256": hashlib.sha256(sql.encode()).hexdigest(),
        "candidate": candidate,
        "sites": rendered_sites,
    }


def build_file(path: str, *, benchmark: str = "", query: str = "") -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return build_plan(handle.read(), benchmark=benchmark, query=query, sql_path=path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sql")
    ap.add_argument("--benchmark", default="")
    ap.add_argument("--query", default="")
    ap.add_argument("--out", help="Write JSON here instead of stdout.")
    args = ap.parse_args(argv)
    plan = build_file(args.sql, benchmark=args.benchmark, query=args.query)
    text = json.dumps(plan, indent=2) + "\n"
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
