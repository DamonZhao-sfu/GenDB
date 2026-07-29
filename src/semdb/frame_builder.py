#!/usr/bin/env python3
"""Execute a deterministic EComm prefix before semantic pair validation.

The self-join path targets EComm's ``product_selection`` CTE (q7/q9). The filtered
cross-table path targets q8's left-side ``product_selection`` CTE. The
normalized product view has already performed the ordinary image mapping joins, so
this module extracts the CTE's ordinary WHERE expression, validates/transpiles that
expression with SQLGlot, and executes it in an in-memory DuckDB connection.

Failing closed is intentional: using every product when a prefix cannot be proved
equivalent changes the validation population and can make a rare-positive join look
perfect for the wrong reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import Any

import predicate
import validation_plan


def _cte_body(sql: str, name: str) -> str:
    clean = predicate._strip_strings(sql)
    match = re.search(
        rf"(?:\bWITH\b|,)\s*{re.escape(name)}\s+AS\s*\(",
        clean, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"CTE {name!r} not found")
    open_index = clean.find("(", match.start())
    close_index = predicate._match_paren(sql, open_index)
    if close_index < 0:
        raise ValueError(f"CTE {name!r} has no matching closing parenthesis")
    return sql[open_index + 1:close_index]


def _where_expression(select_sql: str) -> str:
    """The top-level WHERE expression in one SELECT/CTE body."""
    clean = predicate._strip_strings(select_sql)
    depth = 0
    where_start = None
    i = 0
    while i < len(clean):
        ch = clean[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0:
            match = re.match(r"\bWHERE\b", clean[i:], flags=re.IGNORECASE)
            if match:
                where_start = i + match.end()
                break
        i += 1
    if where_start is None:
        return "TRUE"
    expression = select_sql[where_start:].strip().rstrip(";")
    # The deterministic prefix may not itself depend on an AI decision.
    if predicate.AI_CALL.search(predicate._strip_strings(expression)):
        raise ValueError("the CTE WHERE expression contains an AI call")
    return expression


def deterministic_spec(sql: str, *, benchmark: str, query: str = "") -> dict[str, Any]:
    plan = validation_plan.build_plan(sql, benchmark=benchmark, query=query)
    candidate = plan["candidate"]
    if benchmark.lower() != "ecomm":
        raise ValueError(
            f"automatic deterministic self-join frames are not implemented for "
            f"benchmark {benchmark!r}")
    if candidate.get("unit") not in {"pair", "tuple"} or not candidate.get("base"):
        raise ValueError("query has no supported pair/tuple deterministic CTE")
    body = _cte_body(sql, candidate["base"])
    where = _where_expression(body)
    return {"plan": plan, "where": where}


def deterministic_cross_left_spec(
        sql: str, *, benchmark: str, query: str = "") -> dict[str, Any]:
    """Q8's provable structured-side prefix over the normalized product view.

    The source SQL names a nested BigQuery struct field
    ``productDescriptors.description.value``. ``materialize.py`` deliberately
    exposes that exact scalar as ``description`` before CSV serialization; rewrite
    only this known equivalence and let SQLGlot/DuckDB validate everything else.
    """
    plan = validation_plan.build_plan(sql, benchmark=benchmark, query=query)
    candidate = plan["candidate"]
    bases = {str(base).lower() for base in candidate.get("bases", [])}
    if benchmark.lower() != "ecomm" or query.lower() != "q8":
        raise ValueError(
            f"automatic filtered cross-table frames are not implemented for "
            f"{benchmark}.{query}")
    structured_bases = {"styles_details", "product_selection"}
    if (candidate.get("unit") != "pair"
            or "images" not in bases
            or len(bases & structured_bases) != 1
            or len(bases) != 2):
        raise ValueError("ecomm q8 does not have the expected styles_details x images "
                         "pair candidate domain")
    body = _cte_body(sql, "product_selection")
    source_where = _where_expression(body)
    where = re.sub(
        r"\b([A-Za-z_]\w*)\.productDescriptors\.description\.value\b",
        r"\1.description",
        source_where,
        flags=re.IGNORECASE,
    )
    if "productDescriptors.description.value" in where:
        raise ValueError("q8 nested description reference could not be normalized")
    return {
        "plan": plan,
        "where": where,
        "source_where": source_where,
    }


def build_ecomm_frame(products_csv: str, sql_path: str, out_path: str, *,
                      query: str = "") -> dict[str, Any]:
    try:
        import duckdb
        import pandas as pd
        from sqlglot import parse_one
    except ImportError as exc:
        raise SystemExit(
            f"frame_builder needs duckdb, pandas and sqlglot: {exc}") from exc

    with open(sql_path, encoding="utf-8") as handle:
        sql = handle.read()
    spec = deterministic_spec(sql, benchmark="ecomm", query=query)
    sql_sha = hashlib.sha256(sql.encode()).hexdigest()
    product_stat = os.stat(products_csv)
    product_signature = {
        "path": os.path.abspath(products_csv),
        "size": product_stat.st_size,
        "mtime_ns": product_stat.st_mtime_ns,
    }
    try:
        with open(out_path + ".meta.json", encoding="utf-8") as handle:
            cached = json.load(handle)
        if (os.path.exists(out_path)
                and cached.get("sql_sha256") == sql_sha
                and cached.get("products_signature") == product_signature):
            print(f"[frame_builder] inputs unchanged; reusing {out_path}")
            return cached
    except (OSError, ValueError, TypeError):
        pass

    # keep_default_na=False is semantically important: EComm has both the literal
    # string "NA" and the empty string, while q9 explicitly asks for colour1 = ''.
    products = pd.read_csv(products_csv, keep_default_na=False)
    required = {
        "id", "filename", "price", "baseColour", "colour1", "colour2",
        "semantic_text",
    }
    missing = sorted(required - set(products.columns))
    if missing:
        raise SystemExit(
            f"normalized product view {products_csv} lacks: {', '.join(missing)}")
    if products["id"].astype(str).duplicated().any():
        raise SystemExit("normalized product ids must be unique")

    statement = parse_one(
        "SELECT * FROM products AS styles_details WHERE " + spec["where"],
        read="bigquery")
    executable = statement.sql(dialect="duckdb")
    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("products", products)
        filtered = connection.execute(executable).fetch_df()
    finally:
        connection.close()
    filtered["id"] = filtered["id"].astype(str)
    filtered = filtered.sort_values("id", kind="stable")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    filtered.to_csv(out_path, index=False)
    meta = {
        "version": 2,
        "benchmark": "ecomm",
        "query": query,
        "sql_path": os.path.abspath(sql_path),
        "sql_sha256": sql_sha,
        "products_path": os.path.abspath(products_csv),
        "products_signature": product_signature,
        "input_rows": len(products),
        "output_rows": len(filtered),
        "where": spec["where"],
        "candidate": spec["plan"]["candidate"],
        "plan": spec["plan"],
    }
    with open(out_path + ".meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")
    print(
        f"[frame_builder] deterministic prefix: {len(products)} -> "
        f"{len(filtered)} rows; WHERE {spec['where']}")
    print(f"[frame_builder] wrote {out_path}")
    return meta


def build_ecomm_cross_left_frame(
        products_csv: str, sql_path: str, out_path: str, *,
        query: str = "") -> dict[str, Any]:
    """Materialize q8's filtered description rows before STYLES_DETAILS x IMAGES."""
    try:
        import duckdb
        import pandas as pd
        from sqlglot import parse_one
    except ImportError as exc:
        raise SystemExit(
            f"frame_builder needs duckdb, pandas and sqlglot: {exc}") from exc

    with open(sql_path, encoding="utf-8") as handle:
        sql = handle.read()
    spec = deterministic_cross_left_spec(sql, benchmark="ecomm", query=query)
    sql_sha = hashlib.sha256(sql.encode()).hexdigest()
    product_stat = os.stat(products_csv)
    product_signature = {
        "path": os.path.abspath(products_csv),
        "size": product_stat.st_size,
        "mtime_ns": product_stat.st_mtime_ns,
    }
    try:
        with open(out_path + ".meta.json", encoding="utf-8") as handle:
            cached = json.load(handle)
        if (os.path.exists(out_path)
                and cached.get("mode") == "cross_left"
                and cached.get("sql_sha256") == sql_sha
                and cached.get("products_signature") == product_signature):
            print(f"[frame_builder] inputs unchanged; reusing {out_path}")
            return cached
    except (OSError, ValueError, TypeError):
        pass

    products = pd.read_csv(products_csv, keep_default_na=False)
    required = {"id", "productDisplayName", "description"}
    missing = sorted(required - set(products.columns))
    if missing:
        raise SystemExit(
            f"normalized product view {products_csv} lacks: {', '.join(missing)}")
    if products["id"].astype(str).duplicated().any():
        raise SystemExit("normalized product ids must be unique")

    statement = parse_one(
        "SELECT * FROM products AS styles_details WHERE " + spec["where"],
        read="bigquery")
    executable = statement.sql(dialect="duckdb")
    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("products", products)
        filtered = connection.execute(executable).fetch_df()
    finally:
        connection.close()
    filtered["id"] = filtered["id"].astype(str)
    filtered["predicate_text"] = (
        filtered["productDisplayName"].fillna("").astype(str).str.strip()
        + " "
        + filtered["description"].fillna("").astype(str).str.strip()
    ).str.strip()
    filtered = filtered.sort_values("id", kind="stable")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    filtered.to_csv(out_path, index=False)
    meta = {
        "version": 3,
        "mode": "cross_left",
        "benchmark": "ecomm",
        "query": query,
        "sql_path": os.path.abspath(sql_path),
        "sql_sha256": sql_sha,
        "products_path": os.path.abspath(products_csv),
        "products_signature": product_signature,
        "input_rows": len(products),
        "output_rows": len(filtered),
        "source_where": spec["source_where"],
        "where": spec["where"],
        "candidate": spec["plan"]["candidate"],
        "plan": spec["plan"],
    }
    with open(out_path + ".meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")
    print(
        f"[frame_builder] deterministic cross-left prefix: {len(products)} -> "
        f"{len(filtered)} rows; WHERE {spec['source_where']}")
    print(f"[frame_builder] wrote {out_path}")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--query", default="")
    ap.add_argument("--sql", required=True)
    ap.add_argument("--products", required=True,
                    help="Normalized ecomm_products.csv.")
    ap.add_argument("--cross-left", action="store_true",
                    help="Build q8's filtered structured-side frame instead of a "
                         "self-join prefix.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.benchmark.lower() != "ecomm":
        raise SystemExit(
            f"frame_builder does not yet support benchmark {args.benchmark!r}")
    if args.cross_left:
        build_ecomm_cross_left_frame(
            args.products, args.sql, args.out, query=args.query)
    else:
        build_ecomm_frame(args.products, args.sql, args.out, query=args.query)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
