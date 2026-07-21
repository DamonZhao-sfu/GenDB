#!/usr/bin/env python3
"""Generate ground truth query results for TPC-DS using DuckDB.

Reads tables.json (table metadata) and queries.sql, loads the pipe-delimited
.tbl data files, runs each of the 99 queries, and saves results to
benchmarks/tpc-ds/query_results/Q<N>.csv.

DuckDB acts as the correctness oracle (same approach as TPC-H). Run
setup_data.sh first to produce data/, schema.sql, queries.sql and tables.json.

Usage:
    python3 benchmarks/tpc-ds/generate_ground_truth.py --sf 1
    python3 benchmarks/tpc-ds/generate_ground_truth.py --sf 10 --data-dir /path/to/data
"""

import argparse
import csv
import json
import os
import re
import sys

import duckdb


def parse_queries(queries_path):
    """Parse queries.sql into a dict of {query_name: sql}.

    Blocks are delimited by ``-- Q<N>: <description>`` headers (same format as
    the TPC-H parser).
    """
    with open(queries_path, "r") as f:
        content = f.read()

    queries = {}
    parts = re.split(r"--\s*(Q\d+):\s*([^\n]*)\n", content)
    # parts[0] is preamble, then groups of (name, description, sql)
    i = 1
    while i + 2 <= len(parts):
        name = parts[i].strip()
        sql = parts[i + 2].strip()
        # Drop trailing line comments / blank tail.
        sql = re.sub(r"--[^\n]*$", "", sql, flags=re.MULTILINE).strip()
        if sql and not sql.endswith(";"):
            sql += ";"
        if sql:
            queries[name] = sql
        i += 3

    return queries


def load_tpcds_data(con, tables_json_path, data_dir):
    """Create tables from tables.json and load .tbl data into DuckDB."""
    with open(tables_json_path, "r") as f:
        meta = json.load(f)

    load_order = meta["load_order"]
    columns = meta["columns"]

    for table in load_order:
        cols = columns[table]
        col_defs = ", ".join(f"{name} {ctype}" for name, ctype in cols)
        con.execute(f"CREATE TABLE {table} ({col_defs})")

        tbl_path = os.path.join(data_dir, f"{table}.tbl")
        if not os.path.exists(tbl_path):
            print(f"  Warning: {tbl_path} not found, skipping {table}")
            continue
        print(f"  Loading {table} from {tbl_path}...")
        # Empty fields are read back as NULL (nullstr defaults to '').
        con.execute(
            f"COPY {table} FROM '{tbl_path}' "
            f"(FORMAT csv, DELIMITER '|', HEADER false, AUTO_DETECT false)"
        )
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"    {table}: {count:,} rows")


def main():
    parser = argparse.ArgumentParser(description="Generate TPC-DS ground truth results")
    parser.add_argument("--sf", type=int, default=1, help="Scale factor (default: 1)")
    parser.add_argument("--data-dir", type=str, help="Path to .tbl files directory")
    parser.add_argument("--output-dir", type=str, help="Output dir for query results")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    queries_path = os.path.join(script_dir, "queries.sql")
    tables_json_path = os.path.join(script_dir, "tables.json")

    data_dir = args.data_dir or os.path.join(script_dir, "data", f"sf{args.sf}")
    output_dir = args.output_dir or os.path.join(script_dir, "query_results")

    for required, hint in [
        (queries_path, "queries.sql"),
        (tables_json_path, "tables.json"),
    ]:
        if not os.path.exists(required):
            print(f"Error: {hint} not found at {required}")
            print("Run: bash benchmarks/tpc-ds/setup_data.sh <SCALE_FACTOR>")
            sys.exit(1)

    if not os.path.exists(data_dir):
        print(f"Error: data directory not found: {data_dir}")
        print("Run: bash benchmarks/tpc-ds/setup_data.sh <SCALE_FACTOR>")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Scale factor: {args.sf}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    con = duckdb.connect(":memory:")
    print("\nLoading TPC-DS data...")
    load_tpcds_data(con, tables_json_path, data_dir)

    queries = parse_queries(queries_path)
    ordered = sorted(queries.items(), key=lambda x: int(x[0][1:]))
    print(f"\nFound {len(queries)} queries: {', '.join(n for n, _ in ordered)}")

    n_ok, n_err = 0, 0
    for name, sql in ordered:
        print(f"\nRunning {name}...")
        try:
            result = con.execute(sql)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()

            output_path = os.path.join(output_dir, f"{name}.csv")
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                for row in rows:
                    processed = []
                    for val in row:
                        if isinstance(val, float):
                            processed.append(f"{val:.2f}")
                        elif val is None:
                            processed.append("")
                        else:
                            processed.append(str(val))
                    writer.writerow(processed)

            print(f"  {name}: {len(rows)} rows -> {output_path}")
            n_ok += 1
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
            n_err += 1

    con.close()
    print(f"\nGround truth generated in: {output_dir}")
    print(f"  {n_ok} succeeded, {n_err} failed")
    if n_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
