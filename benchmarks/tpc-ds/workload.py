"""TPC-DS workload configuration and setup functions.

Scope: GenDB core path + a DuckDB baseline. PostgreSQL / ClickHouse / Umbra /
MonetDB are intentionally not wired up for TPC-DS yet — the 99 queries need
per-dialect porting, which is left as incremental follow-up work. Those systems
are disabled via the ``skip_*`` flags, so their setup functions are never called.
"""

import json
import re
from pathlib import Path

import duckdb

from benchmarks.lib.config import WorkloadConfig
from benchmarks.lib.utils import strip_fk_constraints


# ---------------------------------------------------------------------------
# Queries — loaded from the auto-generated queries.sql (99 TPC-DS queries)
# ---------------------------------------------------------------------------

def load_queries(queries_path: Path) -> dict:
    """Parse queries.sql (``-- Q<N>: ...`` blocks) into {name: sql}."""
    content = Path(queries_path).read_text()
    queries = {}
    parts = re.split(r"--\s*(Q\d+):\s*([^\n]*)\n", content)
    i = 1
    while i + 2 <= len(parts):
        name = parts[i].strip()
        sql = parts[i + 2].strip()
        sql = re.sub(r"--[^\n]*$", "", sql, flags=re.MULTILINE).strip()
        sql = sql.rstrip(";").strip()
        if sql:
            queries[name] = sql
        i += 3
    return queries


def get_queries(scale_factor: int = 1) -> dict:
    """Return the 99 TPC-DS queries from the generated queries.sql."""
    queries_path = Path(__file__).parent / "queries.sql"
    if not queries_path.exists():
        raise FileNotFoundError(
            f"queries.sql not found at {queries_path}. "
            "Run: bash benchmarks/tpc-ds/setup_data.sh <SCALE_FACTOR>"
        )
    return load_queries(queries_path)


# ---------------------------------------------------------------------------
# DuckDB baseline setup — build a persistent .duckdb from the .tbl data files
# ---------------------------------------------------------------------------

def duckdb_setup(config, force_setup=False):
    db_path = config.duckdb_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = config.data_dir
    tables_json = config.benchmark_root / "tables.json"

    if not force_setup and db_path.exists():
        try:
            conn = duckdb.connect(str(db_path), read_only=True)
            count = conn.execute(f"SELECT COUNT(*) FROM {config.check_table}").fetchone()[0]
            conn.close()
            if count > 0:
                print(f"  Database '{db_path.name}' already exists with data, skipping setup.")
                print(f"  Use --setup flag to force data reload.")
                return
        except Exception:
            pass

    if force_setup and db_path.exists():
        print(f"  Removing existing database '{db_path.name}'...")
        db_path.unlink()

    if not tables_json.exists():
        raise FileNotFoundError(
            f"tables.json not found at {tables_json}. "
            "Run: bash benchmarks/tpc-ds/setup_data.sh <SCALE_FACTOR>"
        )
    meta = json.loads(tables_json.read_text())

    print(f"  Creating persistent database '{db_path.name}'...")
    conn = duckdb.connect(str(db_path))

    # Prefer the generated schema.sql (matches gendb's view); fall back to
    # tables.json column types if schema.sql is absent.
    schema_path = config.benchmark_root / "schema.sql"
    if schema_path.exists():
        schema_no_fk = strip_fk_constraints(schema_path.read_text())
        for stmt in schema_no_fk.split(";"):
            stmt = stmt.strip()
            if stmt and stmt.upper().startswith("CREATE"):
                conn.execute(stmt)
    else:
        for table in meta["load_order"]:
            cols = meta["columns"][table]
            col_defs = ", ".join(f"{n} {t}" for n, t in cols)
            conn.execute(f"CREATE TABLE {table} ({col_defs})")

    for table in meta["load_order"]:
        tbl_file = data_dir / f"{table}.tbl"
        if not tbl_file.exists():
            print(f"  Warning: {tbl_file} not found, skipping")
            continue
        print(f"  Loading {table}...", end="", flush=True)
        conn.execute(
            f"COPY {table} FROM '{tbl_file}' "
            f"(FORMAT csv, DELIMITER '|', HEADER false)"
        )
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f" {count:,} rows")

    conn.close()


# ---------------------------------------------------------------------------
# Stubs for systems not yet supported on TPC-DS (never called: skip_* = True)
# ---------------------------------------------------------------------------

def _not_supported(*_args, **_kwargs):
    raise NotImplementedError(
        "This baseline system is not wired up for TPC-DS yet "
        "(only DuckDB is supported). It should have been skipped."
    )


def get_config(args) -> WorkloadConfig:
    benchmark_root = Path(__file__).parent
    sf = args.sf

    data_dir = getattr(args, "data_dir", None)
    if data_dir is None:
        data_dir = benchmark_root / "data" / f"sf{sf}"
    data_dir = Path(data_dir).resolve()

    queries = get_queries(sf)

    if getattr(args, "output", None) is None:
        figures_dir = benchmark_root.parent / "figures" / "tpc-ds"
        figures_dir.mkdir(parents=True, exist_ok=True)
        args.output = figures_dir / "benchmark_results_per_query.png"

    duckdb_dir = benchmark_root / "duckdb"
    duckdb_dir.mkdir(exist_ok=True)

    return WorkloadConfig(
        name="tpc-ds",
        scale_label=f"SF={sf}",
        scale_value=sf,
        db_name=f"tpcds_sf{sf}",
        check_table="store_sales",
        benchmark_root=benchmark_root,
        queries=queries,
        duckdb_path=duckdb_dir / f"tpcds_sf{sf}.duckdb",
        umbra_port=5442,
        pk_columns={},
        pg_setup_fn=_not_supported,
        duckdb_setup_fn=duckdb_setup,
        clickhouse_setup_fn=_not_supported,
        umbra_setup_fn=_not_supported,
        monetdb_setup_fn=_not_supported,
        monetdb_install_fn=_not_supported,
        clickhouse_tables={},
        clickhouse_order_keys={},
        data_dir=data_dir,
        # DuckDB baseline only for now; other systems skipped.
        skip_postgres=True,
        skip_duckdb=getattr(args, "skip_duckdb", False),
        skip_clickhouse=True,
        skip_umbra=True,
        skip_monetdb=True,
    )
