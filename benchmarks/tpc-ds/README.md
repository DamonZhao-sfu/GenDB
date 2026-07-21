# TPC-DS benchmark (99 queries)

TPC-DS support follows the same convention-based layout as TPC-H, but its
assets are **generated on setup** rather than committed, because DuckDB's
self-contained `tpcds` extension can produce the data and all 99 standard
queries deterministically.

## Files

| File | Committed? | Purpose |
|---|---|---|
| `gen_tpcds_assets.py` | yes | Uses DuckDB's `tpcds` extension to emit `schema.sql`, `queries.sql`, `tables.json`, and `data/sf<N>/*.tbl`. |
| `setup_data.sh` | yes | Thin wrapper around `gen_tpcds_assets.py` (TPC-H's `dbgen` analogue). |
| `generate_ground_truth.py` | yes | Loads the `.tbl` data, runs all 99 queries in DuckDB, writes `query_results/Q<N>.csv`. |
| `workload.py` | yes | Baseline config for `benchmark.py`. DuckDB baseline only for now. |
| `schema.sql`, `queries.sql`, `tables.json` | generated | Produced by `setup_data.sh`. |
| `data/sf<N>/*.tbl` | generated (gitignored) | Pipe-delimited data files GenDB ingests. |
| `query_results/Q<N>.csv` | generated | Ground-truth oracle results (commit if you want them versioned). |

## Quick start

```bash
# 1. Generate data + schema.sql + queries.sql + tables.json  (needs network on first run
#    to download the DuckDB tpcds extension from extensions.duckdb.org)
bash benchmarks/tpc-ds/setup_data.sh 1          # scale factor 1 (~1 GB)

# 2. Generate ground-truth results (DuckDB as the correctness oracle)
python3 benchmarks/tpc-ds/generate_ground_truth.py --sf 1

# 3. Run GenDB on TPC-DS
node src/gendb/orchestrator.mjs --benchmark tpc-ds --sf 1

# 4. Compare GenDB vs the DuckDB baseline
python3 benchmarks/benchmark.py --benchmark tpc-ds --sf 1 --gendb-run output/tpc-ds-sf1
```

## Baseline scope

Only **DuckDB** is wired up as a baseline for TPC-DS. PostgreSQL, ClickHouse,
Umbra, and MonetDB require per-dialect porting of the 99 queries and are
disabled via the `skip_*` flags in `workload.py` — add them incrementally by
following the TPC-H `workload.py` as a template.
