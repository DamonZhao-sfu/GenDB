#!/usr/bin/env bash
#
# Setup TPC-DS benchmark data + queries using DuckDB's built-in tpcds extension.
#
# Unlike TPC-H (which clones and builds the external dbgen tool), TPC-DS uses
# DuckDB's self-contained `tpcds` extension to generate both the data and the
# 99 standard queries. See gen_tpcds_assets.py for details.
#
# Usage:
#   bash benchmarks/tpc-ds/setup_data.sh [SCALE_FACTOR]
#
# SCALE_FACTOR defaults to 1 (≈1 GB of data).
# Assets are written to benchmarks/tpc-ds/:
#   schema.sql, queries.sql, tables.json, data/sf${SCALE_FACTOR}/*.tbl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCALE_FACTOR="${1:-1}"
DATA_DIR="${SCRIPT_DIR}/data/sf${SCALE_FACTOR}"

echo "=== TPC-DS Data Setup ==="
echo "Scale factor: ${SCALE_FACTOR}"
echo "Data directory: ${DATA_DIR}"

# Step 1: Ensure the DuckDB Python package is available (used for generation).
if ! python3 -c "import duckdb" 2>/dev/null; then
    echo "Installing duckdb Python package..."
    pip3 install --quiet duckdb || pip install --quiet duckdb
fi

# Step 2: Generate data + queries + schema via the tpcds extension.
echo "Generating TPC-DS assets (data, schema, 99 queries)..."
python3 "${SCRIPT_DIR}/gen_tpcds_assets.py" --sf "${SCALE_FACTOR}"

# Step 3: Summarize.
echo ""
echo "=== TPC-DS data ready ==="
echo "Files in ${DATA_DIR}:"
ls -lh "${DATA_DIR}"/*.tbl
echo ""
echo "Row counts:"
for f in "${DATA_DIR}"/*.tbl; do
    name=$(basename "$f")
    count=$(wc -l < "$f")
    echo "  ${name}: ${count} rows"
done
echo ""
echo "Next: generate ground truth results with"
echo "  python3 benchmarks/tpc-ds/generate_ground_truth.py --sf ${SCALE_FACTOR}"
