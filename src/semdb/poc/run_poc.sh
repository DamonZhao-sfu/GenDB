#!/usr/bin/env bash
#
# run_poc.sh — end-to-end, GPU-free verification of the SemDB compilation idea.
#
#   Phase B (Extractor)      : one pass over the images -> img_attrs.json schema
#   Phase C (Code Generator) : compiled hash-join + residual VLM  -> compiled.csv
#   Oracle (naive plan)      : M×N VLM.IF                          -> baseline.csv
#   Check                    : compiled.csv MUST equal baseline.csv
#
# Exit code is non-zero if the compiled plan disagrees with the oracle.
set -euo pipefail

cd "$(dirname "$0")"
OUT="./out"
mkdir -p "$OUT"

echo "=== Phase B: Extractor (small VLM, one pass over all images) ==="
python3 mock_extractor.py data/images.csv "$OUT/img_attrs.json"
echo

echo "=== Phase C: compiled query (reuse schema + hash join + residual VLM) ==="
python3 compiled_q7.py data/airlines.csv "$OUT/img_attrs.json" "$OUT/compiled.csv"
echo

echo "=== Oracle: naive M×N semantic execution ==="
python3 baseline_q7.py data/airlines.csv data/images.csv "$OUT/baseline.csv"
echo

echo "=== Check: compiled result == oracle result ==="
if diff -u "$OUT/baseline.csv" "$OUT/compiled.csv"; then
  echo "PASS: compiled plan is result-equivalent to the naive semantic plan."
else
  echo "FAIL: compiled plan diverged from the oracle." >&2
  exit 1
fi

echo
echo "=== Cost summary (single query) ==="
python3 - "$OUT/img_attrs.json" data/airlines.csv <<'PY'
import json, sys, csv
attrs = json.load(open(sys.argv[1]))
airlines = list(csv.DictReader(open(sys.argv[2])))
M, N = len(airlines), len(attrs)
unsure = sum(1 for a in attrs if a["logo_brand"] == "none" or a["conf"] < 0.5)
naive = M * N
compiled = unsure * M                       # residual pairs only
print(f"  images (N)                 : {N}")
print(f"  airlines (M)               : {M}")
print(f"  naive VLM.IF calls         : {naive}  (M x N)")
print(f"  extractions (amortized)    : {N}      (paid once, shared by every logo query)")
print(f"  residual VLM.IF calls      : {compiled}")
print(f"  per-query model calls      : {naive} -> {compiled}")
print(f"  amortized over K logo queries: K*{naive} -> {N} + K*{compiled}")
PY
