#!/usr/bin/env bash
# run_image_queries.sh — run ALL of SemBench's image queries through the SemDB orchestrator.
#
# For every scenario, this runs the whole query dir with `--image-only`, so the orchestrator
# self-selects only the queries whose corpus is an image table (skipping text- and
# audio-only queries). By default it uses DIRECT mode: the strict VADAR 3 agents
# (Signature -> API -> Solver) synthesize ONE end-to-end solve_<q>.py per query that calls
# the local vision API (CLIP/OCR/CV) and answers the whole query — no schema design, no
# extract/compile split. Results (P/R/F1 + telemetry) are appended to $OUT/results.csv.
#
# Image queries selected (movie is text-only, so it has none):
#   mmqa    q2a q2b q7                      (logo recognition)
#   cars    Q3 Q5 Q6 Q7 Q8 Q9              (car image predicates)
#   medical Q3 Q5 Q6 Q7 Q8 Q9 Q11 (+Q7_filter)  (x-ray / skin image predicates)
#   animals Q1 Q3 Q5 Q6 Q7 Q8 Q9 Q10       (animal image predicates)
#   ecomm   q2 q4 q6 q8 q9 q10 q11 q12 q13 q14  (fashion image predicates)
#
# Usage:
#   ./run_image_queries.sh                 # all scenarios, DIRECT mode
#   ./run_image_queries.sh mmqa cars       # only these scenarios
#   MODE=compiled ./run_image_queries.sh   # use Schema-Designer + extract/compile instead
#   FORCE=1 ./run_image_queries.sh mmqa    # regenerate agents/code even if cached
#   DRY=1 ./run_image_queries.sh           # print rendered prompts only (no agent calls)
#
# Requires: `python3` on PATH must have torch/transformers (the vision API). e.g. first run
#   conda activate gendb   (or:  export PATH=$HOME/miniconda3/envs/gendb/bin:$PATH )
set -uo pipefail

# --- config (override via env) ---
ROOT="${SEMBENCH_ROOT:-/localhome/hza214/SemBench/files}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"        # GenDB repo root
OUT="${OUT:-$REPO/src/semdb/runs}"
PROVIDER="${PROVIDER:-claude}"
CLIP="${CLIP:-openai/clip-vit-base-patch32}"
MODE="${MODE:-direct}"                                            # direct | compiled
ORCH="$REPO/src/semdb/orchestrator.mjs"

mode_flag=""; [ "$MODE" = "direct" ] && mode_flag="--direct"
force_flag=""; [ "${FORCE:-0}" = "1" ] && force_flag="--force"
dry_flag="";   [ "${DRY:-0}" = "1" ]   && dry_flag="--dry-run"

# scenario | query-dir | data-dir (relative to $ROOT/<scenario>) | ground-truth-dir
SCENARIOS=(
  "mmqa|query/bigquery|data/sf_200|raw_results/ground_truth"
  "cars|query/bigquery|data/sf_200|raw_results/ground_truth"
  "medical|query/bigquery|data|raw_results/ground_truth"          # flat data/ (FULL corpus)
  "animals|query/bigquery|data/sf_200|raw_results/ground_truth"
  "ecomm|queries/dialects/bigquery|data/sf_250|raw_results/ground_truth"
)

# optional scenario filter from CLI args
want=(); [ "$#" -gt 0 ] && want=("$@")
selected() { [ "${#want[@]}" -eq 0 ] && return 0; for w in "${want[@]}"; do [ "$w" = "$1" ] && return 0; done; return 1; }

echo "[run] provider=$PROVIDER mode=$MODE clip=$CLIP out=$OUT"
echo "[run] python3 -> $(command -v python3)  (must have torch/transformers)"
echo

fail=0
for row in "${SCENARIOS[@]}"; do
  IFS='|' read -r bench qdir ddir gtdir <<< "$row"
  selected "$bench" || continue
  QDIR="$ROOT/$bench/$qdir"; DDIR="$ROOT/$bench/$ddir"; GTDIR="$ROOT/$bench/$gtdir"
  if [ ! -d "$QDIR" ] || [ ! -d "$DDIR" ]; then
    echo "[run] SKIP $bench — missing $QDIR or $DDIR"; continue
  fi
  echo "==================================================================="
  echo "[run] SCENARIO $bench  (image-only, $MODE)"
  echo "==================================================================="
  node "$ORCH" \
    --query-dir "$QDIR" \
    --data-dir  "$DDIR" \
    --ground-truth-dir "$GTDIR" \
    --benchmark "$bench" \
    --agent-provider "$PROVIDER" \
    --clip-model "$CLIP" \
    --out "$OUT" \
    --image-only $mode_flag $force_flag $dry_flag
  rc=$?
  [ $rc -ne 0 ] && { echo "[run] $bench exited $rc"; fail=1; }
  echo
done

echo "==================================================================="
echo "[run] DONE. Aggregated metrics -> $OUT/results.csv"
[ $fail -ne 0 ] && echo "[run] (one or more scenarios reported a non-zero exit — check the logs above)"
exit $fail
