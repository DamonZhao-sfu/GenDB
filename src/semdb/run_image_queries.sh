#!/usr/bin/env bash
# run_image_queries.sh — run SemBench queries through the SemDB orchestrator.
#
# For every scenario this runs the whole query dir with `--image-only`, so the
# orchestrator self-selects only the queries whose corpus is an image table (skipping
# text- and audio-only queries). By default it uses DIRECT mode: the strict VADAR 3
# agents (Signature -> API -> Solver) synthesize ONE end-to-end solve_<q>.py per query
# that calls the local vision API (CLIP/OCR/CV) and answers the whole query — no schema
# design, no extract/compile split. Results (P/R/F1 + telemetry) are appended to
# $OUT/results.csv.
#
# SCALE FACTORS MATTER. The ground truth is one file per query covering the FULL
# dataset, so running a small sf subset scores as massive recall loss: cars Q3 on
# sf_200 reported recall=0.2 purely because 8 of its 10 ground-truth rows were not in
# the subset. Each scenario below points at the sf whose rows the ground truth
# actually covers.
#
# Queries selected per scenario (--image-only picks these automatically):
#   mmqa    q2a q2b q7                     (logo recognition; joins a table to images)
#   cars    Q3 Q5 Q6 Q7 Q8 Q9              (car image predicates)
#   medical Q3 Q5 Q6 Q7 Q8 Q9 Q11          (x-ray / skin image predicates)
#   animals Q1 Q3 Q5 Q6 Q7 Q8 Q9 Q10       (animal image predicates)
#   ecomm   q2 q4 q6 q8 q9 q10 q11 q12 q13 q14  (fashion image predicates)
#   movie   (none — text-only; needs ALL=1)
#
# Usage:
#   ./run_image_queries.sh                     # all image scenarios, DIRECT, 5 iterations
#   ./run_image_queries.sh mmqa cars           # only these scenarios
#   ITERS=3 ./run_image_queries.sh animals     # cap the refinement loop at 3 iterations
#   ITERS=0 ./run_image_queries.sh             # single-shot (no refinement)
#   QUERIES=q2a,q7 ./run_image_queries.sh mmqa # only these queries
#   ALL=1 ./run_image_queries.sh movie         # drop --image-only (text/audio too)
#   AGENT_EXECUTION=structured ./run_image_queries.sh mmqa
#                                                # opt into tool-free Planner/Optimizer
#   FORCE=1 ./run_image_queries.sh mmqa        # regenerate agents/code even if cached
#   DRY=1 ./run_image_queries.sh               # print rendered prompts only
#
# Refining against a VALIDATION SET instead of the ground truth (recommended — without
# it every iteration is scored on the full ground truth and the agent's feedback shows
# ground-truth rows, i.e. it tunes on the test set):
#   VAL_RATE=0.5 VAL_PAIR_TOP=400 ENDPOINT=http://localhost:8000/v1 \
#     ORACLE=Qwen/Qwen2.5-VL-7B-Instruct ./run_image_queries.sh mmqa
# VAL_RATE requires ENDPOINT (the oracle labeler is a model call, and it must be a
# VISION model for image corpora). VAL_PAIR_TOP prunes a cross-table pair frame to its
# N most similar pairs before sampling — see USAGE.md; on mmqa q2a the full frame at
# rate 0.05 captures 0 of 5 positives while --top 400 at rate 0.3 captures 4 of 5 for
# a SMALLER labeling bill.
#
# Requires: `python3` on PATH must have torch/transformers (the vision API). e.g. first
#   conda activate gendb   (or:  export PATH=$HOME/miniconda3/envs/gendb/bin:$PATH )
set -uo pipefail

# --- config (override via env) ---
ROOT="${SEMBENCH_ROOT:-/localhome/hza214/SemBench/files}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"        # GenDB repo root
OUT="${OUT:-$REPO/src/semdb/runs}"
PROVIDER="${PROVIDER:-vllm}"
CLIP="${CLIP:-openai/clip-vit-base-patch32}"
ITERS="${ITERS:-5}"                                               # --max-iterations
AGENT_EXECUTION="${AGENT_EXECUTION:-agent}"                       # agent | structured
ORCH="$REPO/src/semdb/orchestrator.mjs"

case "$AGENT_EXECUTION" in
  structured|agent) ;;
  *)
    echo "[run] ERROR: AGENT_EXECUTION must be 'structured' or 'agent' (got '$AGENT_EXECUTION')." >&2
    exit 2
    ;;
esac

force_flag=""; [ "${FORCE:-0}" = "1" ] && force_flag="--force"
dry_flag="";   [ "${DRY:-0}" = "1" ]   && dry_flag="--dry-run"
# --image-only self-selects image queries; ALL=1 keeps text corpora too (movie).
img_flag="--image-only"; [ "${ALL:-0}" = "1" ] && img_flag=""

# ITERS=0 means single-shot. The orchestrator spells that --no-refine; passing
# --max-iterations 0 alone would still run iteration 0 and then log a stop decision.
iter_flags=(--max-iterations "$ITERS")
[ "$ITERS" = "0" ] && iter_flags=(--no-refine)

# Per-query filter: --query accepts a comma/space separated list.
query_flags=()
[ -n "${QUERIES:-}" ] && query_flags=(--query "$QUERIES")

# Validation-set refinement. Without VAL_RATE the loop scores every iteration against
# the FULL ground truth — correct for a quick benchmark run, wrong if you care that the
# agent never sees the test set.
val_flags=()
if [ -n "${VAL_RATE:-}" ]; then
  if [ -z "${ENDPOINT:-}" ]; then
    echo "[run] ERROR: VAL_RATE=$VAL_RATE needs ENDPOINT (the oracle labeler is a model call)." >&2
    echo "[run]        e.g. ENDPOINT=http://localhost:8000/v1  (the /v1 suffix is required)" >&2
    exit 2
  fi
  val_flags=(--val-rate "$VAL_RATE" --endpoint "$ENDPOINT")
  [ -n "${VAL_PAIR_TOP:-}" ] && val_flags+=(--val-pair-top "$VAL_PAIR_TOP")
  [ -n "${ORACLE:-}" ]       && val_flags+=(--oracle-model "$ORACLE")
  [ -n "${VAL_CERT_RATE:-}" ] && val_flags+=(--val-cert-rate "$VAL_CERT_RATE")
fi

# scenario | query-dir | data-dir (relative to $ROOT/<scenario>) | ground-truth-dir | modality
# The data-dir is the scale factor the ground truth covers — see the note above.
SCENARIOS=(
  "mmqa|query/bigquery|data/sf_200|raw_results/ground_truth|image"
  "cars|query/bigquery|data/sf_9836|raw_results/ground_truth|image"
  "medical|query/bigquery|data|raw_results/ground_truth|image"        # flat data/ (FULL corpus)
  "animals|query/bigquery|data/sf_100|raw_results/ground_truth|image"
  "movie|query/bigquery|data/sf_1000|raw_results/ground_truth|text"   # text-only: needs ALL=1
  "ecomm|queries/dialects/bigquery|data/sf_250|raw_results/ground_truth|image"
)

# optional scenario filter from CLI args
want=(); [ "$#" -gt 0 ] && want=("$@")
selected() { [ "${#want[@]}" -eq 0 ] && return 0; for w in "${want[@]}"; do [ "$w" = "$1" ] && return 0; done; return 1; }

echo "[run] provider=$PROVIDER agent-execution=$AGENT_EXECUTION iters=$ITERS clip=$CLIP out=$OUT"
echo "[run] scope=${img_flag:-all-modalities}${QUERIES:+  queries=$QUERIES}"
if [ ${#val_flags[@]} -gt 0 ]; then
  echo "[run] validation-set refinement: rate=$VAL_RATE endpoint=$ENDPOINT${VAL_PAIR_TOP:+ pair-top=$VAL_PAIR_TOP}"
else
  echo "[run] NO validation set — every iteration is scored on the FULL ground truth."
  echo "[run] (set VAL_RATE+ENDPOINT to refine against oracle-labeled samples instead)"
fi
echo "[run] python3 -> $(command -v python3)  (must have torch/transformers)"
echo

fail=0
for row in "${SCENARIOS[@]}"; do
  IFS='|' read -r bench qdir ddir gtdir modality <<< "$row"
  selected "$bench" || continue
  # A text-only scenario has nothing to contribute under --image-only; say so instead
  # of running it to a silent "0 image queries".
  if [ -n "$img_flag" ] && [ "$modality" = "text" ]; then
    echo "[run] SKIP $bench — text-only scenario; re-run with ALL=1 to include it."
    continue
  fi
  QDIR="$ROOT/$bench/$qdir"; DDIR="$ROOT/$bench/$ddir"; GTDIR="$ROOT/$bench/$gtdir"
  if [ ! -d "$QDIR" ] || [ ! -d "$DDIR" ]; then
    echo "[run] SKIP $bench — missing $QDIR or $DDIR"; continue
  fi
  echo "==================================================================="
  echo "[run] SCENARIO $bench  ($ddir, ${ITERS} iteration(s))"
  echo "==================================================================="
  node "$ORCH" \
    --query-dir "$QDIR" \
    --data-dir  "$DDIR" \
    --ground-truth-dir "$GTDIR" \
    --benchmark "$bench" \
    --agent-provider "$PROVIDER" \
    --agent-execution "$AGENT_EXECUTION" \
    --clip-model "$CLIP" \
    --out "$OUT" \
    "${iter_flags[@]}" "${query_flags[@]}" "${val_flags[@]}" \
    $img_flag $force_flag $dry_flag
  rc=$?
  [ $rc -ne 0 ] && { echo "[run] $bench exited $rc"; fail=1; }
  echo
done

echo "==================================================================="
echo "[run] DONE. Aggregated metrics -> $OUT/results.csv"
[ $fail -ne 0 ] && echo "[run] (one or more scenarios reported a non-zero exit — check the logs above)"
exit $fail
