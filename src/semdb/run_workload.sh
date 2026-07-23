#!/usr/bin/env bash
# run_workload.sh — run SemDB end-to-end over a SemBench workload.
#
# Loops every query in a scenario (schema-design → generate extractor → extract once
# per corpus → compile → execute → evaluate). Per-corpus artifacts cache under
# runs/_corpus/, so a corpus is extracted once and reused by all its queries. Each
# query is scored into runs/results.csv (mmqa via the stdlib metric; movie/cars/
# medical/animals/ecomm via scenario_metrics — faithful SemBench-parity, needs the
# sembench conda env for pandas/sklearn/scipy).
#
# AUDIO queries are skipped automatically by the orchestrator. ecomm is PARQUET and
# not yet runnable end-to-end (extraction reads CSV) — it will warn.
#
# Usage:
#   src/semdb/run_workload.sh --bench movie [--sf 1000] [--concurrency 32]
#                             [--queries "Q1 Q2"] [--force] [--provider codex]
# Config via env: SEMBENCH, ENDPOINT, EXTRACT_MODEL, API_KEY, CONCURRENCY,
#   AGENT_PROVIDER, IMAGE_MODEL/TEXT_MODEL, IMAGE_ENDPOINT/TEXT_ENDPOINT (optional).
set -uo pipefail

SEMBENCH="${SEMBENCH:-/localhome/hza214/SemBench}"
BENCH="mmqa"
SF=""
AGENT_PROVIDER="${AGENT_PROVIDER:-codex}"
ENDPOINT="${ENDPOINT:-http://localhost:8000/v1}"
EXTRACT_MODEL="${EXTRACT_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
API_KEY="${API_KEY:-EMPTY}"
CONCURRENCY="${CONCURRENCY:-32}"
FORCE=0
QUERIES=""
IMAGE_ENDPOINT="${IMAGE_ENDPOINT:-}"; IMAGE_MODEL="${IMAGE_MODEL:-}"
TEXT_ENDPOINT="${TEXT_ENDPOINT:-}";  TEXT_MODEL="${TEXT_MODEL:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --bench) BENCH="$2"; shift 2;;
    --sf) SF="$2"; shift 2;;
    --concurrency) CONCURRENCY="$2"; shift 2;;
    --provider) AGENT_PROVIDER="$2"; shift 2;;
    --endpoint) ENDPOINT="$2"; shift 2;;
    --model) EXTRACT_MODEL="$2"; shift 2;;
    --queries) QUERIES="$2"; shift 2;;
    --force) FORCE=1; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "[run] unknown flag: $1" >&2; exit 2;;
  esac
done

# Per-benchmark defaults: scale factor + query subdir (ecomm uses queries/dialects/).
case "$BENCH" in
  mmqa)    QSUB="query/bigquery";           DEF_SF=200;;
  movie)   QSUB="query/bigquery";           DEF_SF=1000;;
  cars)    QSUB="query/bigquery";           DEF_SF=200;;
  animals) QSUB="query/bigquery";           DEF_SF=200;;
  ecomm)   QSUB="queries/dialects/bigquery"; DEF_SF=100;;
  medical) QSUB="query/bigquery";           DEF_SF="";;   # flat data/, no sf
  *) echo "[run] unknown/unsupported bench: $BENCH (mmqa|movie|cars|medical|animals|ecomm)" >&2; exit 2;;
esac
[ -n "$SF" ] || SF="$DEF_SF"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ORCH="$REPO/src/semdb/orchestrator.mjs"
QDIR="$SEMBENCH/files/$BENCH/$QSUB"
RESULTS="$REPO/src/semdb/runs/results.csv"
LOGDIR="$REPO/src/semdb/runs/_logs/$(date +%Y%m%d-%H%M%S)-$BENCH"
mkdir -p "$LOGDIR"
[ -d "$QDIR" ] || { echo "[run] query dir not found: $QDIR" >&2; exit 1; }
command -v node >/dev/null || { echo "[run] node not on PATH" >&2; exit 1; }
[ "$BENCH" = "ecomm" ] && echo "[run] WARNING: ecomm is parquet — extraction is not yet wired for parquet; expect failures."

img_ep="${IMAGE_ENDPOINT:-$ENDPOINT}"; txt_ep="${TEXT_ENDPOINT:-$ENDPOINT}"
img_model="${IMAGE_MODEL:-$EXTRACT_MODEL}"; txt_model="${TEXT_MODEL:-$EXTRACT_MODEL}"
health(){ curl -fsS --max-time 5 "${1%/}/models" -H "Authorization: Bearer $API_KEY" >/dev/null 2>&1; }
for ep in $(printf '%s\n%s\n' "$img_ep" "$txt_ep" | sort -u); do
  health "$ep" && echo "[run] endpoint OK: $ep" || echo "[run] WARNING: $ep/models unreachable — is vLLM up?"
done

[ "$FORCE" = 1 ] && { echo "[run] --force: clearing runs/_corpus cache"; rm -rf "$REPO/src/semdb/runs/_corpus"; }

if [ -n "$QUERIES" ]; then read -r -a qs <<< "$QUERIES"
else mapfile -t qs < <(ls "$QDIR"/*.sql 2>/dev/null | xargs -r -n1 basename | sed 's/\.sql$//' | sort); fi
[ "${#qs[@]}" -gt 0 ] || { echo "[run] no queries in $QDIR" >&2; exit 1; }
echo "[run] bench=$BENCH sf=${SF:-flat} ${#qs[@]} queries: ${qs[*]}  (logs -> $LOGDIR)"

is_image(){ grep -qiE "image|\.uri|\.ref|thalamusdb_images|logo|images\b|x_ray|skin_cancer|car_mm|image_data_mm" "$QDIR/$1.sql"; }

pass=0; fail=0
for q in "${qs[@]}"; do
  if is_image "$q"; then ep="$img_ep"; model="$img_model"; mod=image; else ep="$txt_ep"; model="$txt_model"; mod=text; fi
  log="$LOGDIR/$q.log"
  echo "[run] === $BENCH/$q [$mod] model=$model === -> $log"
  if node "$ORCH" --benchmark "$BENCH" ${SF:+--sf "$SF"} --sembench-dir "$SEMBENCH" \
      --query "$q" --agent-provider "$AGENT_PROVIDER" \
      --endpoint "$ep" --extract-model "$model" --api-key "$API_KEY" \
      --concurrency "$CONCURRENCY" >"$log" 2>&1; then
    echo "[run]   OK   $q"; pass=$((pass + 1))
  else
    echo "[run]   FAIL $q (rc=$?) — see $log"; fail=$((fail + 1))
  fi
done

echo; echo "[run] done: $pass ok, $fail failed. Results: $RESULTS"
if [ -f "$RESULTS" ]; then
  echo "[run] ---- metrics summary ----"
  python3 - "$RESULTS" "$BENCH" "${qs[@]}" <<'PY'
import csv, sys
path, bench, wanted = sys.argv[1], sys.argv[2], set(sys.argv[3:])
seen = {}
for r in csv.DictReader(open(path)):
    if r.get("query") in wanted and (r.get("benchmark","") in (bench,"")):
        seen[r["query"]] = r
for q in sorted(seen):
    r = seen[q]; m = r.get("metric") or "f1"
    core = {"retrieval_f1":"f1","id_set_f1":"f1","top1":"f1","macro_f1":"f1",
            "aggregation":"relative_error","ranking":"spearman","ari":"ari"}.get(m,"f1")
    print(f"  {q:5s} [{m:12s}] {core}={r.get(core,'')}")
PY
fi
