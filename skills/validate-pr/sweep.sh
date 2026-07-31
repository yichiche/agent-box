#!/usr/bin/env bash
# conc sweep vs a running sglang server. Usage: sweep.sh <outdir> <tag> [ISL] [OSL] [CONCS...]
set -uo pipefail
OUTDIR="$1"; TAG="$2"; ISL="${3:-8192}"; OSL="${4:-1024}"; shift 4 2>/dev/null || shift $#
CONCS=("$@"); [ ${#CONCS[@]} -eq 0 ] && CONCS=(4 8 16 32 64 128 256)
CLIENT=${CLIENT:-$HOME/bench_serving/benchmark_serving.py}
MODEL=${MODEL:-/data/amd/Qwen3.5-397B-A17B-MXFP4}
PORT=${PORT:-8000}; RR=${RANGE_RATIO:-0.8}
mkdir -p "$OUTDIR"
for c in "${CONCS[@]}"; do
  np=$((c*10))
  echo "-- BENCH $TAG il${ISL} ol${OSL} conc=$c np=$np $(date +%H:%M:%S)"
  python3 "$CLIENT" --model="$MODEL" --backend=sglang --port="$PORT" \
    --dataset-name=random --random-input-len="$ISL" --random-output-len="$OSL" --random-range-ratio="$RR" \
    --num-prompts="$np" --max-concurrency="$c" --request-rate=inf --ignore-eos \
    --percentile-metrics=ttft,tpot,itl,e2el --save-result --result-dir="$OUTDIR" \
    --result-filename="bench_${TAG}_il${ISL}_ol${OSL}_conc${c}.json" > "$OUTDIR/bench_${TAG}_conc${c}.log" 2>&1
  sleep 3
done
echo "=== SWEEP $TAG DONE $(date) ==="
