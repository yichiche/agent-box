#!/usr/bin/env bash
# E2E throughput sweep via `python -m sglang.bench_serving` (neutral cwd /tmp).
# Usage: PHASE=before|after bash run_bench.sh
set -uo pipefail

PHASE="${PHASE:-before}"
MODEL="/data/amd/Qwen3.5-397B-A17B-MoE-MXFP4"
PORT="${PORT:-9000}"
OUTDIR="${WORKDIR:-/tmp/pr3986_bench}/results"
mkdir -p "${OUTDIR}"
RANGE_RATIO=0.8
CONCS=(4 8 16 32 64 128)
# (IL OL) workloads
WORKLOADS=("1024 1024" "8192 1024")

cd /tmp
for wl in "${WORKLOADS[@]}"; do
  read -r IL OL <<< "$wl"
  for c in "${CONCS[@]}"; do
    np=$(( c * 10 ))
    out="${OUTDIR}/${PHASE}_il${IL}_ol${OL}_conc${c}.json"
    echo "=== [${PHASE}] IL=${IL} OL=${OL} conc=${c} num_prompts=${np} ==="
    python -m sglang.bench_serving \
      --backend sglang --model "${MODEL}" --port "${PORT}" \
      --dataset-name random \
      --random-input-len "${IL}" --random-output-len "${OL}" \
      --random-range-ratio "${RANGE_RATIO}" \
      --num-prompts "${np}" --max-concurrency "${c}" \
      --request-rate inf \
      --output-file "${out}" 2>&1 | tail -25
  done
done
echo "=== sweep done (${PHASE}) ==="
