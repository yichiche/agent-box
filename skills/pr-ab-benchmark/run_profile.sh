#!/usr/bin/env bash
# Decode-phase profiling at conc 4 and 64 (IL8192/OL1024) via client --profile.
# Server must be launched with SGLANG_TORCH_PROFILER_DIR=<dir>.
# Usage: PHASE=before|after bash run_profile.sh
set -uo pipefail

PHASE="${PHASE:-before}"
MODEL="/data/amd/Qwen3.5-397B-A17B-MoE-MXFP4"
PORT="${PORT:-9000}"
IL=8192
OL=1024
RANGE_RATIO=0.8

mkdir -p "${WORKDIR:-/tmp/pr3986_bench}/results"
cd /tmp
for c in 4 64; do
  echo "=== [PROFILE ${PHASE}] IL=${IL} OL=${OL} conc=${c} ==="
  python -m sglang.bench_serving \
    --backend sglang --model "${MODEL}" --port "${PORT}" \
    --dataset-name random \
    --random-input-len "${IL}" --random-output-len "${OL}" \
    --random-range-ratio "${RANGE_RATIO}" \
    --num-prompts "${c}" --max-concurrency "${c}" \
    --request-rate inf \
    --profile --profile-start-step 60 --profile-steps 40 \
    --profile-prefix "${PHASE}_conc${c}" \
    --output-file "${WORKDIR:-/tmp/pr3986_bench}/results/${PHASE}_profile_conc${c}.json" 2>&1 | tail -20
  sleep 3
done
echo "=== profiling done (${PHASE}) ==="
