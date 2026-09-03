#!/usr/bin/env bash
# Saturate an 8-GPU node with 4 concurrent TP2 InferenceMax slots.
#
# Each slot gets its own GPU pair, port, run dir and private /workspace mount
# namespace, so the recipes cannot collide on the paths they hardcode.
#
# NOTE ON FIDELITY: InferenceX runs one job per node. Four concurrent TP2 jobs
# share host CPU, PCIe and the node power cap, so absolute numbers here are not
# directly comparable to the dashboard -- they are internally consistent with
# each other, which is what a before/after needs. Say so when reporting.
#
#   BASE_DIR=... bash fanout.sh <slotspec> [<slotspec> ...]
#   slotspec = "<mode>:<conc,conc,...>"    e.g. "agent:1" "fixed:4,8,16"
set -uo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-/home/yichiche/agent-runs/inferencemax/fanout_$(date +%Y%m%d_%H%M%S)}"
MODEL_PREFIX="${MODEL_PREFIX:-qwen3.5}"
MODEL_PATH="${MODEL_PATH:-/shared_nfs/models/Qwen/Qwen3.5-397B-A17B-MXFP4}"
TP="${TP:-2}"
SGLANG_PY="${SGLANG_PY:-}"        # prepend to PYTHONPATH, e.g. a clean worktree

GPU_PAIRS=(0,1 2,3 4,5 6,7)
PORTS=(8891 8892 8893 8894)

mkdir -p "$BASE_DIR"
echo "fanout base: $BASE_DIR"

i=0
pids=()
for spec in "$@"; do
  mode="${spec%%:*}"
  concs="${spec#*:}"
  concs="${concs//,/ }"
  gpus="${GPU_PAIRS[$i]}"
  port="${PORTS[$i]}"
  slot="$BASE_DIR/slot$((i+1))_${mode}_gpu${gpus//,/}"
  mkdir -p "$slot"

  echo "slot$((i+1)): mode=$mode conc=[$concs] gpus=$gpus port=$port -> $slot"
  (
    export MODEL_PREFIX MODEL_PATH TP
    export MODE="$mode"
    export CONCURRENCIES="$concs"
    export CUDA_VISIBLE_DEVICES="$gpus"
    export PORT="$port"
    export RUN_DIR="$slot"
    [ -n "$SGLANG_PY" ] && export PYTHONPATH="$SGLANG_PY${PYTHONPATH:+:$PYTHONPATH}"
    bash "$SKILL_DIR/run_infmax.sh"
  ) > "$slot/slot.log" 2>&1 &
  pids+=($!)
  i=$((i+1))
  sleep 20   # stagger model loads so four 397B reads do not hit the NFS at once
done

echo "launched ${#pids[@]} slots; waiting"
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo "all slots finished (rc=$rc); base=$BASE_DIR"
exit "$rc"
