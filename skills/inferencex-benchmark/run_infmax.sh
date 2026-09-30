#!/usr/bin/env bash
# Local runner for an InferenceX (InferenceMax) benchmark arm.
#
# This is the in-container stand-in for what CI does around the recipe:
#   runners/launch_mi355x-amds.sh      -> container mounts + env + `bash <recipe>`
#   .github/workflows/benchmark-tmpl.yml -> RESULT_FILENAME, RESULT_DIR, thresholds
#
# It deliberately re-expresses NO server or client flag. The upstream recipe
# under $INFERENCEX_DIR/benchmarks/single_node/ owns the whole command line,
# including server launch, readiness wait, the client, and teardown. Keeping
# alignment therefore means `git -C $INFERENCEX_DIR pull`, not editing this file.
#
# Env-driven, no flags, so it can be driven through `docker exec -e ...`.
#
#   MODEL_PREFIX=qwen3.5 MODE=fixed MODEL_PATH=/shared_nfs/models/Qwen/... \
#     bash run_infmax.sh
set -uo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCEX_DIR="${INFERENCEX_DIR:-/home/yichiche/InferenceX}"

MODEL_PREFIX="${MODEL_PREFIX:?set MODEL_PREFIX, e.g. qwen3.5}"
MODE="${MODE:?set MODE=fixed or MODE=agent}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the local weights dir}"
SPEC="${SPEC:-mtp}"
PRECISION_IN="${PRECISION:-fp4}"
FRAMEWORK_IN="${FRAMEWORK:-sglang}"
HW="${HW:-mi355x}"
TP_IN="${TP:-}"
PORT="${PORT:-8888}"
DRY_RUN="${DRY_RUN:-0}"
FULL="${FULL:-0}"
RUNNER_NAME="${RUNNER_NAME:-local}"
RUNS_ROOT="${RUNS_ROOT:-/home/yichiche/agent-runs/inferencemax}"

log() { printf '[infmax %s] %s\n' "$(date +%T)" "$*"; }
die() { printf '[infmax] ERROR: %s\n' "$*" >&2; exit 1; }

# CI gets a fresh container per concurrency point, so it never has to think about
# the previous server. We reuse one process, and the recipe's teardown
# (stop_background_process_tree, 60 s grace) returns well before the driver has
# actually released ~230 GB/shard. Starting the next point immediately gives you
# either an OOM on load, or -- worse -- a healthy-looking OLD server on the same
# port that the client happily benchmarks with the previous --max-running-requests.
# That failure is silent: TPOT freezes at the first point's value and TTFT explodes.
wait_for_slot_idle() {
  local port="$1" devs="$2" timeout="${3:-900}" t=0
  while [ "$t" -lt "$timeout" ]; do
    local busy=0
    if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$port/health" 2>/dev/null; then
      busy=1
    fi
    if [ "$busy" -eq 0 ] && ! python3 - "$devs" <<'PY'
import json,subprocess,sys
want={int(x) for x in sys.argv[1].split(',') if x != ''}
out=subprocess.run(['python3','/home/yichiche/agent-box/skills/gpu-status/gpu_status.py','--json'],
                   capture_output=True,text=True).stdout
gpus=json.load(open('/dev/stdin')) if False else json.loads(out)['gpus']
sys.exit(0 if all(g['vram_used_gib'] < 5 for g in gpus if g['cuda_index'] in want) else 1)
PY
    then busy=1; fi
    [ "$busy" -eq 0 ] && { [ "$t" -gt 0 ] && log "slot idle after ${t}s"; return 0; }
    sleep 15; t=$((t+15))
  done
  log "WARNING: port $port or GPUs $devs still busy after ${timeout}s"
  return 1
}

# The recipe's server can outlive its EXIT trap and reparent to init, holding
# ~230 GB/shard indefinitely. Reap only the server bound to OUR port -- never a
# broad `pkill -f sglang.launch_server`, which would kill other tenants, and never
# SIGKILL first, which orphans the TP workers into unreclaimable VRAM.
reap_our_server() {
  local port="$1"
  local pids
  pids=$(pgrep -f "launch_server.*--port[= ]$port" 2>/dev/null)
  [ -n "$pids" ] || return 0
  log "reaping leftover server(s) on port $port: $pids"
  for p in $pids; do
    local pg; pg=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$pg" ] && kill -TERM -"$pg" 2>/dev/null
  done
  sleep 120
}

# ---------------------------------------------------------------- 1. checkout
[ -d "$INFERENCEX_DIR/.git" ] || die "INFERENCEX_DIR is not a git checkout: $INFERENCEX_DIR"
log "InferenceX $(git -C "$INFERENCEX_DIR" log --oneline -1)"
log "         committed $(git -C "$INFERENCEX_DIR" log -1 --format=%ci)"
if git -C "$INFERENCEX_DIR" fetch --quiet origin main 2>/dev/null; then
  behind=$(git -C "$INFERENCEX_DIR" rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
  if [ "${behind:-0}" -gt 0 ]; then
    log "WARNING: checkout is $behind commits behind origin/main."
    log "         The recipe you are about to run may not be the current standard."
    log "         Run: git -C $INFERENCEX_DIR pull --recurse-submodules"
  fi
else
  log "note: could not reach origin; cannot tell whether the recipe is current"
fi

# ------------------------------------------------------------------- 2. arm
eval "$(INFERENCEX_DIR="$INFERENCEX_DIR" python3 "$SKILL_DIR/resolve_arm.py" \
  --model-prefix "$MODEL_PREFIX" --mode "$MODE" --spec "$SPEC" \
  --precision "$PRECISION_IN" --framework "$FRAMEWORK_IN" --hw "$HW" \
  --kv-offloading "${KV_OFFLOADING:-none}" \
  ${TP_IN:+--tp "$TP_IN"})" || die "arm resolution failed"

[ -n "${ARM_NAME:-}" ] || die "arm resolution produced nothing"
log "arm       $ARM_NAME"
log "recipe    $RECIPE"
log "image(CI) $IMAGE   <- CI runs the recipe in this image; you are running it in the current container"

CONCURRENCIES="${CONCURRENCIES:-$CONC_LIST}"
log "tp=$TP ep=$EP_SIZE spec=$SPEC_DECODING conc=[$CONCURRENCIES]"

# -------------------------------------------------------------- 3. workspace
# CI mounts the checkout at /workspace (-v $GITHUB_WORKSPACE:/workspace/) and the
# fixed-seq recipe hardcodes /workspace/server.log and --result-dir /workspace/.
# Those two paths are the only /workspace uses on the fixed path, and both are
# WRITES (`workspace_dir=$(pwd)` is what the client is read from), so we give each
# run its own /workspace via a private mount namespace. That is what makes several
# slots runnable at once on one node -- otherwise they interleave into one
# server.log and race on the result file.
if [ -L /workspace ]; then
  rm -f /workspace   # a symlink cannot be bind-mounted onto; the bind resolves through it
fi
if [ -e /workspace ] && [ -n "$(ls -A /workspace 2>/dev/null)" ]; then
  die "/workspace exists and is not empty; refusing to touch it"
fi
mkdir -p /workspace
unshare -m true 2>/dev/null || die "need CAP_SYS_ADMIN for 'unshare -m' to isolate /workspace per run"

if [ "$MODE" = agent ]; then
  [ -f "$INFERENCEX_DIR/utils/aiperf/pyproject.toml" ] || {
    log "initialising utils/aiperf submodule"
    git -C "$INFERENCEX_DIR" submodule update --init utils/aiperf || die "submodule init failed"
  }
  [ -d "$INFERENCEX_DIR/utils/agentic-benchmark" ] || die "utils/agentic-benchmark missing"
fi

# ------------------------------------------------------------------- 4. GPUs
need=$TP
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  log "using caller-supplied CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
  free=$(python3 "$SKILL_DIR/../gpu-status/gpu_status.py" --json \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(",".join(str(g["cuda_index"]) for g in d["gpus"] if g["status"]=="FREE"))')
  n=$(awk -F, '{print NF}' <<<"${free:-}")
  [ -n "$free" ] && [ "$n" -ge "$need" ] || die "need $need free GPUs, have ${n:-0} (${free:-none}). Not sharing cards -- a contended GPU makes the number meaningless."
  export CUDA_VISIBLE_DEVICES="$(cut -d, -f1-"$need" <<<"$free")"
  log "picked CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # runners/launch_mi325x-tw.sh sets this

# ------------------------------------------------------------- 5. static env
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$RUNS_ROOT/${MODEL_PREFIX}_${MODE}_tp${TP}_${TS}}"
mkdir -p "$RUN_DIR"
log "run dir   $RUN_DIR"

export INFMAX_CONTAINER_WORKSPACE="$INFERENCEX_DIR"
export MODEL MODEL_PREFIX MODEL_PATH IMAGE RUNNER_TYPE FRAMEWORK PRECISION
export TP EP_SIZE SPEC_DECODING DISAGG PORT
export PP_SIZE=1 DCP_SIZE=1 PCP_SIZE=1 DP_SIZE=1 DP_ATTENTION=false
export GPU_COUNT=$((TP * PP_SIZE * PCP_SIZE))
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/tmp/inferencex-pycache

if [ "$MODE" = fixed ]; then
  export ISL OSL
  export RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"   # benchmark-tmpl.yml:159
  export MEM_FRAC_STATIC="${MEM_FRAC_STATIC:-0.8}"
  export RUN_EVAL="${RUN_EVAL:-false}"
  export EVAL_ONLY="${EVAL_ONLY:-false}"
  # The MTP recipe calls `hf download "$MODEL"` unguarded. We pass a local path,
  # so make that call fail fast and quietly instead of reaching for the network.
  # (The recipe has no `set -e`, so the failure is not fatal.)
  export HF_HUB_OFFLINE=1
  # Fixed-seq uses MODEL directly as --model-path.
  export MODEL="$MODEL_PATH"
else
  # Agentic keeps MODEL as the HF repo id: it feeds --tokenizer-path and aiperf's
  # --tokenizer, and a bare "Qwen3.5-397B-A17B-MXFP4" is not a repo (404).
  # The recipe skips the weight download because MODEL_PATH is a non-empty dir.
  [ -n "$(ls -A "$MODEL_PATH" 2>/dev/null)" ] || die "MODEL_PATH is empty: $MODEL_PATH (the recipe would download the full repo)"
  export IS_AGENTIC=1
  export SCENARIO_TYPE=agentic-coding
  # KV_OFFLOADING / KV_OFFLOAD_BACKEND come from the resolved arm. benchmark_lib.sh
  # hard-errors at source time unless they are consistent, and for dram it also
  # demands a positive TOTAL_CPU_DRAM_GB. The recipe then adds the --hicache-* flags
  # itself via require_agentic_kv_offload_backend; we add no flag of our own.
  export KV_OFFLOADING KV_OFFLOAD_BACKEND
  mem_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)
  util="${DRAM_UTILIZATION:-0.80}"
  export TOTAL_CPU_DRAM_GB=$(awk -v k="$mem_kb" -v u="$util" 'BEGIN{printf "%d", k*1024*u/1e9}')
  if [ "$KV_OFFLOADING" = dram ]; then
    # HiCache pushes KV into host DRAM. Concurrent slots each believe they own
    # TOTAL_CPU_DRAM_GB, so N slots oversubscribe host RAM N-fold. Refuse to
    # pretend otherwise.
    log "HiCache: KV_OFFLOAD_BACKEND=$KV_OFFLOAD_BACKEND, TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB"
    [ "${ALLOW_PARALLEL_HICACHE:-0}" = 1 ] || \
      log "NOTE: do not run several HiCache slots at once on this node (host DRAM is shared and not partitioned)"
  fi
  export DURATION="${DURATION:-$([ "$FULL" = 1 ] && echo 3600 || echo 1200)}"
  [ "$DURATION" -ge 900 ] || die "DURATION=$DURATION is below 900s; aiperf would add --unsafe-override and the run would not be a valid submission"
  [ "$FULL" = 1 ] || export AIPERF_EXPERIMENTAL_FAST=1
  export AIPERF_FAILED_REQUEST_THRESHOLD="${AIPERF_FAILED_REQUEST_THRESHOLD:-0.10}"
  export AIPERF_DATASET_MMAP_CACHE_DIR="${AIPERF_DATASET_MMAP_CACHE_DIR:-/tmp/aiperf_mmap_cache}"
  mkdir -p "$AIPERF_DATASET_MMAP_CACHE_DIR"
fi

# ------------------------------------------------------------------ 6. sweep
rc_total=0
for CONC in $CONCURRENCIES; do
  export CONC
  if [ "$MODE" = fixed ]; then
    EXP_NAME="$EXP_NAME_STEM"
  else
    EXP_NAME="${EXP_NAME_STEM}_tp${TP}_conc${CONC}_${KV_SUFFIX}"
    [ "$SPEC_DECODING" = none ] || EXP_NAME="${EXP_NAME}_spec-${SPEC_DECODING}"
  fi
  export EXP_NAME
  # benchmark-tmpl.yml:299
  export RESULT_FILENAME="${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-pp${PP_SIZE}-dcp${DCP_SIZE}-pcp${PCP_SIZE}-ep${EP_SIZE}-dpa${DP_ATTENTION}_disagg-${DISAGG}_spec-${SPEC_DECODING}_conc${CONC}_${RUNNER_NAME}"

  CONC_DIR="$RUN_DIR/conc${CONC}"
  mkdir -p "$CONC_DIR"
  # benchmark_lib.sh:136 -- relative by default, i.e. shared. Pin it per run.
  export GPU_METRICS_CSV="$CONC_DIR/gpu_metrics.csv"
  if [ "$MODE" = agent ]; then
    export RESULT_DIR="$CONC_DIR"
    export AGENTIC_OUTPUT_DIR="$CONC_DIR"
  fi

  log "=== conc=$CONC  RESULT_FILENAME=$RESULT_FILENAME"
  if [ "$DRY_RUN" != 1 ]; then
    if ! wait_for_slot_idle "$PORT" "$CUDA_VISIBLE_DEVICES" 600; then
      reap_our_server "$PORT"
      wait_for_slot_idle "$PORT" "$CUDA_VISIBLE_DEVICES" 600 || \
        die "conc=$CONC: slot still busy after reaping. Refusing to run -- the client would silently benchmark the previous server (frozen TPOT, exploding TTFT)."
    fi
  fi
  if [ "$DRY_RUN" = 1 ]; then
    { echo "# resolved env for conc=$CONC"; env | grep -E \
      '^(ARM_NAME|MODEL|MODEL_PATH|MODEL_PREFIX|IMAGE|RUNNER_TYPE|FRAMEWORK|PRECISION|TP|EP_SIZE|PP_SIZE|DCP_SIZE|PCP_SIZE|DP_SIZE|DP_ATTENTION|SPEC_DECODING|DISAGG|CONC|ISL|OSL|RANDOM_RANGE_RATIO|MEM_FRAC_STATIC|RUN_EVAL|EVAL_ONLY|PORT|GPU_COUNT|EXP_NAME|RESULT_FILENAME|RESULT_DIR|AGENTIC_OUTPUT_DIR|INFMAX_CONTAINER_WORKSPACE|IS_AGENTIC|SCENARIO_TYPE|KV_OFFLOADING|KV_OFFLOAD_BACKEND|TOTAL_CPU_DRAM_GB|DURATION|AIPERF_|HF_HUB_OFFLINE|GPU_METRICS_CSV|CUDA_VISIBLE_DEVICES|CUDA_DEVICE_ORDER|PYTHON)' | sort
      echo "# command (in a private mount ns with $CONC_DIR bound at /workspace)"
      echo "cd $INFERENCEX_DIR && bash $RECIPE"; } | tee "$CONC_DIR/dry_run.txt"
    continue
  fi

  # Private mount namespace: this run's /workspace IS its conc dir, so the
  # recipe's hardcoded /workspace writes land where they belong and cannot
  # collide with a sibling slot.
  unshare -m -- bash -c '
    mount --bind "$1" /workspace || exit 97
    cd "$2" && exec bash "$3"
  ' _ "$CONC_DIR" "$INFERENCEX_DIR" "$RECIPE" 2>&1 | tee "$CONC_DIR/recipe.log"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { log "conc=$CONC recipe exited $rc"; rc_total=$rc; }

  if [ "$MODE" = fixed ]; then
    if [ -f "$CONC_DIR/$RESULT_FILENAME.json" ]; then
      # process_result.py is stdlib-only and reads "$RESULT_FILENAME.json" relative
      # to cwd, so run it where the result now lives; agg_*.json lands beside it.
      ( cd "$CONC_DIR" && python3 "$INFERENCEX_DIR/utils/process_result.py" ) \
        >"$CONC_DIR/process_result.log" 2>&1 \
        && log "conc=$CONC aggregated" \
        || log "conc=$CONC process_result.py failed, see $CONC_DIR/process_result.log"
    else
      log "conc=$CONC produced no result JSON -- check $CONC_DIR/recipe.log"
    fi
  fi
done

# ------------------------------------------------------------------ 7. report
# The recipe's own teardown can return before the server is gone, leaving it
# reparented to init holding ~230 GB/shard. Between points wait_for_slot_idle
# catches that; after the LAST point nothing would, so reap here too.
if [ "$DRY_RUN" != 1 ]; then
  reap_our_server "$PORT"
  wait_for_slot_idle "$PORT" "$CUDA_VISIBLE_DEVICES" 300 \
    || log "WARNING: GPUs $CUDA_VISIBLE_DEVICES not released; check for an orphan before the next run"
fi

if [ "$DRY_RUN" = 1 ]; then
  log "dry run only; nothing was launched. Env blocks in $RUN_DIR/conc*/dry_run.txt"
  exit 0
fi

log "leftovers in the checkout (should be none):"
git -C "$INFERENCEX_DIR" status -s | head -20

python3 "$SKILL_DIR/render_table.py" --run-dir "$RUN_DIR" --mode "$MODE" \
  | tee "$RUN_DIR/summary.md"

log "done: $RUN_DIR"
exit "$rc_total"
