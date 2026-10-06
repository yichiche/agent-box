#!/usr/bin/env bash
# Local runner for an InferenceX (InferenceMax) benchmark arm.
#
# This is the in-container stand-in for what CI does around an srt-slurm recipe:
#   .github/workflows/benchmark-tmpl.yml  -> job env, RESULT_FILENAME, runtime_settings.sh
#   runners/slurm_utils.sh                -> launch_srt_single_node + apply_srt_recipe
#   srtctl (frontend.type sglang)         -> sglang.launch_server, then the recipe's
#                                            benchmark.command against it
#
# It re-expresses NO server or client flag. srt_point.py runs CI's own
# select_recipe/validate_recipe and synthetic_acceptance code from the checkout and
# renders the worker command line the way srtctl does; the client is the recipe's
# own benchmark command. Keeping alignment means `git -C $INFERENCEX_DIR pull`.
#
# Env-driven, no flags, so it can be driven through `docker exec -e ...`.
#
#   MODEL_PREFIX=qwen3.5 MODE=fixed MODEL_PATH=/data/Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2 \
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
SYNTHETIC_ACCEPTANCE="${SYNTHETIC_ACCEPTANCE:-1}"
# benchmark-tmpl.yml exports thinking_on; it picks the golden acceptance curve.
THINKING_MODE="${THINKING_MODE:-thinking_on}"
# runners/srt-slurm/mi355x-amds.yaml visible_devices_env
VISIBLE_DEVICES_ENV="${VISIBLE_DEVICES_ENV:-ROCR_VISIBLE_DEVICES}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-2400}"
RUNNER_NAME="${RUNNER_NAME:-local}"
RUNS_ROOT="${RUNS_ROOT:-/home/yichiche/agent-runs/inferencemax}"

log() { printf '[infmax %s] %s\n' "$(date +%T)" "$*"; }
die() { printf '[infmax] ERROR: %s\n' "$*" >&2; exit 1; }

# CI gets a fresh node per concurrency point, so it never has to think about the
# previous server. We reuse one container, and a server's teardown can return well
# before the driver has released ~230 GB/shard. Starting the next point
# immediately gives either an OOM on load or -- worse -- a healthy-looking OLD
# server on the same port that the client happily benchmarks with the previous
# --max-running-requests. That failure is silent: TPOT freezes at the first
# point's value and TTFT explodes.
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
gpus=json.loads(out)['gpus']
sys.exit(0 if all(g['vram_used_gib'] < 5 for g in gpus if g['cuda_index'] in want) else 1)
PY
    then busy=1; fi
    [ "$busy" -eq 0 ] && { [ "$t" -gt 0 ] && log "slot idle after ${t}s"; return 0; }
    sleep 15; t=$((t+15))
  done
  log "WARNING: port $port or GPUs $devs still busy after ${timeout}s"
  return 1
}

# A server can outlive its parent and reparent to init, holding ~230 GB/shard
# indefinitely. Reap only the server bound to OUR port -- never a broad
# `pkill -f sglang.launch_server`, which would kill other tenants, and never
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

# Same contract as benchmark_lib.sh stop_background_process_tree: SIGTERM the
# whole process group, give it a grace period, then SIGKILL what is left.
stop_server() {
  local pgid="$1" t=0
  kill -TERM -"$pgid" 2>/dev/null || return 0
  while kill -0 -"$pgid" 2>/dev/null && [ "$t" -lt 60 ]; do sleep 2; t=$((t+2)); done
  kill -KILL -"$pgid" 2>/dev/null
  wait "$pgid" 2>/dev/null
  return 0
}

wait_for_server() {
  local pid="$1" port="$2" timeout="$3" t=0
  while [ "$t" -lt "$timeout" ]; do
    curl -sf -o /dev/null --max-time 2 "http://127.0.0.1:$port/health" && return 0
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 10; t=$((t+10))
  done
  return 1
}

# ---------------------------------------------------------------- 1. checkout
[ -d "$INFERENCEX_DIR/.git" ] || die "INFERENCEX_DIR is not a git checkout: $INFERENCEX_DIR"
# The benchmark harness lives in inferencex-e2e/ since the repo split; older checkouts keep it at the root.
if [ -f "$INFERENCEX_DIR/inferencex-e2e/configs/amd-master.yaml" ]; then
  E2E_DIR="$INFERENCEX_DIR/inferencex-e2e"
else
  E2E_DIR="$INFERENCEX_DIR"
fi
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
eval "$(INFERENCEX_E2E_DIR="$E2E_DIR" python3 "$SKILL_DIR/resolve_arm.py" \
  --model-prefix "$MODEL_PREFIX" --mode "$MODE" --spec "$SPEC" \
  --precision "$PRECISION_IN" --framework "$FRAMEWORK_IN" --hw "$HW" \
  --kv-offloading "${KV_OFFLOADING:-none}" \
  ${TP_IN:+--tp "$TP_IN"})" || die "arm resolution failed"

[ -n "${ARM_NAME:-}" ] || die "arm resolution produced nothing"
log "arm       $ARM_NAME"
log "recipe    $RECIPE"
log "image(CI) $IMAGE   <- CI runs the recipe in this image; you are running it in the current container"

# srt_agentic.sh reads an inherited CONC_LIST as a multi-node batch of points
# against one server, so the arm's list must never reach a client.
ARM_CONC_LIST="$CONC_LIST"
unset CONC_LIST
CONCURRENCIES="${CONCURRENCIES:-$ARM_CONC_LIST}"
log "tp=$TP ep=$EP_SIZE spec=$SPEC_DECODING conc=[$CONCURRENCIES]"

if [ "$MODE" = agent ]; then
  [ -f "$E2E_DIR/utils/aiperf/pyproject.toml" ] || {
    log "initialising the aiperf submodule"
    git -C "$INFERENCEX_DIR" submodule update --init "${E2E_DIR#"$INFERENCEX_DIR"/}/utils/aiperf" \
      || die "submodule init failed"
  }
fi

# ------------------------------------------------------------------- 4. GPUs
need=$TP
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  DEVICES="$CUDA_VISIBLE_DEVICES"
  log "using caller-supplied GPUs $DEVICES"
else
  free=$(python3 "$SKILL_DIR/../gpu-status/gpu_status.py" --json \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(",".join(str(g["cuda_index"]) for g in d["gpus"] if g["status"]=="FREE"))')
  n=$(awk -F, '{print NF}' <<<"${free:-}")
  [ -n "$free" ] && [ "$n" -ge "$need" ] || die "need $need free GPUs, have ${n:-0} (${free:-none}). Not sharing cards -- a contended GPU makes the number meaningless."
  DEVICES="$(cut -d, -f1-"$need" <<<"$free")"
  log "picked GPUs $DEVICES"
fi
log "server sees them through $VISIBLE_DEVICES_ENV=$DEVICES"

# ------------------------------------------------------------- 5. static env
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$RUNS_ROOT/${MODEL_PREFIX}_${MODE}_tp${TP}_${TS}}"
mkdir -p "$RUN_DIR"
log "run dir   $RUN_DIR"

# benchmark-tmpl.yml job env. ARM_MODEL keeps the HF id the recipe is validated
# against; MODEL is what the client sends.
export ARM_MODEL="$MODEL"
export INFMAX_CONTAINER_WORKSPACE="$E2E_DIR"
export MODEL MODEL_PREFIX MODEL_PATH IMAGE RUNNER_TYPE FRAMEWORK PRECISION
export TP EP_SIZE SPEC_DECODING DISAGG PORT THINKING_MODE SCENARIO_SUBDIR
export PP_SIZE=1 DCP_SIZE=1 PCP_SIZE=1 DP_SIZE=1 DP_ATTENTION=false
export GPU_COUNT=$((TP * PP_SIZE * PCP_SIZE))
export RUN_EVAL="${RUN_EVAL:-false}"
export EVAL_ONLY="${EVAL_ONLY:-false}"
export GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
export REQUIRE_POWER="${REQUIRE_POWER:-0}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/tmp/inferencex-pycache

if [ "$MODE" = fixed ]; then
  export IS_AGENTIC=0
  export ISL OSL
  export RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"   # benchmark-tmpl.yml
  export KV_OFFLOADING="" KV_OFFLOAD_BACKEND="" KV_OFFLOAD_BACKEND_METADATA=""
  export TOTAL_CPU_DRAM_GB=0
else
  # Agentic keeps MODEL as the HF repo id: aiperf's --model and --tokenizer use it.
  [ -n "$(ls -A "$MODEL_PATH" 2>/dev/null)" ] || die "MODEL_PATH is empty: $MODEL_PATH"
  export IS_AGENTIC=1
  export SCENARIO_TYPE=agentic-coding
  export ISL=0 OSL=0 MAX_MODEL_LEN=0
  # benchmark_lib.sh rejects a mismatch: KV_OFFLOAD_BACKEND is the plain backend
  # name, KV_OFFLOAD_BACKEND_METADATA its JSON, both empty when KV stays on GPU.
  export KV_OFFLOADING KV_OFFLOAD_BACKEND KV_OFFLOAD_BACKEND_METADATA
  # infx/matrix/generate.py agentic_dram_offload_gb: 0 unless KV goes to DRAM,
  # then node DRAM x dram-utilization x this point's share of the node's GPUs.
  export TOTAL_CPU_DRAM_GB=0
  if [ "$KV_OFFLOADING" = dram ]; then
    mem_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)
    util="${DRAM_UTILIZATION:-0.80}"
    TOTAL_CPU_DRAM_GB=$(awk -v k="$mem_kb" -v u="$util" -v g="$GPU_COUNT" -v n="${GPUS_PER_NODE:-8}" \
      'BEGIN{printf "%d", k*1024*u*g/n/1e9}')
    log "HiCache: KV_OFFLOAD_BACKEND=$KV_OFFLOAD_BACKEND, TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB (this point's GPU share)"
  fi
  export DURATION="${DURATION:-$([ "$FULL" = 1 ] && echo 3600 || echo 1200)}"
  [ "$DURATION" -ge 900 ] || die "DURATION=$DURATION is below 900s; aiperf would add --unsafe-override and the run would not be a valid submission"
  export AIPERF_EXPERIMENTAL_FAST="$([ "$FULL" = 1 ] && echo 0 || echo 1)"
  export AIPERF_FAILED_REQUEST_THRESHOLD="${AIPERF_FAILED_REQUEST_THRESHOLD:-0.10}"
  export AIPERF_DATASET_MMAP_CACHE_DIR="${AIPERF_DATASET_MMAP_CACHE_DIR:-/tmp/aiperf_mmap_cache}"
  mkdir -p "$AIPERF_DATASET_MMAP_CACHE_DIR"
fi
if [ "$MODE" = agent ] && [ "$SPEC_DECODING" != none ]; then
  if [ "$SYNTHETIC_ACCEPTANCE" = 1 ]; then
    log "acceptance: golden curve ($THINKING_MODE), as CI's apply_srt_recipe adds it"
  else
    log "WARNING: SYNTHETIC_ACCEPTANCE=0 -- real draft acceptance; NOT comparable to the dashboard"
  fi
fi
POINT_ARGS=()
[ "$SYNTHETIC_ACCEPTANCE" = 1 ] || POINT_ARGS+=(--no-synthetic-acceptance)

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
  export RESULT_FILENAME="${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-pp${PP_SIZE}-dcp${DCP_SIZE}-pcp${PCP_SIZE}-ep${EP_SIZE}-dpa${DP_ATTENTION}_disagg-${DISAGG}_spec-${SPEC_DECODING}_conc${CONC}_${RUNNER_NAME}"

  CONC_DIR="$RUN_DIR/conc${CONC}"
  SRT_DIR="$CONC_DIR/srt"
  mkdir -p "$SRT_DIR"

  log "=== conc=$CONC  RESULT_FILENAME=$RESULT_FILENAME"
  point_env=$(python3 "$SKILL_DIR/srt_point.py" --e2e-dir "$E2E_DIR" \
    --recipe "$E2E_DIR/$RECIPE" --mode "$MODE" --model-path "$MODEL_PATH" \
    --port "$PORT" --devices "$DEVICES" --visible-env "$VISIBLE_DEVICES_ENV" \
    "${POINT_ARGS[@]}" --out-dir "$SRT_DIR") \
    || { log "conc=$CONC: no single recipe variant for this point"; rc_total=1; continue; }
  eval "$point_env"
  log "variant   ${SRT_VARIANT##*:}  (expander: $SRT_EXPANDER)"
  log "overrides $SRT_OVERRIDES"

  if [ "$DRY_RUN" = 1 ]; then
    { echo "# variant ${SRT_VARIANT}"
      echo "# overrides ${SRT_OVERRIDES}"
      echo "# server ($SRT_DIR/launch_server.sh)"; cat "$SRT_DIR/launch_server.sh"
      echo "# recipe benchmark env ($SRT_DIR/bench_env.sh)"; cat "$SRT_DIR/bench_env.sh"
      echo "# client ($SRT_DIR/client.sh)"; cat "$SRT_DIR/client.sh"
      echo "# job env"; env | grep -E \
      '^(ARM_NAME|ARM_MODEL|MODEL|MODEL_PATH|MODEL_PREFIX|IMAGE|RUNNER_TYPE|FRAMEWORK|PRECISION|TP|EP_SIZE|PP_SIZE|DCP_SIZE|PCP_SIZE|DP_SIZE|DP_ATTENTION|SPEC_DECODING|THINKING_MODE|DISAGG|CONC|ISL|OSL|RANDOM_RANGE_RATIO|RUN_EVAL|EVAL_ONLY|PORT|GPU_COUNT|EXP_NAME|RESULT_FILENAME|INFMAX_CONTAINER_WORKSPACE|IS_AGENTIC|SCENARIO_TYPE|KV_OFFLOAD|TOTAL_CPU_DRAM_GB|DURATION|AIPERF_|REQUIRE_POWER|GPU_MONITOR_INTERVAL|PYTHON)' | sort
    } | tee "$CONC_DIR/dry_run.txt"
    continue
  fi

  if ! wait_for_slot_idle "$PORT" "$DEVICES" 600; then
    reap_our_server "$PORT"
    wait_for_slot_idle "$PORT" "$DEVICES" 600 || \
      die "conc=$CONC: slot still busy after reaping. Refusing to run -- the client would silently benchmark the previous server (frozen TPOT, exploding TTFT)."
  fi

  setsid bash "$SRT_DIR/launch_server.sh" > "$CONC_DIR/server.log" 2>&1 &
  SERVER_PGID=$!
  log "server pgid $SERVER_PGID, log $CONC_DIR/server.log"
  if ! wait_for_server "$SERVER_PGID" "$PORT" "$SERVER_READY_TIMEOUT"; then
    log "conc=$CONC server never became healthy:"
    tail -20 "$CONC_DIR/server.log"
    stop_server "$SERVER_PGID"
    rc_total=1
    continue
  fi
  log "conc=$CONC server healthy"

  # Env layering follows CI: job env, then runtime_settings.sh, then the recipe's
  # benchmark.env, then the runtime-owned bindings. Anything the caller exported
  # explicitly (e.g. a relaxed live failure threshold) still beats runtime_settings.
  (
    caller_env=$(export -p)
    set -a
    source "$E2E_DIR/benchmarks/runtime_settings.sh"
    eval "$caller_env"
    source "$SRT_DIR/bench_env.sh"
    set +a
    if [ "$MODE" = fixed ]; then
      # The served name is the local dir (srt_point.py); keep the tokenizer offline.
      export MODEL="$MODEL_PATH" HF_HUB_OFFLINE=1
    fi
    unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
    export "$VISIBLE_DEVICES_ENV=$DEVICES"
    export CONC RESULT_FILENAME
    export RESULT_DIR="$CONC_DIR" AGENTIC_OUTPUT_DIR="$CONC_DIR"
    export GPU_METRICS_CSV="$CONC_DIR/gpu_metrics.csv"
    export SRT_FRONTEND_HOST=127.0.0.1 SRT_FRONTEND_PORT="$PORT" SRT_AGG_ENDPOINTS="127.0.0.1:$PORT"
    exec bash "$SRT_DIR/client.sh"
  ) 2>&1 | tee "$CONC_DIR/recipe.log"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { log "conc=$CONC client exited $rc"; rc_total=$rc; }
  stop_server "$SERVER_PGID"

  if [ "$MODE" = fixed ]; then
    if [ -f "$CONC_DIR/$RESULT_FILENAME.json" ]; then
      # CI's "Process result" step: reads $RESULT_FILENAME.json from cwd, writes agg_*.json.
      ( cd "$CONC_DIR" && PYTHONPATH="$E2E_DIR" python3 -P -m infx.results.fixed_sequence ) \
        >"$CONC_DIR/process_result.log" 2>&1 \
        && log "conc=$CONC aggregated" \
        || log "conc=$CONC result processing failed, see $CONC_DIR/process_result.log"
    else
      log "conc=$CONC produced no result JSON -- check $CONC_DIR/recipe.log"
    fi
  fi
done

# ------------------------------------------------------------------ 7. report
if [ "$DRY_RUN" = 1 ]; then
  log "dry run only; nothing was launched. Per-point plans in $RUN_DIR/conc*/dry_run.txt"
  exit "$rc_total"
fi

# A server that ignored SIGTERM and SIGKILL's process group can still be found by port.
reap_our_server "$PORT"
wait_for_slot_idle "$PORT" "$DEVICES" 300 \
  || log "WARNING: GPUs $DEVICES not released; check for an orphan before the next run"

python3 "$SKILL_DIR/render_table.py" --run-dir "$RUN_DIR" --mode "$MODE" \
  | tee "$RUN_DIR/summary.md"

log "done: $RUN_DIR"
exit "$rc_total"
