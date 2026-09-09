#!/usr/bin/env bash
# Profile an InferenceX AgentX (agent-mode) trace replay.
#
# This adds NOTHING to the benchmark itself. It reuses
# ~/agent-box/skills/inferencemax-benchmark/run_infmax.sh verbatim -- same arm
# resolution, same recipe, same server and client command lines, same GPU
# picking and teardown -- and only attaches a sidecar that drives SGLang's
# /start_profile HTTP endpoint once the replay reaches its measurement window.
#
# Why a sidecar and not PROFILE=1: benchmark_lib.sh's PROFILE=1 path only adds
# `--profile` to the fixed-seq bench_serving client. aiperf has no such flag,
# and the agentic recipe owns the server command line, so the trace has to be
# requested from the engine over HTTP while the replay runs.
#
#   MODEL_PREFIX=qwen3.5 TP=2 PROFILE_CONCS="4 64" \
#     MODEL_PATH=/shared_nfs/models/Qwen/Qwen3.5-397B-A17B-MXFP4 \
#     bash run_agent_profile.sh
set -uo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFMAX_DIR="${INFMAX_DIR:-$SKILL_DIR/../inferencemax-benchmark}"
RUNS_ROOT="${RUNS_ROOT:-$HOME/agent-runs/inferencemax}"

MODEL_PREFIX="${MODEL_PREFIX:?set MODEL_PREFIX, e.g. qwen3.5}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the local weights dir}"
# Global convention: profiling/kernel-confirm anchors are conc4 and conc64.
PROFILE_CONCS="${PROFILE_CONCS:-4 64}"
NUM_STEPS="${NUM_STEPS:-100}"
ACTIVITIES="${ACTIVITIES:-CPU,GPU}"
PROFILE_BY_STAGE="${PROFILE_BY_STAGE:-0}"
RECORD_SHAPES="${RECORD_SHAPES:-0}"
WITH_STACK="${WITH_STACK:-0}"
MERGE_PROFILES="${MERGE_PROFILES:-0}"
PORT="${PORT:-8888}"
DRY_RUN="${DRY_RUN:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-$RUNS_ROOT/${MODEL_PREFIX}_agentprof_tp${TP:-auto}_${TS}}"

log() { printf '[agent-profile %s] %s\n' "$(date +%T)" "$*"; }
die() { printf '[agent-profile] ERROR: %s\n' "$*" >&2; exit 1; }

[ -f "$INFMAX_DIR/run_infmax.sh" ] || die "run_infmax.sh not found at $INFMAX_DIR (set INFMAX_DIR)"
mkdir -p "$RUN_ROOT"
log "run root $RUN_ROOT"

rc_total=0
for CONC in $PROFILE_CONCS; do
  RUN_DIR="$RUN_ROOT/conc${CONC}_run"
  CONC_DIR="$RUN_DIR/conc${CONC}"
  PROF_DIR="$RUN_ROOT/profiles/conc${CONC}"
  mkdir -p "$RUN_DIR" "$PROF_DIR"

  log "=== conc=$CONC  traces -> $PROF_DIR"

  if [ "$DRY_RUN" = 1 ]; then
    log "dry run: would launch run_infmax.sh (MODE=agent CONCURRENCIES=$CONC) + sidecar"
    MODE=agent CONCURRENCIES="$CONC" RUN_DIR="$RUN_DIR" PORT="$PORT" DRY_RUN=1 \
      bash "$INFMAX_DIR/run_infmax.sh"
    continue
  fi

  # The replay is the unmodified upstream run. Backgrounded so the sidecar can
  # watch it; its own EXIT trap still owns teardown.
  #
  # The failed-request threshold is relaxed because a torch-profiler window
  # stalls the scheduler for seconds and can trip aiperf's live error gate,
  # aborting the replay mid-capture. A profiled run is not a valid submission
  # anyway -- never report its aggregate as a benchmark number.
  MODE=agent \
  CONCURRENCIES="$CONC" \
  RUN_DIR="$RUN_DIR" \
  PORT="$PORT" \
  MODEL_PREFIX="$MODEL_PREFIX" \
  MODEL_PATH="$MODEL_PATH" \
  AIPERF_LIVE_FAILED_REQUEST_THRESHOLD="${AIPERF_LIVE_FAILED_REQUEST_THRESHOLD:-0.5}" \
    bash "$INFMAX_DIR/run_infmax.sh" > "$RUN_DIR/run_infmax.log" 2>&1 &
  REPLAY_PID=$!
  log "replay pid $REPLAY_PID, log $RUN_DIR/run_infmax.log"

  SIDECAR_ARGS=(
    --port "$PORT"
    --conc "$CONC"
    --out-dir "$PROF_DIR"
    --num-steps "$NUM_STEPS"
    --activities "$ACTIVITIES"
    --recipe-log "$CONC_DIR/recipe.log"
    --collect-timeout "${COLLECT_TIMEOUT:-1800}"
  )
  [ "$PROFILE_BY_STAGE" = 1 ] && SIDECAR_ARGS+=(--profile-by-stage)
  [ "$RECORD_SHAPES" = 1 ] && SIDECAR_ARGS+=(--record-shapes)
  [ "$WITH_STACK" = 1 ] && SIDECAR_ARGS+=(--with-stack)
  [ "$MERGE_PROFILES" = 1 ] && SIDECAR_ARGS+=(--merge-profiles)
  [ -n "${TRIGGER_REGEX:-}" ] && SIDECAR_ARGS+=(--trigger-regex "$TRIGGER_REGEX")

  python3 "$SKILL_DIR/profile_sidecar.py" "${SIDECAR_ARGS[@]}" \
    2>&1 | tee "$PROF_DIR/sidecar.log"
  srrc=${PIPESTATUS[0]}
  [ "$srrc" -eq 0 ] || { log "conc=$CONC sidecar exited $srrc (replay continues)"; rc_total=$srrc; }

  # Let the replay finish its duration and tear its own server down. Never add
  # a pkill here: upstream uses stop_background_process_tree, and a broad
  # `pkill -9 -f sglang` orphans TP workers into unreclaimable VRAM.
  log "conc=$CONC waiting for replay to finish"
  wait "$REPLAY_PID"; rrc=$?
  [ "$rrc" -eq 0 ] || { log "conc=$CONC replay exited $rrc"; rc_total=$rrc; }
done

log "traces:"
find "$RUN_ROOT/profiles" -name '*.trace.json.gz' -printf '  %p  %sB\n' 2>/dev/null
log "done: $RUN_ROOT"
exit "$rc_total"
