#!/usr/bin/env bash
# perf_sweep.sh — accuracy-gated concurrency-sweep benchmark for an SGLang server.
#
# Runs INSIDE the target container (invoke via `docker exec ... bash perf_sweep.sh`).
# One model load serves: (1) an optional GSM8K accuracy gate, then (2) a concurrency
# sweep, with optional per-concurrency profiling. Designed to be model-agnostic:
# everything is env-overridable, and server flags can be inherited from a reference
# script under /home/yichiche (see REF_SCRIPT / SERVER_ARGS).
#
# It encodes the gotchas learned the hard way on this box:
#   * GPU selection uses HIP_VISIBLE_DEVICES with CUDA/torch indices (same enumeration
#     torch uses). NEVER pass an empty string — empty VISIBLE var = hide ALL GPUs.
#     Get the free CUDA indices from the gpu-status skill (cached map).
#   * The container's benchmark_serving.py is a NEWER sglang variant: ignore-eos is the
#     default (only --disable-ignore-eos exists), results save via --output-file (NOT
#     --save-result/--result-dir/--result-filename), and percentiles print by default
#     (no --percentile-metrics). This script PROBES --help and adapts the arg list, so
#     it works across old and new benchmark_serving.py.
#   * Server is launched once and KEPT alive (no per-conc reload); torn down at the end
#     unless KEEP_SERVER=1.
set -uo pipefail

log()  { echo "[perf-sweep] $(date +%H:%M:%S) $*"; }
warn() { echo "[perf-sweep] WARNING: $*" >&2; }
die()  { echo "[perf-sweep] FATAL: $*" >&2; exit 1; }

# ── Config (all env-overridable) ─────────────────────────────────────────────
MODEL="${MODEL:?set MODEL=/path/to/model}"
PORT="${PORT:-8001}"
TP="${TP:-2}"
GPUS="${GPUS:?set GPUS=6,7  (CUDA/torch indices from gpu-status; used as HIP_VISIBLE_DEVICES)}"
[ -z "$GPUS" ] && die "GPUS is empty — an empty VISIBLE var hides ALL GPUs. Pass real indices."

INPUT_LEN="${INPUT_LEN:-8192}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
RANGE_RATIO="${RANGE_RATIO:-0.8}"
DATASET="${DATASET:-random}"
BACKEND="${BACKEND:-sglang}"
CONCURRENCIES="${CONCURRENCIES:-4 8 16 32 64 128 256}"
NUM_PROMPTS_MULT="${NUM_PROMPTS_MULT:-8}"     # num_prompts = conc * MULT
NUM_PROMPTS_CAP="${NUM_PROMPTS_CAP:-0}"       # 0 = uncapped

# Accuracy gate
ACCURACY="${ACCURACY:-1}"                     # 1 = run GSM8K gate before sweep
ACC_THRESHOLD="${ACC_THRESHOLD:-0.92}"
ACC_NUM_Q="${ACC_NUM_Q:-200}"
ACC_PARALLEL="${ACC_PARALLEL:-2000}"
ACC_NUM_SHOTS="${ACC_NUM_SHOTS:-5}"
GSM8K_SCRIPT="${GSM8K_SCRIPT:-/sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py}"

# Profiling (only at the listed concurrencies)
PROFILE_CONCS="${PROFILE_CONCS:-}"            # e.g. "4" or "4 16"; empty = none
PROFILE_DIR="${PROFILE_DIR:-/tmp/perf_sweep_profiles}"

# Server lifecycle
LAUNCH_SERVER="${LAUNCH_SERVER:-1}"           # 0 = assume server already up on $PORT
KEEP_SERVER="${KEEP_SERVER:-0}"               # 1 = leave server running at the end
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1200}"
HOST="${HOST:-127.0.0.1}"

# Paths
BENCH_SERVING_DIR="${BENCH_SERVING_DIR:-}"    # autodetected if empty
RESULT_DIR="${RESULT_DIR:-/tmp/perf_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULT_DIR"
SERVER_LOG="${SERVER_LOG:-$RESULT_DIR/server.log}"
SUMMARY_CSV="$RESULT_DIR/summary.csv"

# Server flags / env (default = the Qwen3.5-MXFP4 perf-agent contract; override freely)
# NOTE: keep SERVER_ARGS free of embedded JSON/quotes — it is word-split unquoted.
# To enable multithread load, pass it yourself, properly quoted, in your own launch,
# or extend launch_server. Default is a plain, robust Qwen3.5-MXFP4 aiter flag set.
SERVER_ARGS="${SERVER_ARGS:---attention-backend aiter --trust-remote-code --chunked-prefill-size 32768 --watchdog-timeout 1200 --mem-fraction-static 0.9 --disable-radix-cache --enable-aiter-allreduce-fusion --max-running-requests 512 --page-size 16}"
SERVER_ENV="${SERVER_ENV:-AITER_FLYDSL_FORCE=1 SGLANG_USE_AITER_UNIFIED_ATTN=1 SGLANG_USE_AITER=1}"

SERVER_PID=""
cleanup() {
  if [ "$KEEP_SERVER" != "1" ] && [ -n "$SERVER_PID" ]; then
    log "Stopping server (pkill on port $PORT)"
    pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ── Locate benchmark_serving.py ──────────────────────────────────────────────
find_bench_serving() {
  [ -n "$BENCH_SERVING_DIR" ] && { echo "$BENCH_SERVING_DIR/benchmark_serving.py"; return; }
  for d in "$HOME/bench_serving" /sgl-workspace/bench_serving /root/bench_serving; do
    [ -f "$d/benchmark_serving.py" ] && { echo "$d/benchmark_serving.py"; return; }
  done
  # last resort: sglang's module
  echo "MODULE:sglang.bench_serving"
}
BENCH="$(find_bench_serving)"
log "benchmark client: $BENCH"

# Probe which args this benchmark_serving.py supports (the portability fix).
BENCH_HELP="$( { [ "${BENCH#MODULE:}" != "$BENCH" ] && python3 -m "${BENCH#MODULE:}" --help \
                 || python3 "$BENCH" --help; } 2>&1 )"
has_arg() { grep -q -- "$1" <<<"$BENCH_HELP"; }

# ── Server ───────────────────────────────────────────────────────────────────
launch_server() {
  [ "$LAUNCH_SERVER" = "1" ] || { log "LAUNCH_SERVER=0 — using existing server on $PORT"; return; }
  log "Launching server: model=$(basename "$MODEL") port=$PORT tp=$TP CUDA_VISIBLE_DEVICES=$GPUS"
  local prof_env=""
  [ -n "$PROFILE_CONCS" ] && { mkdir -p "$PROFILE_DIR"; prof_env="SGLANG_TORCH_PROFILER_DIR=$PROFILE_DIR"; }
  # GPU pinning: SGLang reads CUDA_VISIBLE_DEVICES (NOT HIP_VISIBLE_DEVICES) and rocm-smi index != CUDA index.
  # GPUS must be CUDA/torch indices (from gpu-status). Do NOT also set HIP_VISIBLE_DEVICES (double-remap).
  # shellcheck disable=SC2086
  setsid env CUDA_VISIBLE_DEVICES="$GPUS" $SERVER_ENV $prof_env \
    python3 -m sglang.launch_server --model-path "$MODEL" --tp "$TP" \
      --host 0.0.0.0 --port "$PORT" $SERVER_ARGS \
    > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
}

wait_health() {
  log "Waiting for server health (timeout ${HEALTH_TIMEOUT}s)"
  local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health_generate" 2>/dev/null)" = "200" ]; then
      log "Server ready"; return 0
    fi
    grep -qiE "Traceback|OutOfMemory|HIP error|CUDA error|FATAL" "$SERVER_LOG" 2>/dev/null \
      && die "Server crashed during load — see $SERVER_LOG"
    sleep 5
  done
  die "Server not ready within ${HEALTH_TIMEOUT}s — see $SERVER_LOG"
}

# ── Accuracy gate (GSM8K) ────────────────────────────────────────────────────
run_accuracy() {
  [ "$ACCURACY" = "1" ] || { log "ACCURACY=0 — skipping accuracy gate"; return 0; }
  local alog="$RESULT_DIR/gsm8k_accuracy.log"
  log "GSM8K accuracy: $ACC_NUM_Q questions, parallel $ACC_PARALLEL, threshold $ACC_THRESHOLD"
  python3 "$GSM8K_SCRIPT" --num-questions "$ACC_NUM_Q" --num-shots "$ACC_NUM_SHOTS" \
    --parallel "$ACC_PARALLEL" --host "http://${HOST}" --port "$PORT" 2>&1 | tee "$alog"
  local acc
  acc="$(grep -oiE 'Accuracy[: ]+[0-9.]+' "$alog" | grep -oE '[0-9.]+' | tail -1)"
  [ -z "$acc" ] && die "Could not parse accuracy from $alog"
  echo "$acc" > "$RESULT_DIR/accuracy.txt"
  if awk "BEGIN{exit !($acc >= $ACC_THRESHOLD)}"; then
    log "ACCURACY PASS: $acc >= $ACC_THRESHOLD"
    return 0
  fi
  die "ACCURACY GATE FAILED: $acc < $ACC_THRESHOLD — NOT running the sweep."
}

# ── One concurrency point ────────────────────────────────────────────────────
bench_one() {
  local conc="$1"
  local nump=$(( conc * NUM_PROMPTS_MULT ))
  [ "$NUM_PROMPTS_CAP" -gt 0 ] && [ "$nump" -gt "$NUM_PROMPTS_CAP" ] && nump="$NUM_PROMPTS_CAP"
  local out="$RESULT_DIR/result_conc${conc}.json"
  local clog="$RESULT_DIR/bench_conc${conc}.log"
  log "conc=$conc num_prompts=$nump  (IL${INPUT_LEN}/OL${OUTPUT_LEN})"

  local args=( --model "$MODEL" --backend "$BACKEND" --host "$HOST" --port "$PORT"
               --dataset-name "$DATASET" --random-input-len "$INPUT_LEN"
               --random-output-len "$OUTPUT_LEN" --num-prompts "$nump"
               --max-concurrency "$conc" --request-rate inf )
  has_arg "--random-range-ratio" && args+=( --random-range-ratio "$RANGE_RATIO" )
  # ignore-eos: new variant has it on by default (only --disable-ignore-eos); old needs --ignore-eos
  if has_arg "--ignore-eos" && ! has_arg "--disable-ignore-eos"; then args+=( --ignore-eos ); fi
  # result saving: new=--output-file, old=--save-result/--result-dir/--result-filename
  if has_arg "--output-file"; then
    args+=( --output-file "$out" )
  elif has_arg "--save-result"; then
    args+=( --save-result --result-dir "$RESULT_DIR" --result-filename "$(basename "$out")" )
  fi
  has_arg "--percentile-metrics" && args+=( --percentile-metrics ttft,tpot,itl,e2el )

  if [ "${BENCH#MODULE:}" != "$BENCH" ]; then
    python3 -m "${BENCH#MODULE:}" "${args[@]}" 2>&1 | tee "$clog"
  else
    python3 "$BENCH" "${args[@]}" 2>&1 | tee "$clog"
  fi
}

# ── Profiling pass (separate from the perf table; --profile distorts throughput) ──
profile_pass() {
  [ -n "$PROFILE_CONCS" ] || return 0
  has_arg "--profile" || { warn "client has no --profile; skipping profiling pass"; return 0; }
  mkdir -p "$PROFILE_DIR"
  for conc in $PROFILE_CONCS; do
    local nump=$(( conc * 4 )); [ "$nump" -lt 8 ] && nump=8   # short run, just for a trace
    local plog="$RESULT_DIR/profile_conc${conc}.log"
    log "PROFILING pass conc=$conc (num_prompts=$nump) -> $PROFILE_DIR"
    local pargs=( --model "$MODEL" --backend "$BACKEND" --host "$HOST" --port "$PORT"
                  --dataset-name "$DATASET" --random-input-len "$INPUT_LEN"
                  --random-output-len "$OUTPUT_LEN" --num-prompts "$nump"
                  --max-concurrency "$conc" --request-rate inf --profile )
    has_arg "--random-range-ratio" && pargs+=( --random-range-ratio "$RANGE_RATIO" )
    has_arg "--profile-output-dir" && pargs+=( --profile-output-dir "$PROFILE_DIR" )
    if [ "${BENCH#MODULE:}" != "$BENCH" ]; then
      python3 -m "${BENCH#MODULE:}" "${pargs[@]}" 2>&1 | tee "$plog"
    else
      python3 "$BENCH" "${pargs[@]}" 2>&1 | tee "$plog"
    fi
    # the client may print "Traces are saved to: <dir>" (server-side) — capture it
    grep -oiE "saved to[: ]+[^ ]+" "$RESULT_DIR/../"*.log "$SERVER_LOG" 2>/dev/null | tail -1 || true
  done
}

# ── Summarize ────────────────────────────────────────────────────────────────
summarize() {
  python3 - "$RESULT_DIR" "$SUMMARY_CSV" <<'PY'
import json, os, sys, glob
rd, csv_out = sys.argv[1], sys.argv[2]
# NOTE: key names match the newer sglang benchmark_serving.py JSON schema.
cols = ["max_concurrency","completed","duration","request_throughput",
        "input_throughput","output_throughput","total_throughput",
        "mean_ttft_ms","median_ttft_ms","p99_ttft_ms",
        "mean_tpot_ms","median_tpot_ms","p99_tpot_ms",
        "mean_itl_ms","median_itl_ms","p99_itl_ms",
        "mean_e2e_latency_ms","median_e2e_latency_ms","p99_e2e_latency_ms"]
rows=[]
for f in glob.glob(os.path.join(rd,"result_conc*.json")):
    try: d=json.load(open(f))
    except Exception: continue
    rows.append(d)
def k(d): return d.get("max_concurrency",0)
rows.sort(key=k)
with open(csv_out,"w") as o:
    o.write(",".join(cols)+"\n")
    for d in rows:
        o.write(",".join(str(d.get(c,"")) for c in cols)+"\n")
print("\n=== SWEEP SUMMARY (%s) ===" % csv_out)
hdr=["conc","out_tok/s","tot_tok/s","med_TTFT","med_TPOT","med_ITL","med_E2E"]
print("{:>6} {:>10} {:>10} {:>10} {:>9} {:>9} {:>10}".format(*hdr))
for d in rows:
    print("{:>6} {:>10.1f} {:>10.1f} {:>10.1f} {:>9.2f} {:>9.2f} {:>10.1f}".format(
        d.get("max_concurrency",0), d.get("output_throughput",0), d.get("total_throughput",0),
        d.get("median_ttft_ms",0), d.get("median_tpot_ms",0), d.get("median_itl_ms",0),
        d.get("median_e2e_latency_ms",0)))
PY
}

# ── Main ─────────────────────────────────────────────────────────────────────
log "RESULT_DIR=$RESULT_DIR"
launch_server
wait_health
run_accuracy
for c in $CONCURRENCIES; do bench_one "$c"; done
profile_pass
summarize
log "DONE. Results in $RESULT_DIR (summary.csv, per-conc JSON/logs, gsm8k_accuracy.log)"
