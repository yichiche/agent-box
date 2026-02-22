#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_local_benchmark_e2e.sh --image <docker_image> [options]

Required:
  --image <image>                    Docker image used for benchmark run.
  --model-path <path>                Model path for sglang.launch_server.

Common options:
  --container-name <name>            Optional container name.
                                     Default: sglang-bench-<image-tag>-<timestamp>
  --concurrencies <csv>              Concurrency list, comma separated.
                                     Default: 1,2,4,8,16
  --profile                          Enable --profile in bench_serving.
  --no-profile                       Disable profile mode (skip interactive prompt).
  --prompts-multiplier <int>         num_prompts = concurrency * multiplier.
                                     Default: 8 (or 2 when --profile is set)
  --random-input-len <int>           Random input token length. Default: 70000
  --random-output-len <int>          Random output token length. Default: 200
  --random-range-ratio <float>       Random range ratio. Default: 1.0
  --dataset-name <name>              Dataset for bench_serving. Default: random
  --result-dir <path>                Host directory for collected outputs.
                                     Default: ./benchmark_runs/<timestamp>
  --wait-timeout-sec <int>           Max wait for server health. Default: 1800
  --keep-container                   Do not stop/remove container on exit.
  --no-sudo-docker                   Use docker directly instead of sudo docker.
  --accuracy                         Run GSM8K accuracy benchmark after perf bench.
  --no-accuracy                      Skip accuracy benchmark (skip interactive prompt).
  --accuracy-num-questions <int>     Number of GSM8K questions. Default: 2000
  --accuracy-parallel <int>          GSM8K parallel requests. Default: 1000
  --accuracy-num-shots <int>         GSM8K few-shot count. Default: 5

Advanced options:
  --port <int>                       Server port. Default: 9000
  --tp-size <int>                    Tensor parallel size. Default: 8
  --max-running-requests <int>       Max running requests. Default: 64
  --chunked-prefill-size <int>       Chunked prefill size. Default: 131072
  --mem-fraction-static <float>      mem_fraction_static. Default: 0.8
  --kv-cache-dtype <dtype>           KV cache dtype. Default: fp8_e4m3
  --attention-backend <name>         Attention backend. Default: aiter
  --server-extra-args "<args>"       Extra args appended to launch_server.
  --bench-extra-args "<args>"        Extra args appended to bench_serving.

Example:
  ./run_local_benchmark_e2e.sh \
    --image rocm/sgl-dev:v0.5.7-rocm700-mi35x-20260108 \
    --model-path /data/Qwen/Qwen3-Coder-Next/ \
    --concurrencies 1,2,4,8
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

quote_join() {
  local out=""
  local arg
  for arg in "$@"; do
    printf -v out '%s%q ' "$out" "$arg"
  done
  echo "${out% }"
}

quote_one() {
  printf '%q' "$1"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"
CONFIG_FILE="${AGENT_BOX_DIR}/.bench_config"

# Load saved values from config file (if exists)
SAVED_MODEL_PATH=""
SAVED_HOST_HOME_DIR=""
if [[ -f "$CONFIG_FILE" ]]; then
  SAVED_MODEL_PATH="$(grep -m1 '^MODEL_PATH=' "$CONFIG_FILE" 2>/dev/null | cut -d= -f2- || true)"
  SAVED_HOST_HOME_DIR="$(grep -m1 '^HOST_HOME_DIR=' "$CONFIG_FILE" 2>/dev/null | cut -d= -f2- || true)"
fi

IMAGE=""
MODEL_PATH=""
CONTAINER_NAME=""
CONCURRENCIES="1,2,4,8,16"
CONCURRENCIES_SET=0
PROFILE_MODE=0
PROFILE_SET=0
PROMPTS_MULTIPLIER=""
RANDOM_INPUT_LEN=70000
RANDOM_OUTPUT_LEN=200
RANDOM_RANGE_RATIO="1.0"
DATASET_NAME="random"
RESULT_DIR=""
WAIT_TIMEOUT_SEC=1800
KEEP_CONTAINER=0
USE_SUDO_DOCKER=1
HOST_HOME_DIR=""
MTP_MODE=""
ACCURACY_MODE=""
ACCURACY_NUM_QUESTIONS=2000
ACCURACY_PARALLEL=1000
ACCURACY_NUM_SHOTS=5

SERVER_PORT=9000
TP_SIZE=8
MAX_RUNNING_REQUESTS=64
CHUNKED_PREFILL_SIZE=131072
MEM_FRACTION_STATIC="0.8"
KV_CACHE_DTYPE="fp8_e4m3"
ATTENTION_BACKEND="aiter"
SERVER_EXTRA_ARGS=""
BENCH_EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="${2:-}"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="${2:-}"
      shift 2
      ;;
    --container-name)
      CONTAINER_NAME="${2:-}"
      shift 2
      ;;
    --concurrencies)
      CONCURRENCIES="${2:-}"
      CONCURRENCIES_SET=1
      shift 2
      ;;
    --profile)
      PROFILE_MODE=1
      PROFILE_SET=1
      shift
      ;;
    --no-profile)
      PROFILE_MODE=0
      PROFILE_SET=1
      shift
      ;;
    --prompts-multiplier)
      PROMPTS_MULTIPLIER="${2:-}"
      shift 2
      ;;
    --random-input-len)
      RANDOM_INPUT_LEN="${2:-}"
      shift 2
      ;;
    --random-output-len)
      RANDOM_OUTPUT_LEN="${2:-}"
      shift 2
      ;;
    --random-range-ratio)
      RANDOM_RANGE_RATIO="${2:-}"
      shift 2
      ;;
    --dataset-name)
      DATASET_NAME="${2:-}"
      shift 2
      ;;
    --result-dir)
      RESULT_DIR="${2:-}"
      shift 2
      ;;
    --wait-timeout-sec)
      WAIT_TIMEOUT_SEC="${2:-}"
      shift 2
      ;;
    --keep-container)
      KEEP_CONTAINER=1
      shift
      ;;
    --no-sudo-docker)
      USE_SUDO_DOCKER=0
      shift
      ;;
    --accuracy)
      ACCURACY_MODE=1
      shift
      ;;
    --no-accuracy)
      ACCURACY_MODE=0
      shift
      ;;
    --accuracy-num-questions)
      ACCURACY_NUM_QUESTIONS="${2:-}"
      shift 2
      ;;
    --accuracy-parallel)
      ACCURACY_PARALLEL="${2:-}"
      shift 2
      ;;
    --accuracy-num-shots)
      ACCURACY_NUM_SHOTS="${2:-}"
      shift 2
      ;;
    --home-dir)
      HOST_HOME_DIR="${2:-}"
      shift 2
      ;;
    --mtp)
      MTP_MODE=1
      shift
      ;;
    --no-mtp)
      MTP_MODE=0
      shift
      ;;
    --port)
      SERVER_PORT="${2:-}"
      shift 2
      ;;
    --tp-size)
      TP_SIZE="${2:-}"
      shift 2
      ;;
    --max-running-requests)
      MAX_RUNNING_REQUESTS="${2:-}"
      shift 2
      ;;
    --chunked-prefill-size)
      CHUNKED_PREFILL_SIZE="${2:-}"
      shift 2
      ;;
    --mem-fraction-static)
      MEM_FRACTION_STATIC="${2:-}"
      shift 2
      ;;
    --kv-cache-dtype)
      KV_CACHE_DTYPE="${2:-}"
      shift 2
      ;;
    --attention-backend)
      ATTENTION_BACKEND="${2:-}"
      shift 2
      ;;
    --server-extra-args)
      SERVER_EXTRA_ARGS="${2:-}"
      shift 2
      ;;
    --bench-extra-args)
      BENCH_EXTRA_ARGS="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

if [[ -z "$IMAGE" ]]; then
  read -r -p "Docker image: " IMAGE
fi
if [[ -z "$MODEL_PATH" ]]; then
  if [[ -n "$SAVED_MODEL_PATH" ]]; then
    read -r -p "Model path (default: ${SAVED_MODEL_PATH}): " MODEL_PATH_INPUT
    MODEL_PATH="${MODEL_PATH_INPUT:-$SAVED_MODEL_PATH}"
  else
    read -r -p "Model path: " MODEL_PATH
  fi
fi
# Save will happen after all interactive prompts are done
if (( PROFILE_SET == 0 )); then
  read -r -p "Enable profile mode? [y/N]: " PROFILE_INPUT
  case "${PROFILE_INPUT,,}" in
    y|yes) PROFILE_MODE=1 ;;
  esac
fi
if (( CONCURRENCIES_SET == 0 )); then
  read -r -p "Concurrencies (default: 1,2,4,8,16): " CONCURRENCIES_INPUT
  if [[ -n "$CONCURRENCIES_INPUT" ]]; then
    CONCURRENCIES="$CONCURRENCIES_INPUT"
  fi
fi
if [[ -z "$HOST_HOME_DIR" ]]; then
  DEFAULT_HOME_DIR="${SAVED_HOST_HOME_DIR:-$HOME}"
  read -r -p "Host home directory (default: ${DEFAULT_HOME_DIR}): " HOME_DIR_INPUT
  HOST_HOME_DIR="${HOME_DIR_INPUT:-$DEFAULT_HOME_DIR}"
fi
if [[ -z "$MTP_MODE" ]]; then
  read -r -p "Enable MTP (speculative decoding)? [Y/n]: " MTP_INPUT
  case "${MTP_INPUT,,}" in
    n|no) MTP_MODE=0 ;;
    *)    MTP_MODE=1 ;;
  esac
fi
if [[ -z "$ACCURACY_MODE" ]]; then
  read -r -p "Run GSM8K accuracy benchmark? [y/N]: " ACCURACY_INPUT
  case "${ACCURACY_INPUT,,}" in
    y|yes) ACCURACY_MODE=1 ;;
    *)     ACCURACY_MODE=0 ;;
  esac
fi

# Save config for next run
{
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "HOST_HOME_DIR=${HOST_HOME_DIR}"
} > "$CONFIG_FILE"

[[ -n "$IMAGE" ]] || die "--image is required"
[[ -n "$MODEL_PATH" ]] || die "--model-path is required"
[[ -d "$MODEL_PATH" ]] || die "Model path does not exist or is not a directory: $MODEL_PATH"
command -v docker >/dev/null 2>&1 || die "docker not found in PATH"
command -v curl >/dev/null 2>&1 || die "curl not found in PATH"

if [[ -z "$PROMPTS_MULTIPLIER" ]]; then
  if (( PROFILE_MODE == 1 )); then
    PROMPTS_MULTIPLIER=2
  else
    PROMPTS_MULTIPLIER=8
  fi
fi

[[ "$PROMPTS_MULTIPLIER" =~ ^[0-9]+$ ]] || die "--prompts-multiplier must be an integer"
(( PROMPTS_MULTIPLIER > 0 )) || die "--prompts-multiplier must be > 0"
[[ "$WAIT_TIMEOUT_SEC" =~ ^[0-9]+$ ]] || die "--wait-timeout-sec must be an integer"
(( WAIT_TIMEOUT_SEC > 0 )) || die "--wait-timeout-sec must be > 0"

IFS=',' read -r -a CONCURRENCY_ARRAY <<< "$CONCURRENCIES"
(( ${#CONCURRENCY_ARRAY[@]} > 0 )) || die "--concurrencies cannot be empty"
for i in "${!CONCURRENCY_ARRAY[@]}"; do
  c="${CONCURRENCY_ARRAY[$i]//[[:space:]]/}"
  CONCURRENCY_ARRAY[$i]="$c"
  [[ "$c" =~ ^[0-9]+$ ]] || die "Invalid concurrency value: $c"
  (( c > 0 )) || die "Concurrency must be > 0: $c"
done

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

if [[ -z "$CONTAINER_NAME" ]]; then
  IMAGE_TAG="${IMAGE##*:}"
  IMAGE_TAG="${IMAGE_TAG//[^a-zA-Z0-9_.-]/_}"
  CONTAINER_NAME="jacky-sglang-bench-${IMAGE_TAG}-${TIMESTAMP}"
fi

if [[ -z "$RESULT_DIR" ]]; then
  RESULT_DIR="${HOST_HOME_DIR}/benchmark_runs/${TIMESTAMP}"
fi
mkdir -p "$RESULT_DIR"

CONTAINER_RESULT_DIR="/tmp/sglang-bench-${TIMESTAMP}"
CONTAINER_SERVER_LOG="${CONTAINER_RESULT_DIR}/server.log"
CONTAINER_BENCH_JSONL="${CONTAINER_RESULT_DIR}/bench_results.jsonl"
CONTAINER_TRACE_ANALYSIS_DIR="${CONTAINER_RESULT_DIR}/trace_analysis"
HOST_BENCH_JSONL="${RESULT_DIR}/bench_results.jsonl"
HOST_SUMMARY_CSV="${RESULT_DIR}/bench_summary.csv"
HOST_SERVER_LOG="${RESULT_DIR}/server.log"
HOST_CLIENT_LOG="${RESULT_DIR}/client.log"
HOST_TRACE_ANALYSIS_DIR="${RESULT_DIR}/trace_analysis"
CONTAINER_ACCURACY_LOG="${CONTAINER_RESULT_DIR}/accuracy_gsm8k.log"
HOST_ACCURACY_LOG="${RESULT_DIR}/accuracy_gsm8k.log"

docker_cmd() {
  if (( USE_SUDO_DOCKER == 1 )); then
    sudo docker "$@"
  else
    docker "$@"
  fi
}

kill_tail() {
  if [[ -n "${TAIL_PID:-}" ]]; then
    # Kill the entire process group (docker exec + tee pipeline)
    kill -- -"$TAIL_PID" 2>/dev/null || kill "$TAIL_PID" 2>/dev/null || true
    wait "$TAIL_PID" 2>/dev/null || true
    TAIL_PID=""
  fi
}

cleanup() {
  set +e
  kill_tail
  if [[ -n "${CONTAINER_NAME:-}" ]]; then
    if docker_cmd ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
      # Always kill the server process inside the container so the port is freed
      log "Killing server process inside container"
      docker_cmd exec "$CONTAINER_NAME" bash -c 'pkill -f "sglang.launch_server" 2>/dev/null; exit 0' || true
      if (( KEEP_CONTAINER == 1 )); then
        log "Keeping container: ${CONTAINER_NAME}"
      else
        # Interactive prompt: ask user whether to remove the container (default: yes)
        local answer=""
        read -r -p "Remove container '${CONTAINER_NAME}'? [Y/n]: " answer 2>/dev/tty </dev/tty || true
        case "${answer,,}" in
          n|no)
            log "Keeping container: ${CONTAINER_NAME}"
            ;;
          *)
            log "Stopping/removing container: ${CONTAINER_NAME}"
            docker_cmd rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
            ;;
        esac
      fi
    fi
  fi
}
trap cleanup EXIT

if docker_cmd ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  die "Container already exists: $CONTAINER_NAME (pick another name or remove it)"
fi

log "Starting container: ${CONTAINER_NAME}"
DOCKER_RUN_ARGS=(
  -d
  --privileged
  --name "$CONTAINER_NAME"
  --network=host
  --device=/dev/kfd
  --device=/dev/dri
  --group-add video
  --cap-add SYS_PTRACE
  --security-opt seccomp=unconfined
  --ipc=host
  --shm-size 16G
)

add_mount_if_exists() {
  local host_path="$1"
  local container_path="$2"
  if [[ -e "$host_path" ]]; then
    DOCKER_RUN_ARGS+=(-v "${host_path}:${container_path}")
  fi
}

add_mount_if_exists "$HOST_HOME_DIR" "$HOST_HOME_DIR"
add_mount_if_exists /data /data
add_mount_if_exists /data2 /data2
add_mount_if_exists /mnt /mnt
add_mount_if_exists /raid /raid

docker_cmd run "${DOCKER_RUN_ARGS[@]}" "$IMAGE" tail -f /dev/null >/dev/null

docker_cmd exec "$CONTAINER_NAME" mkdir -p "$CONTAINER_RESULT_DIR"

SERVER_ARGS=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --tensor-parallel-size "$TP_SIZE"
  --trust-remote-code
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --host 0.0.0.0
  --port "$SERVER_PORT"
  --log-requests
  --disable-radix-cache
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --attention-backend "$ATTENTION_BACKEND"
)

if (( MTP_MODE == 1 )); then
  SERVER_ARGS+=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
  )
fi

if [[ -n "$SERVER_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  SERVER_EXTRA_ARRAY=($SERVER_EXTRA_ARGS)
  SERVER_ARGS+=("${SERVER_EXTRA_ARRAY[@]}")
fi

SERVER_CMD="$(quote_join "${SERVER_ARGS[@]}")"
START_SERVER_CMD="cd /sgl-workspace/sglang/python && SGLANG_AITER_MLA_PERSIST=1 ${SERVER_CMD} > $(quote_one "$CONTAINER_SERVER_LOG") 2>&1"

log "Launching server inside container"
docker_cmd exec -d "$CONTAINER_NAME" bash -lc "$START_SERVER_CMD"

log "Streaming server log to host: ${HOST_SERVER_LOG}"
set -m  # enable job control so the pipeline gets its own process group
docker_cmd exec "$CONTAINER_NAME" bash -lc "tail -n +1 -F $(quote_one "$CONTAINER_SERVER_LOG")" \
  | tee -a "$HOST_SERVER_LOG" >/dev/null &
TAIL_PID=$!
set +m

wait_for_health() {
  local deadline=$((SECONDS + WAIT_TIMEOUT_SEC))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://localhost:${SERVER_PORT}/health_generate" >/dev/null 2>&1; then
      return 0
    fi

    if ! docker_cmd exec "$CONTAINER_NAME" pgrep -f "sglang.launch_server" >/dev/null 2>&1; then
      log "Server process is not running. Last server log lines:"
      docker_cmd exec "$CONTAINER_NAME" tail -n 80 "$CONTAINER_SERVER_LOG" || true
      return 1
    fi

    sleep 5
  done
  return 1
}

log "Waiting for server health on port ${SERVER_PORT}"
if ! wait_for_health; then
  die "Server failed health check within ${WAIT_TIMEOUT_SEC}s"
fi
log "Server is healthy"

if (( ACCURACY_MODE == 1 )); then
  log "Running GSM8K accuracy benchmark: num_questions=${ACCURACY_NUM_QUESTIONS}, parallel=${ACCURACY_PARALLEL}, num_shots=${ACCURACY_NUM_SHOTS}"
  ACCURACY_CMD="cd /sgl-workspace/sglang/python && python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py"
  ACCURACY_CMD+=" --num-questions ${ACCURACY_NUM_QUESTIONS}"
  ACCURACY_CMD+=" --parallel ${ACCURACY_PARALLEL}"
  ACCURACY_CMD+=" --num-shots ${ACCURACY_NUM_SHOTS}"
  ACCURACY_CMD+=" --port ${SERVER_PORT}"
  docker_cmd exec "$CONTAINER_NAME" bash -lc \
    "${ACCURACY_CMD} 2>&1 | tee $(quote_one "$CONTAINER_ACCURACY_LOG")" \
    | tee -a "$HOST_ACCURACY_LOG"
fi

for c in "${CONCURRENCY_ARRAY[@]}"; do
  NUM_PROMPTS=$((c * PROMPTS_MULTIPLIER))
  CONTAINER_BENCH_LOG="${CONTAINER_RESULT_DIR}/bench_c${c}.log"

  BENCH_ARGS=(
    python3 -m sglang.bench_serving
    --backend sglang
    --host localhost
    --port "$SERVER_PORT"
    --model "$MODEL_PATH"
    --dataset-name "$DATASET_NAME"
    --random-input-len "$RANDOM_INPUT_LEN"
    --random-output-len "$RANDOM_OUTPUT_LEN"
    --random-range-ratio "$RANDOM_RANGE_RATIO"
    --max-concurrency "$c"
    --num-prompts "$NUM_PROMPTS"
    --disable-tqdm
    --output-file "$CONTAINER_BENCH_JSONL"
  )

  if (( PROFILE_MODE == 1 )); then
    BENCH_ARGS+=(--profile)
  fi

  if [[ -n "$BENCH_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    BENCH_EXTRA_ARRAY=($BENCH_EXTRA_ARGS)
    BENCH_ARGS+=("${BENCH_EXTRA_ARRAY[@]}")
  fi

  BENCH_CMD="$(quote_join "${BENCH_ARGS[@]}")"
  log "Running benchmark: concurrency=${c}, num_prompts=${NUM_PROMPTS}"
  docker_cmd exec "$CONTAINER_NAME" bash -lc \
    "cd /sgl-workspace/sglang/python && ${BENCH_CMD} 2>&1 | tee -a $(quote_one "$CONTAINER_BENCH_LOG")" \
    | tee -a "$HOST_CLIENT_LOG"
done

if (( PROFILE_MODE == 1 )); then
  log "Collecting TP0 trace from /tmp"
  TRACE_PATH_IN_CONTAINER="$(docker_cmd exec "$CONTAINER_NAME" bash -lc "ls -t /tmp/*-TP-0.trace.json.gz 2>/dev/null | head -n 1" || true)"
  if [[ -n "$TRACE_PATH_IN_CONTAINER" ]]; then
    TRACE_BASENAME="$(basename "$TRACE_PATH_IN_CONTAINER")"
    log "Running trace analyzer in container"
    docker_cmd exec "$CONTAINER_NAME" mkdir -p "$CONTAINER_TRACE_ANALYSIS_DIR"
    docker_cmd exec "$CONTAINER_NAME" pip install openpyxl -q
    docker_cmd cp "${AGENT_BOX_DIR}/profile/trace_analyzer.py" \
      "${CONTAINER_NAME}:/tmp/trace_analyzer.py"
    docker_cmd exec "$CONTAINER_NAME" bash -lc \
      "python3 /tmp/trace_analyzer.py $(quote_one "$TRACE_PATH_IN_CONTAINER") \
      --export-csv $(quote_one "${CONTAINER_TRACE_ANALYSIS_DIR}/profile.csv") \
      --export-layers 58-65 \
      --debug-layers 2-5 \
      > $(quote_one "${CONTAINER_TRACE_ANALYSIS_DIR}/trace_analyzer.log") 2>&1"
    mkdir -p "$HOST_TRACE_ANALYSIS_DIR"
    docker_cmd cp "${CONTAINER_NAME}:${TRACE_PATH_IN_CONTAINER}" \
      "${HOST_TRACE_ANALYSIS_DIR}/${TRACE_BASENAME}"
  else
    log "No TP0 trace found in /tmp (expected *-TP-0.trace.json.gz)"
  fi
fi

# Stop the background log-streaming pipeline before collecting artifacts
kill_tail

log "Collecting benchmark artifacts to host: ${RESULT_DIR}"
docker_cmd cp "${CONTAINER_NAME}:${CONTAINER_RESULT_DIR}/." "${RESULT_DIR}/"

if [[ -f "$HOST_BENCH_JSONL" ]]; then
  log "Building CSV summary: ${HOST_SUMMARY_CSV}"
  python3 - "$HOST_BENCH_JSONL" "$HOST_SUMMARY_CSV" <<'PY'
import csv
import json
import pathlib
import sys

jsonl_path = pathlib.Path(sys.argv[1])
csv_path = pathlib.Path(sys.argv[2])

rows = []
for raw in jsonl_path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line:
        continue
    item = json.loads(line)
    rows.append(
        {
            "concurrency": item.get("max_concurrency"),
            "completed_requests": item.get("completed"),
            "benchmark_duration_s": item.get("duration"),
            "request_throughput_req_s": item.get("request_throughput"),
            "input_throughput_tok_s": item.get("input_throughput"),
            "output_throughput_tok_s": item.get("output_throughput"),
            "total_throughput_tok_s": item.get("total_throughput"),
            "median_e2e_latency_ms": item.get("median_e2e_latency_ms"),
            "median_ttft_ms": item.get("median_ttft_ms"),
            "median_itl_ms": item.get("median_itl_ms"),
        }
    )

with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "concurrency",
            "completed_requests",
            "benchmark_duration_s",
            "request_throughput_req_s",
            "input_throughput_tok_s",
            "output_throughput_tok_s",
            "median_e2e_latency_ms",
            "total_throughput_tok_s",
            "median_ttft_ms",
            "median_itl_ms",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
PY
else
  log "No JSONL output found at ${HOST_BENCH_JSONL}; skipping CSV generation"
fi

log "Benchmark run completed"
log "Results directory: ${RESULT_DIR}"
log "JSONL: ${HOST_BENCH_JSONL}"
log "CSV:   ${HOST_SUMMARY_CSV}"
