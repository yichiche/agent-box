#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./compare-benchmarks.sh --image-a <img> --image-b <img> [options]

Required:
  --image-a <image>      First Docker image (baseline).
  --image-b <image>      Second Docker image (candidate).

Comparison options:
  --label-a <label>      Display label for image A. Default: image name.
  --label-b <label>      Display label for image B. Default: image name.
  --args-a "<args>"      Extra args passed only to Image A benchmark run.
  --args-b "<args>"      Extra args passed only to Image B benchmark run.
  --reconfigure-b        Re-prompt interactive config for Image B run.
                         Default: Image B reuses the same config as Image A.
  --output-csv <path>    Write machine-readable comparison CSV.

All other options are forwarded to both runs of run-local-benchmark-e2e.sh:
  --model-path, --concurrencies, --home-dir, --profile, --no-profile,
  --accuracy, --no-accuracy, --mtp, --no-mtp, --port, --tp-size, etc.

Non-interactive defaults (injected if not explicitly provided):
  --no-profile, --no-accuracy

Example:
  ./compare-benchmarks.sh \
    --image-a rocm/sgl-dev:v0.5.7-old \
    --image-b rocm/sgl-dev:v0.5.8-new \
    --model-path /data/Qwen/Qwen3-Coder-Next/ \
    --concurrencies 1,2,4,8 \
    --home-dir "$HOME" \
    --no-profile \
    --mtp \
    --no-accuracy

  # Compare same image with different aiter versions:
  ./compare-benchmarks.sh \
    --image-a rocm/sgl-dev:v0.5.9 \
    --image-b rocm/sgl-dev:v0.5.9 \
    --model-path /data/model/ \
    --args-b "--aiter-pr 123"
EOF
}

log() {
  printf '[%s] [compare] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"
BENCH_SCRIPT="${SCRIPT_DIR}/run-local-benchmark-e2e.sh"
COMPARE_SCRIPT="${SCRIPT_DIR}/compare_bench_results.py"

[[ -x "$BENCH_SCRIPT" ]] || die "Benchmark script not found: $BENCH_SCRIPT"
[[ -f "$COMPARE_SCRIPT" ]] || die "Comparison script not found: $COMPARE_SCRIPT"

IMAGE_A=""
IMAGE_B=""
LABEL_A=""
LABEL_B=""
OUTPUT_CSV=""
EXTRA_ARGS_A=""
EXTRA_ARGS_B=""
RECONFIGURE_B=0
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image-a)
      IMAGE_A="${2:-}"
      shift 2
      ;;
    --image-b)
      IMAGE_B="${2:-}"
      shift 2
      ;;
    --label-a)
      LABEL_A="${2:-}"
      shift 2
      ;;
    --label-b)
      LABEL_B="${2:-}"
      shift 2
      ;;
    --output-csv)
      OUTPUT_CSV="${2:-}"
      shift 2
      ;;
    --args-a)
      EXTRA_ARGS_A="${2:-}"
      shift 2
      ;;
    --args-b)
      EXTRA_ARGS_B="${2:-}"
      shift 2
      ;;
    --reconfigure-b)
      RECONFIGURE_B=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      # Collect remaining args to forward to the benchmark script.
      # Handle both --flag and --flag value forms.
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

[[ -n "$IMAGE_A" ]] || die "--image-a is required"
[[ -n "$IMAGE_B" ]] || die "--image-b is required"

# Default labels to image names
[[ -n "$LABEL_A" ]] || LABEL_A="$IMAGE_A"
[[ -n "$LABEL_B" ]] || LABEL_B="$IMAGE_B"

# Helper: check if a flag is already present in PASSTHROUGH_ARGS
has_flag() {
  local flag="$1"
  for arg in "${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}"; do
    if [[ "$arg" == "$flag" ]]; then
      return 0
    fi
  done
  return 1
}

# Inject non-interactive defaults if not already specified
if ! has_flag "--profile" && ! has_flag "--no-profile"; then
  PASSTHROUGH_ARGS+=("--no-profile")
fi
if ! has_flag "--accuracy" && ! has_flag "--no-accuracy"; then
  PASSTHROUGH_ARGS+=("--no-accuracy")
fi

# Determine home-dir for result storage
HOME_DIR="$HOST_HOME"
for i in "${!PASSTHROUGH_ARGS[@]}"; do
  if [[ "${PASSTHROUGH_ARGS[$i]}" == "--home-dir" ]]; then
    HOME_DIR="${PASSTHROUGH_ARGS[$((i + 1))]:-$HOST_HOME}"
    break
  fi
done

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
COMPARISON_DIR="${HOME_DIR}/benchmark_comparisons/${TIMESTAMP}"
RESULT_DIR_A="${COMPARISON_DIR}/image_a"
RESULT_DIR_B="${COMPARISON_DIR}/image_b"
mkdir -p "$RESULT_DIR_A" "$RESULT_DIR_B"

CONTAINER_A="compare-bench-a-${TIMESTAMP}"
CONTAINER_B="compare-bench-b-${TIMESTAMP}"

CSV_A="${RESULT_DIR_A}/bench_summary.csv"
CSV_B="${RESULT_DIR_B}/bench_summary.csv"

log "Comparison directory: ${COMPARISON_DIR}"
log "Image A: ${LABEL_A}"
log "Image B: ${LABEL_B}"

# Resolved config file: Run A saves its config here, Run B loads it by default
RESOLVED_CONFIG_FILE="${COMPARISON_DIR}/.resolved_config"

# ─── Run A ───────────────────────────────────────────────────────────────────
# shellcheck disable=SC2206
EXTRA_ARGS_A_ARRAY=($EXTRA_ARGS_A)
log "Starting benchmark for Image A: ${LABEL_A}"
RUN_A_OK=1
if ! "$BENCH_SCRIPT" \
    --image "$IMAGE_A" \
    --result-dir "$RESULT_DIR_A" \
    --container-name "$CONTAINER_A" \
    --keep-container \
    --resolved-config "$RESOLVED_CONFIG_FILE" \
    "${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}" \
    "${EXTRA_ARGS_A_ARRAY[@]+"${EXTRA_ARGS_A_ARRAY[@]}"}"; then
  log "WARNING: Benchmark for Image A failed"
  RUN_A_OK=0
fi

# Clean up container A to free GPU memory
log "Removing container for Image A: ${CONTAINER_A}"
sudo docker rm -f "$CONTAINER_A" 2>/dev/null || docker rm -f "$CONTAINER_A" 2>/dev/null || true

# ─── Run B ───────────────────────────────────────────────────────────────────
# shellcheck disable=SC2206
EXTRA_ARGS_B_ARRAY=($EXTRA_ARGS_B)

# By default, reuse Image A's config for Image B (no re-prompting)
LOAD_CONFIG_ARGS=()
if (( RECONFIGURE_B == 0 )) && [[ -f "$RESOLVED_CONFIG_FILE" ]]; then
  log "Reusing Image A config for Image B (use --reconfigure-b to re-prompt)"
  LOAD_CONFIG_ARGS=(--load-config "$RESOLVED_CONFIG_FILE")
elif (( RECONFIGURE_B == 1 )); then
  log "Re-prompting config for Image B (--reconfigure-b)"
fi

log "Starting benchmark for Image B: ${LABEL_B}"
RUN_B_OK=1
if ! "$BENCH_SCRIPT" \
    --image "$IMAGE_B" \
    --result-dir "$RESULT_DIR_B" \
    --container-name "$CONTAINER_B" \
    --keep-container \
    "${LOAD_CONFIG_ARGS[@]+"${LOAD_CONFIG_ARGS[@]}"}" \
    "${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}" \
    "${EXTRA_ARGS_B_ARRAY[@]+"${EXTRA_ARGS_B_ARRAY[@]}"}"; then
  log "WARNING: Benchmark for Image B failed"
  RUN_B_OK=0
fi

# Clean up container B
log "Removing container for Image B: ${CONTAINER_B}"
sudo docker rm -f "$CONTAINER_B" 2>/dev/null || docker rm -f "$CONTAINER_B" 2>/dev/null || true

# ─── Compare ─────────────────────────────────────────────────────────────────
if (( RUN_A_OK == 0 && RUN_B_OK == 0 )); then
  die "Both benchmark runs failed. No comparison possible."
fi

if (( RUN_A_OK == 0 )); then
  log "Image A benchmark failed — skipping comparison."
  log "Image B results are in: ${RESULT_DIR_B}"
  exit 1
fi

if (( RUN_B_OK == 0 )); then
  log "Image B benchmark failed — skipping comparison."
  log "Image A results are in: ${RESULT_DIR_A}"
  exit 1
fi

if [[ ! -f "$CSV_A" ]]; then
  die "Image A CSV not found: ${CSV_A}"
fi
if [[ ! -f "$CSV_B" ]]; then
  die "Image B CSV not found: ${CSV_B}"
fi

SUMMARY_CSV="${COMPARISON_DIR}/summary.csv"
COMPARE_ARGS=(
  python3 "$COMPARE_SCRIPT"
  --csv-a "$CSV_A"
  --csv-b "$CSV_B"
  --label-a "$LABEL_A"
  --label-b "$LABEL_B"
  --summary-csv "$SUMMARY_CSV"
)
if [[ -n "$OUTPUT_CSV" ]]; then
  COMPARE_ARGS+=(--output-csv "$OUTPUT_CSV")
fi

log "Running comparison"
"${COMPARE_ARGS[@]}"

log "Comparison complete"
log "Results directory: ${COMPARISON_DIR}"
log "  Image A results: ${RESULT_DIR_A}"
log "  Image B results: ${RESULT_DIR_B}"
log "  Summary CSV:     ${SUMMARY_CSV}"
