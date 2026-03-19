#!/usr/bin/env bash
# bisect.sh — Manual binary search across commits to find the exact regression.
# Runs inside an already-running container with full git history available.
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
GOOD=""
BAD=""
MODE=""
LIB=""
REPO_DIR=""
SERVER_SCRIPT=""
CLIENT_SCRIPT=""
ACCURACY_SCRIPT=""
ACCURACY_BASELINE=""
ACCURACY_THRESHOLD="2.0"
BUILD_SCRIPT=""
PORT="30000"
MODEL_PATH=""
TIMEOUT="600"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_BOX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TRACE_ANALYZER_SCRIPT="${AGENT_BOX_DIR}/profile/trace_module_analyzer.py"
EVALUATE_SCRIPT="${AGENT_BOX_DIR}/profile/evaluate_module_parsing.py"
PROFILE_REF_CSV=""  # auto-generated from good commit

# ── Usage ────────────────────────────────────────────────────────────────────
usage() {
    cat <<'EOF'
Usage: bisect.sh --good <sha> --bad <sha> --mode <launch|perf|accuracy|profile> --lib <sglang|aiter> --server-script <path> [OPTIONS]

Required:
  --good <commit>              Known good commit SHA
  --bad  <commit>              Known bad commit SHA
  --mode <launch|perf|accuracy|profile> What to test
  --lib  <sglang|aiter>        Which library to rebuild after checkout
  --server-script <path>       Script to launch the server

Options:
  --repo-dir <path>            Repository root (default: /sgl-workspace/sglang or /sgl-workspace/aiter)
  --client-script <path>       perf/profile mode: benchmark script (perf: exit 0=GOOD, 1=BAD)
  --accuracy-script <path>     accuracy mode: script whose stdout has "Accuracy: ..."
  --accuracy-baseline <float>  accuracy mode: expected accuracy %
  --accuracy-threshold <float> accuracy mode: allowed drop % (default: 2.0)
  --build-script <path>        Override the default rebuild step
  --port <int>                 Server port (default: 30000)
  --model-path <path>          Model path, exported to server/client scripts
  --timeout <seconds>          Health check timeout (default: 600)
  --export-layers <range>      (deprecated, ignored)

Modes:
  launch    - Test if the server starts successfully
  perf      - Run benchmark script, exit code determines GOOD/BAD
  accuracy  - Parse accuracy from script output, compare against baseline
  profile   - Run profiling, analyze traces, compare S1-S4 structural scores
              against reference scores from the good commit. Any regression = BAD.
EOF
    exit 1
}

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --good)              GOOD="$2";               shift 2 ;;
        --bad)               BAD="$2";                shift 2 ;;
        --mode)              MODE="$2";               shift 2 ;;
        --lib)               LIB="$2";                shift 2 ;;
        --repo-dir)          REPO_DIR="$2";           shift 2 ;;
        --server-script)     SERVER_SCRIPT="$2";      shift 2 ;;
        --client-script)     CLIENT_SCRIPT="$2";      shift 2 ;;
        --accuracy-script)   ACCURACY_SCRIPT="$2";    shift 2 ;;
        --accuracy-baseline) ACCURACY_BASELINE="$2";  shift 2 ;;
        --accuracy-threshold) ACCURACY_THRESHOLD="$2"; shift 2 ;;
        --build-script)      BUILD_SCRIPT="$2";       shift 2 ;;
        --port)              PORT="$2";               shift 2 ;;
        --model-path)        MODEL_PATH="$2";         shift 2 ;;
        --timeout)           TIMEOUT="$2";            shift 2 ;;
        --export-layers)     EXPORT_LAYERS="$2";      shift 2 ;;
        -h|--help)           usage ;;
        *)                   echo "Unknown option: $1"; usage ;;
    esac
done

# ── Validation ───────────────────────────────────────────────────────────────
fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -n "$GOOD" ]] || fail "--good is required"
[[ -n "$BAD"  ]] || fail "--bad is required"
[[ -n "$MODE" ]] || fail "--mode is required"
[[ -n "$LIB"  ]] || fail "--lib is required"

[[ "$MODE" =~ ^(launch|perf|accuracy|profile)$ ]] || fail "--mode must be launch, perf, accuracy, or profile"
[[ "$LIB"  =~ ^(sglang|aiter)$ ]]         || fail "--lib must be sglang or aiter"

[[ -n "$SERVER_SCRIPT" ]] || fail "--server-script is required"
[[ -f "$SERVER_SCRIPT" ]] || fail "--server-script not found: $SERVER_SCRIPT"

if [[ "$MODE" == "perf" ]]; then
    [[ -n "$CLIENT_SCRIPT" ]] || fail "--client-script is required for perf mode"
    [[ -f "$CLIENT_SCRIPT" ]] || fail "--client-script not found: $CLIENT_SCRIPT"
fi

if [[ "$MODE" == "profile" ]]; then
    [[ -n "$CLIENT_SCRIPT" ]] || fail "--client-script is required for profile mode"
    [[ -f "$CLIENT_SCRIPT" ]] || fail "--client-script not found: $CLIENT_SCRIPT"
    [[ -f "$TRACE_ANALYZER_SCRIPT" ]] || fail "trace_module_analyzer.py not found: $TRACE_ANALYZER_SCRIPT"
    [[ -f "$EVALUATE_SCRIPT" ]] || fail "evaluate_module_parsing.py not found: $EVALUATE_SCRIPT"
fi

if [[ "$MODE" == "accuracy" ]]; then
    [[ -n "$ACCURACY_BASELINE" ]] || fail "--accuracy-baseline is required for accuracy mode"
fi

if [[ -n "$ACCURACY_SCRIPT" && ! -f "$ACCURACY_SCRIPT" ]]; then
    fail "--accuracy-script not found: $ACCURACY_SCRIPT"
fi
if [[ -n "$BUILD_SCRIPT" && ! -f "$BUILD_SCRIPT" ]]; then
    fail "--build-script not found: $BUILD_SCRIPT"
fi

# Default repo dir based on --lib
if [[ -z "$REPO_DIR" ]]; then
    if [[ "$LIB" == "sglang" ]]; then
        REPO_DIR="/sgl-workspace/sglang"
    else
        REPO_DIR="/sgl-workspace/aiter"
    fi
fi
if [[ "$LIB" != "aiter" ]]; then
    [[ -d "$REPO_DIR" ]] || fail "repo-dir not found: $REPO_DIR"
fi

# Auto-detect port from server script if user didn't override --port
if [[ "$PORT" == "30000" ]]; then
    detected_port=$(grep -oP -- '--port\s+\K[0-9]+' "$SERVER_SCRIPT" | head -1 || true)
    if [[ -n "$detected_port" ]]; then
        PORT="$detected_port"
    fi
fi

# Export for child scripts
export PORT MODEL_PATH

# ── Logging helpers ──────────────────────────────────────────────────────────
log()  { echo "  $*"; }
step() { echo -n "$*"; }

# ── Rebuild library ──────────────────────────────────────────────────────────
rebuild() {
    if [[ -n "$BUILD_SCRIPT" ]]; then
        bash "$BUILD_SCRIPT"
        return $?
    fi
    case "$LIB" in
        sglang)
            (cd "$REPO_DIR/python" && pip install -e . --no-deps -q)
            ;;
        aiter)
            (cd "$REPO_DIR" \
                && git submodule update --init --recursive \
                && GPU_ARCHS="gfx950" python setup.py develop)
            ;;
    esac
}

# ── Aiter clean source management ────────────────────────────────────────
AITER_CLEAN_DIR=""

prepare_aiter_clean_source() {
    AITER_CLEAN_DIR="$(cd "$AGENT_BOX_DIR/.." && pwd)/aiter"

    if [[ -d "$AITER_CLEAN_DIR/.git" ]]; then
        log "Found existing aiter source at $AITER_CLEAN_DIR"
        local so_count
        so_count=$(find "$AITER_CLEAN_DIR" -name "*.so" -not -path "*/.git/*" 2>/dev/null | wc -l)
        if [[ "$so_count" -gt 0 ]]; then
            log "WARNING: Found $so_count compiled .so files, cleaning..."
            (cd "$AITER_CLEAN_DIR" && git clean -fdx -q)
        fi
        (cd "$AITER_CLEAN_DIR" && git fetch --all -q --no-recurse-submodules)
    else
        log "No clean aiter source found at $AITER_CLEAN_DIR, cloning..."
        local remote_url=""
        if [[ -d "$REPO_DIR/.git" ]]; then
            remote_url=$(cd "$REPO_DIR" && git remote get-url origin 2>/dev/null || true)
        fi
        [[ -n "$remote_url" ]] || fail "Cannot determine aiter remote URL. Provide a clean clone at $AITER_CLEAN_DIR or ensure $REPO_DIR exists with a valid remote."
        git clone "$remote_url" "$AITER_CLEAN_DIR"
    fi
}

copy_clean_aiter() {
    log "replacing $REPO_DIR with clean aiter source..."
    rm -rf "$REPO_DIR"
    cp -a "$AITER_CLEAN_DIR" "$REPO_DIR"
    git config --global --add safe.directory "$REPO_DIR" 2>/dev/null || true
    git config --global --add safe.directory '*' 2>/dev/null || true
}

# ── Launch server ────────────────────────────────────────────────────────────
SERVER_PID=""
SERVER_LOG=""
BISECT_LOG_DIR="/tmp/bisect_logs"
mkdir -p "$BISECT_LOG_DIR"

launch_server() {
    local short_sha="$1"
    SERVER_LOG="${BISECT_LOG_DIR}/server_${short_sha}.log"
    setsid bash "$SERVER_SCRIPT" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
}

# ── Fatal error detection ────────────────────────────────────────────────────
# Check the server log for patterns that indicate an unrecoverable crash.
# Returns 0 if a fatal pattern is found, 1 otherwise.
FATAL_REASON=""
check_server_log_fatal() {
    [[ -f "$SERVER_LOG" ]] || return 1
    local match=""
    match=$(grep -m1 -oP \
        'Memory access fault|Fatal Python error|Segmentation fault|SIGSEGV|SIGABRT|SIGBUS|RuntimeError.*CUDA|OutOfMemoryError|GPU core dump|unhandled cuda error|HIP error|ROCM_ERROR' \
        "$SERVER_LOG" 2>/dev/null || true)
    if [[ -n "$match" ]]; then
        FATAL_REASON="$match"
        return 0
    fi
    return 1
}

# ── Health check ─────────────────────────────────────────────────────────────
wait_for_server() {
    local deadline=$(( $(date +%s) + TIMEOUT ))
    local elapsed=0
    while [[ $(date +%s) -lt $deadline ]]; do
        # Check if the server process is still alive
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            check_server_log_fatal && log "*** Fatal error in log: $FATAL_REASON"
            return 1
        fi
        if curl -sf "http://localhost:${PORT}/health_generate" > /dev/null 2>&1; then
            return 0
        fi
        # Early exit: scan log for fatal errors after each poll
        if check_server_log_fatal; then
            echo ""
            log "*** Fatal error detected in server log: $FATAL_REASON"
            log "*** (not waiting for full timeout)"
            return 1
        fi
        sleep 5
        elapsed=$(( elapsed + 5 ))
        # Print a dot every 30s to show progress
        if (( elapsed % 30 == 0 )); then
            step "."
        fi
    done
    return 1
}

# ── Kill server ──────────────────────────────────────────────────────────────
kill_server() {
    if [[ -n "$SERVER_PID" ]]; then
        # Kill the entire process group spawned by the server script
        kill -- -"$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
    # Clean up any leftover sglang server processes on our port
    pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
    sleep 2
}

# ── Parse accuracy from stdout ───────────────────────────────────────────────
# Accepts:
#   Accuracy: 1850/2000 = 92.50%
#   Accuracy: 0.925
parse_accuracy() {
    local output="$1"
    local acc=""

    # Try "Accuracy: N/M = XX.XX%"
    acc=$(echo "$output" | grep -oP 'Accuracy:\s*\d+/\d+\s*=\s*\K[0-9]+(\.[0-9]+)' | tail -1)
    if [[ -n "$acc" ]]; then
        echo "$acc"
        return
    fi

    # Try "Accuracy: 0.XXXX" (decimal fraction)
    acc=$(echo "$output" | grep -oP 'Accuracy:\s*\K0\.[0-9]+' | tail -1)
    if [[ -n "$acc" ]]; then
        echo "$acc" | awk '{printf "%.2f", $1 * 100}'
        return
    fi

    # Try "Accuracy: XX.XX" (already a percentage)
    acc=$(echo "$output" | grep -oP 'Accuracy:\s*\K[0-9]+\.[0-9]+' | tail -1)
    if [[ -n "$acc" ]]; then
        echo "$acc"
        return
    fi

    echo ""
}

# ── Profile evaluation helper ───────────────────────────────────────────────
PROFILE_SCORES=""  # "S1:100.0 S2:60.0 S3:100.0 S4:100.0"
REF_SCORES=""

run_profile_eval() {
    local short_sha="$1"
    local analysis_dir="${BISECT_LOG_DIR}/profile_${short_sha}"
    mkdir -p "$analysis_dir"
    PROFILE_SCORES=""

    # Clean any old traces
    find /tmp -maxdepth 2 -name '*-TP-0.trace.json.gz' -delete 2>/dev/null || true

    # Run profiling benchmark
    if ! bash "$CLIENT_SCRIPT" > "${analysis_dir}/client.log" 2>&1; then
        return 1
    fi

    # Find trace
    local trace_file
    trace_file=$(find /tmp -maxdepth 2 -name '*-TP-0.trace.json.gz' -type f 2>/dev/null | sort | tail -1)
    [[ -n "$trace_file" ]] || return 2

    local trace_dir
    trace_dir=$(dirname "$trace_file")

    # Run trace_module_analyzer
    python3 "$TRACE_ANALYZER_SCRIPT" "$trace_file" \
        -o "${analysis_dir}/analysis.xlsx" \
        > "${analysis_dir}/trace_analyzer.log" 2>&1 || return 3

    # Run evaluate_module_parsing on the generated xlsx
    local excel_path="${analysis_dir}/analysis.xlsx"
    local eval_json
    eval_json=$(python3 "$EVALUATE_SCRIPT" "$excel_path" --json 2>/dev/null) || return 4
    echo "$eval_json" > "${analysis_dir}/evaluation.json"

    # Extract S1-S4 scores from JSON (evaluate_module_parsing may print extra lines after JSON)
    PROFILE_SCORES=$(echo "$eval_json" | python3 -c "
import sys, json, json.decoder
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except json.decoder.JSONDecodeError:
    dec = json.JSONDecoder()
    d, _ = dec.raw_decode(raw)
sr = d['structural_rules']
print(f\"S1:{sr['s1_phase_coverage']['score']} S2:{sr['s2_architecture_sig']['score']} S3:{sr['s3_instance_consistency']['score']} S4:{sr['s4_time_distribution']['score']}\")
" 2>/dev/null) || return 5

    # Cleanup trace file to save disk
    rm -f "$trace_file"
    return 0
}

# ── Evaluate a single commit ────────────────────────────────────────────────
# Returns via global VERDICT: "GOOD", "BAD", or "SKIP"
VERDICT=""
VERDICT_DETAIL=""

evaluate_commit() {
    local sha="$1"
    local short_sha="$2"

    # For aiter: start from clean uncompiled source each time
    if [[ "$LIB" == "aiter" ]]; then
        copy_clean_aiter
    fi

    # Checkout
    log "checking out..."
    (cd "$REPO_DIR" && git checkout "$sha" --quiet >/dev/null 2>&1)

    # Rebuild
    log "rebuilding..."
    if ! rebuild > /tmp/bisect_build.log 2>&1; then
        VERDICT="SKIP"
        VERDICT_DETAIL="build failed (see /tmp/bisect_build.log)"
        return
    fi

    # Launch server
    launch_server "$short_sha"
    log "launching server... (log: $SERVER_LOG)"
    step "  waiting for health check"
    if ! wait_for_server; then
        echo ""
        kill_server
        if [[ "$MODE" == "launch" ]]; then
            VERDICT="BAD"
            VERDICT_DETAIL="server failed to start within ${TIMEOUT}s (log: $SERVER_LOG)"
        else
            VERDICT="SKIP"
            VERDICT_DETAIL="server failed to start (log: $SERVER_LOG)"
        fi
        return
    fi
    echo ""

    # Mode-specific evaluation
    case "$MODE" in
        launch)
            VERDICT="GOOD"
            VERDICT_DETAIL="server started successfully"
            ;;
        perf)
            log "running benchmark..."
            if bash "$CLIENT_SCRIPT" > /tmp/bisect_client.log 2>&1; then
                VERDICT="GOOD"
                VERDICT_DETAIL="client exited 0"
            else
                VERDICT="BAD"
                VERDICT_DETAIL="client exited $?"
            fi
            ;;
        accuracy)
            echo ""
            log "running accuracy test..."
            local acc_output
            if [[ -n "$ACCURACY_SCRIPT" ]]; then
                acc_output=$(bash "$ACCURACY_SCRIPT" 2>/dev/null) || true
            else
                acc_output=$(python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py \
                    --num-questions 2000 --parallel 1000 --num-shots 5 --port "$PORT" 2>/dev/null) || true
            fi

            local acc
            acc=$(parse_accuracy "$acc_output")
            if [[ -z "$acc" ]]; then
                VERDICT="SKIP"
                VERDICT_DETAIL="could not parse accuracy from output"
            else
                local min_acc
                min_acc=$(awk "BEGIN {printf \"%.4f\", $ACCURACY_BASELINE - $ACCURACY_THRESHOLD}")
                if awk "BEGIN {exit !($acc >= $min_acc)}"; then
                    VERDICT="GOOD"
                    VERDICT_DETAIL="accuracy=${acc}% (>= ${min_acc}%)"
                else
                    VERDICT="BAD"
                    VERDICT_DETAIL="accuracy=${acc}% (< ${min_acc}%)"
                fi
            fi
            ;;
        profile)
            log "running profiling benchmark..."
            if ! run_profile_eval "$short_sha"; then
                VERDICT="SKIP"
                VERDICT_DETAIL="profile evaluation failed (step returned $?)"
            else
                # Compare each S1-S4 score against reference
                local regressed=""
                for rule in S1 S2 S3 S4; do
                    ref_val=$(echo "$REF_SCORES" | grep -oP "${rule}:\K[0-9.]+")
                    cur_val=$(echo "$PROFILE_SCORES" | grep -oP "${rule}:\K[0-9.]+")
                    if awk "BEGIN {exit !($cur_val < $ref_val)}"; then
                        regressed="${regressed} ${rule}:${cur_val}<${ref_val}"
                    fi
                done

                if [[ -z "$regressed" ]]; then
                    VERDICT="GOOD"
                    VERDICT_DETAIL="scores=$PROFILE_SCORES (no regression)"
                else
                    VERDICT="BAD"
                    VERDICT_DETAIL="regression:${regressed} | current=$PROFILE_SCORES"
                fi
            fi
            ;;
    esac

    kill_server
}

# ── Main ─────────────────────────────────────────────────────────────────────

# Aiter: prepare clean source and do initial copy before any git operations
if [[ "$LIB" == "aiter" ]]; then
    echo "=== Preparing aiter clean source ==="
    prepare_aiter_clean_source
    copy_clean_aiter
    echo ""
fi

# Save original HEAD so we can restore it at the end
ORIG_HEAD=$(cd "$REPO_DIR" && git rev-parse HEAD)

cleanup() {
    kill_server
    echo ""
    if [[ "$LIB" == "aiter" ]]; then
        echo "Restoring clean aiter source..."
        copy_clean_aiter 2>/dev/null || true
    else
        echo "Restoring original HEAD..."
        (cd "$REPO_DIR" && git checkout "$ORIG_HEAD" --quiet >/dev/null 2>&1) || true
    fi
}
trap cleanup EXIT

# Get the list of commits between good and bad (exclusive of good, inclusive of bad)
mapfile -t COMMITS < <(cd "$REPO_DIR" && git rev-list --reverse "$GOOD".."$BAD")
NUM_COMMITS=${#COMMITS[@]}

if [[ "$NUM_COMMITS" -eq 0 ]]; then
    fail "No commits found between $GOOD and $BAD. Check that --good is an ancestor of --bad."
fi

STEPS=$(awk "BEGIN {printf \"%d\", log($NUM_COMMITS)/log(2) + 1}")

# Print header
GOOD_SHORT=$(cd "$REPO_DIR" && git rev-parse --short "$GOOD")
BAD_SHORT=$(cd "$REPO_DIR" && git rev-parse --short "$BAD")

echo ""
echo "=== Git Bisect: $MODE mode ==="
echo "  Library: $LIB"
echo "  Good: $GOOD_SHORT  Bad: $BAD_SHORT"
echo "  Commits to test: $NUM_COMMITS (~$STEPS steps)"
[[ -n "$MODEL_PATH" ]] && echo "  Model: $MODEL_PATH"
echo "  Port: $PORT"
echo "  Server logs: $BISECT_LOG_DIR/"
[[ "$LIB" == "aiter" && -n "$AITER_CLEAN_DIR" ]] && echo "  Clean source: $AITER_CLEAN_DIR"
echo ""

# Generate reference profile from good commit (profile mode only)
if [[ "$MODE" == "profile" ]]; then
    echo "Generating reference profile from good commit ($GOOD_SHORT)..."
    if [[ "$LIB" == "aiter" ]]; then
        copy_clean_aiter
    fi
    (cd "$REPO_DIR" && git checkout "$GOOD" --quiet)
    rebuild > /tmp/bisect_build.log 2>&1 || fail "Cannot build good commit"
    launch_server "$GOOD_SHORT"
    step "  waiting for health check"
    if ! wait_for_server; then
        echo ""
        kill_server
        fail "Good commit server failed to start"
    fi
    echo ""

    run_profile_eval "$GOOD_SHORT" || fail "Profile evaluation failed on good commit"
    REF_SCORES="$PROFILE_SCORES"
    echo "  Reference scores: $REF_SCORES"
    kill_server
    echo ""
fi

# ── Pre-bisect: verify good and bad commits ──────────────────────────────
if [[ "$MODE" != "profile" ]]; then
    echo "--- Verifying good commit ($GOOD_SHORT) ---"
    evaluate_commit "$GOOD" "$GOOD_SHORT"
    if [[ "$VERDICT" != "GOOD" ]]; then
        fail "Good commit $GOOD_SHORT did not pass as GOOD ($VERDICT: $VERDICT_DETAIL). Fix --good."
    fi
    log "Good commit verified: $VERDICT_DETAIL"
    echo ""
fi

echo "--- Verifying bad commit ($BAD_SHORT) ---"
evaluate_commit "$BAD" "$BAD_SHORT"
if [[ "$VERDICT" != "BAD" ]]; then
    fail "Bad commit $BAD_SHORT did not fail as BAD ($VERDICT: $VERDICT_DETAIL). Fix --bad."
fi
log "Bad commit verified: $VERDICT_DETAIL"
echo ""

# Binary search
lo=0
hi=$(( NUM_COMMITS - 1 ))
current_step=0
total_steps=$STEPS
first_bad_idx=""

while [[ $lo -le $hi ]]; do
    mid=$(( (lo + hi) / 2 ))
    sha="${COMMITS[$mid]}"
    short_sha=$(cd "$REPO_DIR" && git rev-parse --short "$sha")
    subject=$(cd "$REPO_DIR" && git log -1 --format='%s' "$sha")
    current_step=$(( current_step + 1 ))

    echo "[Step ${current_step}/${total_steps}] $short_sha  $subject"

    evaluate_commit "$sha" "$short_sha"

    echo ""
    case "$VERDICT" in
        GOOD)
            log "-> GOOD ($VERDICT_DETAIL)"
            lo=$(( mid + 1 ))
            ;;
        BAD)
            log "-> BAD ($VERDICT_DETAIL)"
            first_bad_idx=$mid
            hi=$(( mid - 1 ))
            ;;
        SKIP)
            # On skip, try to continue but mark as uncertain.
            # Move toward the bad end to be safe.
            log "-> SKIP ($VERDICT_DETAIL) — moving toward bad end"
            lo=$(( mid + 1 ))
            ;;
    esac
done

# ── Result ───────────────────────────────────────────────────────────────────
echo ""
echo "=== RESULT ==="
if [[ -n "$first_bad_idx" ]]; then
    bad_sha="${COMMITS[$first_bad_idx]}"
    bad_short=$(cd "$REPO_DIR" && git rev-parse --short "$bad_sha")
    bad_subject=$(cd "$REPO_DIR" && git log -1 --format='%s' "$bad_sha")
    echo "  First bad commit: $bad_short"
    echo "  Subject: $bad_subject"
    echo "  Full SHA: $bad_sha"
else
    echo "  Could not identify the first bad commit."
    echo "  This may happen if some commits were skipped."
fi
echo "  Total steps: $current_step"
echo ""
