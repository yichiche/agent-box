#!/usr/bin/env bash
# Daily performance regression benchmark runner.
#
# Usage:
#   crontab: 0 2 * * * /home/yichiche/agent-box/debug/perf-regression/run_daily.sh
#   Manual:  nohup bash /home/yichiche/agent-box/debug/perf-regression/run_daily.sh &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/daily_${TIMESTAMP}.log"

echo "[$(date)] Starting daily benchmark run" | tee -a "$LOG_FILE"

cd "$SCRIPT_DIR"
python3 "${SCRIPT_DIR}/orchestrator.py" --lookback-days 3 2>&1 | tee -a "$LOG_FILE"

echo "[$(date)] Daily benchmark run completed" | tee -a "$LOG_FILE"
