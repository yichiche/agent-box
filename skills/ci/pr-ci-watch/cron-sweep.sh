#!/usr/bin/env bash
# Host-side Phase A sweep for pr-ci-watch. No Claude turn.
#
#   cron-sweep.sh [high|regular|all]
#
# Posts conflict notices and updates dashboard state. Red NVIDIA CI that still
# needs a log read stays queued for /pr-ci-watch triage.
#
# Set PR_CI_WATCH_ONLY_HOUR=07 to no-op unless the current hour in
# Asia/Taipei is 07. The host cron daemon has no per-user timezone, so the
# installed crontab fires every hour at :00 and this gate keeps the real
# sweep at 07:00 Taipei through US Central DST changes.
set -uo pipefail

TRACK="${1:-regular}"
case "$TRACK" in
  high|regular|all) ;;
  *) echo "usage: cron-sweep.sh [high|regular|all]" >&2; exit 2 ;;
esac

if [ -n "${PR_CI_WATCH_ONLY_HOUR:-}" ]; then
  hour="$(TZ=Asia/Taipei date +%H)"
  [ "$hour" = "$PR_CI_WATCH_ONLY_HOUR" ] || exit 0
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Parent of the agent-box checkout is the host user home. $HOME is /root
# inside a container, so walk up to the agent-box directory instead.
dir="$HERE"
while [ "$dir" != "/" ] && [ "$(basename "$dir")" != "agent-box" ]; do
  dir="$(dirname "$dir")"
done
HOST_HOME="$(dirname "$dir")"
LOG_DIR="${PR_CI_WATCH_DIR:-${AGENT_SCRATCH_DIR:-$HOST_HOME/agent-scratch}/pr-ci-watch}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/cron.log"
LOCK="$LOG_DIR/cron-${TRACK}.lock"

# cron hands us a bare PATH (/usr/bin:/bin), and `gh` lives in the user's own
# bin dir — without this every sweep dies on FileNotFoundError: 'gh'.
export PATH="$HOST_HOME/bin:/usr/local/bin:$PATH"

{
  echo "===== $(date -Is) track=$TRACK host=$(hostname) taipei=$(TZ=Asia/Taipei date +%H:%M) ====="
  if ! flock -n 9; then
    echo "skipped: another $TRACK sweep is still running"
    exit 0
  fi
  if ! command -v gh >/dev/null; then
    echo "abort: gh not on PATH ($PATH) — install it or fix PATH in cron-sweep.sh"
    exit 127
  fi
  python3 "$HERE/watch.py" sweep --track "$TRACK" --apply
  rc=$?
  echo "===== exit $rc ====="
  exit "$rc"
} >>"$LOG" 2>&1 9>"$LOCK"
