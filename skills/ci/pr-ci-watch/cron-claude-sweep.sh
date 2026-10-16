#!/usr/bin/env bash
# Host-side Phase A+B sweep for pr-ci-watch, via headless Claude.
#
#   cron-claude-sweep.sh [high|regular|all]
#
# `cron-sweep.sh` is the pure-Python Phase A: it re-runs CI and posts conflict
# notices, but it cannot run `/ci-analysis`, which is the one thing a failure
# #17050 has never seen actually needs. That gap is why PRs used to sit in
# `awaiting-triage` indefinitely — nothing unattended could read a log.
#
# This runs the same sweep inside a Claude turn, so triage happens unattended
# too. Unlike a Claude `CronCreate` job it has **no 7-day expiry** and does not
# need a REPL to be idle, which is the whole reason it exists.
#
# Keep `cron-sweep.sh` installed alongside it. If Claude auth breaks, the
# pure-Python sweep still re-runs CI; losing triage is much better than losing
# re-runs as well.
set -uo pipefail

TRACK="${1:-high}"
case "$TRACK" in
  high|regular|all) ;;
  *) echo "usage: cron-claude-sweep.sh [high|regular|all]" >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Parent of the agent-box checkout is the host user home. $HOME is /root
# inside a container, so walk up to the agent-box directory instead.
dir="$HERE"
while [ "$dir" != "/" ] && [ "$(basename "$dir")" != "agent-box" ]; do
  dir="$(dirname "$dir")"
done
AGENT_BOX="$dir"
HOST_HOME="$(dirname "$dir")"
LOG_DIR="${PR_CI_WATCH_DIR:-${AGENT_SCRATCH_DIR:-$HOST_HOME/agent-scratch}/pr-ci-watch}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/cron.log"
LOCK="$LOG_DIR/cron-claude-${TRACK}.lock"

# cron hands a bare PATH (/usr/bin:/bin). `claude` and `gh` both live in the
# user's own dirs, and without this every run dies on "command not found" —
# silently, because the message goes to cron.log and the dashboard shows the
# Python sweep's results, which were fine.
export PATH="$HOST_HOME/.local/bin:$HOST_HOME/bin:/usr/local/bin:$PATH"
# Claude resolves skills and project settings from the working directory.
export CLAUDE_PROJECT_DIR="$AGENT_BOX"

# The instruction, not just the command: the sweep prints a TRIAGE REQUIRED
# list, and the whole point of running it under Claude is that the same turn
# clears that list instead of leaving it for a human to click.
read -r -d '' PROMPT <<EOF
Run: /pr-ci-watch sweep --track $TRACK --apply

Then, for every PR the sweep lists under "TRIAGE REQUIRED", run /ci-analysis on
it and immediately record the verdict with \`watch.py apply-verdict ... --apply\`.
Never leave a PR sitting in awaiting-triage — if a decision needs a log read,
do the log read now and act on it.

Two rules that override anything else:
  * Do NOT run update-branch. \`merge-main\` is recorded only; applying it is a
    manual button the user presses.
  * \`wait-upstream\` requires a #17050 row saying main is broken, or a failure
    that already repeated after its re-runs. Otherwise record \`re-run\`.

Finish by printing one line per PR you acted on.
EOF

{
  echo "===== $(date -Is) claude track=$TRACK host=$(hostname) taipei=$(TZ=Asia/Taipei date +%H:%M) ====="
  if ! flock -n 9; then
    echo "skipped: another claude $TRACK sweep is still running"
    exit 0
  fi
  if ! command -v claude >/dev/null; then
    echo "abort: claude not on PATH ($PATH)"
    echo "       find it on the host with: command -v claude"
    echo "       then set CLAUDE_BIN in this script or add its dir to PATH above"
    exit 127
  fi
  if ! command -v gh >/dev/null; then
    echo "abort: gh not on PATH ($PATH)"
    exit 127
  fi
  cd "$AGENT_BOX" || exit 1
  # 25m ceiling: a sweep plus a few /ci-analysis passes, with room to spare.
  # Without it a wedged turn holds the lock until the next reboot.
  timeout 1500 claude -p "$PROMPT"
  rc=$?
  [ "$rc" = 124 ] && echo "TIMED OUT after 25m — killed"
  echo "===== exit $rc ====="
  exit "$rc"
} >>"$LOG" 2>&1 9>"$LOCK"
