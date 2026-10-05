#!/usr/bin/env bash
# Start / stop the pr-ci-watch dashboard and print how to reach it.
#
# Loopback only — reach it through your editor's port forwarding or an SSH
# tunnel, never by binding the port to the LAN.
#
#   serve_dashboard.sh [--port N]   start (idempotent) and print URLs
#   serve_dashboard.sh --stop       stop it
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PR_CI_WATCH_PORT:-8812}"
LOG="/tmp/pr-ci-watch-dashboard.log"
STOP=0

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --stop) STOP=1; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ "$STOP" = 1 ]; then
  pkill -f "dashboard.py $PORT" && echo "stopped dashboard on :$PORT" \
    || echo "nothing running on :$PORT"
  exit 0
fi

if ! curl -sf -o /dev/null -m 3 "http://127.0.0.1:$PORT/"; then
  # </dev/null matters: without it the background child holds this script's
  # stdin/stdout open and the script never returns when its output is piped.
  (setsid nohup python3 "$HERE/dashboard.py" "$PORT" </dev/null >"$LOG" 2>&1 &)
  for _ in $(seq 15); do
    curl -sf -o /dev/null -m 2 "http://127.0.0.1:$PORT/" && break
    sleep 1
  done
fi
curl -sf -o /dev/null -m 3 "http://127.0.0.1:$PORT/" \
  || { echo "dashboard did not start; see $LOG" >&2; exit 1; }

# Print a literal username, never $USER: the tunnel is usually run from
# PowerShell on Windows, which does not expand $USER and silently turns the
# destination into "@host" -> ssh prints its usage text.
# Parent of the agent-box checkout is the host username. $HOME is /root
# inside a container, so walk up to the agent-box directory instead.
dir="$HERE"
while [ "$dir" != "/" ] && [ "$(basename "$dir")" != "agent-box" ]; do
  dir="$(dirname "$dir")"
done
SSH_USER="${PR_CI_WATCH_SSH_USER:-$(basename "$(dirname "$dir")")}"
HOST_NAME="${PR_CI_WATCH_SSH_HOST:-$(hostname -f 2>/dev/null || hostname)}"
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
# Forward to a high local port. Windows reserves dynamic TCP ranges for
# Hyper-V/WSL/Docker, and a low port inside one fails with
# "bind [127.0.0.1]:<port>: Permission denied" even when nothing is listening.
LOCAL_PORT="${PR_CI_WATCH_LOCAL_PORT:-1$PORT}"

echo "Dashboard up on 127.0.0.1:$PORT ($HOST_NAME) — log: $LOG"
echo
echo "If your editor is connected to this host over Remote-SSH (Cursor/VS Code),"
echo "do NOT open an ssh tunnel yourself - it forwards ports for you:"
echo "  PORTS panel -> Forward a Port -> $PORT -> Open in Browser"
echo "  or Ctrl+Shift+P -> 'Simple Browser: Show' -> the URL below"
echo
echo "Otherwise, from a plain terminal on your laptop (keep the session open):"
echo "  ssh -L $LOCAL_PORT:localhost:$PORT $SSH_USER@$HOST_NAME"
[ -n "$HOST_IP" ] && echo "  # if the hostname does not resolve:"
[ -n "$HOST_IP" ] && echo "  ssh -L $LOCAL_PORT:localhost:$PORT $SSH_USER@$HOST_IP"
echo
echo "URLs:"
echo "  http://localhost:$PORT/         # editor-forwarded"
echo "  http://localhost:$LOCAL_PORT/   # manual ssh -L"
