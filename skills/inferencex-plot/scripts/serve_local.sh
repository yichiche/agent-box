#!/usr/bin/env bash
# Serve the InferenceX Curve app locally with a CSV already loaded on the chart.
#
# Why local: the published GitHub Pages app cannot load a CSV without a manual
# download → Import File round trip, and its InferenceX Sync is blocked by CORS
# (the /api/v1/* routes send no Access-Control-Allow-Origin). Running the real
# app under Vite fixes both: ?seed=<file> imports on load, and the dev proxy
# makes Sync work.
#
# Usage: serve_local.sh [csv ...]     (defaults to every CSV in the plots dir)
set -euo pipefail

APP="${INFERENCEX_CURVE_DIR:-/home/yichiche/InferenceXCurve}"
PLOTS="${INFERENCEX_PLOTS_DIR:-/home/yichiche/inferencex-plots}"
PORT="${INFERENCEX_CURVE_PORT:-5173}"

[ -d "$APP/node_modules" ] || { echo "run 'npm install' in $APP first" >&2; exit 1; }

mkdir -p "$APP/public"
files=("$@")
[ ${#files[@]} -eq 0 ] && files=("$PLOTS"/*.csv)
for f in "${files[@]}"; do
  [ -f "$f" ] || { echo "no such CSV: $f" >&2; exit 1; }
  cp "$f" "$APP/public/"
done

# Bind to loopback only; reach it with an SSH tunnel rather than exposing the
# port to the network.
if ! curl -sf -o /dev/null -m 3 "http://127.0.0.1:$PORT/InferenceXCurve/"; then
  (cd "$APP" && nohup npx vite --host 127.0.0.1 --port "$PORT" --strictPort \
      > /tmp/inferencex-curve.log 2>&1 &)
  for _ in $(seq 30); do
    curl -sf -o /dev/null -m 2 "http://127.0.0.1:$PORT/InferenceXCurve/" && break
    sleep 1
  done
fi
curl -sf -o /dev/null -m 3 "http://127.0.0.1:$PORT/InferenceXCurve/" \
  || { echo "server did not start; see /tmp/inferencex-curve.log" >&2; exit 1; }

# Print a literal username, never $USER: the tunnel is usually run from
# PowerShell on Windows, which does not expand $USER and silently turns the
# destination into "@host" -> ssh prints its usage text.
SSH_USER="${INFERENCEX_SSH_USER:-$(basename "$(dirname "$PLOTS")")}"
# Prefer the FQDN: the short name often does not resolve off-cluster.
HOST_NAME="${INFERENCEX_SSH_HOST:-$(hostname -f 2>/dev/null || hostname)}"
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
# Forward to a high local port. Windows reserves dynamic TCP ranges for
# Hyper-V/WSL/Docker, and a low port inside one fails with
# "bind [127.0.0.1]:<port>: Permission denied" even when nothing is listening.
LOCAL_PORT="${INFERENCEX_LOCAL_PORT:-1$PORT}"

echo "Server up on 127.0.0.1:$PORT ($HOST_NAME)"
echo
echo "If your editor is connected to this host over Remote-SSH (Cursor/VS Code),"
echo "do NOT open an ssh tunnel yourself - it forwards ports for you:"
echo "  PORTS panel -> Forward a Port -> $PORT -> Open in Browser"
echo "  or Ctrl+Shift+P -> 'Simple Browser: Show' -> the URL below"
echo "(This container shares the host network namespace, so 127.0.0.1:$PORT is"
echo " the host's loopback and the remote server can see it.)"
echo
echo "Otherwise, from a plain terminal on your laptop (keep the session open):"
echo "  ssh -L $LOCAL_PORT:localhost:$PORT $SSH_USER@$HOST_NAME"
[ -n "$HOST_IP" ] && echo "  # if the hostname does not resolve:"
[ -n "$HOST_IP" ] && echo "  ssh -L $LOCAL_PORT:localhost:$PORT $SSH_USER@$HOST_IP"
echo
echo
echo "URLs (use the editor-forwarded port if you took that route):"
for f in "${files[@]}"; do
  echo "  http://localhost:$PORT/InferenceXCurve/?seed=$(basename "$f")   # editor-forwarded"
  echo "  http://localhost:$LOCAL_PORT/InferenceXCurve/?seed=$(basename "$f")   # manual ssh -L"
done
