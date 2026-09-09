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

echo "Server up on 127.0.0.1:$PORT ($(hostname))"
echo
echo "On your laptop:"
echo "  ssh -L $PORT:localhost:$PORT \$USER@$(hostname)"
echo
echo "Then open:"
for f in "${files[@]}"; do
  echo "  http://localhost:$PORT/InferenceXCurve/?seed=$(basename "$f")"
done
