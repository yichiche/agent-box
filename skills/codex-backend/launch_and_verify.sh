#!/usr/bin/env bash
# launch_and_verify.sh — (re)launch an SGLang server from a launch script, wait until
# ready, then probe whether it parses tool calls (required for Codex agentic use).
#
# Usage:
#   bash launch_and_verify.sh <launch_script.sh> [--port N] [--no-restart] [--timeout SECS]
#
# Prints a machine-readable summary block at the end:
#   BASE_URL=http://127.0.0.1:<port>/v1
#   MODEL=<served model id>
#   TOOLCALL_OK=yes|no
#   REASONING_OK=yes|no
#
# It does NOT touch the Codex config — patch_codex_config.py does that.
set -uo pipefail

SCRIPT=""
PORT=""
RESTART=1
TIMEOUT=900
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2;;
    --no-restart) RESTART=0; shift;;
    --timeout) TIMEOUT="$2"; shift 2;;
    *) SCRIPT="$1"; shift;;
  esac
done

if [[ -z "$SCRIPT" || ! -f "$SCRIPT" ]]; then
  echo "ERROR: launch script not found: '$SCRIPT'" >&2
  exit 2
fi

# Derive the port from the script if not given (look for --port N).
if [[ -z "$PORT" ]]; then
  PORT=$(grep -oE -- '--port[= ]+[0-9]+' "$SCRIPT" | grep -oE '[0-9]+' | head -1)
fi
if [[ -z "$PORT" ]]; then
  echo "ERROR: could not determine --port from $SCRIPT; pass --port N" >&2
  exit 2
fi
BASE="http://127.0.0.1:${PORT}"

port_pid() { (ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null) | grep -E "[:.]${PORT}\b" | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2; }

is_up() { curl -s -m 3 "${BASE}/v1/models" >/dev/null 2>&1; }

if [[ "$RESTART" -eq 1 ]]; then
  PID=$(port_pid)
  if [[ -n "$PID" ]]; then
    PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
    echo ">> killing existing server on :${PORT} (pid=$PID pgid=$PGID)"
    [[ -n "$PGID" ]] && kill -TERM -"$PGID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null
    for _ in $(seq 1 30); do is_up || break; sleep 1; done
    [[ -n "$(port_pid)" ]] && { echo ">> still up, sending KILL"; PID=$(port_pid); PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' '); [[ -n "$PGID" ]] && kill -KILL -"$PGID" 2>/dev/null; sleep 3; }
  fi
else
  if is_up; then echo ">> --no-restart and server already up on :${PORT}"; fi
fi

if ! is_up; then
  LOG="$(dirname "$SCRIPT")/codex_backend_$(basename "$SCRIPT" .sh).out"
  echo ">> launching $SCRIPT (log: $LOG)"
  ( cd "$(dirname "$SCRIPT")" && nohup bash "$SCRIPT" > "$LOG" 2>&1 & )
  echo ">> waiting for ${BASE}/v1/models (timeout ${TIMEOUT}s) ..."
  ELAPSED=0
  while ! is_up; do
    sleep 10; ELAPSED=$((ELAPSED+10))
    if [[ $ELAPSED -ge $TIMEOUT ]]; then
      echo "ERROR: server not ready after ${TIMEOUT}s. Last log lines:" >&2
      tail -n 20 "$LOG" >&2
      exit 1
    fi
    if [[ $((ELAPSED % 30)) -eq 0 ]]; then echo "   ...${ELAPSED}s"; tail -n 1 "$LOG" 2>/dev/null; fi
  done
  echo ">> server is up after ~${ELAPSED}s"
fi

# Discover the served model id and its context window (max_model_len).
MODELS_JSON=$(curl -s -m 5 "${BASE}/v1/models")
MODEL=$(printf '%s' "$MODELS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
CONTEXT_WINDOW=$(printf '%s' "$MODELS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0].get('max_model_len') or '')" 2>/dev/null)
if [[ -z "$MODEL" ]]; then echo "ERROR: could not read model id from ${BASE}/v1/models" >&2; exit 1; fi

# Probe tool-call + reasoning parsing.
PROBE=$(curl -s -m 60 "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"${MODEL}\",
  \"messages\": [{\"role\":\"user\",\"content\":\"What is the weather in Paris? Use the get_weather tool.\"}],
  \"tools\": [{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],
  \"max_tokens\": 256
}")
read -r TOOLCALL_OK REASONING_OK <<<"$(printf '%s' "$PROBE" | python3 -c "
import sys,json
try:
    m=json.load(sys.stdin)['choices'][0]['message']
    tc='yes' if m.get('tool_calls') else 'no'
    rc='yes' if m.get('reasoning_content') else 'no'
except Exception:
    tc,rc='no','no'
print(tc,rc)
")"

echo
echo "================ SUMMARY ================"
echo "BASE_URL=${BASE}/v1"
echo "MODEL=${MODEL}"
echo "CONTEXT_WINDOW=${CONTEXT_WINDOW}"
echo "TOOLCALL_OK=${TOOLCALL_OK}"
echo "REASONING_OK=${REASONING_OK}"
echo "========================================"
if [[ "$TOOLCALL_OK" != "yes" ]]; then
  echo "WARNING: server did NOT return structured tool_calls — Codex agentic use will NOT work." >&2
  echo "         Fix --tool-call-parser in $SCRIPT (see SKILL.md parser table) and relaunch." >&2
  exit 3
fi
