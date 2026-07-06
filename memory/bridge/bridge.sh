#!/usr/bin/env bash
# bridge.sh — one entry point for host <-> container agent coordination.
#
# Two layers, by design:
#   * File bus (durable, async, git-tracked): STATUS.md / INBOX.md / OUTBOX.md
#       -> subcommands: status, msg, watch
#   * Direct exec (near-real-time, one-shot request/reply):
#       -> subcommand: exec  (docker exec into a container, run claude/codex headless)
#
# SAFETY: `exec` is restricted to containers that bind-mount /home/yichiche
# (i.e. yichiche's own containers). This is a shared, multi-tenant host —
# never exec into another user's container. Everything is logged to bridge.log.
#
# Usage:
#   bridge.sh list                                   # yichiche-owned running containers
#   bridge.sh status [--role R] [--task T] [--note N]# refresh STATUS.md (delegates to remote_snapshot.sh)
#   bridge.sh msg to-container|to-host "message"     # append to INBOX/OUTBOX
#   bridge.sh exec <container> "prompt" [--agent claude|codex] [--cwd DIR] [--dangerous] [--timeout S]
#   bridge.sh watch [--interval S]                   # poll INBOX; print new @container messages (for a container agent)
set -euo pipefail

BRIDGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$BRIDGE_DIR/scripts"
LOG="$BRIDGE_DIR/bridge.log"
HOST_MOUNT="/home/yichiche"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

# --- container allowlist ----------------------------------------------
# A container is "ours" iff it bind-mounts $HOST_MOUNT. Filter by mount,
# never by name (names belong to whoever launched them).
owned_containers() {
  local c
  for c in $(docker ps --format '{{.Names}}' 2>/dev/null); do
    if docker inspect "$c" --format '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}' 2>/dev/null \
        | grep -qx "$HOST_MOUNT"; then
      echo "$c"
    fi
  done
  return 0  # never fail just because the last container wasn't ours (pipefail)
}

is_owned() { local list; list="$(owned_containers)"; grep -qxF -- "$1" <<<"$list"; }

cmd_list() {
  echo "yichiche-owned running containers (exec-eligible):"
  owned_containers | sed 's/^/  - /' || true
}

# --- file bus ----------------------------------------------------------
cmd_status() { bash "$SCRIPTS/remote_snapshot.sh" "$@"; }
cmd_msg()    { bash "$SCRIPTS/remote_msg.sh" "$@"; }

# --- direct exec -------------------------------------------------------
cmd_exec() {
  local container="${1:-}"; shift || true
  local prompt="${1:-}"; shift || true
  local agent=claude cwd="/sgl-workspace/sglang" dangerous=0 timeout=300
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --agent) agent="$2"; shift 2 ;;
      --cwd) cwd="$2"; shift 2 ;;
      --dangerous) dangerous=1; shift ;;
      --timeout) timeout="$2"; shift 2 ;;
      *) echo "exec: unknown arg $1" >&2; return 2 ;;
    esac
  done
  if [[ -z "$container" || -z "$prompt" ]]; then
    echo "usage: bridge.sh exec <container> \"prompt\" [--agent claude|codex] [--cwd DIR] [--dangerous] [--timeout S]" >&2
    return 2
  fi
  if ! docker ps --format '{{.Names}}' | grep -qxF "$container"; then
    echo "exec: no running container named '$container'. Try: bridge.sh list" >&2; return 1
  fi
  if ! is_owned "$container"; then
    echo "exec: REFUSED — '$container' does not mount $HOST_MOUNT (not yours). Shared host; not exec-ing." >&2
    log "REFUSED exec into non-owned container=$container"
    return 1
  fi

  # claude/codex install into $HOME/.local/bin which docker exec's default
  # PATH misses — prepend it so headless invocation resolves the binary.
  local pathfix='export PATH="$HOME/.local/bin:$PATH";' cdto="cd '$cwd' 2>/dev/null || cd /;"
  local inner flags=""
  case "$agent" in
    claude)
      [[ "$dangerous" == 1 ]] && flags="--dangerously-skip-permissions"
      inner="$pathfix $cdto claude -p $flags \"\$PROMPT\"" ;;
    codex)
      inner="$pathfix $cdto codex exec \"\$PROMPT\"" ;;
    *) echo "exec: --agent must be claude or codex" >&2; return 2 ;;
  esac

  echo "→ [$container] running $agent (cwd=$cwd, timeout=${timeout}s)…" >&2
  log "exec container=$container agent=$agent cwd=$cwd dangerous=$dangerous prompt=${prompt:0:120}"

  local out rc=0
  out="$(timeout "$timeout" docker exec -e PROMPT="$prompt" "$container" \
        bash -lc "$inner" 2>&1)" || rc=$?

  printf '%s\n' "$out"

  # mirror the reply into OUTBOX so it survives as durable state
  local ts; ts="$(date -u +'%Y-%m-%d %H:%M')"
  local entry="## [${ts} ${container}] @host (via bridge exec, agent=$agent)
$prompt

\`\`\`
$(printf '%s' "$out" | head -60)
\`\`\`
- rc: $rc
- status: done
"
  python3 - "$BRIDGE_DIR/OUTBOX.md" "$entry" <<'PY'
import sys, pathlib
p, entry = pathlib.Path(sys.argv[1]), sys.argv[2]
text = p.read_text()
marker = "\n---\n\n"
i = text.find(marker)
p.write_text(text[:i+len(marker)] + entry + "\n" + text[i+len(marker):] if i>=0 else text.rstrip()+"\n\n"+entry)
PY
  log "exec done container=$container rc=$rc bytes=${#out}"
  return "$rc"
}

# --- watch (container-side poller) ------------------------------------
cmd_watch() {
  local interval=15
  [[ "${1:-}" == "--interval" ]] && interval="$2"
  local inbox="$BRIDGE_DIR/INBOX.md" last=""
  echo "watching $inbox for @container messages (every ${interval}s, Ctrl-C to stop)…" >&2
  while true; do
    local cur; cur="$(sha256sum "$inbox" | awk '{print $1}')"
    if [[ "$cur" != "$last" && -n "$last" ]]; then
      echo "=== INBOX changed $(date -u +%FT%TZ) ==="
      grep -n '@container' "$inbox" | grep -i 'pending' || true
    fi
    last="$cur"; sleep "$interval"
  done
}

case "${1:-}" in
  list)   shift; cmd_list "$@" ;;
  status) shift; cmd_status "$@" ;;
  msg)    shift; cmd_msg "$@" ;;
  exec)   shift; cmd_exec "$@" ;;
  watch)  shift; cmd_watch "$@" ;;
  -h|--help|"") sed -n '2,30p' "$0" ;;
  *) echo "unknown subcommand: $1 (try: list status msg exec watch)" >&2; exit 2 ;;
esac
