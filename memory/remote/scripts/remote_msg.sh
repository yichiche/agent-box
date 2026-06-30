#!/usr/bin/env bash
# Append a message to INBOX or OUTBOX
# Usage:
#   remote_msg.sh to-container "Compare decode xlsx 0618 vs 0625"
#   remote_msg.sh to-host "MoE dispatch is the bottleneck"
set -euo pipefail

AGENT_BOX_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
REMOTE_DIR="$AGENT_BOX_DIR/memory/remote"
TS="$(date -u +"%Y-%m-%d %H:%M")"

DIR="${1:-}"
MSG="${2:-}"
if [[ -z "$DIR" || -z "$MSG" ]]; then
  echo "Usage: remote_msg.sh {to-container|to-host} \"message\"" >&2
  exit 1
fi

case "$DIR" in
  to-container)
    TARGET="$REMOTE_DIR/INBOX.md"
    TAG="host"
    ADDR="@container"
    ;;
  to-host)
    TARGET="$REMOTE_DIR/OUTBOX.md"
    TAG="container"
    ADDR="@host"
    ;;
  *)
    echo "Unknown direction: $DIR" >&2
    exit 1
    ;;
esac

ROLE="$TAG"
if [[ -f /.dockerenv ]]; then
  [[ "$DIR" == "to-host" ]] && ROLE="container" || ROLE="host-in-container"
else
  [[ "$DIR" == "to-container" ]] && ROLE="host" || ROLE="host"
fi

ENTRY="## [${TS} ${ROLE}] ${ADDR}
${MSG}

- status: pending
"

# Insert after header block (first ---)
python3 - <<PY
from pathlib import Path
p = Path("$TARGET")
text = p.read_text()
entry = """$ENTRY"""
marker = "\n---\n\n"
idx = text.find(marker)
if idx == -1:
    new = text.rstrip() + "\n\n" + entry
else:
    insert_at = idx + len(marker)
    new = text[:insert_at] + entry + text[insert_at:]
p.write_text(new)
print(f"[remote_msg] appended to {p}")
PY

# Refresh status timestamp
bash "$(dirname "$0")/remote_snapshot.sh" --role "$([[ -f /.dockerenv ]] && echo container || echo host)" --note "msg: ${MSG:0:80}"
