---
type: architecture
---

# Cross-container Claude Code bridge

## Problem

GPU work runs in Docker (`run_docker.sh`); planning/discussion may happen on host (Cursor) or phone. Need shared situational awareness without duplicating context.

## Solution: dual layer

```mermaid
flowchart LR
  subgraph host [Host]
    H[Cursor / Claude Code]
  end
  subgraph shared [Shared volume]
    ST[STATUS.md]
    IN[INBOX.md]
    OUT[OUTBOX.md]
    CL[.claude/sessions/]
  end
  subgraph container [Container]
    C[Claude Code + GPUs]
  end
  H <-->|read/write| ST
  H -->|to-container| IN
  OUT -->|to-host| H
  C <-->|read/write| ST
  IN --> C
  C --> OUT
  H --- CL
  C --- CL
```

1. **File bus** (`memory/remote/`) — always works with AMD API gateway; async messages + snapshot
2. **Native RC** (`/remote-control`) — real-time if container uses claude.ai `/login`

## Docker mounts (already in run_docker.sh)

- `$HOME/.claude:/root/.claude` — session IDs visible on both sides
- `$HOME:/home/yichiche/` — agent-box memory including `remote/`

## Agent ritual

| Side | On start | On task change | On finish |
|---|---|---|---|
| Container | read INBOX, `remote_snapshot.sh --role container` | update `--task` | `remote_msg.sh to-host`, mark INBOX done |
| Host | read STATUS + OUTBOX | `remote_msg.sh to-container` | capture learnings `/memory-capture` |

Related: [[../remote/README]], skill `/remote-bridge`
