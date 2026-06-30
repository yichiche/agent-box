---
description: Cross-container Claude Code bridge — read/update STATUS, send INBOX/OUTBOX messages between host and container. Use when coordinating host Cursor with container Claude Code, or before /remote-control.
---

# Remote Bridge (Host ↔ Container)

File bus at `memory/remote/`. Full protocol: `memory/remote/README.md`.

## When to use

- Host wants to know what container Claude Code is doing
- Container needs architecture guidance from host
- Before/after enabling native `/remote-control`
- User says "跟 container 說…" or "外面 Claude 想知道進度"

## Session start (both sides)

1. Read `memory/remote/STATUS.md`
2. Read `memory/remote/INBOX.md` if you are **container**; `OUTBOX.md` if you are **host**
3. Update snapshot:
   ```bash
   bash ~/agent-box/memory/remote/scripts/remote_snapshot.sh \
     --role container \
     --task "<current task>" \
     --note "<one line>"
   ```
   Use `--role host` on the host.

## Send a message

```bash
# Host → Container
bash ~/agent-box/memory/remote/scripts/remote_msg.sh to-container "請分析 pr28666 decode xlsx"

# Container → Host
bash ~/agent-box/memory/remote/scripts/remote_msg.sh to-host "瓶頸在 fused_moe，不是 gemm"
```

Or edit `INBOX.md` / `OUTBOX.md` directly (prepend, set `status: pending`).

## Native Remote Control (optional layer)

Requires **claude.ai `/login`** — does **not** work with AMD API gateway alone.

Inside container:
```bash
cd /sgl-workspace/sglang
claude --remote-control "MI355 $(hostname -s)"
# or in session: /remote-control
```

After RC starts, record URL in snapshot:
```bash
bash ~/agent-box/memory/remote/scripts/remote_snapshot.sh \
  --role container --rc-url "https://claude.ai/code/..."
```

Host opens that URL in claude.ai/code for **live** chat; file bus stays for **persistent** context.

## Find active Claude session

```bash
ls -lt ~/.claude/sessions/*.json | head -3
# or read STATUS.md claude_session_id
```

Shared via `-v $HOME/.claude:/root/.claude` in `run_docker.sh`.

## Mark message done

Edit the message block: `status: pending` → `status: done`

## Integrate with memory vault

Important cross-container findings → `/memory-capture` into `memory/gotchas/`
