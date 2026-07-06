---
description: Cross-container agent bridge — file bus (STATUS/INBOX/OUTBOX) plus direct `docker exec` into your own containers to run claude/codex headless and get a reply. Use when coordinating host with container agents, or to send a live one-shot command into a container.
---

# Bridge (Host ↔ Container)

One entry point: `memory/bridge/bridge.sh`. Two layers:

- **File bus** (durable, async, git-tracked): `STATUS.md` / `INBOX.md` / `OUTBOX.md`
- **Direct exec** (near-real-time, one-shot): `docker exec` into a container, run `claude`/`codex` headless, reply is printed and mirrored to `OUTBOX.md`.

Full protocol + mounts: `memory/bridge/README.md`.

## List your containers (exec-eligible)

```bash
bash ~/agent-box/memory/bridge/bridge.sh list
```

Only containers that bind-mount `/home/yichiche` are listed — `exec` **refuses** any other user's container (shared host).

## Send a live command into a container (host → container agent → reply)

```bash
bash ~/agent-box/memory/bridge/bridge.sh exec <container> "your prompt" \
  [--agent claude|codex] [--cwd /sgl-workspace/sglang] [--timeout 300] [--dangerous]
```

- Default `--agent claude`, prints the reply and appends it to `OUTBOX.md`.
- `--dangerous` passes `--dangerously-skip-permissions` so the container agent can actually run tools (edits/commands). Omit it for read-only Q&A.
- Everything is logged to `memory/bridge/bridge.log`.

## Async file-bus messages

```bash
# Host → Container INBOX
bash ~/agent-box/memory/bridge/bridge.sh msg to-container "分析 pr28666 decode xlsx"
# Container → Host OUTBOX
bash ~/agent-box/memory/bridge/bridge.sh msg to-host "瓶頸在 fused_moe，不是 gemm"
```

## Refresh STATUS snapshot

```bash
bash ~/agent-box/memory/bridge/bridge.sh status --role container --task "<task>" --note "<one line>"
```

Use `--role host` on the host. Container agents read `INBOX.md` at session start; host reads `OUTBOX.md`.

## Container-side auto-read (optional)

```bash
bash ~/agent-box/memory/bridge/bridge.sh watch --interval 15   # poll INBOX for @container pending
```

## Native Remote Control (optional third layer)

Requires claude.ai `/login` (not the AMD gateway). Inside container: `claude --remote-control "MI355 $(hostname -s)"`, then record the URL via `bridge.sh status --rc-url ...` for live claude.ai/code chat.

## Integrate with memory vault

Stable cross-container findings → `/memory-capture`. The bridge is coordination; the vault (`memory/journal`, `gotchas/`) is durable knowledge.
