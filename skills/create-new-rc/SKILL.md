---
name: create-new-rc
category: orchestration
description: "Start a persistent Claude Code Remote Control session on the GPU host (Command Center) for phone/laptop monitoring of container agents. Runs command-center.sh in tmux with subscription auth (no API key). Use when the user says '/create-new-rc', 'start remote control', 'open command center', 'start RC for abox-live', or wants a always-on RC session to track container agents away from their desk."
---

# create-new-rc — start the Agent Command Center (Remote Control)

Launches **`~/agent-box/skills/create-new-rc/command-center.sh`**: a **subscription**
Claude Code Remote Control session in tmux on the host. Use it from your phone or laptop
(same claude.ai account) to ask about container agent progress — typically via `/abox-live`.

Container **workers** stay on the AMD gateway API key. Only this coordinator uses
claude.ai subscription. Do not set `ANTHROPIC_API_KEY` in the same process.

## Run (do this immediately)

```bash
bash ~/agent-box/skills/create-new-rc/command-center.sh
bash ~/agent-box/skills/create-new-rc/command-center.sh "My RC Name"
```

Print the script output verbatim. If it already reports a running session, say so and
give attach/stop commands — do not start a second one.

## After it starts

Tell the user:

1. On phone or browser — Claude app or **claude.ai/code**, same claude.ai account
2. Open the Remote Control session with the printed name
3. Ask e.g. `/abox-live`, 「列出現在所有 container agent」, 「有沒有 agent 卡住」

Host attach (optional): `tmux attach -t agent-command-center` (detach: Ctrl-b then d)

Stop: `tmux kill-session -t agent-command-center`

## If it fails

Run in the foreground to see the error:

```bash
cd ~/agent-box && claude --remote-control "AMD Agent Command Center"
```

Common causes:

- **tmux missing** — `apt-get install tmux`
- **Not logged in to claude.ai** on the host — Remote Control needs subscription OAuth
  (`user:sessions:claude_code` scope); one-time device login may be required
- **API key env leaked into the RC process** — command-center.sh unsets it; if you started
  RC manually, unset `ANTHROPIC_*` first

## Related

- **`/abox-live`** — monitor live container ↔ session table (what the RC session queries)
- **`command-center.sh`** — this skill's launcher script (same directory as this file)
- Artifact dashboard — see `/abox-live` skill § Remote dashboard (no RC needed)
