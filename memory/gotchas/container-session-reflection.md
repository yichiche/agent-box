---
type: gotcha
aliases: [container session reflect, abox monitor, watch container agents, claude-code-key footgun]
---

# Watching container Claude agents from the host (and its footguns)

**Mechanism (verified 2026-07-19):** a `claude` running *inside* a container writes its
transcript to `/root/.claude/projects/<proj>/<id>.jsonl`. `/root/.claude` is bind-mounted
to the host's `~/.claude` (same device), so **every container agent's session appears on
the host** under `~/.claude/projects/...`, **appended per turn** (live, ~1 turn lag).
This is how you monitor container agents — **no docker exec, no resume, no stream-json,
no `--forward-subagent-text` (that flag doesn't exist in 2.1.206) needed for read-only
monitoring.**

- Project dir = hash of the container **cwd** (e.g. `/sgl-workspace/aiter` →
  `-sgl-workspace-aiter`). Run workers from `/home/yichiche/agent-box` and the project
  key matches host↔container (that path is bind-mounted at the same location).
- Files are written **`root:root 0600`** (container runs as root) → the host user
  (`yichiche`, uid 11065) can't read them directly; use **sudo** (passwordless here).
- Tooling: **`abox-live`** (`skills/abox-live/`, absorbed the old `abox`) — reader
  engine `abox_report.py`/`abox_parse.py` now lives there: `abox-live ps` / `tail <id>`
  / `watch <id>` / `path <id>`, plus the live process-based view + `stop`/`say`/`web`.
  `ps`/`tail` run the whole report as **one `sudo python3`** (root reads all sessions);
  per-row `sudo` in a bash loop intermittently returned empty.
- To let the host user own the read path instead of sudo: run the container as
  `--user 11065` or `sudo chown` the project dir.

## Footguns hit while building this

1. **`claude-code-key.sh` env doesn't reach headless runs.** It appends the `ANTHROPIC_*`
   exports to `~/.bashrc` and `source`s it — but a non-interactive `bash -lc` hits the
   bashrc "return if not interactive" guard *before* those lines → the worker starts with
   no gateway auth → **"Not logged in · Please run /login"**. **For headless/orchestrated
   workers, export the gateway env explicitly**, don't rely on sourcing that script:
   `ANTHROPIC_API_KEY=dummy ANTHROPIC_BASE_URL=https://llm-api.amd.com/Anthropic
   ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key:$(cat ~/.claude_api_key)"
   ANTHROPIC_MODEL="claude-opus-4-8[1m]"`.
2. **`claude-code-key.sh` can wipe the host subscription login.** Its credential step
   reads `~/.claude/.credentials.json`, `pop("claudeAiOauth")`, and writes
   `/root/.claude/.credentials.json` — which **is the same host file via the mount**. If
   the host creds have other keys it rewrites the file *without* `claudeAiOauth`, breaking
   host Remote Control / subscription. **Never source it on the shared mount** for
   orchestration; set env explicitly (footgun 1) and skip the cred-strip.
3. **A fresh container from the image has no `claude`.** But the host binary
   `/home/yichiche/.local/bin/claude` is reachable in-container via the `/home/yichiche`
   mount at the same path — invoke it directly, no per-container install.
4. **`--dangerously-skip-permissions` is blocked** (harness auto-mode classifier) when
   launching a nested claude. Use `--allowedTools WebSearch WebFetch Read Grep Glob` etc.
5. **Remote Control needs subscription + the `user:sessions:claude_code` scope**, so even
   with a valid host `claudeAiOauth` it triggers a **one-time OAuth device login** — a
   user action, not scriptable. The Command Center (`/create-new-rc`) is subscription;
   container workers stay on the API key. The two must be **separate processes** (one
   process can't be both — the "Both claude.ai and ANTHROPIC_API_KEY set" warning).

## Related

- Skill `/abox-live` (incl. `claude-container-auth.sh` for container workers); `/create-new-rc`; [[container-bench-flags]]
- [[../models/qwen35-mxfp4-mi355]] · CLAUDE.md § Container Task Execution
