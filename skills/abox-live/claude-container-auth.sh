#!/usr/bin/env bash
# Container-side Claude Code setup.
#
# Design goals:
#   1. The HOST's claude.ai login can NEVER be touched by a container.
#   2. Containers authenticate via the AMD gateway API key — no interactive login, ever.
#   3. Shared content (skills / settings / CLAUDE.md / commands) is LIVE-SYNCED by
#      symlink, so editing on the host is instantly visible in every container.
#
# Replaces the credential-stripping block in claude-code-key.sh, which wrote its
# "stripped" copy back onto the SAME inode as the host credential file (both
# $HOME and $HOME/.claude were bind-mounted), deleting the host subscription login
# on every container init.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../env.sh"

# Container config dir (overlay). Credentials stay here so the host login is
# never overwritten. Session files are symlinked onto NFS ~/.claude below.
CFG="${CLAUDE_CONFIG_DIR:-/root/.claude-local}"
HOST_CLAUDE="${HOST_HOME}/.claude"
mkdir -p "$CFG"

# ---------------------------------------------------------------- 1. live sync
# Symlink (not copy) -> edit on host, container sees it immediately. No sync step.
link() { [ -e "$1" ] && ln -sfn "$1" "$CFG/$2" || true; }
link "${HOST_HOME}/agent-box/skills"       skills
link "${HOST_HOME}/agent-box/setting.json" settings.json
link "${HOST_CLAUDE}/CLAUDE.md"            CLAUDE.md
link "${HOST_CLAUDE}/commands"             commands
link "${HOST_CLAUDE}/agents"               agents
link "${HOST_CLAUDE}/plugins"              plugins

# Session store lives on the host NFS home so `claude resume` still works after
# the container is recreated on another node. Credentials / statsig stay local.
link_session() {
    local host_item="$1" name="$2"
    local cfg_item="$CFG/$name"
    mkdir -p "$(dirname "$host_item")" "$(dirname "$cfg_item")"
    if [[ -L "$cfg_item" ]]; then
        ln -sfn "$host_item" "$cfg_item"
        return 0
    fi
    if [[ -d "$cfg_item" ]]; then
        mkdir -p "$host_item"
        if [[ -n "$(ls -A "$cfg_item" 2>/dev/null || true)" ]]; then
            cp -a "$cfg_item"/. "$host_item"/ 2>/dev/null || true
        fi
        rm -rf "$cfg_item"
    elif [[ -f "$cfg_item" ]]; then
        if [[ -s "$cfg_item" ]]; then
            if [[ -s "$host_item" ]]; then
                cat "$cfg_item" >> "$host_item" || true
                rm -f "$cfg_item"
            else
                mv "$cfg_item" "$host_item"
            fi
        else
            rm -f "$cfg_item"
        fi
    fi
    ln -sfn "$host_item" "$cfg_item"
}
mkdir -p "${HOST_CLAUDE}/projects" "${HOST_CLAUDE}/sessions" "${HOST_CLAUDE}/todos"
[[ -f "${HOST_CLAUDE}/history.jsonl" ]] || : > "${HOST_CLAUDE}/history.jsonl"
link_session "${HOST_CLAUDE}/projects"     projects
link_session "${HOST_CLAUDE}/sessions"     sessions
link_session "${HOST_CLAUDE}/todos"        todos
link_session "${HOST_CLAUDE}/history.jsonl" history.jsonl

# Still container-local (must not touch the host): shell-snapshots/, statsig/,
# .credentials.json

# ------------------------------------------------- 2. MCP OAuth tokens (Jira)
# Copy host MCP tokens in, MINUS the claude.ai subscription login, so /mcp works
# without a browser flow. Writes ONLY to the container-local dir.
if [ -f "${HOST_CLAUDE}/.credentials.json" ]; then
  python3 - "${HOST_CLAUDE}/.credentials.json" "$CFG/.credentials.json" <<'PY'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
# The guard that the old script was missing: never write onto the host file.
if os.path.exists(dst) and os.path.samefile(src, dst):
    sys.exit("[claude] REFUSING to write onto the host credential file (same inode)")
creds = json.load(open(src))
creds.pop("claudeAiOauth", None)  # subscription login never enters a container
if creds:
    with open(dst, "w") as f:
        json.dump(creds, f)
    print(f"[claude] MCP OAuth shared into container ({', '.join(sorted(creds))})")
else:
    if os.path.exists(dst):
        os.remove(dst)
    print("[claude] no MCP OAuth tokens — container credentials cleared")
PY
  chmod 600 "$CFG/.credentials.json" 2>/dev/null || true
fi

# --------------------------------------------------------- 3. API-key auth
KEY_FILE="${HOST_HOME}/.claude_api_key"
if [[ ! -f "$KEY_FILE" && -f /root/.claude_api_key ]]; then
  KEY_FILE="/root/.claude_api_key"
fi
if [ ! -f "$KEY_FILE" ]; then
  echo "[claude] ERROR: $KEY_FILE not found — cannot authenticate container." >&2
  echo "[claude] Run ~/run_docker.sh once on the host to save the gateway key locally." >&2
  exit 1
fi
CLAUDE_KEY=$(cat "$KEY_FILE")

# Idempotent: strip legacy claude-code-key.sh and prior container-env blocks.
sed -i '/# >>> claude container env >>>/,/# <<< claude container env <<</d' ~/.bashrc 2>/dev/null || true
sed -i '/^# Claude Code environment$/,/^export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1$/d' ~/.bashrc 2>/dev/null || true
cat >> ~/.bashrc <<EOF
# >>> claude container env >>>
export CLAUDE_CONFIG_DIR="${CFG}"
export ANTHROPIC_API_KEY="dummy"
export ANTHROPIC_BASE_URL="https://llm-api.amd.com/Anthropic"
export ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key:${CLAUDE_KEY}"
export ANTHROPIC_MODEL="claude-opus-5[1m]"
export ANTHROPIC_DEFAULT_OPUS_MODEL="claude-opus-5[1m]"
export ANTHROPIC_DEFAULT_SONNET_MODEL="claude-sonnet-5[1m]"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="Claude-Haiku-4.5"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
# <<< claude container env <<<
EOF

echo "[claude] container config: $CFG (host login untouched, API-key auth wired)"
