 #!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

mkdir -p /sgl-workspace/.claude
cp ${HOST_HOME}/agent-box/CLAUDE.md /sgl-workspace/
ln -sf ${HOST_HOME}/agent-box/setting.json ~/.claude/settings.json

if [ -f "${HOST_HOME}/.claude_api_key" ]; then
  ln -sf "${HOST_HOME}/.claude.json" /root/.claude.json
  KEY_FILE="${HOST_HOME}/.claude_api_key"
  CLAUDE_KEY=$(cat "$KEY_FILE")
else
  KEY_FILE="${HOST_HOME}/.claude_api_key"
  if [ -f "$KEY_FILE" ]; then
    CLAUDE_KEY=$(cat "$KEY_FILE")
  else
    echo ""
    echo "============================================"
    echo "  First-time setup: API key required"
    echo "============================================"
    read -rp "Enter your API key: " CLAUDE_KEY
    if [ -z "$CLAUDE_KEY" ]; then
      echo "[claude] Error: API key cannot be empty."
      exit 1
    fi
    echo "$CLAUDE_KEY" > "$KEY_FILE"
    chmod 600 "$KEY_FILE"
    echo "[claude] API key saved to $KEY_FILE"
  fi

  cat > /root/.claude.json <<EOF
{
  "hasCompletedOnboarding": true
}
EOF
  cp /root/.claude.json "${HOST_HOME}/"
  echo "[claude] First-time initialization"
fi

# Share the host's MCP OAuth tokens (e.g. Atlassian/Jira remote MCP) with the container,
# but STRIP the claude.ai subscription login — containers must bill the API key, never the
# subscription (see CLAUDE.md). The server list itself arrives via the ~/.claude.json symlink;
# this copies only the matching OAuth tokens so /mcp connects without a browser flow.
mkdir -p /root/.claude
if [ -f "${HOST_HOME}/.claude/.credentials.json" ]; then
  python3 - "${HOST_HOME}/.claude/.credentials.json" /root/.claude/.credentials.json <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
creds = json.load(open(src))
creds.pop("claudeAiOauth", None)  # never carry the subscription login into a container
if creds:
    with open(dst, "w") as f:
        json.dump(creds, f)
    print(f"[claude] MCP OAuth tokens shared into container ({', '.join(sorted(creds))})")
else:
    print("[claude] no MCP OAuth tokens on host yet (authenticate via /mcp on the host first)")
PY
  chmod 600 /root/.claude/.credentials.json 2>/dev/null
fi

# Write env vars to .bashrc (EOF must be at column 0, double-quotes expand CLAUDE_KEY)
cat >> ~/.bashrc <<EOF
# Claude Code environment
export ANTHROPIC_API_KEY="dummy"
export ANTHROPIC_BASE_URL="https://llm-api.amd.com/Anthropic"
export ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key:${CLAUDE_KEY}"
export ANTHROPIC_MODEL="claude-opus-4-8[1m]"
export ANTHROPIC_DEFAULT_OPUS_MODEL="claude-opus-4-8[1m]"
export ANTHROPIC_DEFAULT_SONNET_MODEL="claude-sonnet-4-6[1m]"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="Claude-Haiku-4.5"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
EOF
source ~/.bashrc


