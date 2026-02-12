 #!/usr/bin/env bash
HOST_HOME="/home/yichiche"

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

# Write env vars to .bashrc (EOF must be at column 0, double-quotes expand CLAUDE_KEY)
cat >> ~/.bashrc <<EOF
# Claude Code environment
export ANTHROPIC_API_KEY="dummy"
export ANTHROPIC_BASE_URL="https://llm-api.amd.com/Anthropic"
export ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key:${CLAUDE_KEY}"
export ANTHROPIC_MODEL="Claude-Opus-4.6"
export ANTHROPIC_DEFAULT_OPUS_MODEL="Claude-Opus-4.6"
export ANTHROPIC_DEFAULT_SONNET_MODEL="Claude-Sonnet-4.5"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="Claude-Haiku-4.5"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
EOF
source ~/.bashrc


