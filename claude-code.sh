#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
# HOST_HOME is set by env.sh (auto-detected from agent-box parent directory)

echo "[claude] start"

# Download installer
curl -L --progress-bar --connect-timeout 10 --max-time 300 --retry 3 \
  -o /tmp/claude-install.sh https://claude.ai/install.sh
echo "[claude] downloaded"

# Run installer with logging
timeout 300 bash -x /tmp/claude-install.sh 2>&1 | tee /tmp/claude-install.log

# Ensure PATH for current shell and future shells
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Claude initialization / config linking
echo "[claude] finished"

# Install Codex CLI (pinned to 0.132.0 — newer breaks AMD Gateway)
if ! command -v codex >/dev/null 2>&1; then
  echo "[codex] Installing @openai/codex@0.132.0..."
  npm install -g @openai/codex@0.132.0 2>&1 | tail -1
  echo "[codex] finished"
else
  echo "[codex] already installed ($(codex --version 2>/dev/null))"
fi
