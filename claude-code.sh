#!/usr/bin/env bash

HOST_HOME="/home/yichiche"   # host-mounted home directory inside container

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

mkdir -p /sgl-workspace/.claude
cp ${HOST_HOME}/agent-box/CLAUDE.md /sgl-workspace/
cp ${HOST_HOME}/agent-box/setting.json /sgl-workspace/.claude/setting.json 
if [ -f "${HOST_HOME}/.claude.json" ]; then
  ln -sf "${HOST_HOME}/.claude.json" /root/.claude.json
else
  claude
  cp /root/.claude.json "${HOST_HOME}/"
  echo "[claude] First-time initialization"
fi
