#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  OpenAB + Codex one-click setup
#  Usage: bash /home/yichiche/agent-box/setup-openab.sh
#
#  What this does:
#    1. Installs system deps (Node.js, npm)
#    2. Installs codex-cli + codex-acp
#    3. Clones & builds openab from source
#    4. Writes codex config (sandbox + approval policy)
#    5. Writes openab config.toml
#    6. Starts openab (single instance)
#
#  After running: do `codex login --device-auth` to authenticate.
# ============================================================

OPENAB_REPO="https://github.com/openabdev/openab.git"
OPENAB_DIR="/sgl-workspace/openab"
AGENT_BOX_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_SRC="$AGENT_BOX_DIR/configs/openab-config.toml"
CONFIG_FILE="/sgl-workspace/config.toml"
LOG_FILE="/sgl-workspace/openab.log"

# Provide these via the environment (do not hardcode secrets):
#   export DISCORD_BOT_TOKEN=...  DISCORD_CHANNEL_ID=...
DISCORD_BOT_TOKEN="${DISCORD_BOT_TOKEN:-}"
DISCORD_CHANNEL_ID="${DISCORD_CHANNEL_ID:-}"

# ---- helpers ----
info()  { echo "[openab] $*"; }
error() { echo "[openab] ERROR: $*" >&2; exit 1; }

# ---- 1. System deps ----
info "Installing system dependencies..."
if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y nodejs
else
  info "Node.js already installed: $(node --version)"
fi

# ---- 2. Codex CLI + codex-acp ----
if ! command -v codex >/dev/null 2>&1; then
  info "Installing @openai/codex@0.132.0..."
  npm install -g @openai/codex@0.132.0 2>&1 | tail -1
else
  info "codex already installed: $(codex --version 2>&1 | head -1)"
fi

if ! command -v codex-acp >/dev/null 2>&1; then
  info "Installing @zed-industries/codex-acp..."
  npm install -g @zed-industries/codex-acp 2>&1 | tail -1
else
  info "codex-acp already installed"
fi

# ---- 3. Clone & build openab ----
if [ ! -d "$OPENAB_DIR" ]; then
  info "Cloning openab..."
  git clone "$OPENAB_REPO" "$OPENAB_DIR"
else
  info "openab repo already exists at $OPENAB_DIR"
fi

OPENAB_BIN="$OPENAB_DIR/target/release/openab"
if [ ! -f "$OPENAB_BIN" ]; then
  info "Building openab (this takes ~3 minutes)..."
  if ! command -v cargo >/dev/null 2>&1; then
    error "cargo not found. Install Rust first: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
  fi
  cd "$OPENAB_DIR"
  cargo build --release 2>&1 | tail -3
  cd /sgl-workspace
  info "Build complete"
else
  info "openab binary already exists"
fi

# ---- 4. Codex config ----
info "Writing codex config..."

mkdir -p /root/.codex
if [ ! -f /root/.codex/config.toml ]; then
  cat > /root/.codex/config.toml <<'EOF'
model = "gpt-5.5"
model_reasoning_effort = "xhigh"
EOF
fi

# Remove AMD gateway proxy URL if present (we connect to OpenAI directly)
if grep -q "openai_base_url" /root/.codex/config.toml 2>/dev/null; then
  sed -i '/openai_base_url/d' /root/.codex/config.toml
  info "Removed openai_base_url from codex config (using OpenAI directly)"
fi

mkdir -p /home/node/.codex
cat > /home/node/.codex/config.toml <<'EOF'
sandbox_mode = "danger-full-access"
approval_policy = "auto-review"
EOF

# ---- 5. OpenAB config (symlink from agent-box) ----
info "Linking openab config..."
if [ ! -f "$CONFIG_SRC" ]; then
  error "Config not found at $CONFIG_SRC"
fi
ln -sf "$CONFIG_SRC" "$CONFIG_FILE"
info "Symlinked $CONFIG_FILE -> $CONFIG_SRC"

codex login --device-auth  

# ---- 6. Start openab (kill any existing instances first) ----
info "Starting openab..."
pkill -f "openab run" 2>/dev/null || true
sleep 2

nohup "$OPENAB_BIN" run -c "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
OPENAB_PID=$!

sleep 4
if kill -0 "$OPENAB_PID" 2>/dev/null; then
  info "openab running (PID: $OPENAB_PID)"
  if grep -q "discord bot connected" "$LOG_FILE" 2>/dev/null; then
    info "Discord bot connected successfully!"
  else
    info "Waiting for Discord connection..."
    sleep 3
    if grep -q "discord bot connected" "$LOG_FILE" 2>/dev/null; then
      info "Discord bot connected!"
    else
      info "Check logs: tail -f $LOG_FILE"
    fi
  fi
else
  error "openab failed to start. Check: tail $LOG_FILE"
fi

echo ""
echo "============================================"
echo "  OpenAB setup complete!"
echo "============================================"
echo ""
echo "  Bot:     openABAgent"
echo "  Agent:   codex-acp (OpenAI Codex)"
echo "  Channel: ${DISCORD_CHANNEL_ID}"
echo "  Logs:    tail -f ${LOG_FILE}"
echo "  PID:     ${OPENAB_PID}"
echo "  Then @openABAgent in Discord to test!"
echo "============================================"
