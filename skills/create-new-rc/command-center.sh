#!/usr/bin/env bash
# AMD Agent Command Center — a subscription Remote Control session on the host that
# your phone connects to. It watches container agents via `abox-live`.
#
# WHY subscription (not the gateway API key): Remote Control only works under a
# claude.ai login; if ANTHROPIC_API_KEY / gateway env is set the two collide
# ("auth may not work"). This launcher strips that env so RC uses your claude.ai login.
# The container WORKERS stay on the API key — only this coordinator is subscription.
set -uo pipefail

NAME="${1:-AMD Agent Command Center}"
SES="agent-command-center"

command -v tmux >/dev/null || { echo "tmux not installed (apt-get install tmux)"; exit 1; }

if tmux has-session -t "$SES" 2>/dev/null; then
  echo "Command Center already running (tmux '$SES')."
  echo "  attach: tmux attach -t $SES    stop: tmux kill-session -t $SES"
  exit 0
fi

CLAUDE_BIN="$(command -v claude || echo "$HOME/.local/bin/claude")"

tmux new-session -d -s "$SES" bash -lc "
  unset ANTHROPIC_API_KEY ANTHROPIC_BASE_URL ANTHROPIC_CUSTOM_HEADERS \
        ANTHROPIC_MODEL ANTHROPIC_DEFAULT_OPUS_MODEL \
        ANTHROPIC_DEFAULT_SONNET_MODEL ANTHROPIC_DEFAULT_HAIKU_MODEL 2>/dev/null
  cd '$HOME/agent-box'
  exec '$CLAUDE_BIN' --remote-control '$NAME'
"

sleep 2
if tmux has-session -t "$SES" 2>/dev/null; then
  echo "✅ Command Center '$NAME' starting in tmux '$SES'."
  echo "   host pane : tmux attach -t $SES   (detach: Ctrl-b then d)"
  echo "   stop      : tmux kill-session -t $SES"
  echo
  echo "On your phone — Claude app or claude.ai/code, SAME claude.ai account:"
  echo "   open the '$NAME' Remote Control session, then ask e.g."
  echo "   「列出現在所有 container agent 在做什麼」 / 「有沒有 agent 卡住等輸入」"
else
  echo "❌ did not start. Run in the foreground to see why:"
  echo "   cd ~/agent-box && claude --remote-control \"$NAME\""
fi
