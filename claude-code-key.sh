#!/usr/bin/env bash
# DEPRECATED: use skills/abox-live/claude-container-auth.sh (isolated CLAUDE_CONFIG_DIR).
# Kept as a thin wrapper so old docs/scripts keep working without touching host OAuth.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[claude] claude-code-key.sh is deprecated — running claude-container-auth.sh instead" >&2
exec bash "${SCRIPT_DIR}/skills/abox-live/claude-container-auth.sh"
