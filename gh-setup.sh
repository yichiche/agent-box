#!/usr/bin/env bash
# agent-box/gh-setup.sh — runs once on Docker container launch (chained from run_docker.sh).
#
# What it does (idempotent):
#   1. Installs `gh` (pinned version) to ${HOST_HOME}/bin/gh so the binary is
#      persistent across container destroy / re-create — host-mounted, downloaded once.
#   2. Appends a managed block to ~/.bashrc that on every interactive shell:
#        - puts ${HOST_HOME}/bin on PATH so `gh` is on $PATH
#        - exports GH_TOKEN from $HOME/.git-credentials or ${HOST_HOME}/.git-credentials
#      The block is bracketed by markers so it is appended at most once.
#
# Why this lives in agent-box (not just the skill): the matching skill is the
# on-demand fix; this script is the proactive setup that runs at container init.
# Both share the same install URL + token-extraction logic.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
# env.sh defines HOST_HOME (e.g. /home/yichiche)

GH_VERSION="${GH_VERSION:-2.62.0}"
GH_BIN_DIR="${HOST_HOME}/bin"
GH_BIN="${GH_BIN_DIR}/gh"
GH_URL="https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.tar.gz"

echo "[gh-setup] start"

# ---------------------------------------------------------------------------
# 1) Install gh persistently to ${HOST_HOME}/bin/gh
# ---------------------------------------------------------------------------
if [ -x "$GH_BIN" ]; then
  echo "[gh-setup] gh already at $GH_BIN ($("$GH_BIN" --version | head -1))"
else
  mkdir -p "$GH_BIN_DIR"
  tmp="$(mktemp -d)"
  echo "[gh-setup] downloading gh v${GH_VERSION}"
  curl -fsSL "$GH_URL" -o "$tmp/gh.tar.gz"
  tar -xzf "$tmp/gh.tar.gz" -C "$tmp"
  cp "$tmp/gh_${GH_VERSION}_linux_amd64/bin/gh" "$GH_BIN"
  chmod +x "$GH_BIN"
  rm -rf "$tmp"
  echo "[gh-setup] installed $GH_BIN"
fi

# ---------------------------------------------------------------------------
# 2) Append managed PATH + GH_TOKEN block to ~/.bashrc (idempotent)
#    Inside docker, $HOME is typically /root; on host it's the user home.
#    Either way, we write to whichever ~/.bashrc the calling shell uses.
# ---------------------------------------------------------------------------
BASHRC="${HOME}/.bashrc"
MARKER_BEGIN="# >>> gh-setup (agent-box managed) >>>"
MARKER_END="# <<< gh-setup (agent-box managed) <<<"

if [ -f "$BASHRC" ] && grep -qF "$MARKER_BEGIN" "$BASHRC"; then
  echo "[gh-setup] bashrc block already present in $BASHRC"
else
  mkdir -p "$(dirname "$BASHRC")"
  # Use unquoted heredoc so ${GH_BIN_DIR} / ${HOST_HOME} get expanded NOW,
  # but escape any $ that should stay dynamic in the bashrc itself.
  cat >>"$BASHRC" <<EOF

${MARKER_BEGIN}
# Managed by agent-box/gh-setup.sh — do not edit between these markers.
# Adds gh to PATH and exports GH_TOKEN from ~/.git-credentials at shell start.
export PATH="${GH_BIN_DIR}:\$PATH"
if [ -z "\${GH_TOKEN:-}" ]; then
  for _gh_cred in "\$HOME/.git-credentials" "${HOST_HOME}/.git-credentials"; do
    if [ -f "\$_gh_cred" ]; then
      GH_TOKEN=\$(grep -oP 'https://[^:]+:\K[^@]+' "\$_gh_cred" 2>/dev/null | head -1)
      if [ -n "\$GH_TOKEN" ]; then
        export GH_TOKEN
        break
      fi
    fi
  done
  unset _gh_cred
fi
${MARKER_END}
EOF
  echo "[gh-setup] appended managed block to $BASHRC"
fi

# ---------------------------------------------------------------------------
# 3) Best-effort sanity check (does not fail the script)
# ---------------------------------------------------------------------------
export PATH="${GH_BIN_DIR}:$PATH"
if command -v gh >/dev/null 2>&1; then
  if [ -z "${GH_TOKEN:-}" ] && [ -f "${HOST_HOME}/.git-credentials" ]; then
    GH_TOKEN=$(grep -oP 'https://[^:]+:\K[^@]+' "${HOST_HOME}/.git-credentials" 2>/dev/null | head -1) || true
    [ -n "${GH_TOKEN:-}" ] && export GH_TOKEN
  fi
  if [ -n "${GH_TOKEN:-}" ] && gh auth status >/dev/null 2>&1; then
    echo "[gh-setup] verified: gh auth status OK"
  else
    echo "[gh-setup] gh installed; auth not verified (next interactive shell will set GH_TOKEN from ~/.git-credentials)"
  fi
fi

echo "[gh-setup] finished"
