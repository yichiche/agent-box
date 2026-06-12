#!/usr/bin/env bash
# agent-box/gh-setup.sh — runs once on Docker container launch (chained from run_docker.sh).
#
# What it does (idempotent):
#   1. Installs `gh` (pinned version) to ${HOST_HOME}/bin/gh so the binary is
#      persistent across container destroy / re-create — host-mounted, downloaded once.
#   2. Logs in to github.com with HTTPS git protocol via non-interactive
#      `gh auth login --with-token`, using the PAT from ~/.git-credentials
#      (or ${HOST_HOME}/.git-credentials inside Docker where $HOME=/root).
#   3. Writes a managed block at the *start* of ~/.bashrc (prepend) so
#      `source ~/.bashrc` in non-interactive bash (e.g. docker bash -c init) still
#      adds PATH before Debian's "if not interactive, return" guard.
#      Does NOT export GH_TOKEN — that env var blocks `gh auth login` and stored creds.
#
# Usage:
#   - Container init (run_docker.sh): bash .../gh-setup.sh && . /root/.bashrc && ...
#   - Existing shell (immediate gh): source .../gh-setup.sh
#
# Running with `bash` only updates ~/.bashrc and future shells; `source` also
# puts gh on PATH in the current shell.

set -euo pipefail

# True when sourced (e.g. `source gh-setup.sh`), false when executed as a subprocess.
_GH_SETUP_SOURCED=false
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  _GH_SETUP_SOURCED=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
# env.sh defines HOST_HOME (e.g. /home/yichiche)

GH_VERSION="${GH_VERSION:-2.62.0}"
GH_BIN_DIR="${HOST_HOME}/bin"
GH_BIN="${GH_BIN_DIR}/gh"
GH_URL="https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.tar.gz"
GH_HOST="${GH_HOST:-github.com}"
# Use a host-owned dir (not $HOME/.config/gh) so Docker root shells can persist creds.
GH_CONFIG_DIR="${GH_CONFIG_DIR:-${HOST_HOME}/.gh}"

echo "[gh-setup] start"

_gh_read_token() {
  local _gh_cred _gh_token=""
  for _gh_cred in "${HOME}/.git-credentials" "${HOST_HOME}/.git-credentials"; do
    if [ -f "$_gh_cred" ]; then
      _gh_token=$(grep -oP 'https://[^:]+:\K[^@]+' "$_gh_cred" 2>/dev/null | head -1) || true
      if [ -n "$_gh_token" ]; then
        printf '%s' "$_gh_token"
        unset _gh_cred _gh_token
        return 0
      fi
    fi
  done
  unset _gh_cred _gh_token
  return 1
}

_gh_apply_shell_env() {
  case ":$PATH:" in
    *":${GH_BIN_DIR}:"*) ;;
    *) export PATH="${GH_BIN_DIR}:$PATH" ;;
  esac
  export GH_CONFIG_DIR
  # GH_TOKEN overrides stored gh credentials and blocks `gh auth login`.
  unset GH_TOKEN
}

_gh_auth_login() {
  local _gh_token
  if ! _gh_token="$(_gh_read_token)"; then
    echo "[gh-setup] no PAT in ~/.git-credentials — skip gh auth login"
    return 0
  fi

  unset GH_TOKEN
  mkdir -p "$GH_CONFIG_DIR"

  if gh auth status -h "$GH_HOST" >/dev/null 2>&1; then
    echo "[gh-setup] already logged in to ${GH_HOST} (stored credentials)"
    unset _gh_token
    return 0
  fi

  echo "[gh-setup] gh auth login --hostname ${GH_HOST} --git-protocol https --with-token"
  printf '%s\n' "$_gh_token" | gh auth login \
    --hostname "$GH_HOST" \
    --git-protocol https \
    --with-token
  unset _gh_token
}

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
# 2) Managed PATH block at start of ~/.bashrc (replaces legacy GH_TOKEN block)
#    Inside docker, $HOME is typically /root; on host it's the user home.
# ---------------------------------------------------------------------------
BASHRC="${HOME}/.bashrc"
MARKER_BEGIN="# >>> gh-setup (agent-box managed) >>>"
MARKER_END="# <<< gh-setup (agent-box managed) <<<"

_gh_remove_bashrc_block() {
  if [ -f "$BASHRC" ] && grep -qF "$MARKER_BEGIN" "$BASHRC"; then
    local tmp="${BASHRC}.gh-setup.tmp"
    awk -v begin="$MARKER_BEGIN" -v end="$MARKER_END" '
      $0 == begin { skip=1; next }
      $0 == end { skip=0; next }
      !skip { print }
    ' "$BASHRC" >"$tmp" && mv "$tmp" "$BASHRC"
  fi
}

_gh_write_bashrc_block() {
  _gh_remove_bashrc_block
  mkdir -p "$(dirname "$BASHRC")"
  [ -f "$BASHRC" ] || touch "$BASHRC"
  local tmp="${BASHRC}.gh-setup.tmp"
  {
    echo "${MARKER_BEGIN}"
    echo "# Managed by agent-box/gh-setup.sh — do not edit between these markers."
    echo "# Prepended so \`source ~/.bashrc\` applies PATH even in non-interactive bash (e.g. docker init)."
    echo "# Auth is stored via \`gh auth login\` (HTTPS), not GH_TOKEN."
    echo "export PATH=\"${GH_BIN_DIR}:\$PATH\""
    echo "export GH_CONFIG_DIR=\"${GH_CONFIG_DIR}\""
    echo "unset GH_TOKEN"
    echo "${MARKER_END}"
    echo ""
    cat "$BASHRC"
  } >"$tmp" && mv "$tmp" "$BASHRC"
  echo "[gh-setup] wrote managed block to start of $BASHRC"
}

_gh_write_bashrc_block

# ---------------------------------------------------------------------------
# 3) PATH + gh auth login, then best-effort sanity check (non-fatal)
# ---------------------------------------------------------------------------
_gh_apply_shell_env

if command -v gh >/dev/null 2>&1; then
  _gh_auth_login || echo "[gh-setup] gh auth login failed (non-fatal)"
  if gh auth status -h "$GH_HOST" >/dev/null 2>&1; then
    echo "[gh-setup] verified: gh auth status OK (${GH_HOST}, git protocol https)"
  else
    echo "[gh-setup] gh installed; auth not verified"
  fi
else
  echo "[gh-setup] WARNING: gh not on PATH after setup (unexpected)"
fi

if "$_GH_SETUP_SOURCED"; then
  echo "[gh-setup] finished (current shell ready — try: gh auth status)"
else
  echo "[gh-setup] finished"
  echo "[gh-setup] NOTE: running as a subprocess does not update your current shell."
  echo "[gh-setup]       To use gh here now, run one of:"
  echo "[gh-setup]         source \"${BASHRC}\""
  echo "[gh-setup]         source \"${SCRIPT_DIR}/gh-setup.sh\""
fi
