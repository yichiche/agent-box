#!/usr/bin/env bash
# gh-setup: ensure `gh` is installed (~/bin) and GH_TOKEN is exported from ~/.git-credentials.
#
# Designed to be SOURCED, not executed:
#     source ~/agent-box/skills/gh-setup/scripts/setup_gh.sh
#
# Idempotent: safe to source multiple times.
# Requires: curl, tar, grep -P, write access to $HOME/bin, and a valid PAT in ~/.git-credentials.

# Pinned gh version known to work with inferencex-table.
GH_VERSION="${GH_VERSION:-2.62.0}"
GH_ARCH="linux_amd64"
GH_BIN_DIR="$HOME/bin"
GH_BIN="$GH_BIN_DIR/gh"

_gh_log() { printf '[gh-setup] %s\n' "$*"; }
_gh_err() { printf '[gh-setup] ERROR: %s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# Step 1: install gh if missing
# ---------------------------------------------------------------------------

# Make sure ~/bin is on PATH first so an existing ~/bin/gh is discoverable.
case ":$PATH:" in
  *":$GH_BIN_DIR:"*) ;;
  *) export PATH="$GH_BIN_DIR:$PATH" ;;
esac

if command -v gh >/dev/null 2>&1; then
  _gh_log "gh already on PATH ($(command -v gh))"
elif [ -x "$GH_BIN" ]; then
  _gh_log "gh already at $GH_BIN"
else
  _gh_log "Installing gh v${GH_VERSION} to $GH_BIN_DIR (no sudo)"
  mkdir -p "$GH_BIN_DIR" || { _gh_err "could not create $GH_BIN_DIR"; return 1 2>/dev/null || exit 1; }

  _gh_tmp="$(mktemp -d)"
  _gh_url="https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_${GH_ARCH}.tar.gz"

  if ! curl -fsSL "$_gh_url" -o "$_gh_tmp/gh.tar.gz"; then
    _gh_err "download failed: $_gh_url"
    rm -rf "$_gh_tmp"
    return 1 2>/dev/null || exit 1
  fi

  if ! tar -xzf "$_gh_tmp/gh.tar.gz" -C "$_gh_tmp"; then
    _gh_err "tar extraction failed"
    rm -rf "$_gh_tmp"
    return 1 2>/dev/null || exit 1
  fi

  cp "$_gh_tmp/gh_${GH_VERSION}_${GH_ARCH}/bin/gh" "$GH_BIN" \
    && chmod +x "$GH_BIN" \
    || { _gh_err "install copy failed"; rm -rf "$_gh_tmp"; return 1 2>/dev/null || exit 1; }

  rm -rf "$_gh_tmp"
  _gh_log "gh installed at $GH_BIN"
fi

# ---------------------------------------------------------------------------
# Step 2: export GH_TOKEN from ~/.git-credentials (if not already set)
# ---------------------------------------------------------------------------

if [ -n "$GH_TOKEN" ]; then
  _gh_log "GH_TOKEN already set in env (length ${#GH_TOKEN})"
elif [ -n "$GITHUB_TOKEN" ]; then
  export GH_TOKEN="$GITHUB_TOKEN"
  _gh_log "GH_TOKEN set from GITHUB_TOKEN (length ${#GH_TOKEN})"
elif [ -f "$HOME/.git-credentials" ]; then
  _gh_token="$(grep -oP 'https://[^:]+:\K[^@]+' "$HOME/.git-credentials" 2>/dev/null | head -1)"
  if [ -n "$_gh_token" ]; then
    export GH_TOKEN="$_gh_token"
    _gh_log "GH_TOKEN set from ~/.git-credentials (length ${#GH_TOKEN})"
  else
    _gh_err "could not parse a PAT from ~/.git-credentials"
  fi
  unset _gh_token
else
  _gh_err "no GH_TOKEN/GITHUB_TOKEN env var and no ~/.git-credentials — gh will not authenticate"
fi

# ---------------------------------------------------------------------------
# Step 3: best-effort verification
# ---------------------------------------------------------------------------

if command -v gh >/dev/null 2>&1; then
  _gh_log "$(gh --version | head -1)"
  if [ -n "$GH_TOKEN" ]; then
    if gh auth status >/dev/null 2>&1; then
      _gh_log "gh auth status OK"
    else
      _gh_log "gh auth status reported issues (PAT may be expired or missing scopes: repo, actions:read)"
    fi
  fi
else
  _gh_err "gh still not on PATH after setup"
fi

unset _gh_log _gh_err _gh_tmp _gh_url
