#!/usr/bin/env bash
# Shared AMD LLM Gateway subscription key for Claude and Codex.
#
# Either file is enough. A missing or empty sibling is created from the one
# that exists. Existing non-empty files are never overwritten with each other.
#   $HOST_HOME/.claude_api_key
#   $HOST_HOME/.codex_api_key
#
# Container bootstrap runs as root and used to leave these mode 600 root:root,
# which the host user cannot read. Files are mode 600 and owned by the owner
# of $HOST_HOME. Root in the container can still read them.
#
# Source this, then: gateway_key_ensure "$HOST_HOME/.claude_api_key"
#   sets GATEWAY_KEY
# Or run: bash gateway-key.sh --ensure

gateway_key_host_owner() {
  stat -c '%u:%g' "$HOST_HOME" 2>/dev/null || echo "$(id -u):$(id -g)"
}

# Read a non-empty key file the current user or passwordless sudo can read.
# Prints the key with surrounding whitespace removed. Nothing is logged.
gateway_key_read() {
  local f="$1" data=""
  if [[ -f "$f" && -r "$f" ]]; then
    data="$(cat "$f")"
  elif [[ -f "$f" ]] && sudo -n test -r "$f" 2>/dev/null; then
    data="$(sudo -n cat "$f")"
  else
    return 1
  fi
  data="${data//$'\r'/}"
  data="${data//$'\n'/}"
  data="${data#"${data%%[![:space:]]*}"}"
  data="${data%"${data##*[![:space:]]}"}"
  [[ -n "$data" ]] || return 1
  printf '%s' "$data"
}

gateway_key_fix_owner() {
  local f="$1" owner cur
  [[ -e "$f" ]] || return 0
  owner="$(gateway_key_host_owner)"
  if [[ "$(id -u)" -eq 0 ]]; then
    cur="$(stat -c '%u:%g' "$f")"
  else
    cur="$(stat -c '%u:%g' "$f" 2>/dev/null || sudo -n stat -c '%u:%g' "$f" 2>/dev/null || true)"
  fi
  if [[ -n "$cur" && "$cur" != "$owner" ]]; then
    if [[ "$(id -u)" -eq 0 ]]; then
      chown "$owner" "$f"
    else
      sudo -n chown "$owner" "$f"
    fi
  fi
  if [[ "$(id -u)" -eq 0 || -O "$f" ]]; then
    chmod 600 "$f"
  else
    sudo -n chmod 600 "$f"
  fi
}

gateway_key_copy() {
  local src="$1" dest="$2"
  if [[ -r "$src" ]]; then
    cp -f "$src" "$dest"
  elif sudo -n test -r "$src" 2>/dev/null; then
    sudo -n cp -f "$src" "$dest"
  else
    echo "[gateway] ERROR: cannot read ${src}" >&2
    return 1
  fi
  gateway_key_fix_owner "$dest"
}

gateway_key_unreadable() {
  local f="$1"
  [[ -e "$f" ]] || return 1
  [[ -r "$f" ]] && return 1
  sudo -n test -r "$f" 2>/dev/null && return 1
  return 0
}

# Ensure $1 exists, is non-empty, mode 600, and owned by the host user.
# The other of .claude_api_key / .codex_api_key is used when $1 is missing.
gateway_key_ensure() {
  local preferred="$1"
  local claude_f="${HOST_HOME}/.claude_api_key"
  local codex_f="${HOST_HOME}/.codex_api_key"
  local other="" key=""

  if [[ "$preferred" == "$claude_f" ]]; then
    other="$codex_f"
  elif [[ "$preferred" == "$codex_f" ]]; then
    other="$claude_f"
  else
    echo "[gateway] ERROR: unsupported key path ${preferred}" >&2
    return 1
  fi

  if key="$(gateway_key_read "$preferred")"; then
    gateway_key_fix_owner "$preferred"
  elif gateway_key_unreadable "$preferred"; then
    echo "[gateway] ERROR: ${preferred} exists but is not readable." >&2
    echo "[gateway] Re-run as root, or: sudo chown \$(stat -c '%u:%g' \"${HOST_HOME}\") ${preferred}" >&2
    return 1
  elif key="$(gateway_key_read "$other")"; then
    gateway_key_copy "$other" "$preferred"
    echo "[gateway] reused AMD LLM Gateway key from ${other} -> ${preferred}" >&2
  elif gateway_key_unreadable "$other"; then
    echo "[gateway] ERROR: ${other} exists but is not readable." >&2
    echo "[gateway] Re-run as root, or: sudo chown \$(stat -c '%u:%g' \"${HOST_HOME}\") ${other}" >&2
    return 1
  elif [[ -t 0 ]]; then
    echo "" >&2
    echo "AMD LLM Gateway key required (one key for Claude and Codex)." >&2
    read -rp "Enter your AMD LLM Gateway API key: " key || true
    if [[ -z "${key}" ]]; then
      echo "[gateway] Error: API key cannot be empty." >&2
      return 1
    fi
    (umask 077; printf '%s\n' "$key" > "$preferred")
    gateway_key_fix_owner "$preferred"
    gateway_key_copy "$preferred" "$other"
    echo "[gateway] API key saved to ${preferred} and ${other}" >&2
  else
    echo "[gateway] ERROR: no AMD LLM Gateway key at ${claude_f} or ${codex_f}." >&2
    echo "[gateway] Save the subscription key to either file (mode 600) and re-run container-dep.sh." >&2
    return 1
  fi

  if ! gateway_key_read "$other" >/dev/null 2>&1; then
    if gateway_key_unreadable "$other"; then
      gateway_key_fix_owner "$other"
    else
      gateway_key_copy "$preferred" "$other"
      echo "[gateway] mirrored AMD LLM Gateway key to ${other}" >&2
    fi
  else
    gateway_key_fix_owner "$other"
  fi

  GATEWAY_KEY="$key"
  echo "[gateway] AMD LLM Gateway key ready (${preferred})" >&2
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  # shellcheck source=env.sh
  source "${SCRIPT_DIR}/env.sh"
  case "${1:-}" in
    --ensure|"") gateway_key_ensure "${HOST_HOME}/.claude_api_key" ;;
    *)
      echo "usage: bash gateway-key.sh --ensure" >&2
      exit 2
      ;;
  esac
fi
