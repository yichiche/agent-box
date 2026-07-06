#!/usr/bin/env bash
# lib.sh — shared helpers for agent-box memory automation.
# Source this; it defines paths, role detection, and small utilities.
# Safe to source from host or from inside a container (mounts make the
# vault path identical in both: /home/yichiche/agent-box/memory).

# --- paths -------------------------------------------------------------
# BIN dir = memory/bin ; MEM = memory ; AGENT_BOX = repo root
_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEM_DIR="$(cd "$_LIB_DIR/.." && pwd)"
AGENT_BOX_DIR="$(cd "$MEM_DIR/.." && pwd)"
# shellcheck source=/dev/null
[[ -f "$AGENT_BOX_DIR/env.sh" ]] && source "$AGENT_BOX_DIR/env.sh"
HOST_HOME="${AGENT_BOX_HOST_HOME:-$(dirname "$AGENT_BOX_DIR")}"

JOURNAL_DIR="$MEM_DIR/journal"
META_DIR="$MEM_DIR/meta"
PROVENANCE="$META_DIR/provenance.tsv"
SYNC_LOG="$META_DIR/sync.log"

CLAUDE_PROJECTS="${HOST_HOME}/.claude/projects"
CODEX_MEM="${HOST_HOME}/.codex/memories"

# --- role detection ----------------------------------------------------
# Prints "container" if running inside a Docker container, else "host".
mem_role() {
  if [[ -f /.dockerenv ]] || grep -qa 'docker\|kubepods' /proc/1/cgroup 2>/dev/null; then
    echo container
  else
    echo host
  fi
}

# --- vault-loop guard --------------------------------------------------
# A project's memory dir may be a symlink back into the vault (the
# agent-box project does exactly this). Importing from it would re-ingest
# the whole vault forever. Return 0 (skip) if $1 resolves under the vault.
mem_is_vault_path() {
  local real; real="$(realpath -m "$1" 2>/dev/null)" || return 1
  [[ "$real" == "$MEM_DIR" || "$real" == "$MEM_DIR/"* ]]
}

# --- logging -----------------------------------------------------------
mem_log() {
  mkdir -p "$META_DIR"
  printf '%s [%s] %s\n' "$(date -u +%FT%TZ)" "$(mem_role)" "$*" >> "$SYNC_LOG"
}

# sha256 of a file, first field only
mem_sha() { sha256sum "$1" | awk '{print $1}'; }
