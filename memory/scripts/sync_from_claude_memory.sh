#!/usr/bin/env bash
# Merge ~/.claude/projects/*/memory/*.md into agent-box/memory/imported/
# Does not overwrite existing vault notes; writes side-by-side for manual merge.
set -euo pipefail

AGENT_BOX_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
IMPORT_DIR="$AGENT_BOX_DIR/memory/imported"
CLAUDE_MEM_ROOT="${HOME}/.claude/projects"

mkdir -p "$IMPORT_DIR"

count=0
while IFS= read -r -d '' src; do
  base="$(basename "$src")"
  project="$(basename "$(dirname "$(dirname "$src")")")"
  dest="$IMPORT_DIR/${project}__${base}"
  if [[ -f "$dest" ]]; then
    if cmp -s "$src" "$dest"; then
      continue
    fi
    dest="$IMPORT_DIR/${project}__${base%.md}__$(date +%Y%m%d_%H%M%S).md"
  fi
  cp "$src" "$dest"
  ((count++)) || true
done < <(find "$CLAUDE_MEM_ROOT" -path '*/memory/*.md' ! -name 'MEMORY.md' -print0 2>/dev/null)

echo "[sync] imported $count note(s) → $IMPORT_DIR"
echo "[sync] review and promote stable facts into memory/gotchas|models|workflows/"
