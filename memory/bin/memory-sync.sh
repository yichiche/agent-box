#!/usr/bin/env bash
# memory-sync.sh — converge per-session memory shards into the shared vault.
#
# Because ~/.claude and ~/.codex are bind-mounted into every yichiche
# container, all session shards already land under the host stores. This
# script is therefore a host-local convergence job: it copies new shards
# verbatim into memory/journal/YYYY-MM/ (append-only, source of truth for
# history) and records provenance. It never mutates or overwrites shards,
# and de-dupes by content hash so re-running is a no-op.
#
# Usage:
#   memory-sync.sh                 run the sync
#   memory-sync.sh --dry-run       show what would be imported, change nothing
#   memory-sync.sh --install-hook  register as a Claude Code Stop hook
#   memory-sync.sh --uninstall-hook
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

SETTINGS="${HOST_HOME}/.claude/settings.json"
DRY=0

# ---------------------------------------------------------------- import
sha_seen() { [[ -f "$PROVENANCE" ]] && cut -f5 "$PROVENANCE" | grep -qxF "$1"; }

import_one() {  # $1=src file  $2=source-tag (claude|codex)  $3=project
  local src="$1" tag="$2" proj="$3"
  local sha base month destdir dest rel
  sha="$(mem_sha "$src")"
  if sha_seen "$sha"; then return 0; fi
  base="$(basename "$src")"
  month="$(date -u -r "$src" +%Y-%m 2>/dev/null || date -u +%Y-%m)"
  destdir="$JOURNAL_DIR/$month"
  dest="$destdir/${proj}__${base}"
  # name clash with different content → suffix short sha
  if [[ -e "$dest" ]] && ! cmp -s "$src" "$dest"; then
    dest="$destdir/${proj}__${base%.md}__${sha:0:8}.md"
  fi
  rel="${dest#"$MEM_DIR"/}"
  if [[ "$DRY" == 1 ]]; then
    echo "  + $tag  $proj/$base  ->  $rel"
    return 0
  fi
  mkdir -p "$destdir"
  cp -p "$src" "$dest"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%FT%TZ)" "$tag" "$proj" \
    "$(date -u -r "$src" +%FT%TZ 2>/dev/null || echo '?')" "$sha" "$rel" \
    >> "$PROVENANCE"
  return 0
}

run_sync() {
  mkdir -p "$JOURNAL_DIR" "$META_DIR"
  [[ -f "$PROVENANCE" ]] || printf 'added_utc\tsource\tproject\tsrc_mtime\tsha256\tjournal_path\n' > "$PROVENANCE"

  local before after
  before="$(wc -l < "$PROVENANCE")"

  # --- Claude shards: ~/.claude/projects/<project>/memory/*.md ---------
  if [[ -d "$CLAUDE_PROJECTS" ]]; then
    while IFS= read -r memdir; do
      # skip project memory dirs that resolve into the vault (symlink loop)
      if mem_is_vault_path "$memdir"; then
        [[ "$DRY" == 1 ]] && echo "  (skip vault-linked: $memdir)"
        continue
      fi
      local proj; proj="$(basename "$(dirname "$memdir")")"
      while IFS= read -r -d '' f; do
        import_one "$f" claude "$proj"
      done < <(find "$memdir" -maxdepth 1 -name '*.md' ! -name 'MEMORY.md' -print0 2>/dev/null)
    done < <(find -L "$CLAUDE_PROJECTS" -mindepth 2 -maxdepth 2 -type d -name memory 2>/dev/null)
  fi

  # --- Codex shards: ~/.codex/memories/**/*.md ------------------------
  if [[ -d "$CODEX_MEM" ]]; then
    while IFS= read -r -d '' f; do
      mem_is_vault_path "$f" && continue
      import_one "$f" codex "codex" || true
    done < <(find -L "$CODEX_MEM" -name '*.md' ! -name 'MEMORY.md' -print0 2>/dev/null)
  fi

  after="$(wc -l < "$PROVENANCE")"
  local n=$(( after - before ))
  if [[ "$DRY" == 1 ]]; then
    echo "[sync] dry-run complete"
  else
    mem_log "sync: imported $n new note(s) into journal"
    echo "[sync] imported $n new note(s) → $JOURNAL_DIR (log: $SYNC_LOG)"
  fi
}

# ------------------------------------------------------------- hook mgmt
manage_hook() {  # $1 = install|uninstall
  local action="$1" cmd
  cmd="bash ${MEM_DIR}/bin/memory-sync.sh >/dev/null 2>&1 || true"
  python3 - "$SETTINGS" "$action" "$cmd" <<'PY'
import json, sys, pathlib
path, action, cmd = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path(path)
data = json.loads(p.read_text()) if p.exists() else {}
hooks = data.setdefault("hooks", {})
stop = hooks.setdefault("Stop", [])
def has(group): return any(h.get("command") == cmd for h in group.get("hooks", []))
present = any(has(g) for g in stop)
if action == "install":
    if not present:
        stop.append({"hooks": [{"type": "command", "command": cmd}]})
        print("[hook] installed Stop hook")
    else:
        print("[hook] already installed")
elif action == "uninstall":
    for g in stop:
        g["hooks"] = [h for h in g.get("hooks", []) if h.get("command") != cmd]
    data["hooks"]["Stop"] = [g for g in stop if g.get("hooks")]
    if not data["hooks"]["Stop"]:
        del data["hooks"]["Stop"]
    if not data["hooks"]:
        del data["hooks"]
    print("[hook] removed Stop hook")
p.write_text(json.dumps(data, indent=2) + "\n")
PY
}

case "${1:-}" in
  --dry-run)        DRY=1; run_sync ;;
  --install-hook)   manage_hook install ;;
  --uninstall-hook) manage_hook uninstall ;;
  ""|--run)         run_sync ;;
  -h|--help) sed -n '2,14p' "$0" ;;
  *) echo "unknown arg: $1" >&2; exit 2 ;;
esac
