#!/usr/bin/env bash
# Wire agent-box skills into Cursor (~/.cursor/skills) and Claude Code (~/.claude/skills).
# Uses symlinks so edits under agent-box/skills/ are live everywhere — no copy step.
set -Eeuo pipefail

SKILLS_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURSOR_DIR="${HOME}/.cursor"
CLAUDE_DIR="${HOME}/.claude"

link_skills() {
    local parent="$1"
    local label="$2"
    mkdir -p "$parent"
    local dest="${parent}/skills"
    if [[ -L "$dest" ]]; then
        local target
        target="$(readlink -f "$dest")"
        if [[ "$target" == "$SKILLS_SRC" ]]; then
            echo "OK  $label: $dest -> $SKILLS_SRC"
            return 0
        fi
        rm "$dest"
    elif [[ -e "$dest" ]]; then
        echo "WARN $label: $dest exists and is not a symlink — leaving it alone" >&2
        echo "     Remove or move it manually, then re-run this script." >&2
        return 1
    fi
    ln -sfn "$SKILLS_SRC" "$dest"
    echo "OK  $label: $dest -> $SKILLS_SRC"
}

echo "Syncing agent-box skills from: $SKILLS_SRC"
echo

rc=0
link_skills "$CURSOR_DIR" "Cursor" || rc=1
link_skills "$CLAUDE_DIR" "Claude" || rc=1

echo
count="$(find "$SKILLS_SRC" -mindepth 1 -maxdepth 1 -type d ! -name '.*' ! -name '_*' \
    -exec test -f '{}/SKILL.md' \; -print | wc -l)"
echo "Skills available: $count (each with SKILL.md)"
echo
echo "In Cursor:"
echo "  - Type / in chat to pick a skill (reload window if the list looks stale)"
echo "  - Or ask naturally, e.g. \"run /gpu-status\" or \"follow the validate skill\""
echo
echo "List all skills:"
echo "  python3 \"$SKILLS_SRC/skills-list/list_skills.py\""
echo

exit "$rc"
