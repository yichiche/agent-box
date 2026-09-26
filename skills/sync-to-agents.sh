#!/usr/bin/env bash
# Wire agent-box skills into Claude Code (~/.claude/skills) and Codex
# (${CODEX_HOME:-~/.codex}/skills) with one symlink per skill, so edits under
# agent-box/skills/ are live everywhere — no copy step.
#
# Per-skill links (instead of linking the whole skills/ dir) leave the tools'
# own entries alone: ~/.claude/skills/synced (claude.ai) and
# ~/.codex/skills/.system (Codex built-ins).
#
#   bash sync-to-agents.sh            link / refresh
#   bash sync-to-agents.sh --remove   remove links that point into this dir
set -Eeuo pipefail

SKILLS_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGETS=(
    "Claude:${HOME}/.claude/skills"
    "Codex:${CODEX_HOME:-${HOME}/.codex}/skills"
)
REMOVE=0
[[ "${1:-}" == "--remove" ]] && REMOVE=1

# Skills = top-level dirs with a SKILL.md; skip _shared and hidden dirs.
skills=()
for d in "$SKILLS_SRC"/*/; do
    name="$(basename "$d")"
    [[ "$name" == _* || "$name" == .* ]] && continue
    [[ -f "$d/SKILL.md" ]] || continue
    skills+=("$name")
done

rc=0
for entry in "${TARGETS[@]}"; do
    label="${entry%%:*}"
    dest_dir="${entry#*:}"
    mkdir -p "$dest_dir"
    linked=0 skipped=0 pruned=0

    # Drop links we own that are stale (skill deleted/renamed) or all of them on --remove.
    for link in "$dest_dir"/*; do
        [[ -L "$link" ]] || continue
        target="$(readlink "$link")"
        [[ "$target" == "$SKILLS_SRC"/* ]] || continue
        if (( REMOVE )) || [[ ! -f "$target/SKILL.md" ]]; then
            rm "$link"; pruned=$((pruned + 1))
        fi
    done

    if (( ! REMOVE )); then
        for name in "${skills[@]}"; do
            link="$dest_dir/$name"
            if [[ -e "$link" && ! -L "$link" ]]; then
                echo "WARN $label: $link exists and is not a symlink — leaving it alone" >&2
                skipped=$((skipped + 1)); rc=1
                continue
            fi
            ln -sfn "$SKILLS_SRC/$name" "$link"
            linked=$((linked + 1))
        done
    fi
    echo "OK  $label: $dest_dir  linked=$linked pruned=$pruned skipped=$skipped"
done

(( REMOVE )) || echo "Skills available: ${#skills[@]} (restart claude / codex to pick up changes)"
exit "$rc"
