#!/usr/bin/env bash
# Initialize submodules and activate local git hooks
git submodule update --init
git config core.hooksPath .githooks
git config submodule.recurse true
chmod +x .githooks/*

# Link skills into Claude Code and Codex (one symlink per skill)
bash "$(cd "$(dirname "$0")" && pwd)/skills/sync-to-agents.sh"

# Symlink workflows into Claude Code config
WORKFLOWS_SRC="$(cd "$(dirname "$0")" && pwd)/workflow"
WORKFLOWS_DST="$HOME/.claude/workflows"
if [ -L "$WORKFLOWS_DST" ]; then
    echo "Workflows symlink already exists: $WORKFLOWS_DST -> $(readlink "$WORKFLOWS_DST")"
elif [ -d "$WORKFLOWS_DST" ]; then
    echo "Warning: $WORKFLOWS_DST is a directory (not a symlink). Back it up and re-run, or manually replace with:"
    echo "  rm -rf $WORKFLOWS_DST && ln -s $WORKFLOWS_SRC $WORKFLOWS_DST"
else
    ln -s "$WORKFLOWS_SRC" "$WORKFLOWS_DST"
    echo "Workflows symlinked: $WORKFLOWS_DST -> $WORKFLOWS_SRC"
fi

echo "Setup complete: submodules initialized, git hooks activated, skills linked, workflows linked."
