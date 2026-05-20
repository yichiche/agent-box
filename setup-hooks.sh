#!/usr/bin/env bash
# Initialize submodules and activate local git hooks
git submodule update --init
git config core.hooksPath .githooks
git config submodule.recurse true
chmod +x .githooks/*

# Symlink skills into Claude Code config
SKILLS_SRC="$(cd "$(dirname "$0")" && pwd)/skills"
SKILLS_DST="$HOME/.claude/skills"
mkdir -p "$HOME/.claude"
if [ -L "$SKILLS_DST" ]; then
    echo "Skills symlink already exists: $SKILLS_DST -> $(readlink "$SKILLS_DST")"
elif [ -d "$SKILLS_DST" ]; then
    echo "Warning: $SKILLS_DST is a directory (not a symlink). Back it up and re-run, or manually replace with:"
    echo "  rm -rf $SKILLS_DST && ln -s $SKILLS_SRC $SKILLS_DST"
else
    ln -s "$SKILLS_SRC" "$SKILLS_DST"
    echo "Skills symlinked: $SKILLS_DST -> $SKILLS_SRC"
fi

echo "Setup complete: submodules initialized, git hooks activated, skills linked."
