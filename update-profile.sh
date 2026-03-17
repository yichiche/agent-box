#!/usr/bin/env bash
# Update profile/ submodule to the latest tagged release of torch-profiler-parser.
# Usage: bash update-profile.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMODULE_DIR="$SCRIPT_DIR/profile"

# Initialize submodule if not yet populated
if [ ! -f "$SUBMODULE_DIR/.git" ] && [ ! -d "$SUBMODULE_DIR/.git" ]; then
    git -C "$SCRIPT_DIR" submodule update --init profile
fi

# Fetch latest tags
git -C "$SUBMODULE_DIR" fetch --tags origin

# Find the latest version tag (vX.Y or vX.Y.Z, sorted by version)
LATEST_TAG=$(git -C "$SUBMODULE_DIR" tag -l 'v*' --sort=-version:refname | head -1)

if [ -z "$LATEST_TAG" ]; then
    echo "No version tags found in torch-profiler-parser. Staying on current commit."
    exit 0
fi

CURRENT=$(git -C "$SUBMODULE_DIR" describe --tags --exact-match 2>/dev/null || echo "none")

if [ "$CURRENT" = "$LATEST_TAG" ]; then
    echo "profile/ already at latest tag: $LATEST_TAG"
else
    git -C "$SUBMODULE_DIR" checkout "$LATEST_TAG"
    echo "profile/ updated to $LATEST_TAG"
fi
