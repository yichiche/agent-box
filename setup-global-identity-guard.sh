#!/usr/bin/env bash
# Install the global identity guard for THIS environment (idempotent).
# Applies to every git repo here (including aiter/sglang you don't maintain) via
# core.hooksPath, and sets the required identity. RE-RUN in each container: the
# global config lives in $HOME/.gitconfig which is ephemeral in containers.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.githooks-global"
git config --global core.hooksPath "$DIR"
git config --global user.name  "jacky.cheng"
git config --global user.email "yichiche@amd.com"
echo "OK: core.hooksPath=$DIR"
echo "OK: global identity = jacky.cheng <yichiche@amd.com>"
echo "NOTE: agent-box keeps its own repo-local .githooks (local overrides global)."
echo "NOTE: re-run this in every fresh container (\$HOME/.gitconfig is ephemeral)."
