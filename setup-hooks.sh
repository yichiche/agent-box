#!/usr/bin/env bash
# Activate local git hooks from .githooks/
git config core.hooksPath .githooks
chmod +x .githooks/*
echo "Git hooks activated from .githooks/"

# Update profile/ submodule to latest tagged release
bash "$(dirname "${BASH_SOURCE[0]}")/update-profile.sh"
