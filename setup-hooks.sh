#!/usr/bin/env bash
# Initialize submodules and activate local git hooks
git submodule update --init
git config core.hooksPath .githooks
git config submodule.recurse true
chmod +x .githooks/*
echo "Setup complete: submodules initialized, git hooks activated, submodule.recurse enabled."
