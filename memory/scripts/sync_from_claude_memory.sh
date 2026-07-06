#!/usr/bin/env bash
# DEPRECATED — replaced by memory/bin/memory-sync.sh (2026-07-05 redesign).
# The old flat imported/ dump is gone; shards now converge into journal/YYYY-MM/
# with provenance, and this runs automatically via a Claude Code Stop hook.
# This shim forwards so any old muscle-memory / cron still works.
exec bash "$(cd "$(dirname "$0")/../bin" && pwd)/memory-sync.sh" "$@"
