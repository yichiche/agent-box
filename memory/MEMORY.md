# Agent Memory Index

> Obsidian-style knowledge vault for **all** agents (Cursor, Claude Code, Codex).
> Atomic notes live here; stable procedures live in `skills/`; always-on context is distilled into `AGENTS.md` / `CLAUDE.md` by `/memory-consolidate`.

## Quick routing

| You want to… | Read first |
|---|---|
| Launch a model server | [[models/INDEX]] → pick model card |
| Run benchmark / perf sweep | [[workflows/benchmark]] |
| Validate a code change | [[workflows/validate]] + skill `/validate` |
| Profile & parse trace | [[workflows/profiling]] + skill `/parse-trace` |
| Pick GPUs | skill `/gpu-status` + [[gotchas/gpu-pinning]] |
| Commit / PR | skill `/commit-push-pr` + `skills/_shared/repo-config.md` |

## Model cards

See [[models/INDEX]] for server script, client script, TP, accuracy gate, and env flags per model/hardware combo.

## Workflows

- [[workflows/benchmark]] — concurrency sweep, InferenceX, perf tables
- [[workflows/validate]] — before/after benchmark + accuracy + profile for PRs
- [[workflows/profiling]] — trace capture, `trace_module_analyzer.py`, kernel diff
- [[workflows/accuracy]] — GSM8K, thinking models, thresholds
- [[workflows/remote-bridge]] — Host ↔ container Claude Code (STATUS / INBOX / OUTBOX)

## Gotchas (read before benchmarking)

- [[gotchas/bench-cwd-shadow]] — never launch from `$HOME`; stale `aiter/` + `sglang/` shadows
- [[gotchas/container-bench-flags]] — newer `bench_serving` CLI on rocm images
- [[gotchas/no-edit-running-script]] — don't edit a `.sh` while it's executing

## Script catalog

- [[scripts/INDEX]] — all `$HOME/run_*.sh` with purpose, port, IL/OL defaults

## Journal (raw session captures)

- `journal/` — dated notes; promote stable facts into gotchas/models/workflows

## Maintenance

- **Capture:** end of session → `/memory-capture` (or ask agent to "remember this")
- **Consolidate:** weekly or after major workflow change → `/memory-consolidate` updates `AGENTS.md` + `CLAUDE.md`
- **Import:** `scripts/sync_from_claude_memory.sh` merges `~/.claude/projects/*/memory/` into this vault
- **Cross-container:** [`remote/README.md`](remote/README.md) — STATUS / INBOX / OUTBOX for host ↔ container Claude Code (`/remote-bridge`)
