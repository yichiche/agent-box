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
- [[workflows/remote-bridge]] — Host ↔ container agents (file bus + `bridge.sh exec`)

## Gotchas (read before benchmarking)

- [[gotchas/bench-cwd-shadow]] — never launch from `$HOME`; stale `aiter/` + `sglang/` shadows
- [[gotchas/container-bench-flags]] — newer `bench_serving` CLI on rocm images
- [[gotchas/no-edit-running-script]] — don't edit a `.sh` while it's executing

## Script catalog

- [[scripts/INDEX]] — all `$HOME/run_*.sh` with purpose, port, IL/OL defaults

## Journal (raw session captures)

- `journal/YYYY-MM/` — verbatim session shards, auto-imported. Append-only history; **not** hand-edited. Provenance (source + time + sha) in `meta/provenance.tsv`.
- Promote stable facts **up** into `gotchas/` / `models/` / `workflows/`.

## How memory flows (architecture)

`~/.claude` and `~/.codex` are bind-mounted into every yichiche container, so all
session shards already converge on the host. Convergence is therefore host-local —
no message bus, no `docker exec` needed for memory.

```
session shards                     curated vault              always-on
~/.claude/projects/*/memory/  ─┐   gotchas/ models/           AGENTS.md
~/.codex/memories/            ─┼─▶ journal/YYYY-MM/  ─promote▶ workflows/ ─distill▶ CLAUDE.md
                               │      (raw, provenance)        (curated)
                          memory-sync.sh                    /memory-consolidate
                        (Stop hook, auto)                   /skill-suggest (drafts)
```

- **`bin/memory-sync.sh`** — converge shards → `journal/`, dedup by sha, log to `meta/sync.log`. Runs automatically (Claude Code **Stop hook**). Disable: `bin/memory-sync.sh --uninstall-hook`. Preview: `--dry-run`.
- **`bin/skill-suggest.sh`** (`/skill-suggest`) — detect recurring themes → draft review stubs in `meta/suggestions/`. Detect → draft → you approve.
- **`bridge/bridge.sh`** (`/remote-bridge`) — host↔container coordination: file bus (`STATUS`/`INBOX`/`OUTBOX`) + `exec` (allowlisted `docker exec` running `claude`/`codex` headless in your own containers).

## Maintenance

- **Capture:** end of session → `/memory-capture` (or "remember this").
- **Converge:** automatic via Stop hook; manual catch-up `bin/memory-sync.sh`.
- **Consolidate:** weekly → `/memory-consolidate` promotes journal facts and refreshes `AGENTS.md` + `CLAUDE.md`.
- **Suggest:** `/skill-suggest` drafts workflow improvements from the journal.
- **Cross-container:** [`bridge/README.md`](bridge/README.md) — `/remote-bridge`.
