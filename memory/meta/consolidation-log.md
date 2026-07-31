## 2026-07-17 — Qwen3.5 aiter July findings promoted

- Sync: 0 new shards (Stop hook current). Reviewed July journal (a16w4 accuracy, GDN decode, MoE candidate queue).
- **candidates/gdn-decode-ilp4-bf16state.md + README.md:** re-derived from the 2026-07-14 microbench. FlyDSL/ILP4 give ~no win at the REAL head config (the "5× @ conc64" was a num_v=8-vs-32 CSV artifact); the real lever is **bf16 SSM state = 1.7× @ b64 on the existing Triton kernel** (conf 0.85, gated on OL1k accuracy). GDN priority 7.1→3.8; queue re-ranked (MoE sorting now #1).
- **models/qwen35-mxfp4-mi355.md:** added the a16w4 INTERLEAVE second-degrader finding (~doubles GSM8K invalid 0.21→0.52; NOT fp8-branch/truncation/shared-fusion — all dead ends; serve a4w4) to Tier-2.
- **New gotcha** `flydsl-tuned-csv-head-mismatch.md`: phantom speedup from tuned-CSV head mismatch → re-microbench at deployed config÷TP before it sets Amdahl priority. Added to MEMORY.md.
- **scripts/INDEX.md:** +`run_wan2.2_T2V_deterministic.sh`.
- **skill-suggest:** flydsl theme (33 notes) → accepted, resolved by the new gotcha + existing aiter gotchas (no standalone /flydsl skill).
- AGENTS.md / CLAUDE.md: no change — all promotions were model/candidate-specific tier-1 (kept tier 3 lean).

## 2026-07-05 — Memory system redesign (Phase 1: auto-convergence)

- Verified `~/.claude` + `~/.codex` are bind-mounted into all yichiche containers → session shards already converge on the host; sync is a host-local job, no docker exec needed for memory.
- Added `memory/bin/lib.sh` (role detection, paths, vault-loop guard) + `memory/bin/memory-sync.sh` (verbatim shard→`journal/YYYY-MM/`, sha dedup, `meta/provenance.tsv`, `meta/sync.log`, `--dry-run`, `--install-hook`).
- Guard: skips project memory dirs that symlink back into the vault (the agent-box project does this — was a latent infinite-import loop).
- Migrated `imported/` (36) → `journal/` and backfilled the stalled 6/28–6/30 shards. journal now 41 notes; each traceable via provenance.tsv. Retired flat `imported/`.
- Installed `memory-sync` as a Claude Code **Stop hook** (in `setting.json`); disable with `memory-sync.sh --uninstall-hook`.

## 2026-07-05 — Memory system redesign (Phase 4: docs + simplify)

- Rewrote `MEMORY.md` with the dataflow diagram and entry points; updated `CLAUDE.md` (layout, memory section, skills) and `AGENTS.md` (memory + cross-container).
- Updated `/memory-consolidate` to use `memory-sync.sh` + journal (was `sync_from_claude_memory.sh` + `imported/`) and added a `/skill-suggest` step; refreshed `scripts/INDEX.md`.
- Retired `scripts/sync_from_claude_memory.sh` to a thin deprecation shim → `bin/memory-sync.sh`.
- Net simplification: flat `imported/` (36) → dated `journal/` with provenance; `remote/` (2 scripts + phantom watcher) → one `bridge.sh`; curated entry count unchanged (models/workflows/gotchas), raw history now traceable.

## 2026-07-05 — Memory system redesign (Phase 3: workflow feedback)

- Added `memory/bin/skill-suggest.sh` + skill `/skill-suggest`: tokenises journal+gotcha notes (frontmatter stripped), finds themes spanning ≥N notes that no skill/workflow/gotcha/model already owns, and drafts review stubs to `meta/suggestions/`.
- Detect → draft → approve; never auto-creates a skill. First run drafted 6 stubs; the `moe` stub clusters 13 real MoE-tuning notes into a candidate workflow.

## 2026-07-05 — Memory system redesign (Phase 2: unified bridge)

- Renamed `memory/remote/` → `memory/bridge/`; collapsed `remote_msg.sh` + `remote_snapshot.sh` behind one entry `bridge.sh` (`list|status|msg|exec|watch`).
- Added `exec`: host `docker exec`s into one of **its own** containers (allowlisted strictly by the `/home/yichiche` bind-mount — refuses other users' containers on this shared host) and runs `claude -p`/`codex exec` headless; reply prints inline + mirrors to `OUTBOX.md`, logged to `bridge.log`.
- Proven end-to-end: `bridge.sh exec <jacky-container> "…"` → container Claude replied with hostname + branch. Non-owned container (thomas/…) correctly refused.
- Updated `/remote-bridge` skill + `workflows/remote-bridge.md`.

## 2026-07-05 — Memory system redesign (Phase 1: auto-convergence)

- Added `memory/remote/` (STATUS, INBOX, OUTBOX, snapshot scripts)
- Added skill `/remote-bridge`
- Documented native RC vs file-bus dual layer

## 2026-06-27 — Initial vault scaffold

- Created `memory/` Obsidian-style vault (models, workflows, gotchas, scripts)
- Imported 36 notes from `~/.claude/projects/*/memory/` → `memory/imported/`
- Updated AGENTS.md with top gotchas + memory pointer
- Updated CLAUDE.md with memory layout + expanded skills list
- Added skills: `/memory-capture`, `/memory-consolidate`
