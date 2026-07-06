---
description: Distill memory vault into AGENTS.md and CLAUDE.md; sync from Claude native memory. Use weekly, after major workflow changes, or when user says "/memory-consolidate".
---

# Memory Consolidate

Three-tier knowledge model:

| Tier | Location | Loaded when |
|---|---|---|
| 0 Raw | `memory/journal/` | Never auto-loaded |
| 1 Atomic | `memory/{gotchas,models,workflows}/` | On demand via MEMORY.md |
| 2 Procedures | `skills/*/SKILL.md` | Skill invocation |
| 3 Always-on | `AGENTS.md`, `CLAUDE.md` | Every session start |

This skill moves stable facts **up** the tiers (1→3). Never bloat tier 3.

## Step 1: Converge session shards

Normally automatic (a Claude Code **Stop hook** runs it after every session). To
force a catch-up:

```bash
bash "$AGENT_BOX_DIR/memory/bin/memory-sync.sh"          # or --dry-run to preview
```

New Claude/Codex shards land verbatim in `memory/journal/YYYY-MM/` with a row in
`memory/meta/provenance.tsv` (source + time + sha). Review recent journal notes and
promote stable facts **up** into `gotchas/`, `models/`, or `workflows/`.

## Step 2: Refresh script catalog

```bash
ls -1 "$HOME"/run_*.sh
```

Diff against `memory/scripts/INDEX.md`; add missing scripts with one-line purpose (read script header).

## Step 3: Audit model registry

For each active model the user benchmarks (last 30 days of `memory/journal/` notes):
- Server script still exists?
- TP / env flags match script?
- Accuracy threshold still valid?

Update `memory/models/INDEX.md` and detail cards.

## Step 3.5: Draft workflow suggestions

```bash
bash "$AGENT_BOX_DIR/memory/bin/skill-suggest.sh"   # writes review stubs to meta/suggestions/
```

Review `memory/meta/suggestions/*.md`: promote good clusters into a workflow/skill, mark the rest `status: rejected`. See `/skill-suggest`.

## Step 4: Distill into AGENTS.md

**AGENTS.md** = cross-tool, repo-agnostic workspace facts. Add/update ONLY:

- Pointer to `memory/MEMORY.md` as the knowledge index
- Top 5 gotchas (one line each with link path)
- Model→script quick table (compact; detail stays in `memory/models/`)
- Standard dev flows: validate → profile → PR

**Do not** paste full skill bodies or long traces.

## Step 5: Distill into CLAUDE.md

**CLAUDE.md** = Claude Code + SGLang architecture context. Add/update:

- `memory/` vault layout (same as agent-box section)
- Skills list with one-line triggers (sync with actual `skills/` dir)
- Profiling pointer unchanged (`profile/profile.md`)

Keep CLAUDE.md under ~150 lines of agent-relevant content.

## Step 6: Log

Append to `memory/meta/consolidation-log.md`:

```markdown
## YYYY-MM-DD
- Imported N Claude notes
- AGENTS.md: +gotcha bench-cwd, updated model table
- CLAUDE.md: refreshed skills list
```

## Step 7: Symlink (optional, one-time)

Point Claude project memory at vault (run once per machine):

```bash
VAULT="$HOME/agent-box/memory"
PROJ_MEM="$HOME/.claude/projects/-home-yichiche-agent-box/memory"
mkdir -p "$(dirname "$PROJ_MEM")"
if [[ ! -L "$PROJ_MEM" ]]; then
  rm -rf "$PROJ_MEM"
  ln -s "$VAULT" "$PROJ_MEM"
fi
```

## Scheduling

- **Manual:** `/memory-consolidate`
- **Cursor loop:** `/loop 7d /memory-consolidate`
- **After big project:** run immediately

## Quality bar for tier-3 promotion

A fact earns a line in AGENTS.md/CLAUDE.md only if:
1. Wrong without it ≥2 times in past month, OR
2. Needed at **every** session start (repo layout, commit tags), OR
3. Blocks expensive mistakes (cwd shadow, main-branch commits)

Everything else stays in tier 1 with a link from MEMORY.md.
