---
description: Scan the memory vault for recurring themes that no skill/workflow/gotcha covers yet, and draft review stubs proposing a workflow/skill/gotcha. Detect → draft → you approve; never auto-creates a skill. Use weekly or when the journal has grown.
category: meta
---

# /skill-suggest — feed memory back into workflows

Turns accumulated `memory/journal/` + `gotchas/` notes into **review stubs** for new
workflows/skills. It only drafts; you decide what becomes real.

## Run

```bash
bash ~/agent-box/memory/bin/skill-suggest.sh --dry-run   # ranked candidate themes, writes nothing
bash ~/agent-box/memory/bin/skill-suggest.sh             # write stubs to memory/meta/suggestions/
bash ~/agent-box/memory/bin/skill-suggest.sh --min 5     # stricter (theme must span >=5 notes)
```

## What it does

1. Tokenises journal + gotcha notes (filename + prose, frontmatter stripped).
2. Counts themes spanning ≥ `--min` distinct notes.
3. Drops themes already owned by a skill, workflow, gotcha, or model card.
4. Writes `memory/meta/suggestions/<theme>.md` linking the contributing notes and a
   checkbox action list (new gotcha / workflow / skill / reject). Existing stubs are
   never overwritten.

## Your part (the approval step)

Open a stub, then either:
- promote the stable facts with `/memory-capture` into `gotchas/`, `workflows/`, or a new `skills/<name>/SKILL.md`, and set `status: accepted`; or
- set `status: rejected` if it's noise.

Suggestions are drafts, not decisions — nothing here changes agent behaviour until you promote it.
