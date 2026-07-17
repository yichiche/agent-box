---
description: Capture session learnings into agent-box/memory vault. Use at end of session, when user says "remember this", or after discovering a gotcha/workflow/model config.
category: meta
---

# Memory Capture

Promote **one atomic fact** from the current session into the Obsidian-style vault at `memory/`.

## When to run

- User says "記住", "remember", "下次別再…", "這個 gotcha"
- After fixing a non-obvious bug (cwd shadow, CLI flag drift, wrong GPU index)
- After validating a new model config combo (server+client+threshold)

## Where to write

| Content type | Path |
|---|---|
| Model-specific config | `memory/models/<slug>.md` + row in `memory/models/INDEX.md` |
| Workflow change | `memory/workflows/<name>.md` |
| Gotcha / pitfall | `memory/gotchas/<slug>.md` |
| New run script | row in `memory/scripts/INDEX.md` |
| Raw / uncertain | `memory/journal/YYYY-MM-DD.md` (bullet, link session topic) |

## Note format (Obsidian-compatible)

```markdown
---
type: gotcha | model | workflow | feedback
aliases: [user shorthand]
---

# Title

**Problem:** …
**Fix:** …
**Verify:** …

Related: [[other-note]], skill `/validate`
```

Use `[[wikilinks]]` to related notes. Keep each note **one concern** (~30–80 lines max).

## Do NOT

- Dump entire chat logs into memory
- Duplicate content already in a skill — link to the skill instead
- Store secrets, tokens, or credentials

## After capture

1. Add link to `memory/MEMORY.md` index if it's a top-level gotcha or model card
2. Tell user: "Captured to `memory/gotchas/foo.md`. Run `/memory-consolidate` when ready to promote to AGENTS.md."

## Cross-model

This vault is plain markdown — Cursor, Claude Code, and Codex all read the same files. Claude's native memory (`~/.claude/projects/*/memory/`) is a **cache**; this vault is **source of truth**.
