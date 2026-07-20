---
name: skills-list
description: "Print every local skill and its usage as a table, grouped by category. Use when the user says '/skills-list', '/skills', asks 'what skills do I have', 'list my slash commands', 'what can I run', or wants a cheat-sheet of available skills (there is no shell autocomplete for slash commands)."
category: meta
---

# skills-list — table of every local skill + its usage

Prints all skills under the skills root (`agent-box/skills/<name>/SKILL.md`) as a
compact table: `/<name>` + one-line usage, grouped by `category`. It reads each
skill's own frontmatter, so it stays correct as skills are added/removed — nothing
to hand-maintain. Purpose: a quick cheat-sheet when the CLI has no slash-command
autocomplete.

## How to run

```bash
python3 "$HOME/agent-box/skills/skills-list/list_skills.py"
```

Show the table to the user. Relay it as a Markdown table when that reads better in
the client.

### Flags

| Flag | Effect |
|---|---|
| (none) | grouped table, one-line usage per skill |
| `-u` / `--by-usage` | rank by how often each skill was invoked (from `history.jsonl`); `×` = never invoked |
| `-l` / `--long` | full untruncated descriptions |
| `-f` / `--flat` | single alphabetical table, no category grouping |
| `-s STR` / `--search STR` | only skills whose name/description/category contains `STR` |
| `--plain` | disable ANSI color (piping / dumb terminals) |

`--by-usage` counts leading `/<name>` entries in `history.jsonl` (honoring
`CLAUDE_CONFIG_DIR`, falling back to `~/.claude/history.jsonl`), keeping only names
that are real local skills — so path-like `/home/...` and built-ins (`/model`,
`/resume`, `/fast`, …) do not pollute the ranking. It reflects **typed** slash
commands on this host only: a skill invoked by natural language, by the model via the
Skill tool, or inside a container won't be counted (shows `×`).

Examples:

```bash
python3 "$HOME/agent-box/skills/skills-list/list_skills.py" --search gpu     # everything GPU-related
python3 "$HOME/agent-box/skills/skills-list/list_skills.py" --long           # full descriptions
```

## Notes

- Parses YAML-ish frontmatter (`name` / `description` / `category`), tolerant of
  quoted and block-scalar (`>-`, `|`) descriptions; missing `name` falls back to the
  directory name, missing `category` to `uncategorized`.
- Dirs beginning with `.` or `_` (e.g. `_shared`) and any dir without a `SKILL.md`
  are skipped.
- Read-only: it never launches or modifies anything.
