---
description: "Generate a daily standup summary from Claude conversation history, git commits, and PRs. Formats as 'Yesterday / Today' with key accomplishments, metrics, and next steps. Use when the user says '/standup' or asks for a daily summary."
---

# Daily Standup Summary

Generate a standup-format summary of what the user accomplished today (or a specified date), using Claude conversation history, git logs, and GitHub PRs.

## Usage

- `/standup` — summarize today
- `/standup yesterday` — summarize yesterday
- `/standup 2026-05-19` — summarize a specific date

If `$ARGUMENTS` is provided, parse it as a date or relative day. Default: today.

## Step 1: Determine the target date

- Default: today's date (from the system)
- If user says "yesterday", compute yesterday's date
- If user provides a date string (e.g., `2026-05-19`), use that
- Store as `$TARGET_DATE` (YYYY-MM-DD) and `$PREV_DATE` (the day before `$TARGET_DATE`)

## Step 2: Gather data from Claude history

Read `~/.claude/history.jsonl`. Each line is a JSON object with fields:
- `display`: the user's prompt text
- `timestamp`: epoch milliseconds
- `project`: working directory
- `sessionId`: session UUID

Filter entries where `timestamp` falls on `$TARGET_DATE` (convert to local time).

Group by `sessionId`. For each session:
1. Find the transcript file: `~/.claude/projects/*/SESSION_ID.jsonl`
2. Read the transcript and extract:
   - **ai-title**: the session's auto-generated title (look for `type: "ai-title"`)
   - **user messages** (`type: "user"`): scan the `message` field for key activities
3. From history entries, note the `project` (working directory) and `display` text

Summarize each session into a one-line activity description. Focus on:
- What was built, fixed, or analyzed
- Specific tools/skills invoked (e.g., `/compare-kernels`, `/validate`, `/implement-kernel`)
- File paths or PR URLs mentioned

**Keep transcript reading lightweight** — scan for `ai-title` and user messages only, skip assistant content to avoid blowing up context.

## Step 3: Gather data from git

Run across all known repo roots. Check `_shared/repo-config.md` for the repo table, plus detect the active SGLang and scan common locations:

```bash
SGLANG_ROOT=$(python3 -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).resolve().parents[2])" 2>/dev/null)
```

- `$SGLANG_ROOT` (the active SGLang installation)
- `$HOME/agent-box`

For each repo, run:
```bash
git -C <repo> log --author="yichiche" --since="$TARGET_DATE 00:00" --until="$TARGET_DATE 23:59" --oneline --no-merges
```

Collect commit messages and counts.

Also check for PRs created or merged on `$TARGET_DATE`:
```bash
gh pr list --repo <pr-base-repo> --author=yichiche --search "created:$TARGET_DATE" --state all --json number,title,url,state 2>/dev/null
```

## Step 4: Compose the standup

### Format

```
Jacky – M/D
Yesterday:
<project-name>
<bullet points of accomplishments with specific metrics, kernel timings, percentages, PR links>

Today:
<planned next steps — infer from the last session's conversation or ask the user>
```

### Rules

1. **Header**: `Jacky – M/D` where M/D is the target date (no leading zeros, e.g., `5/20` not `05/20`)
2. **"Yesterday" section** contains what was done on `$TARGET_DATE`. Despite the name, this section always describes the target date's work (standup convention: you report "yesterday" in tomorrow's standup).
3. **Group by project/workstream** (e.g., "dsv4 pro", "wan 2.2", "agent-box tooling"). Use short readable names, not full paths.
4. **Max 3 bullets per category.** Merge related items into a single high-level bullet. Prefer outcome over activity — say what changed, not each step taken.
5. **High-level tone.** Write for a manager or cross-team audience, not a detailed changelog. One bullet = one theme (e.g., "Profiled decode phase, MI355 at ~56% of B200" not separate bullets for each kernel).
6. **Include concrete numbers** only when they convey the headline result (e.g., performance ratio, % uplift). Omit per-kernel µs unless it's the main finding.
7. **Include PR links** if any were created or merged.
8. **"Today" section**: Infer from the last conversation's direction. If unclear, put a placeholder and ask the user what they plan to work on next.
9. **Keep it concise** — each bullet should be 1 line max. The whole standup should fit in a Slack message.
10. **No preamble or explanation** — output the standup text directly, ready to copy-paste.

### Example output

```
Jacky – 5/20
Yesterday:
dsv4 pro
- Profiled MI355 vs B200 decode: MI355 at ~56% of B200 E2E, top gaps in GEMM, elementwise, attention
- Tested fused norm+rope in compressor — reverted pending further validation
- Identified Compressor layer as 4.1x slower (13 vs 9 kernels)
agent-box
- Built /perf-summary, /compare-kernels, /implement-kernel skills for kernel optimization workflow
- Made skills portable across containers with symlink setup
- Enhanced trace analyzer: added variant detection, recategorize, and mhc category

Today:
- Validate fused compress-norm-rope kernel and re-land
- Profile decode attention gap, explore FA3 for MI355
```

## Step 5: Present and refine

1. Output the standup text
2. Ask: "Does this look right? Want to adjust anything or add Today's plan?"
3. If the user provides edits, incorporate them and output the final version
