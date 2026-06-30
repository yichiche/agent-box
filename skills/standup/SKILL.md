---
description: "Generate or formalize a daily standup: auto-gather from history/git/gh, or rewrite user-pasted notes into Teams-ready bullets (Yesterday / Today / Blocker). Use when the user says '/standup', 'formalize' standup, or pastes draft standup text."
---

# Daily Standup Summary

Generate a standup-format summary of what the user accomplished (or a specified date range), using Claude conversation history, git commits, and GitHub PRs — **or**, when they already wrote notes, **formalize** those notes into the same Teams-ready output.

## Draft supplied by user ("formalize" mode)

**When this applies:** The user pasted, attached, or `@`-referenced **draft standup material** in the same message as `/standup` or a request to **formalize** / **clean up** / **Teams format** standup. Triggers include: scratch bullets, `Yesterday:` / `Today:` fragments, terminal log snippets, partial PR lists, or multi-line notes that clearly describe what they did.

**Behavior:**

1. Treat the **user-supplied text as the source of truth** for this turn. **Skip Steps 2 and 3** (no `history.jsonl`, no `git log`, no `gh` PR queries) unless the user explicitly asks to verify, refresh PR state, or combine pasted notes **with** an auto-gathered window.
2. **Primary task:** Rewrite into the **Teams-ready format** in Step 4 (template + rules): correct header `Name – M/D` (use the date from their draft if present; otherwise infer from context or the system "today" when they say "today"), normalize sections **Yesterday:** / **Today:** / **Blocker:**, every substantive line under those sections as a `- ` bullet, reference PRs by `PR#####` shorthand in the narrative with full links tabbed beneath (see Rule 8), fix typos and stray indentation, merge to **max 3 bullets** per section when the draft is long (unless the user explicitly asks to keep more structure).
3. **Preserve facts:** PR numbers, repo names, links, metrics — do not invent work that is not in the draft.
4. If the draft has **no Blocker** section, add `Blocker:` with `- None` unless they clearly described a blocker.
5. Still follow **Step 5** after output: one short question about adjusting Today or Blocker.

## Usage

- `/standup` — summarize the **default reporting window** (see Step 1)
- `/standup yesterday` — summarize **calendar yesterday** (typical before a morning standup)
- `/standup today` — summarize **today** (end-of-day recap)
- `/standup 2026-05-19` — summarize a specific date
- `/standup 7 days` or `/standup 7d` — summarize the last 7 days (multi-day range)

If `$ARGUMENTS` is provided, parse it as a date, relative day, or multi-day range.

## Step 1: Determine the target date(s)

- **Default reporting window**: **calendar yesterday** (local date). Rationale: most `/standup` runs are for a **morning Teams update** where "Yesterday" means the prior workday. For an **end-of-day** recap for *today*, use `/standup today` or pass today's date explicitly.
- If user says **`today`**: use today's date for `$START_DATE` and `$END_DATE`.
- If user says **`yesterday`**: use yesterday's date for both.
- If user provides a date string (e.g., `2026-05-19`), use that for both.
- If user provides a number of days (e.g., `7 days`, `7d`, `5 days`), compute a date range: from `today - N + 1` to `today`
- Store as `$START_DATE` and `$END_DATE` (both YYYY-MM-DD). For single-day mode, `$START_DATE == $END_DATE`.
- Also store `$PREV_DATE` (the day before `$START_DATE`)

### Multi-day mode

When a range spans multiple days:

- In Step 2 and 3, query from `$START_DATE 00:00` to `$END_DATE 23:59` instead of a single day
- In Step 4, use the header format `Name – M/D–M/D` (e.g., `Jacky – 5/23–5/29`)
- Group work thematically rather than by day — the audience wants a weekly summary, not 7 separate dailies
- Still follow the same bullet-point rules (high-level, max 3 per category, concrete numbers)
- **Focus on performance results and PR status** — the audience wants to know what shipped, what's in review, and what perf numbers changed. Omit internal tooling (agent-box, skills) unless the user specifically asks.
- **Organize by PR / deliverable**, not by activity type. Each bullet should reference a PR link and its status (merged, open, closed).

## Step 2: Gather data from Claude history

**Skip entirely** if **Draft supplied by user ("formalize" mode)** applies (user pasted or attached standup notes in the same request).

Read `~/.claude/history.jsonl`. Each line is a JSON object with fields:

- `display`: the user's prompt text
- `timestamp`: epoch milliseconds
- `project`: working directory
- `sessionId`: session UUID

Filter entries where `timestamp` falls between `$START_DATE` and `$END_DATE` inclusive (convert to local time).

If `history.jsonl` is missing or unreadable, skip this step and rely more on git/GitHub; note the gap briefly in Step 5 if material.

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

**Skip entirely** if **Draft supplied by user ("formalize" mode)** applies.

Run across all known repo roots. Check `$HOME/agent-box/skills/_shared/repo-config.md` for the repo table, plus detect the active SGLang and scan common locations:

```bash
SGLANG_ROOT=$(python3 -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).resolve().parents[2])" 2>/dev/null)
```

- `$SGLANG_ROOT` (the active SGLang installation), if non-empty
- `$HOME/agent-box`

Resolve **git author** from repo-config (`yichiche` / email in that file) and use `git log --author=...` consistently with local commits.

For each repo, run:

```bash
git -C <repo> log --author="<author-pattern>" --since="$START_DATE 00:00" --until="$END_DATE 23:59" --oneline --no-merges
```

Collect commit messages and counts.

Also check for PRs created or merged in the date range (use **`GH_TOKEN=""`** when calling `gh` against `sgl-project/sglang` if enterprise PAT issues apply — see repo-config):

```bash
GH_TOKEN="" gh pr list --repo sgl-project/sglang --author=<github-username> --search "created:$START_DATE..$END_DATE" --state all --json number,title,url,state
GH_TOKEN="" gh pr list --repo sgl-project/sglang --author=<github-username> --search "merged:$START_DATE..$END_DATE" --state merged --json number,title,url,state
```

## Step 4: Compose the standup

### Display name (header)

Use the first token of `git config user.name` (run in any configured repo), unless the user gave a name in the chat. Fallback: ask or use a neutral placeholder.

### Teams-ready format (required)

Output **only** the standup block: no preamble, no wrapping markdown code fence (plain text pastes cleanly into Teams).

**Grouped layout (preferred).** Under **Yesterday:**, group work under a **ticket / workstream header** line
(e.g. a Jira ticket `GPUAI-6500 — <short title>`, or a project name like `InferenceX`), then list the work as
**indented** `  - ` sub-bullets beneath it. Use one header per distinct ticket/workstream the day's work maps to.
**Today:** and **Blocker:** stay as simple indented bullets (no headers needed unless the day's plan spans
multiple tickets).

**Single-day template:**

```
Name – M/D

Yesterday:

GPUAI-#### — <short workstream title>
  - <what shipped / progressed, referencing PR##### / PR##### inline>
    - PR#####: <short PR title> — <full PR URL>
    - PR#####: <short PR title> — <full PR URL>
  - <related work, referencing PR#####>
    - PR#####: <short PR title> — <full PR URL>

<Project/Workstream>
  - <what shipped / progressed, referencing PR#####>
    - PR#####: <short PR title> — <full PR URL>

Today:
  - <plan>
  - <plan>

Blocker:
  - None
```

**Multi-day template:** same grouped layout, header `Name – M/D–M/D`; group by ticket/workstream across the
range and merge related lines.

### Rules

1. **Header**: `Name – M/D` for single day, or `Name – M/D–M/D` for multi-day (no leading zeros, e.g., `6/5` not `06/05`).
2. **Yesterday:** bullets = completed work in `$START_DATE`…`$END_DATE` (standup naming convention).
3. **Exclude internal tooling**: do **not** mention `agent-box` work — skills, workflows, the standup/profile/compare-kernels tooling itself, or any change under `$HOME/agent-box`. The audience cares about shipped product work (sglang, aiter, InferenceX), not the local automation used to produce it. Only include agent-box work if the user **explicitly** asks for it.
4. **Group by ticket/workstream**: under Yesterday, head each group with its Jira ticket (`GPUAI-####`) or workstream/project name, then indent the work as `  - ` sub-bullets. **Do not** prefix sub-bullets with the repo (no `(sglang)` / `(aiter)` / `(inferencex)`); the PR URL already identifies the repo. **Default Jira ticket**: Qwen3.5 / AMD inference work belongs under `GPUAI-6500` — use it as the ticket header unless a different ticket clearly applies. Never drop the ticket header in favor of a made-up workstream title.
5. **Max ~3 sub-bullets per group** (merge related lines); aim for a short post overall.
6. **High-level tone** for managers / cross-team; one bullet = one outcome or theme.
7. **Include concrete numbers** only for headline results (%, ratio, tok/s).
8. **PR links**: in the narrative sub-bullet, refer to PRs by `PR#####` shorthand (e.g. `PR3693`, `PR28658`) instead of inlining URLs. Place the full link on its own **tabbed** sub-line directly beneath, one per PR, formatted `PR#####: <short PR title> — <full https://github.com/... URL>`. This keeps the prose line readable while still carrying every link. Applies whether the bullet cites one PR or several.
9. **Today:** infer from the latest conversation direction; if unknown, use `- (add plan)` and ask once in Step 5.
10. **Blocker:** always include this section. Use `- None` when there are no blockers; otherwise one bullet with the blocker.
11. **Length**: should fit a short Teams chat post.

### Example output

```
Jacky – 6/17

Yesterday:

GPUAI-6500 — Qwen3.5-397B-A17B-mxfp4
  - CK-Tile interleaved post-activation: fused silu_and_mul (PR3603); cherry-picked Qwen3.5 397B mxfp4 GEMM tuning (PR3693); addressed review feedback
    - PR3603: CK-Tile interleaved post-activation silu_and_mul — https://github.com/ROCm/aiter/pull/3603
    - PR3693: Add qwen3.5 397b mxfp4 GEMM tuning — https://github.com/ROCm/aiter/pull/3693
  - Qwen3.5 HIP fusions: aiter fused topk_gating for MoE routing (PR28399) and fused QK GemmaRMSNorm + MRoPE attn-prep (PR28398); revised earlier fusions PR28361 / PR28362
    - PR28399: aiter fused topk_gating for MoE routing — https://github.com/sgl-project/sglang/pull/28399
    - PR28398: fused QK GemmaRMSNorm + MRoPE attn-prep — https://github.com/sgl-project/sglang/pull/28398

InferenceX
  - Added --kv-cache-dtype fp8_e4m3 on a new branch, opened a PR, kicked off the perf sweep via perf-changelog.yaml

Today:
  - Run 5-source discovery on Qwen3.5 traces and triage candidates
  - Configure 8 NVMe drives as RAID0 on MI355X

Blocker:
  - None
```

## Step 5: Present and refine

1. Output the standup text (Teams format above).
2. Ask once: "Does this look right? Adjust Today or Blocker?"
3. If the user provides edits, incorporate them and output the final version only.
