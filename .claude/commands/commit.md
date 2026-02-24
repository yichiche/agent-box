---
description: Stage, branch, and commit changes following repo conventions (auto-detects commit-msg hook tag format)
---

# Commit Changes

Follow these steps precisely to commit the current changes.

## Step 1: Check current branch, status, and commit-msg hook

Run these in parallel:
- `git status` to see all changed/untracked files
- `git branch --show-current` to get the current branch name
- `git diff` to see unstaged changes
- `git diff --cached` to see staged changes
- `git log --oneline -5` to see recent commit style
- Read the active commit-msg hook to discover allowed tag formats:
  1. Run `git config --get core.hooksPath` to find the hooks directory (defaults to `.git/hooks` if unset)
  2. Read the `commit-msg` file from that directory (e.g., `cat $(git rev-parse --show-toplevel)/.githooks/commit-msg` or `.git/hooks/commit-msg`)
  3. Parse the hook to extract the allowed tags (look for patterns like `Feature|Fix|Refactor|...`)

## Step 2: Determine the commit message tag format

Based on the commit-msg hook analysis from Step 1:

- **If a commit-msg hook exists and enforces specific tags** (e.g., `[Feature]`, `[Fix]`, `[Refactor]`, `[Docs]`, `[Test]`, `[CI]`, `[Chore]`, `[Perf]`):
  Use the repo's required format. Pick the tag that best matches the nature of the change:
  - `[Feature]` — new functionality
  - `[Fix]` — bug fix
  - `[Refactor]` — code restructuring without behavior change
  - `[Docs]` — documentation only
  - `[Test]` — adding or updating tests
  - `[CI]` — CI/CD pipeline changes
  - `[Chore]` — maintenance, dependencies, tooling
  - `[Perf]` — performance improvement

- **If no commit-msg hook exists or it doesn't enforce a tag format**:
  Use `[AMD]` as the default prefix.

## Step 3: Ensure we are on a feature branch (NOT main/master)

If the current branch is `main` or `master`:
1. Determine a short, descriptive branch name based on the changes (e.g., `fix-mla-bf16-attention`, `add-rocm-triton-kernel`)
2. Create and switch to the new branch: `git checkout -b <branch-name>`

If already on a feature branch, stay on it.

## Step 4: Stage the changes

- Stage only the relevant changed files by name (e.g., `git add file1.py file2.py`)
- Do NOT use `git add -A` or `git add .` unless the user explicitly asks
- Do NOT stage files that look like they contain secrets (.env, credentials, tokens, etc.)
- If unsure which files to stage, ask the user

## Step 5: Write the commit message

The commit message MUST follow this format:
- Start with the tag determined in Step 2 (e.g., `[Fix]` or `[AMD]`)
- Followed by a single concise sentence describing what was changed and why
- Example (with repo hook): `[Fix] Resolve bf16 type casting in MLA decode attention for dp-attention mode`
- Example (without hook): `[AMD] Fix bf16 type casting in MLA decode attention for dp-attention mode`
- Do NOT include `Co-Authored-By` or any other trailers — they are forbidden by project convention

If the user provided `$ARGUMENTS`, incorporate that into the commit message description.

If no arguments were provided, analyze the diff to write an appropriate one-sentence summary.

## Step 6: Create the commit

Always use `--author` to set the commit author explicitly:

```bash
git commit --author="jacky.cheng <yichiche@amd.com>" -m "[Tag] <one sentence description>"
```

## Step 7: Verify

Run `git status` and `git log --oneline -3` to confirm the commit was created successfully. Report the branch name and commit hash to the user.

## Important Rules

- NEVER include Co-Authored-By or any trailers in the commit message
- NEVER amend a previous commit unless the user explicitly asks
- NEVER force push
- NEVER skip pre-commit hooks
- If a pre-commit hook fails, fix the issue, re-stage, and create a NEW commit
- NEVER commit .env files, credentials, or secrets
