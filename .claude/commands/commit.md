---
description: Stage, branch, and commit changes following repo conventions (auto-detects commit-msg hook tag format)
---

# Commit Changes

Follow these steps precisely to commit the current changes.
Read `_shared/repo-config.md` for author, repo table, tag format, and safety rules.

## Step 1: Check current branch, status, and commit-msg hook

Run these in parallel:
- `git status` to see all changed/untracked files
- `git branch --show-current` to get the current branch name
- `git diff` to see unstaged changes
- `git diff --cached` to see staged changes
- `git log --oneline -5` to see recent commit style
- `git rev-parse --show-toplevel` to determine which repo this is
- Read the active commit-msg hook to discover allowed tag formats:
  1. Run `git config --get core.hooksPath` to find the hooks directory (defaults to `.git/hooks` if unset)
  2. Read the `commit-msg` file from that directory
  3. Parse the hook to extract the allowed tags (look for patterns like `Feature|Fix|Refactor|...`)

## Step 2: Determine the commit message tag format

Based on the commit-msg hook analysis from Step 1:

- **If a commit-msg hook exists and enforces specific tags**: Use the hook's required format.
- **If no commit-msg hook exists or it doesn't enforce a tag format**: Use `[AMD]` as the default prefix.

## Step 3: Branching strategy

Look up the repo in the repo table (`_shared/repo-config.md`):

- **If "Commit on main" is Yes** (e.g., agent-box): commit directly on `main`. Do NOT create a feature branch.
- **Otherwise**: if the current branch is `main` or `master`, create a feature branch:
  1. Determine a short, descriptive branch name based on the changes (e.g., `fix-mla-bf16-attention`, `add-rocm-triton-kernel`)
  2. Create and switch to the new branch: `git checkout -b <branch-name>`

If already on a feature branch, stay on it.

## Step 4: Stage the changes

- Stage only the relevant changed files by name (e.g., `git add file1.py file2.py`)
- Do NOT use `git add -A` or `git add .` unless the user explicitly asks
- Do NOT stage files that look like they contain secrets (.env, credentials, tokens, etc.)
- If unsure which files to stage, ask the user

## Step 5: Draft the commit message

Analyze the staged diff and draft a commit message:
- Start with the tag determined in Step 2 (e.g., `[Fix]` or `[AMD]`)
- Followed by a single concise sentence describing what was changed and why
- Example: `[AMD] Add fused softmax pool Triton kernels for compressor on ROCm`
- Do NOT include `Co-Authored-By` or any other trailers

If the user provided `$ARGUMENTS`, incorporate that into the commit message description.

## Step 6: Confirm the commit message with the user

**ALWAYS** present the drafted commit message to the user and ask for approval using `AskUserQuestion` before committing.

Show them:
- The list of files that will be committed
- The proposed commit message

Let the user approve, edit, or reject the message. If they provide alternative text, use that instead.

## Step 7: Create the commit

After user approval, use `--author` from `_shared/repo-config.md`:

```bash
git commit --author="jacky.cheng <yichiche@amd.com>" -m "<approved commit message>"
```

## Step 8: Verify

Run `git status` and `git log --oneline -3` to confirm the commit was created successfully. Report the branch name and commit hash to the user.
