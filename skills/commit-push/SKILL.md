---
description: Commit (if needed) and push to the user's fork, with repo-aware remote selection and user confirmation before push
---

# Commit and Push

Follow these steps precisely. This skill chains to `/commit` when a commit is needed.
Read `_shared/repo-config.md` for remote URLs, branch rules, and safety rules.

## Step 1: Check state

Run these in parallel:
- `git rev-parse --show-toplevel` to determine which repo this is
- `git status` to see uncommitted changes
- `git branch --show-current` to get the current branch
- `git diff --cached` to see staged changes
- `git diff` to see unstaged changes
- `git log --oneline -3` to see recent commits
- `git remote -v` to see configured remotes

## Step 2: Commit if needed

Check if there are uncommitted changes (modified files, staged changes, or relevant untracked files):

- **If there are uncommitted changes**: Invoke the `/commit` skill. Wait for it to complete successfully before proceeding.
- **If the working tree is clean** (existing commits on the branch): Skip to Step 3.
- **If HEAD is detached with no changes**: Warn the user there is nothing to push.

## Step 3: Determine the push remote

Look up the repo in the repo table (`_shared/repo-config.md`) to find the push remote URL.

From the `git remote -v` output in Step 1:
- If an existing remote already points to the target URL, use that remote name.
- If no remote matches, add one: `git remote add fork <target-url>` and use `fork`.

## Step 4: Confirm with the user before pushing

**ALWAYS** ask the user for confirmation before pushing. Show them:
- The branch name that will be pushed
- The remote name and URL
- The number of commits that will be pushed (use `git log --oneline <remote>/<branch>..HEAD` or `git log --oneline -N` if the remote branch doesn't exist yet)

Use `AskUserQuestion` to get confirmation.

## Step 5: Push

After user confirms:
```bash
git push -u <remote> <branch-name>
```

## Step 6: Verify and report

Run `git log --oneline -3` and confirm:
- The branch name
- The remote it was pushed to
- The commit hash(es) pushed
