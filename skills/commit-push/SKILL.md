---
description: Commit (if needed) and push to the user's fork, with repo-aware remote selection and user confirmation before push. Changes under ~/agent-box take a standing-authorization fast path — commit and push straight to main with no confirmation.
category: deliver
---

# Commit and Push

Follow these steps precisely. This skill chains to `/commit` when a commit is needed.
Read `_shared/repo-config.md` for remote URLs, branch rules, and safety rules.

## Step 0: Pick the target repo

The cwd repo is not always the one that changed. Skills, memory and workflows live
in `~/agent-box`, so a session working in SGLang or InferenceX routinely leaves its
only changes there.

```bash
git -C "$(git rev-parse --show-toplevel)" status --short
git -C "$HOME/agent-box" status --short
```

- Exactly one of them dirty → that is the target.
- Both dirty → handle the cwd repo first, then agent-box, as two separate commits.
- Neither dirty → report that there is nothing to commit and stop.

## Step 0b: agent-box fast path (standing authorization)

**When the target repo is `~/agent-box`, do not ask anything. Commit on `main` and
push to `origin` (`https://github.com/yichiche/agent-box`) in one go.**

The user has granted standing authorization for this repo: it is their own personal
tooling repo, `main` is the working branch by convention (`_shared/repo-config.md`
marks it "commit on main: Yes"), and there is no CI, no reviewer and no one else
consuming it. Asking each time is pure friction.

```bash
cd "$HOME/agent-box"
git add -A
git commit -m "[Tag] <one sentence description>"
git push -u origin main
```

- Tag from the repo's own history: `[Feature]`, `[Fix]`, `[Chore]`, `[Refactor]`, `[Docs]`.
- No trailers (no `Co-Authored-By`) — see `_shared/repo-config.md`.
- No pre-commit config exists in agent-box; skip Step 2 there.
- Still never commit secrets, and still never force-push.
- Report the message and the pushed hash afterwards rather than asking beforehand.

Then skip to Step 7. Steps 1–6 below apply to every **other** repo, where the
confirmation gate stays in force.

## Step 1: Check state

Run these in parallel:
- `git rev-parse --show-toplevel` to determine which repo this is
- `git status` to see uncommitted changes
- `git branch --show-current` to get the current branch
- `git diff --cached` to see staged changes
- `git diff` to see unstaged changes
- `git log --oneline -3` to see recent commits
- `git remote -v` to see configured remotes

## Step 2: Run pre-commit checks

Before committing, ensure pre-commit is installed and run it on the changed files:

```bash
pip3 install pre-commit 2>&1 | tail -3
cd <repo-root> && pre-commit run --all-files
```

- If pre-commit **modifies files** (auto-formatting), re-stage the modified files before proceeding to commit.
- If pre-commit **fails** with errors that cannot be auto-fixed, report the failures to the user and stop.
- If pre-commit **passes** with no changes, proceed to commit.

## Step 3: Commit if needed

Check if there are uncommitted changes (modified files, staged changes, or relevant untracked files):

- **If there are uncommitted changes**: Invoke the `/commit` skill. Wait for it to complete successfully before proceeding.
- **If the working tree is clean** (existing commits on the branch): Skip to Step 5.
- **If HEAD is detached with no changes**: Warn the user there is nothing to push.

## Step 4: Determine the push remote

Look up the repo in the repo table (`_shared/repo-config.md`) to find the push remote URL.

From the `git remote -v` output in Step 1:
- If an existing remote already points to the target URL, use that remote name.
- If no remote matches, add one: `git remote add fork <target-url>` and use `fork`.

## Step 5: Confirm with the user before pushing

Skip this step entirely for `~/agent-box` (Step 0b). For every other repo,
**ALWAYS** ask the user for confirmation before pushing. Show them:
- The branch name that will be pushed
- The remote name and URL
- The number of commits that will be pushed (use `git log --oneline <remote>/<branch>..HEAD` or `git log --oneline -N` if the remote branch doesn't exist yet)

Use `AskUserQuestion` to get confirmation.

## Step 6: Push

After user confirms:
```bash
git push -u <remote> <branch-name>
```

## Step 7: Verify and report

Run `git log --oneline -3` and confirm:
- The branch name
- The remote it was pushed to
- The commit hash(es) pushed
