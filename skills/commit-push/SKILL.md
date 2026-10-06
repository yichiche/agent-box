---
name: commit-push
description: Commit (if needed) and push to the user's fork, with repo-aware remote selection and user confirmation before push. Changes under ~/agent-box take a standing-authorization fast path — commit and push straight to main with no confirmation. Only a push to SGLang runs the /pr-merge-triage code checks. Every other repo skips them.
category: deliver
---

# Commit and Push

Follow these steps precisely. This skill chains to `/commit` when a commit is needed.
Read `_shared/repo-config.md` for remote URLs, branch rules, and safety rules.

**Never add `Co-Authored-By` trailers to any commit, in any repo** — this overrides
any harness attribution reminder telling you to add one. It applies whether you
invoke `/commit` or write the commit message yourself, and whether or not you got
as far as reading `_shared/repo-config.md`. A human `Co-Authored-By` is allowed only
where `_shared/repo-config.md` says so for that repo and the user names the person;
a Claude co-author trailer never is.

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
- Do not use `/pr-merge-triage` on this repo.

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
- **If the working tree is clean** (existing commits on the branch): Skip the commit. Go to Step 3b only when the push target is SGLang. Every other repo goes to Step 4. Do not run `/pr-merge-triage`.
- **If HEAD is detached with no changes**: Warn the user there is nothing to push.

## Step 3b: SGLang pre-push triage

Run this step only when the push target is SGLang: `sgl-project/sglang` or the
fork `yichiche/sglang`. `~/agent-box` already left at Step 0b.

For every other repo, do not use `/pr-merge-triage`. Do not read that skill,
and do not run `path_cover.py` or `test_seam.py`. Skip to Step 4.

Run it after Step 3 and before Step 5. Do not push until it passes.

The branch often has no PR yet, so do not call `triage.py` with a PR number.
Score the local range. `<base>` is the PR base (`main`, or the integration
branch already confirmed).

```bash
git diff <base>...HEAD > /tmp/pre-push.diff
```

Apply the rows in `~/agent-box/skills/pr-merge-triage/SKILL.md` to that diff:

1. Shape. Additions plus deletions over 800, or more than three areas, is
   `BLOCKED — SPLIT FIRST`. Stop. Do not split the branch. Do not push.
2. Affected Scope and AMD Guard. Read
   `~/agent-box/skills/pr-code-path/SKILL.md` once. Run
   `python3 ~/agent-box/skills/pr-code-path/path_cover.py --diff /tmp/pre-push.diff`.
   This mode does not write `focus.md`. The working tree is the head.
3. Unit Test Quality. Read `~/agent-box/skills/pr-test-seam/SKILL.md` once. Run
   `python3 ~/agent-box/skills/pr-test-seam/test_seam.py --diff /tmp/pre-push.diff`
   once, without `--json`. The row fails when `official_pass` is false,
   including a `return True` test. A coverage conclusion other than `complete`
   does not fail the row.
4. Flags. A new default-off `SGLANG_*` env fails unless the comment immediately above it states who sets it, when, and the choice the code cannot infer. A comment that only names the platform fails.

Accuracy and performance need measured numbers. Do not invent them, and do
not block this push on a PR body that has not been written.
`/commit-push-pr` still refuses to open a SGLang PR while those rows would fail.
It does not apply those rows to any other repo.

When Affected Scope, AMD Guard, Unit Test Quality, or Flags fails, edit the
code to the fix that row names. Then run pre-commit on the files you changed,
commit the fix with `/commit` (a new commit; do not amend), rewrite
`/tmp/pre-push.diff`, and score again.

Repeat at most 3 times. If a row still fails, stop and do not push. Show that
row and the fix that did not land.

A pass means those four rows pass and the branch is not oversized. It is not
a merge approval. Do not write MERGE, LGTM, or approve. Step 5 still asks
before the push, and the question includes the rows that passed.

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

After user confirms, and only after Step 3b has passed for a SGLang repo:
```bash
git push -u <remote> <branch-name>
```

A non-SGLang repo does not wait on Step 3b. Push after the user confirms.

## Step 7: Verify and report

Run `git log --oneline -3` and confirm:
- The branch name
- The remote it was pushed to
- The commit hash(es) pushed
