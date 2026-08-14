---
description: Resolve a conflicting GitHub PR inside a throwaway container built from an image you name. Launches the container, reproduces the merge, auto-resolves mechanical conflicts (asks on semantic ones), verifies the PR's own changes survived, runs pre-commit on the changed files, then pushes the merge back to the PR branch. Use when a PR shows "This branch has conflicts", or the user says '/pr-conflict-fix', 'fix the conflict on PR X', 'rebase/merge main into PR X'.
category: deliver
---

# Fix a PR's merge conflict in a container

Resolve conflicts in the environment the PR is *meant* to run in, not in whatever
checkout happens to be on the host. The image is the point: it pins the Python
env, so `import sglang` and `pre-commit` actually mean something.

Read `_shared/repo-config.md` for the repo table, author identity, and the
`GH_TOKEN=""` rule.

## Invocation

```
/pr-conflict-fix <PR number or URL> --image <docker image> [--fast] [--no-gpu]
```

- **PR** — e.g. `31794` or the full `https://github.com/sgl-project/sglang/pull/31794`.
- **`--image`** — required, no default. e.g. `lmsysorg/sglang-rocm:v0.5.16-rocm700-mi35x-20260803`.
  If the user did not name one, **ask** — do not guess a tag.
- **`--fast`** — bootstrap only `gh` + the identity guard (~30s) instead of the full
  `container-dep.sh` (~minutes). Enough for pure conflict work.
- **`--no-gpu`** — skip `/dev/kfd` + `/dev/dri`. Conflict resolution never needs a GPU;
  the flags are kept by default only for parity with `~/run_docker.sh`.

Runs on the **host** (needs `docker`). If you are already inside a container and
`docker` is missing, say so and stop — do not silently fall back to resolving on the host.

## Step 1 — Preflight (host)

```bash
GH_TOKEN="" gh pr view <PR> --repo sgl-project/sglang \
  --json number,title,state,mergeable,mergeStateStatus,headRefName,baseRefName,headRepositoryOwner,url
```

Read off: **head repo owner** (the fork to push back to), **head branch**, **base branch**.

- `mergeable: MERGEABLE` already → report that and **stop**. Nothing to fix.
- `mergeable: UNKNOWN` → GitHub is still computing. Wait ~15s and re-query.
- `state != OPEN` → stop and say so.

## Step 2 — Launch the container

```bash
bash ~/agent-box/skills/pr-conflict-fix/boot.sh up <image> --pr <PR> [--fast] [--no-gpu]
```

Prints the container name — hold onto it, call it `$C`. The container is `--rm` and
detached (`sleep infinity`), so it lives exactly as long as this skill run.

It mirrors `~/run_docker.sh`: `--privileged`, `--network=host`, `--ipc=host`,
`--shm-size 16G`, host home at `/home/yichiche`, plus `.claude` / `.codex`. That home
mount is what carries `gh` auth (`/home/yichiche/.gh`), the git identity, and the
global pre-push hook into the container.

Run everything after this through:

```bash
bash ~/agent-box/skills/pr-conflict-fix/boot.sh sh "$C" '<command>'
```

which presets `GH_TOKEN=""`, `GH_CONFIG_DIR`, and `PATH`.

Sanity-check the bootstrap before continuing — `gh auth status` must show
`Logged in ... account yichiche`, and `git config --global user.email` must be
`yichiche@amd.com`. If either is wrong, fix it before touching the repo; a merge
commit with the wrong author is a rewrite later.

## Step 3 — Set up the repo inside the container

Work in the image's own checkout, `/sgl-workspace/sglang` (confirm with
`git -C /sgl-workspace/sglang rev-parse --show-toplevel`). It is pristine per image,
so the host clone is never touched.

```bash
cd /sgl-workspace/sglang
git fetch https://github.com/<headOwner>/sglang.git <headRefName>:prwork
git fetch origin <baseRefName>
git worktree add /tmp/prwork prwork
```

Use a **worktree**, not a checkout — the image's main tree may have local state you
do not want to disturb.

## Step 4 — Reproduce the conflict

```bash
cd /tmp/prwork && git merge origin/<baseRefName> --no-edit
git diff --name-only --diff-filter=U     # the conflicted set
```

Merge, don't rebase. The PR branch already carries merge commits from GitHub's
"Update branch" button; rebasing rewrites history that reviewers have seen and
re-authors other people's commits.

If the merge succeeds cleanly, GitHub was stale — jump to Step 8.

## Step 5 — Classify and resolve

For each conflicted file, read every hunk and sort it into one of two buckets.

**Mechanical — resolve yourself, no need to ask.** Both sides are additive and the
union is unambiguous:
- Import-line conflicts where one side's imports are a superset of the other's
  (take the superset; verify every symbol both sides used is still imported).
- Both sides appending distinct entries to a list, dict, enum, or `__all__`.
- One side edits a hunk the other only moved.
- Pure formatting/whitespace divergence.

**Semantic — stop and ask.** Both sides changed the *same* logic: overlapping edits
to one function body, a guard the PR adds where main restructured the surrounding
code, a signature changed on one side and called on the other, or upstream deleting
something the PR modifies. Present the two sides with `AskUserQuestion`, showing each
side's hunk in `preview`, and let the user pick.

**Never resolve by taking a whole side wholesale** (`--ours` / `--theirs`) unless you
have read every hunk in the file and confirmed one side genuinely subsumes the other.

Resolve by editing the file with Edit — delete the `<<<<<<<` / `=======` / `>>>>>>>`
lines as part of the edit, not in a separate pass. Then `git add` the file.

## Step 6 — Verify the PR's intent survived

This is the step that catches a merge that "worked" but quietly dropped the PR.

```bash
git diff origin/<baseRefName> -- <every file the PR touched>
```

Compare against the PR's original intent (`git diff $(git merge-base origin/<base> prwork) prwork --stat`).
**Every** change the PR introduced must still be present. If a hunk vanished, the
merge resolved against the PR rather than for it — go back to Step 5.

Report this as an explicit checklist in your final summary, one line per change.

## Step 7 — Validate

```bash
git grep -n '<<<<<<<\|>>>>>>>' -- '*.py'        # must be empty
git diff --check                                 # whitespace damage
python -c "import ast; [ast.parse(open(f).read()) for f in [<conflicted .py files>]]"
python -c "import sglang; print(sglang.__file__)"
pre-commit run --files <all files changed by the merge>
```

`pre-commit` may **rewrite** files (isort/ruff/black). If it does, re-run Step 6 on
the rewritten files — a formatter reflowing a conflict region is a real way to lose a
hunk — then `git add` the results.

If `pre-commit` is not installed in the image, `pip install pre-commit` and retry; if
it still fails, say so in the summary rather than skipping it silently.

## Step 8 — Commit

```bash
git commit --no-verify -m "Merge branch '<baseRefName>' into <headRefName>"
```

`--no-verify` on the **commit** only, to skip the tag-format `commit-msg` hook, which
rejects merge subjects. Do not invent a `[Tag]` for a merge commit, and never add
trailers.

## Step 9 — Confirm, then push

Pushing mutates a public PR. Show the user, in one message:
- the conflicted files and how each was resolved,
- the Step 6 checklist,
- validation results.

Then `AskUserQuestion` to confirm the push. Only skip this gate if the user already
said to push without asking.

```bash
git push https://github.com/<headOwner>/sglang.git prwork:<headRefName>
```

### The identity hook will probably reject this

`~/agent-box/.githooks-global` blocks pushes containing commits not authored by
`jacky.cheng`. A PR that has ever been updated via GitHub's "Update branch" button
carries a merge commit authored by whoever pressed it — so the hook fires on commits
that are *already on the remote*. Before bypassing, prove it is that false positive:

```bash
git log -1 --format='%an <%ae> | %cn <%ce>'                          # your merge: must be jacky.cheng
git log origin/<baseRefName>..HEAD --format='%h %an <%ae>' | grep -v yichiche@amd.com
```

Every foreign commit listed must already exist on the remote head
(`git branch -r --contains <sha>`). If so, re-push with `--no-verify` and **say in the
summary that you bypassed the hook and why**. If a foreign commit is *not* already on
the remote, do not bypass — stop and ask.

## Step 10 — Confirm and tear down

```bash
sleep 15
GH_TOKEN="" gh pr view <PR> --json mergeable,mergeStateStatus
```

`mergeable: MERGEABLE` is the success condition. `mergeStateStatus: BLOCKED` alongside
it means review/CI gating, **not** a merge problem — report it as such and don't chase it.

```bash
bash ~/agent-box/skills/pr-conflict-fix/boot.sh down "$C"
```

Tear down even on failure — the container is `--rm`, so leaving it costs a running
process. If resolution failed and the user may want to inspect, ask before removing.

## Final report

- Container + image used.
- Each conflicted file: what the conflict was, how it was resolved, and *why* that
  resolution is right.
- Step 6 checklist: every PR change confirmed present.
- Validation output, including anything skipped.
- Whether the identity hook was bypassed.
- Final `mergeable` / `mergeStateStatus`, with `BLOCKED` explained if present.

## Failure modes

| Symptom | Cause | Action |
|---|---|---|
| `docker: command not found` | Running inside a container | Stop. This skill is host-side. |
| `gh` v0.0.4 | pip `gh` shadowing the real CLI | See `_shared/repo-config.md` prerequisites. |
| `gh` 403 / token policy | `GH_TOKEN` PAT blocked by the enterprise policy | Always `GH_TOKEN=""`; `boot.sh sh` already does this. |
| Push rejected on identity | Pre-existing foreign merge commit | Step 9 — verify, then `--no-verify`. |
| `mergeable: UNKNOWN` | GitHub still computing | Wait 15s, re-query. |
| No `/sgl-workspace/sglang` in the image | Non-sglang or slim image | Ask the user for the in-container repo path, or clone the fork to `/tmp`. |
| Conflict in a generated file (`*_pb2.py`) | Regenerated upstream | Take upstream's, then regenerate if the PR changed the `.proto`. |
