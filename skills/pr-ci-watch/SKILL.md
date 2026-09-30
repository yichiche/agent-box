---
name: pr-ci-watch
description: Watch a list of sgl-project/sglang PRs on two cadences (daily, or high-priority every 2h) and babysit them — re-run red NVIDIA CI only after /ci-analysis clears the failure as not-the-PR's-fault, and notify the author once per head SHA when the branch starts conflicting with main. Ships a local dashboard where you paste a PR link to add it. Use when the user says '/pr-ci-watch', 'watch these PRs', 'keep re-running CI on my PRs', 'tell me when a watched PR conflicts', or wants a monitoring list for in-flight PRs.
category: deliver
---

# Watch PRs, re-run only the CI that deserves it

Two kinds of toil this removes:

1. **NVIDIA CI goes red for reasons unrelated to the PR** — GPU busy after
   teardown, profiler teardown decode error, a GSM8K threshold flake, a
   `cutlass-dsl` pin bug. The fix is "re-run the failed group", but only after
   someone confirms it is not a real bug.
2. **A PR silently goes `CONFLICTING`** and sits there because nobody told the
   author.

This skill is the loop around [`/ci-analysis`](../ci-analysis/SKILL.md) (which
does the triage) and [`/pr-conflict-fix`](../pr-conflict-fix/SKILL.md) (which
fixes a conflict on demand). It contributes the watchlist, the schedule, the
memory of what has already been done, and somewhere to look at it all.

Read `_shared/repo-config.md` for the `GH_TOKEN=""` rule. Repo is
`sgl-project/sglang`; `--repo` exists but the CI scope filter is tuned for it.

## Invocation

```
/pr-ci-watch add <pr url|number>… [--high] [--note "..."]
/pr-ci-watch remove <pr url|number>…
/pr-ci-watch list | status
/pr-ci-watch sweep [--track high|regular|all] [--pr N…] [--apply] [--force]
/pr-ci-watch dashboard [--port 8812]
/pr-ci-watch arm            # register the two cron tracks
```

Everything runs through `watch.py` in this directory. **No mutation happens
without `--apply`** — a bare `sweep` prints exactly what it would do and touches
nothing outward-facing.

## What counts as "NVIDIA CI"

Scope is decided **per workflow**, not per job name, using the `workflow` field
from `gh pr checks --json`. That is what makes `PR Test Base / base-a-test-cpu`
stay in scope — it is a CPU job, but it belongs to the NVIDIA workflow.

The filter is an **exclude-list**, so a newly added NVIDIA workflow is picked up
with no code change:

- vendor: `AMD ROCm Arm64 aarch64 MLX MUSA NPU Ascend XPU Xeon Gaudi HPU TPU Mori`
- admin: `Lint`, `Auto Label`, `PR States`, `Release`, `Close Inactive`, `Nightly`, `Docs`
- plus `sgl-router` (Rust-only, no NVIDIA test content)

Verified on PR #41870 (109 checks): keeps **`PR Test Base`** (18) and
**`PR Test Extra`** (8) — covering `call-jit-kernel-tests`, `base-a-test-cpu`,
`call-sgl-kernel-tests`, the b200/h100 shards — and drops all 11 other
workflows, including `PR Test Extra (AMD)` while keeping `PR Test Extra`.

## The sweep is three phases, and the middle one is you

`watch.py` never decides whether a red CI deserves a re-run. That judgement is
`/ci-analysis`'s, and `/ci-analysis` is an agent procedure, not a script.

### Phase A — gather

```bash
python3 ~/agent-box/skills/pr-ci-watch/watch.py sweep --track high --apply
```

Per watched PR it: reads `state / mergeable / mergeStateStatus / headRefOid /
author`; drops the PR from the watchlist if it is no longer `OPEN`; handles the
conflict case (below); otherwise classifies in-scope failed checks into
**real failing jobs** vs **gate-only** and stops with a TRIAGE request.

A PR swept in the last 30 minutes is skipped (so the daily track does not redo a
high sweep that just ran). `--force` or `--pr N` overrides.

**Gate-only** uses the same taxonomy as `/ci-analysis` Phase 2.5: `*-finish`,
`pr-gate`, `Standard Test Results`, `wait-for-*`, `check-pr-test-health`. A
workflow whose only in-scope failures are these has **no root cause here** — the
real failure is in a skipped job or an out-of-scope vendor workflow, and
re-running an aggregation gate cannot turn it green. `apply-verdict` refuses to
re-run those unless you pass `--force-gates`.

### Phase B — triage (you, via `/ci-analysis`)

For each PR the sweep flagged, run `/ci-analysis <pr url>`, read its **Root
Cause Failures** table, and reduce it to **one** action:

| Any root-cause failure with… | Action |
|---|---|
| `Related: yes`, or `Action: code fix` | `code-fix` — **stop, never re-run**; surface to the user |
| any `Action: merge main` | `merge-main` — record, no re-run |
| any `Action: wait upstream` | `wait-upstream` — record, no re-run |
| all remaining `Action: re-run` | `re-run` |

Precedence: `code-fix` > `merge-main` > `wait-upstream` > `re-run`. When
`/ci-analysis` traces the root cause into a **vendor** workflow (AMD, NPU, …),
that is out of scope — record `wait-upstream` or `code-fix` as it directs and do
not re-run the NVIDIA gates.

### Phase C — act

```bash
python3 watch.py apply-verdict --pr 41870 --action re-run \
  --summary "b200 shard 2: CUDA devices busy after teardown; unrelated" --apply
```

On `re-run`, for each in-scope workflow with a real failing job and
`reruns < 2` **for the current head SHA**:

```bash
GH_TOKEN="" gh api -X POST repos/sgl-project/sglang/actions/runs/<run_id>/rerun-failed-jobs
```

Per-run rather than per-job: a run maps 1:1 to a workflow, and the workflow is
already the scope unit. The cap is a backstop behind `/ci-analysis`, not the
primary gate; it resets when the author pushes.

## Conflicts

`mergeable == CONFLICTING` → comment **once per head SHA**, then skip CI work
for that PR (CI is meaningless until the conflict is resolved). The recorded
`conflict_comment_sha` re-arms automatically when the author pushes, so a stuck
PR gets one notice per push, not one per sweep.

```
@<author> heads-up — this PR now conflicts with `main`
(`mergeStateStatus: DIRTY`) as of `<sha8>`, so CI cannot run to completion.

Could you merge or rebase `main` into `<branch>`? Happy to help if the
conflict lands in AMD/ROCm code.
```

English only; no Claude attribution trailer. This is the **only** thing posted
to someone else's PR — `code-fix` verdicts are reported to the user, never
commented on the author's PR.

To actually fix one: [`/pr-conflict-fix <PR> --image <tag>`](../pr-conflict-fix/SKILL.md).

## Dashboard

```bash
bash ~/agent-box/skills/pr-ci-watch/serve_dashboard.sh [--port 8812]
```

Stdlib HTTP server, no dependencies, loopback only (both `127.0.0.1` and `::1`,
so an editor port-forwarder that resolves `localhost` to IPv6 still works). The
script prints the Remote-SSH / `ssh -L` instructions; `--stop` shuts it down.

Columns: PR + title + author, track (click to flip), merge state, red NVIDIA
workflows (gate-only marked as not re-runnable), last verdict + one-line
summary, re-run budget `n/2`, last swept. Auto-refreshes every 60s.

**Quick add:** paste a PR link into the box at the top and pick a track. Accepts
`https://github.com/sgl-project/sglang/pull/41870`, a `/files` deep link, `#41870`,
or `41870`.

> The server reads and writes only `watchlist.json` / `state.json`. It makes no
> `gh` calls and **cannot** re-run a workflow or post a comment — every
> outward-facing action stays in the sweep, which runs under a Claude turn. The
> web layer never needs credentials.

## Scheduling the two tracks

`/pr-ci-watch arm` registers two durable cron jobs via `CronCreate`
(`durable: true`), on off-minutes so they do not pile onto `:00`:

| Track | Cron | Prompt |
|---|---|---|
| high | `23 */2 * * *` | `/pr-ci-watch sweep --track high --apply` |
| regular | `17 9 * * *` | `/pr-ci-watch sweep --track regular --apply` |

Then record it so `status` can warn you later:

```bash
python3 watch.py arm-status --record "high=23 */2 * * *, regular=17 9 * * *"
```

> **Two limits to state up front, not discover later:**
> - Claude cron jobs **auto-expire after 7 days**. `/pr-ci-watch status` flags
>   this at day 6 — re-arm then.
> - They **only fire while the REPL is idle**. If the session is mid-task at
>   `:23`, that sweep is late, not lost.
>
> A headless crontab variant would dodge both, but it cannot run `/ci-analysis`,
> which is the entire gate on re-running. That trade is why it is not the default.

## Data

`$AGENT_SCRATCH_DIR/pr-ci-watch/` (override with `PR_CI_WATCH_DIR`):

| file | holds |
|---|---|
| `watchlist.json` | PR → track, added, note, repo |
| `state.json` | per PR: head SHA, `conflict_comment_sha`, `reruns{workflow:{sha,count}}`, last verdict/action, last sweep |
| `sweeps/<ts>.json` | one record per sweep |
| `sweep.log` | append-only audit of every mutation |

## Load-bearing gotchas

- **`--apply` is the whole safety model.** A sweep without it is a full dry run
  that prints the comment body it would post and the runs it would re-run. Use it
  first on any PR you have not watched before.
- **Never re-run past a `code-fix` verdict.** If `/ci-analysis` attributes a
  failure to the PR, re-running burns CI and hides a real bug. The script will
  not do it; do not work around it with `--force-gates`.
- **Gate-only red is not flaky red.** `pr-test-finish` / `pr-gate` failing alone
  usually means the root cause is in a vendor workflow this skill deliberately
  ignores. Report it; do not re-run it.
- **`mergeable: UNKNOWN`** means GitHub is still computing — the script re-queries
  once after 15s. If it is *still* unknown, the conflict check is deferred to the
  next sweep (CI checks are evaluated as normal); it is never treated as clean.
- **Draft PRs stay on the list** but are worth a lower track; CI on a draft is
  often intentionally red.
