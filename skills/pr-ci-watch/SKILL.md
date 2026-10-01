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
/pr-ci-watch triage <pr>…   # /ci-analysis + apply-verdict for PRs already swept
/pr-ci-watch report         # the <P0><CI clear><PR…> status block
/pr-ci-watch pause | resume # kill switch; a paused sweep is a no-op
/pr-ci-watch dashboard [--port 8812]
/pr-ci-watch arm            # register the two cron tracks
```

Line 2 of each entry is the PR's **own performance claim**, extracted from its
body by `perf.py` — not the CI triage reason. It reads the markdown result
tables two ways: metric-in-first-cell (`| TPOT median | 2.60 | 2.48 | -4.6% |`)
and metric-in-header (`| … | Δ TTFT | … | Δ |`, where a bare Δ column inherits
the metric to its left). Only *signed* percentages count as deltas — an unsigned
one beside them is a standard error. Across several benchmark tables it reports
the **median**, preferring rows the PR marks `median`/`p50`, so a multi-shape
sweep is not cherry-picked.

Which metrics: **TPOT and TTFT when the PR states both**, otherwise up to two of
TPOT / TTFT / E2E / total throughput. Direction is per metric — `-4.6% TPOT` is
an improvement, `-2.9% throughput` is a regression. Reciprocal restatements
(`Interactivity (1 / TPOT)`) and accuracy tables are excluded; counting the
former would report one result as both an improvement and a regression.

Override a poor extraction with `watch.py set <pr> --note "..."`; check one
without a sweep via `python3 perf.py <pr>`.

The report token comes from the same `ci_verdict` the table shows (`CI clear` /
`CI running` / `CI red`), with `code-fix`, `merge-main` and `wait-upstream`
overriding it because they say more than the raw state. Row order is shared via
`row_order()` — the table and the block cannot drift apart.

**`triage`** is Phase B on demand: for each PR named, run
[`/ci-analysis`](../ci-analysis/SKILL.md), reduce its table to one action, and
record it with `apply-verdict --apply`. The dashboard's "Waiting on triage" panel
generates this command for you to paste.

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

### Two failures the sweep resolves without a triage pass

**Watcher-only → auto `re-run`.** A `wait-for-*` job that failed with *no* real
failing job beside it died on its own — GitHub API 5xx, timeout — while the jobs
it watched were still green. (If a watched job had truly failed, it would be in
the failure list too.) That is unambiguous infra, so the sweep records `re-run`
directly instead of spending a `/ci-analysis` pass on it.

**Same failure after a re-run → stop, evaluate `merge main`.** Each re-run stores
a signature of what failed. If the next sweep sees the identical set of failing
jobs for the same head SHA, it prints `SAME FAILURE AFTER RE-RUN` and forces the
PR into triage with an explicit instruction: do **not** record `re-run` again —
check `gh api compare/<head>...main` and prefer `merge-main`, or `code-fix` if it
is the PR's own bug. The dashboard shows the same warning in red.

This is the escalation ladder: *re-run once → if it comes back identical, it is
not flaky, so stop retrying and look at main.*

**Gate-only** uses the same taxonomy as `/ci-analysis` Phase 2.5: `*-finish`,
`pr-gate`, `Standard Test Results`, `wait-for-*`, `check-pr-test-health`. A
workflow whose only in-scope failures are these has **no root cause here** — the
real failure is in a skipped job or an out-of-scope vendor workflow, and
re-running an aggregation gate cannot turn it green. `apply-verdict` refuses to
re-run those unless you pass `--force-gates`.

Because a gate-only red says nothing about *what broke*, every sweep that sees a
real failure stores it as `last_real_failure` (jobs + their log URLs, scoped to
the head SHA). The dashboard falls back to that whenever the live checks show
only gates — which is the normal state once a re-run is in flight — so the
column always answers "what failed?" instead of printing an unactionable label.

**Cascade demotion.** A job that failed at the `check-pr-test-health` step was
killed by fail-fast before running a single test. Its *name* gives no hint —
`base-b-test-2-gpu-large (5)` looks like a real shard either way — so the sweep
spends one API call per real-looking failed job to read its failed step, and
demotes cascades to gates. Without this, a PR whose only problem is a dead
watcher gets sent to triage with a phantom "real" failure beside it.

**Verdicts survive re-sweeps.** `apply-verdict` records a fingerprint of the
failing jobs it judged. A later sweep that sees the same head SHA and the same
failures keeps the verdict instead of resetting to `awaiting-triage` — otherwise
a `code-fix` decision silently evaporates and the PR is re-triaged forever. A new
push, or a different set of failures, correctly re-opens triage.

### Phase B — triage (you, via `/ci-analysis`)

For each PR the sweep flagged, run `/ci-analysis <pr url>`, read its **Root
Cause Failures** table, and reduce it to **one** action:

| Any root-cause failure with… | Action |
|---|---|
| `Related: yes`, or `Action: code fix` | `code-fix` — **stop, never re-run**; surface to the user |
| any `Action: merge main` | `merge-main` — record, no re-run |
| any `Action: wait upstream` | `wait-upstream` — record, no re-run |
| all remaining `Action: re-run` | `re-run` |

Precedence: `code-fix` > `merge-main` > `wait-upstream` > `re-run`.

**When the root cause is not NVIDIA, the verdict is `out-of-scope` — never
`wait-upstream`.** If `/ci-analysis` traces the failure into a vendor workflow
(NPU, AMD, MUSA, …), this tool has nothing to act on: those workflows are
excluded by design, and the red NVIDIA gates are only echoing them. Reserve
`wait-upstream` for an **in-scope NVIDIA job** genuinely blocked on a dependency
(the `cutlass-dsl` pin bug is the archetype).

The sweep now reaches this conclusion on its own: a PR whose in-scope failures
are *all* gate-only is recorded `out-of-scope` and never enters the triage
queue, because there is nothing to re-run and nothing for `/ci-analysis` to
decide.

### Phase C — act

```bash
python3 watch.py apply-verdict --pr 41870 --action re-run \
  --summary "b200 shard 2: CUDA devices busy after teardown; unrelated" --apply
```

On `re-run`, for each in-scope workflow with a real failing job:

```bash
GH_TOKEN="" gh api -X POST repos/sgl-project/sglang/actions/runs/<run_id>/rerun-failed-jobs
```

Per-run rather than per-job: a run maps 1:1 to a workflow, and the workflow is
already the scope unit.

**There is no re-run count cap.** `/ci-analysis` is the gate: a failure
attributed to the PR returns `code fix` and nothing is re-run at all, so a count
limit could only ever block a failure already cleared as unrelated. Attempts are
still counted per head SHA and shown as `re-run ×N` so a workflow being retried
over and over is visible rather than silent.

If GitHub answers `403 This workflow is already running`, the attempt is
recorded as deferred, not counted, and the next sweep retries it.

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
summary, re-run count `×N`, last swept (Taiwan time). Auto-refreshes every 60s.

**Quick add:** paste a PR link into the box at the top and pick a track. Accepts
`https://github.com/sgl-project/sglang/pull/41870`, a `/files` deep link, `#41870`,
or `41870`.

**Buttons, and what each can actually reach:**

| Button | Does | Needs Claude? |
|---|---|---|
| `Monitoring ON/OFF` | Writes `_config.enabled`; every sweep, including a cron-fired one, exits immediately when off | no — takes effect instantly |
| `Refresh now` | Runs `sweep --track all --force` **without `--apply`** in a subprocess: re-reads merge state and red NVIDIA CI for every PR. Comments nothing, re-runs nothing | no |
| `Notify author` | Runs `watch.py notify --pr N --apply` — posts the conflict notice for that PR now instead of waiting for the next sweep. Shown **only** when the PR is conflicting and not yet notified for this head SHA; asks for confirmation first | no |
| `▲ / ▼` | Nudges a row within its sort bucket. Manual order is a tiebreaker only — it cannot drag a row across the Pass / priority / conflict boundaries, because that would silently snap back | no |
| Track dropdown | Sets `regular` or `high` explicitly, both directions | no |
| `Copy` (triage panel) | Copies `/pr-ci-watch triage <prs>` to paste into Claude | yes, to run it |
| `Copy` (status block) | Copies the `<P0><CI clear><PR…>` report as **rich text + plain text**, so `<PR41133>` stays a hyperlink when pasted into Teams | no |

`Refresh now` is deliberately read-only. Deciding **re-run vs merge main vs real
bug** means reading job logs — that is `/ci-analysis`, which needs a Claude turn,
so no button can do it. Registering cron likewise needs a Claude turn.

**Verdict** is the in-scope CI state only — `Pass` / `Pending` (in flight,
first run or re-run) / `Fail`. A red *aggregation gate* never makes it `Fail`: a
gate is not a job, it is mirroring a vendor workflow this tool excludes, so
calling that Fail would report someone else's failure as this PR's NVIDIA
result. Rows are ordered **Pass first** (those are the ones you can go merge), then
P0 → P2, then **conflicts last within each priority** (nothing can progress on
them until the author rebases), then your manual `▲/▼` order, then PR number. **Action** is what to do — `CI re-run` (one is in flight),
`Solve conflict`, `Merge main`, `Code fix`, `Triage`, `Wait upstream`, or `-`.
**Status** carries the job counts (`34 pass, 2 fail, 10 running`), the conflict
notice proof, and the triage reason; it flags the counts as stale when they
predate a re-run.

The internal verdicts map onto those columns as:

| Verdict | Means |
|---|---|
| `green` | nothing to do |
| `awaiting-triage` | real failing NVIDIA jobs, no verdict yet — needs `/ci-analysis` |
| `re-run` | cleared as unrelated and re-run; waiting on CI |
| `merge-main` | **stuck** — no conflict, but main already has the fix for a CI failure and the PR is behind. Action: **Merge main** |
| `conflict` | **stuck** — the branch has git conflicts with main; CI cannot complete. Action: **Solve conflict** (a different problem from `merge-main` — never label them the same) |
| `wait-upstream` | **stuck** — an in-scope NVIDIA job is blocked on an upstream fix |
| `out-of-scope` | nothing to do — the red is entirely from vendor workflows (NPU/AMD/…) |
| `code-fix` | real bug in this PR; author must fix. Never re-run |

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
