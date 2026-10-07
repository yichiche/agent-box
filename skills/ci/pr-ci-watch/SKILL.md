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
/pr-ci-watch auto <pr> [--triage on|off] [--rerun on|off]   # per-PR switches
/pr-ci-watch update-branch --pr N [--apply]   # GitHub's Update branch
/pr-ci-watch report         # the <P0><CI clear><PR…> status block
/pr-ci-watch pause | resume # kill switch; a paused sweep is a no-op
/pr-ci-watch dashboard [--port 8812]
/pr-ci-watch arm            # register the two cron tracks
```

Entries render as a markdown bullet with the perf line nested under it —
`- <P0><CI clear><PR41134>Title` / `  - TPOT 6.3% improvement`. The rich flavour
is a real nested `<ul>`, so a paste into Teams lands as a proper list; the plain
flavour uses `- ` and two-space indent, which Teams also turns into bullets.
In the rich flavour, `CI clear` is wrapped in `<b>`, so that token is bold on
the page and stays bold in Teams. Other tokens are not.

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

A delta column is recognised by `DELTA_HEADER` — `Δ`, `delta`, `change`,
`diff`, `%`. **`change` is load-bearing**: a header of `tok/s change | TPOT
change` is the same table as `… | Δ`, and while the word was missing the whole
table fell through to the metric-in-first-cell path, where the first cell is a
concurrency number and nothing matches — so a PR with perfectly good numbers
reported none at all.

When no table yields a metric, `_prose()` is tried as a last resort: a sentence
like *"median TPOT drops **from 3.50 to 2.78** ms at concurrency 1"* has real
e2e numbers, just absolute and unpercented, so nothing above sees them. It is
scoped tightly — the metric name must be in the same sentence **and before**
the `from A to B`, or "from 4 to 8" in a sentence about concurrency becomes a
100% regression — and it only fills metrics the tables were silent on, because
a table stating its own delta is the PR's considered claim. It reads one pair
per metric mention, so a sentence quoting two shapes contributes the first.

**A blank second line means the PR states no percentage delta, not that it is
slow.** Four shapes legitimately produce nothing, and they are not bugs:

| Shape | Example |
|---|---|
| Bug/crash fix with descriptive tables only | before/after behaviour, log lines, no numbers |
| Absolute numbers with no delta stated | `Conc \| 4 \| 8 \| …` rows of raw tok/s — the extractor reports the PR's *claim*, it does not compute deltas itself |
| Accuracy-only tables | excluded by `EXCLUDE` on purpose |
| Kernel microbenchmarks only | `TP4 speedup 2.30x`, `µs` columns — no e2e metric is named, and a kernel win is not an e2e perf claim (see `memory/workflows/` on claiming perf only on `canonical-8k`) |

A table whose metric is named only in the **prose above it** (`| Batch size |
TP4 aiter | TP4 this PR | TP4 change |`, under a paragraph saying "decode step
time … in ms") still yields nothing from the table itself — headers and first
cells are all that is read. That shape is why the prose fallback exists.

Override a poor extraction with `watch.py set <pr> --note "..."`; check one
without a sweep via `python3 perf.py <pr>`. If a PR you expect numbers from
shows none, run that first — it prints what was collected and tells you whether
the PR is silent or the extractor missed it.

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

## Tabs

Each PR carries a free-text `group` — a model name, `debug`, whatever you want
to manage separately. The dashboard shows one tab per group (plus **All**), and
the tab filters the table, the triage panel **and the status block**, so you can
paste a standup entry for one workstream without editing it by hand.

Set a group by typing in the **Group** column — a plain text field, no
dropdown; an existing name moves the PR to that tab, a new one creates it.
(It had a datalist; the marker that adds costs real width in a column this
narrow, and typing the name was the only thing it ever did.) Or from the CLI:

```bash
python3 watch.py set 41133 --group "Qwen3.5 MoE"
python3 watch.py add 41870 --high --group debug
python3 watch.py report --group GDN     # just that tab
python3 watch.py set 41133 --group ""   # back to Ungrouped
```

**Drag a tab to reorder it**, Chrome-style — grab it anywhere and drop it where
you want; the tab moves as you drag, so the landing spot is visible before you
let go. The order is saved to `state.json` under `_config.group_order`. There is
no on-screen hint for this — the grab cursor on a tab is the affordance.

**All** and **Ungrouped** are fixed ends of the bar and are not draggable, which
is also what keeps a dragged tab inside the named range. The saved order is a
preference, not the source of truth: `all_groups()` reconciles it with the
groups that actually exist on every render, so a group deleted since the last
drag drops out and a group created since is appended alphabetically rather than
vanishing from the bar.

`Ungrouped` is always listed last so it never heads the tab bar, and a tab whose
last PR moved out falls back to **All** instead of leaving you on an empty table
with no way back.

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
python3 ~/agent-box/skills/ci/pr-ci-watch/watch.py sweep --track high --apply
```

Per watched PR it: reads `state / mergeable / mergeStateStatus / headRefOid /
author`; drops the PR from the watchlist if it is no longer `OPEN`; handles the
conflict case (below); otherwise classifies in-scope failed checks into
**real failing jobs** vs **gate-only** and stops with a TRIAGE request.

A PR swept in the last 30 minutes is skipped (so the daily track does not redo a
high sweep that just ran). `--force` or `--pr N` overrides.

### A merged or closed PR stays visible for 3 days

Landing used to be the one thing this tool could not report. The sweep deleted
a non-`OPEN` PR from the watchlist on sight, and only under `--apply` — so
between the merge and the next applying sweep the row was **actively wrong**
(`Pass`, the pre-merge job counts, and live `Re-run CI` / `Update branch`
buttons on a PR that had already landed), and after it the row simply vanished,
indistinguishable from one you removed by hand or never added.

`mergeable` makes it worse: GitHub stops computing it once a PR merges, so the
Merge column rendered the amber *"GitHub had not finished computing
mergeability"* pill — the only hint anything had changed, and a wrong one.

Now a landed PR keeps its row for `LANDED_GRACE_DAYS` (3), and says so:

| | |
|---|---|
| Merge | a filled `merged` (purple) or `closed` (red) pill, winning over draft/clean/conflict |
| Verdict / Action | `—` and `-` — same rule as a draft: nothing is claimed about CI that cannot change, and nothing is asked of anyone |
| Act now | one disabled button naming the state. Re-run and Update branch are meaningless here, and offering them was the bug |
| Status | `merged <when>` plus when the row drops, and that the counts below are history |
| Sort | below everything, including `L` — it is the one row that wants nothing from you |
| Status block | `<merged>` / `<closed>` token instead of `<CI clear>`, keeping the PR's perf claim on line 2 — which is what makes it a standup line |

The sweep **stops fetching CI and `behind_by`** for a landed PR: those cannot
change, and re-reading them each sweep spends API calls redrawing a row nobody
can act on.

`landed_at` is GitHub's `mergedAt`/`closedAt`, not when we noticed — a PR that
merges while nobody sweeps must not get a fresh 3 days from the next sweep.
First detection writes one line to `sweep.log` (`#28650 merged at …`), which is
what the `▸ history` panel reads, so the landing survives the row being dropped.
That flag keys off `landed_at`, **not** `state`: the old code already wrote
`state` before deleting, so keying on it would mean a PR that merged before this
existed never got its history line.

`×` clears a landed row immediately if 3 days is too patient.

### Drafts are skipped

A draft PR is still being written: its CI is the author's own scratchpad and its
code is not up for review. The sweep snapshots a draft and **stops there** — no
conflict notice, no CI triage, no re-run, no `run-ci` label. `pr-gate.yml`
agrees: it fails `Block draft PR` outright, so a watched draft would otherwise
show up as a mysterious all-red gate with nothing behind it.

This is read live from the PR on every sweep, not stored. Draft-ness is a fact
GitHub answers authoritatively and that changes without telling us, so the
watchlist deliberately keeps no second copy of it — a track is a sweep cadence
(`regular`, `high`) and nothing else.

A draft row says so in the **Merge** column — a filled `draft` pill, which
takes precedence over `clean`/`conflict`, because `clean` on a draft invites you
to read the row as ready to land. **Verdict** and **Action** both show a dash:
we do not read a draft's checks, so any verdict would be a claim we have not
earned, and nothing is being asked of anyone on account of its CI. Status is a
single dim line — no tally, no failure block.

The dash on Action holds even for a conflicting draft: a draft's conflicts are
the author's to find in their own time, which is the same reason the sweep does
not comment on them.

### Why a gate said no

A failed `pr-gate` is not one thing. `.github/workflows/pr-gate.yml` fails on
exactly one named step, and each one implies a *different* fix — so the sweep
spends one API call to read the failed step and reports which:

| reason | what actually happened | fix | re-run alone fixes it? |
|---|---|---|---|
| `draft` | PR was a draft when CI ran, and still is | author marks it Ready | no |
| `stale-draft` | CI ran while it was a draft; it is Ready **now** | re-run | **yes** |
| `missing-run-ci` | no `run-ci` label | add the label (`/tag-and-rerun-ci`) | no |
| `missing-label` | a workflow-specific opt-in label is missing | add the label | no |
| `rate-limit` | author is low-permission and inside the cooldown window | re-run **from an account with write access** | **yes** |
| `opt-in-extra` | `PR Test Extra` was never asked to run | nothing — this is expected | n/a |

Two of these are auto-resolved with no triage pass, because there is no log for
`/ci-analysis` to read — *no job ran at all*:

- **`rate-limit`** — the gate checks the run's **triggering actor**, not the PR
  author. A re-run pressed by someone with write access skips the check
  entirely. This is the one gate-only failure worth re-running, so
  `apply-verdict` lets it through without `--force-gates`.
- **`stale-draft`** — marking a PR Ready re-triggers nothing, so the draft gate
  sits red forever until someone re-runs it.

The rest become the `gated` action: the dashboard says **CI never started** and
names the missing label, and the sweep deliberately does *not* burn a re-run
attempt on a gate that will reject it again.

`opt-in-extra` is explicitly **not** a failure. Most PRs have not opted into
`PR Test Extra`, and treating its red gate as a problem would put a permanent
false alarm on most of the watchlist.

### Missing `run-ci` is fixed, not reported

Without the `run-ci` label the entire NVIDIA suite refuses to start, so an open
PR that lacks it is not waiting on CI — it is waiting on a one-word fix. Every
sweep checks the PR's labels **as they are now**, and an `--apply` sweep adds
the label itself.

Checked live rather than inferred from the recorded gate failure, because the
two disagree: #41982's gate died at `Block draft PR`, so the run-ci step was
never even evaluated — yet that label is what actually blocks it today. A live
`missing-run-ci` therefore overrides a stale gate reason, since it is what a
re-run would hit next.

Drafts are exempt: we leave them alone entirely.

**A blocking gate also takes over the Merge and Verdict columns.** `clean` is
true (there is no git conflict) but on its own reads as ready to land, and
`Pass` off the handful of admin checks that survive a blocked gate is a verdict
nobody earned — #41982 showed `clean` / `Pass` beside "4 pass, 4 fail" while
both NVIDIA workflows sat un-started. A blocking gate now renders Merge as
`gate blocked` and Verdict as **Fail**. Fail rather than a dash: nothing ran,
so it is not a test result, but the PR cannot merge and someone has to act —
and a dash reads as "no data yet" and sorts the row down beside the quiet ones. Only a *blocking* gate does this, so
ordinary rows keep the clean/conflict split the column exists for instead of
every row turning amber over an opt-in workflow.

**Every gate reason is still shown, including the non-blocking ones.** It is
not enough to decide a red is harmless — the fail count stays on the row either
way, and a count you cannot account for reads as two broken tests. A blocked
gate renders red (**CI never started**, plus what unblocks it); a harmless one
renders dim (*not a failure · `PR Test Extra` is opt-in…*), both annotated with
how many of the fails they cover, so the numbers always add up.

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
re-run those unless you pass `--force-gates` — except for the two re-runnable
gate reasons above (`rate-limit`, `stale-draft`), where the gate rejected the
*author*, not the code.

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
failing jobs it judged. A later sweep that sees the same head SHA keeps the
verdict instead of resetting to `awaiting-triage` — otherwise a `code-fix`
decision silently evaporates and the PR is re-triaged forever.

The match is a **subset** test, not equality. The failing set legitimately
shrinks between sweeps — a flaky shard goes green on its own, a cascade gets
demoted — and equality reads that as a new situation and throws away a verdict
that still accounts for every red that is left. A genuinely *new* failing job
re-opens triage (nobody has judged it); a new push does too, via the head SHA.

The judged action lives in `verdict_action`, separate from `last_action`.
`last_action` is scratch that any sweep may overwrite with `awaiting-triage`, so
holding the verdict off it loses the decision the moment one sweep re-opens
triage — and nothing can then put it back.

### Phase B — triage (the agent, via `/ci-analysis`)

**Do this in the same turn as the sweep, without being asked.** A sweep that
ends with "N PRs need triage" and stops has done half a job: the PR sits on
`awaiting-triage` until a human notices and pastes a command back, which is the
toil this skill exists to remove. If the sweep prints `TRIAGE REQUIRED`, work
the list immediately, record a verdict for each, and only then report.

Two things that are **not** reasons to stop and ask first:

- *"The live checks have moved on since the failure was recorded."* Re-sweep
  with `--force` and triage what is red now. A re-run finishing between the
  sweep and the triage is the normal case, not an obstacle.
- *"The verdict might be `code-fix`, which is serious."* `code-fix` is recorded
  like any other verdict — it is reported to the user, never commented on the
  author's PR, and it is what *stops* the tool re-running CI. Recording it
  early is the safe direction, not the risky one.

Ask the user only when the evidence genuinely does not separate two actions;
say which two and what would settle it.

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
bash ~/agent-box/skills/ci/pr-ci-watch/serve_dashboard.sh [--port 8812]
```

Stdlib HTTP server, no dependencies, loopback only (both `127.0.0.1` and `::1`,
so an editor port-forwarder that resolves `localhost` to IPv6 still works). The
script prints the Remote-SSH / `ssh -L` instructions; `--stop` shuts it down.

### Signing in to GitHub

**One dashboard per person, on their own machine.** This is not a service other
people log in to — it is loopback-only and has no concept of a user. Sharing the
skill means they run their own copy against their own watchlist, so sign-in
exists to spare a newcomer "first go and figure out `gh auth login`", not to
separate accounts.

The sign-in button **is** `gh auth login --web`. There is no OAuth App to
register, no client ID to configure, and `auth.py` never sees, stores or logs a
token — `gh` writes it to the system credential store, which is where every
`gh` call in `watch.py` already looks. Signing out is `gh auth logout`, so it
signs out the whole `gh` CLI, not just this page; the confirm dialog says so.

The only real work is that `gh auth login --web` is interactive: it prints a
one-time code and waits for Enter before opening a browser. A web handler has no
terminal, so it runs on a **pty**, the code is scraped from the output and shown
on the page in 30px type, and Enter is fed from our side with `BROWSER=true` so
a headless host does not try to launch one. The page polls every 3s while a code
is on screen, so it notices by itself the moment GitHub accepts it.

A **paste-a-PAT** field sits beside it for when a proxy blocks the device flow.
It goes to `gh auth login --with-token` on stdin and is never echoed back or
written to `sweep.log` — the audit line records only that a sign-in happened.

Scopes requested: `repo,read:org,gist`. If you are signed in with a token
missing `repo` or `read:org`, the page says so up front, because the failure
mode otherwise is that everything reads fine and only **Re-run CI** and
**Update branch** break when clicked — which looks like a broken button, not a
missing permission.

### Cross-site POSTs are refused

Loopback keeps the network out, not the browser: any page you visit can POST a
form to `http://127.0.0.1:8813/api/...` and your browser will send it — enough
to re-run CI, merge main into a PR, or start a Claude turn on your machine.
`do_POST` rejects a request whose `Origin` is not a loopback host. A *missing*
`Origin` is allowed: that is `curl`, not a forgery.

Columns: Rank, PR + title + author, group, track, merge state, Verdict, Action,
Auto, Status, last swept (Taiwan time). Auto-refreshes every 60s (every 15s
while a `Triage now` is in flight).

### Auto — the two per-PR switches

Two switches per row, **both default on**, stored at `watchlist[pr].auto`:

| Switch | On (default) | Off |
|---|---|---|
| `triage` | the PR enters the triage queue; the sweep's Claude turn runs `/ci-analysis` and records a verdict | the sweep records `triage-off` and moves on. It still reads and displays the CI — you just see the red without anything being judged |
| `re-run` | a verdict of `re-run` actually POSTs `rerun-failed-jobs` | the verdict is still recorded, the POST is withheld. Flip it back on and the re-run goes out with a reason already attached |

Off means **"keep watching this PR, do not act on it"** — the row is still swept
and still displayed. This is the per-PR version of the global `Monitoring OFF`
switch, and it replaces the old `hold` flag.

The `re-run` switch is enforced inside `apply-verdict`, not only in the sweep,
because that is the single place the POST is made — a hand-run command, the
sweep and the dashboard are all gated by the same check. `--force-auto`
overrides it from the CLI.

```bash
python3 watch.py auto 41133                        # show
python3 watch.py auto 41133 --rerun off            # set
python3 watch.py auto 41133 --triage on --rerun on
```

### Act now — the three buttons that skip the schedule

`Auto` is a *policy* ("from now on…"); `Act now` is an *action* ("this PR,
now"). They are separate columns because stacking them put `re-run` in one cell
twice meaning two different things.

**`Triage now`** spawns `claude -p … --allowedTools …` in the skill directory and
runs the whole pipeline for that one PR: `sweep --pr N --apply --force`, then
`/ci-analysis` and `apply-verdict`. The allowlist is `RUN_ALLOWED_TOOLS` in
`dashboard.py` — `gh`, this directory's `watch.py`, and the read-only tools.
It replaced `--dangerously-skip-permissions`, which Claude Code refuses under
`getuid() === 0`: fine while the dashboard ran on the host as a normal user,
fatal once it runs as root inside a container. It is the manual version of
the `triage` switch — same work, just not waiting up to 2h (or a day, on the
regular track) for the next sweep. Disabled when the `triage` switch is off.

The prompt forbids commenting on the PR and forbids pushing to any branch.
Guardrails, because this is the only button that spends tokens:

- **One PR per click** — there is no "run all".
- **`MAX_CONCURRENT_RUNS = 2`**, and a PR already running refuses a second click.
- **30-minute timeout** per run.
- `PR_CI_WATCH_CLAUDE` overrides the binary, `PR_CI_WATCH_CLAUDE_ARGS` appends
  flags (e.g. `--model` to run these on something cheaper than the default).

**`Re-run CI`** re-runs the failing workflows without waiting for a verdict —
you have decided the failure is not the PR's fault. Routed through
`apply-verdict --action re-run --force-auto`, so the gate-only rules, the
per-SHA attempt counter and the "already running" deferral are the ones the
sweep uses. Offered only when something is failing that a re-run could turn
green; gate-only reds get a disabled button. It **overwrites the stored
verdict** — an explicit click outranks what triage concluded, including
`code-fix`.

It refuses to re-queue a run that is not on the PR's current head, checked two
ways: the live PR head against the swept one, and each run's own `head_sha`.
This is not thrift. `pr-test.yml` sets `concurrency:
pr-test-pull_request-<n>-all` with `cancel-in-progress: true`, so re-running an
older run outranks the one already going and GitHub cancels the real CI —
"Canceling since a higher priority waiting request … exists". The jobs then
read *Cancelled after 5m*, which looks like a timeout and is not one. #39575 on
2026-10-02: a button drawn from a pre-push sweep re-ran `36947209065`
(`321d4ed0`) and killed `37021312519` on the then-current `41b89e5e`.

**`Update branch`** is GitHub's own button:
`PUT /repos/{repo}/pulls/{n}/update-branch`, with `expected_head_sha` so a click
on a stale page fails instead of merging into a head nobody looked at. GitHub
makes the merge commit **server-side, attributed to you**. It does not touch a
working tree, never rewrites the author's commits, and **cannot resolve a
conflict** — which is exactly why it is safe to offer on the 9-in-11 watched PRs
that belong to other people. Both buttons ask for confirmation first.

Four states, and the title attribute says which:

| State | Button |
|---|---|
| `behind_by > 0`, mergeable | live, amber (it is the only button that writes to a branch) |
| `behind_by == 0` | disabled — "up to date with main" |
| `behind_by is None` | disabled — never measured; the next sweep records it |
| `CONFLICTING` | disabled — GitHub would refuse; that is `/pr-conflict-fix` |
| already updated on this head | disabled — `updated ✓` |

```bash
python3 watch.py update-branch --pr 41133            # dry run
python3 watch.py update-branch --pr 41133 --apply
```

### `behind_by`, and why `mergeStateStatus` could not answer this

`BEHIND` only appears in `mergeStateStatus` when the repo **requires** branches
to be up to date, and sglang does not — so every out-of-date PR here reads
`BLOCKED` or `UNSTABLE` like any other, and the column that would have told you
never says anything. The sweep therefore spends one `gh api compare/main...<head>`
per PR and stores `behind_by`. It is shown under the `clean` pill, because
"clean" on a branch 677 commits behind reads as ready-to-land when it is not.

### `Main already has the fix` panel

Every PR whose verdict is `merge-main`, listed with the triage reason. It has no
button on purpose: most watched PRs belong to other people, and a dashboard that
can push to someone else's branch is a different tool with a different blast
radius. **Nothing is posted and nobody is told** — the panel exists so the
decision reaches you instead of sitting in a JSON field. `/pr-conflict-fix <pr>`
is what actually does the merge.

**Status is the only prose column, and it is silent when nothing is wrong.**
There used to be a separate `Red NVIDIA CI` column, but on a clean row it spent
its width saying so — *"no NVIDIA job is failing; the red is aggregation gates
mirroring out-of-scope vendor workflows"* on every passing PR. An empty cell
says that already. What survived is the part that only exists when something
broke (failing jobs with log links, the stale-after-re-run warning, the SAME
failure after re-run warning), and it moved into Status beside the job counts.

Two matching silences, same principle — don't restate `Pass`:

- `green` and `out-of-scope` are in `QUIET_ACTIONS`: their `ACTION_HINT` and
  stored `last_verdict` are both just long ways of saying "nothing to act on",
  and the Action column already shows `-`, so Status prints neither. A passing
  row is job counts and nothing else.
- The **last failure** block is a *fallback*, shown only when the live list is
  empty and the verdict is not `Pass`. That is the case it exists for — a
  re-run is in flight, so live checks have dropped back to "only gates are
  red". Beside a live failure list it would print the same jobs twice; on a
  green PR it is history, not status.

Column widths are pinned by a `<colgroup>` under `table-layout: fixed`. That is
load-bearing, not cosmetic: with auto layout the knobs in Rank / Group / Track
claim their intrinsic width first and squeeze **Status** into a
two-words-per-line ribbon. The knobs are capped at the width of the control
(`td.knob`, `select.mini`); Status gets 29%.

**Prose wraps; form controls do not.** A tight column costs a sentence an extra
line, but it makes a `<select>` clip its own label — `P0` losing its `0` — and
makes a `<button>` overhang, because neither can shrink below its content. So
the knob columns are sized for *label + native dropdown arrow*, the knob
selects run a size smaller (`select.mini`: 11px, 2px side padding), and both
the reorder handle (`⠿`) and the remove button (`×`) are glyphs rather than
words. Widen a knob column before you widen a prose one.

Four traps, each of which put a control's border across the next column:

- **Every `<col>` is a percentage, summing to 100.** Mix `px` and `%` and the
  browser rescales *everything* proportionally once the declared widths exceed
  the table — the `px` columns get squeezed too, which is the overflow you were
  trying to avoid. The floor is held by `table { min-width: 1150px }` plus
  `overflow-x: auto` on the panel: below that width, scroll rather than crush.
  1150px is where the narrowest knob column still fits its control; check that
  before lowering it.
- **A bare `input[type=text]` rule out-specifies `.grp` / `.note`.** An
  attribute selector counts like a class, so `0-1-1` beats `0-1-0`. The add
  box's `min-width: 280px` was therefore applying to the in-table Group field
  and painting its border straight across Track and Merge. That rule is now
  scoped `form.add input[type=text]`; keep it scoped.
- **`.pill` needs `max-width: 100%`** or a long verdict word overhangs its cell.
- **A `<button>` label is a hard width floor.** It neither wraps nor shrinks, so
  the word "remove" in a 3% column simply overhangs. Use a glyph plus `title` +
  `aria-label`.

If you add a column, add a `<col>` and re-balance to 100 — and check the floor
widths (`1250px × n%`) against what each control actually needs, not against
what its text needs.

**Per-PR history (`▸ history`).** The **toggle** sits at the foot of the
**Status** cell, because history is that column's long form — how the state got
here. The **panel** it opens is a full-width `<tr>` below the row, where the
long verdict summaries have room to read; indenting it to Status would spend
two thirds of the table on margin.

The toggle is labelled rather than a bare triangle: under a stack of other
small blocks in Status, a lone glyph reads as punctuation, not a control. Only
the triangle rotates when open.

That the panel is a second `<tr>` is the one thing it costs — row dragging has
to move the pair together (`histOf` / `tailOf` / `draggedHist`), or a panel
ends up describing whichever row it lands under.

Nothing new is recorded for this — `history.py`
reconstructs the timeline from the two files the sweep already writes, neither
of which reads as a story alone: `sweep.log` holds every mutation but
interleaved across all PRs, and `sweeps/*.json` holds every sweep's view of
every PR including the long runs where nothing changed.

So the log is filtered to one PR and the sweep records are collapsed to their
**transitions** — a new head SHA (the author pushed), a changed CI state. The
collapsing is the load-bearing part twice over: 36 sweeps of an idle PR is 36
identical rows, and because only transitions survive, **a state line's
timestamp *is* the moment the PR entered that state**. "Since when has this
been green" is read straight off the top green line, with no arithmetic.

Lines are coloured by how bad the news is — green clean, red failing, amber
in-flight or stuck, grey bookkeeping:

```
10/01 13:44  author pushed — new head cf9e0e67                        (grey)
10/01 13:23  re-run in flight — watcher died on its own (31 pass…)    (amber)
10/01 12:58  verdict merge-main — hicache 3FS eval accuracy 0.005…    (amber)
10/01 12:57  CI red — a real NVIDIA job failed (31 pass, 7 fail…)     (red)
10/01 00:56  verdict code-fix — AttributeError 'GDNAttnBackend'…      (red)
09/30 22:59  conflicts with main — CI cannot complete                 (red)
```

Three judgements behind that, each of which would otherwise mislead:

- **`out-of-scope` IS the green state**, not a third thing: every red check is
  an aggregation gate mirroring a vendor workflow, so no NVIDIA job is failing.
  Calling it anything else would hide the moment a PR came good.
- **…unless jobs are still queued.** `out-of-scope` is recorded as soon as
  nothing *in scope* is failing, which happens while most of the run is still
  pending; that reads as `CI running — nothing failing so far`. Calling it
  green there would put the "went green" timestamp an hour before CI finished.
- **Collapsing is on the state, not the raw outcome.** `needs-triage` followed
  by `verdict-held:code-fix` is one unbroken stretch of red; printing both
  would read as two separate failures.

The job counts ride along as evidence — "CI green" beside `59 pass, 2 fail` is
honest about the two gates still red, where the bare word reads like a
contradiction.

Verdict summaries are free text typed into `apply-verdict`, so everything
lifted out of the log is HTML-escaped in `_kind()`; only the markup that
function adds itself is literal.

**Quick add:** paste a PR link into the box at the top and pick a track. Accepts
`https://github.com/sgl-project/sglang/pull/41870`, a `/files` deep link, `#41870`,
or `41870`.

**Buttons, and what each can actually reach:**

| Button | Does | Needs Claude? |
|---|---|---|
| `Sign in to GitHub` | Runs `gh auth login --web` on a pty and shows the one-time code. No OAuth App, no token stored by this skill. A PAT paste field is the proxy fallback | no |
| `sign out` | `gh auth logout` — signs out the whole `gh` CLI, not just this page. Confirms first | no |
| `Monitoring ON/OFF` | Writes `_config.enabled`; every sweep, including a cron-fired one, exits immediately when off | no — takes effect instantly |
| `Refresh now` | Runs `sweep --track all --force` **without `--apply`** in a subprocess: re-reads merge state and red NVIDIA CI for every PR. Comments nothing, re-runs nothing | no |
| `Notify author` | Runs `watch.py notify --pr N --apply` — posts the conflict notice for that PR now instead of waiting for the next sweep. Shown **only** when the PR is conflicting and not yet notified for this head SHA; asks for confirmation first. The result (sent, with the comment URL — or why not) comes back as a banner, and the PR is re-swept so a button standing on stale state disappears | no |
| `⠿` grip | Drag a row by its grip to reorder it within its sort bucket. Manual order is a tiebreaker only, so a drop into another bucket is **refused** (the target row outlines in red) rather than accepted and sprung back on reload | no |
| `▸ history` (foot of Status) | Expands that PR's history as a full-width row below — every verdict, re-run, conflict notice and push, newest first. Fetched from `/api/history` on first open, so a page that redraws every 60s does not pay for panels nobody opened | no |
| Track dropdown | Sets `regular` or `high` explicitly, both directions | no |
| `triage` / `re-run` (Auto column) | Writes `watchlist[pr].auto.<field>`. Both default on; off means "keep watching this PR, do not act on it" | no — takes effect on the next sweep |
| `Triage now` (Act now) | Spawns a headless `claude -p`: sweep, `/ci-analysis`, record the verdict, re-run if the verdict says so. The manual version of the `triage` switch. Posts no comment, pushes to no branch. Capped at 2 concurrent | **it is one** — the only button that starts a Claude turn, and it costs tokens |
| `Re-run CI` (Act now) | `apply-verdict --action re-run --force-auto` — re-runs the failing workflows now, skipping triage. Overwrites the stored verdict | no |
| `Update branch` (Act now) | `PUT /pulls/{n}/update-branch` — GitHub merges main into the PR branch server-side, attributed to you. Cannot rewrite the author's commits, cannot resolve a conflict. Confirms first | no |
| `Copy` (triage panel) | Copies `/pr-ci-watch triage <prs>` to paste into Claude | yes, to run it |
| `Copy` (status block) | Copies the `<P0><CI clear><PR…>` report as **rich text + plain text**. `CI clear` is `<b>` in the rich flavour, and `<PR41133>` stays a hyperlink when pasted into Teams | no |

Both `Copy` buttons degrade: rich clipboard → `navigator.clipboard.writeText`
→ `document.execCommand` → revealing the plain-text box with the content
selected. `navigator.clipboard` **does not exist outside a secure context**, so
reaching `localhost` by IP rather than through a port forward leaves only the
last two rungs; touching it unguarded throws and the button appears dead.

`Refresh now` is deliberately read-only — it re-reads every PR and decides
nothing. Deciding **re-run vs merge main vs real bug** means reading job logs,
which is `/ci-analysis` and needs a model. `Triage now` is the button that
admits this and starts one, for a single PR; the dashboard process itself still
judges nothing. `Re-run CI` and `Update branch` are you making that call by
hand instead. Registering cron likewise needs a Claude turn.

**Verdict** is the in-scope CI state only — `Pass` / `Pending` (in flight,
first run or re-run) / `Fail`. A red *aggregation gate* never makes it `Fail`: a
gate is not a job, it is mirroring a vendor workflow this tool excludes, so
calling that Fail would report someone else's failure as this PR's NVIDIA
result. Rows are ordered **Pass first** (those are the ones you can go merge), then
P0 → P2, then **conflicts last within each rank** (nothing can progress on
them until the author rebases), then your manual drag order, then PR number.
**L (low priority) is always last**, after every P0–P2 row, whether it is
Pass, pending, conflicting, or any other status. The Rank column is the old
Pri column, with `L` added. **Action** is what to do — `CI re-run` (one is in flight),
`Solve conflict`, `Merge main`, `Code fix`, `Triage`, `Wait upstream`, or `-`.
**Status** carries the job counts (`29 pass, 2 fail, 3 running, 6 queued`), the conflict
notice proof, and the triage reason; it flags the counts as stale when they
predate a re-run.

`gh pr checks` lumps `QUEUED` and `IN_PROGRESS` into one `pending` bucket, but
they answer different questions — queued means the runners are busy, in
progress means it is actually testing — so the sweep splits them using the
`state` field and `tally_bits()` renders `running` as pending minus queued.
`queued` is stored as a *subset* of `pending`, so every existing `pending`
check (the Verdict column, the clean-CI print) keeps working untouched, and a
tally recorded before the split simply shows everything as running.

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

`cron-sweep.sh` is that headless variant, available as a belt-and-braces Phase A
from the host crontab. **It must set its own `PATH`**: cron hands a job
`/usr/bin:/bin`, `gh` lives in `~/bin`, and every sweep it ran died on
`FileNotFoundError: 'gh'` — silently, because the traceback went to `cron.log`
and the dashboard shows Claude-cron results, which were fine. It now exports the
PATH and aborts with a readable message instead of a traceback if `gh` is still
missing. If you ever wonder whether it is working, read `cron.log`, not the
dashboard.

**So does triage happen on its own? Yes.** The schedule is a Claude cron job,
not a shell script, so the scheduled sweep runs Phase B in the same turn and
clears the "Waiting on triage" panel without being asked. The panel says so
explicitly — it used to say "needs a Claude turn" and "the next sweep picks them
up automatically" in one breath, which reads as a contradiction unless you
already know the sweep *is* a Claude turn.

## Data

`$AGENT_SCRATCH_DIR/pr-ci-watch/` (override with `PR_CI_WATCH_DIR`):

| file | holds |
|---|---|
| `watchlist.json` | PR → track, added, note, repo, `auto{triage,rerun}` (absent means on) |
| `state.json` | per PR: head SHA, `behind_by`, `update_branch{sha,at}`, `conflict_comment_sha`, `reruns{workflow:{sha,count}}`, last verdict/action, last sweep. Plus `_config`: `enabled`, `group_order` (the dragged tab order) |
| `sweeps/<ts>.json` | one record per sweep. Also the source of the `▸` history panel's push/outcome transitions — don't prune it without meaning to shorten that |
| `sweep.log` | append-only audit of every mutation (sign-ins are recorded, tokens never are). The other half of the history panel |

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
- **A dry-run sweep still writes its decision to state.** `sweep` without
  `--apply` performs no mutation on GitHub, but it does record `last_action` so
  the dashboard reflects the latest read. A row can therefore say `CI re-run`
  with nothing re-run yet — the Action cell distinguishes *decided, not yet
  applied* from a re-run GitHub actually refused, and only the latter is
  recorded (as `rerun_deferred`) by a real `--apply` attempt.
- **`mergeable: UNKNOWN`** means GitHub is still computing, not that anything is
  wrong — asking is what schedules the computation, so the first answer on a
  recently-touched PR is routinely UNKNOWN. The snapshot polls (5s, 10s, 20s,
  30s), returning the moment it resolves. A single retry was not enough: two PRs
  stored UNKNOWN and sat that way on the dashboard while the real answer had
  been available within the minute — and a stale UNKNOWN survives until the next
  sweep, which on the regular track is a whole day.
  If it is *still* unknown after the backoff, the conflict check is deferred to
  the next sweep (CI checks are evaluated as normal); it is never treated as
  clean. The previous known value is **not** kept either — UNKNOWN appears
  precisely when the branch just changed, which is when a conflict is most
  likely to have appeared, so a stale `clean` is the one wrong answer that
  actually misleads.
- **Draft PRs stay on the list** and keep their track — snapshotted every
  sweep, but never triaged, labelled or re-run. CI on a draft is the author's
  scratchpad and is often intentionally red.
- **"4 pass, 4 fail" can mean CI never ran.** When a `pr-gate` blocks a
  workflow, every job under it is *skipped*, so the counts collapse to a handful
  of admin checks and look like a small test failure. Always read the gate
  reason before concluding anything about the tests.
