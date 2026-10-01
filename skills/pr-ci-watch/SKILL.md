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
/pr-ci-watch sweep [--track high|regular|draft|all] [--pr N…] [--apply] [--force]
/pr-ci-watch triage <pr>…   # /ci-analysis + apply-verdict for PRs already swept
/pr-ci-watch report         # the <P0><CI clear><PR…> status block
/pr-ci-watch pause | resume # kill switch; a paused sweep is a no-op
/pr-ci-watch dashboard [--port 8812]
/pr-ci-watch arm            # register the two cron tracks
```

Entries render as a markdown bullet with the perf line nested under it —
`- <P0><CI clear><PR41134>Title` / `  - TPOT 6.3% improvement`. The rich flavour
is a real nested `<ul>`, so a paste into Teams lands as a proper list; the plain
flavour uses `- ` and two-space indent, which Teams also turns into bullets.

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
python3 ~/agent-box/skills/pr-ci-watch/watch.py sweep --track high --apply
```

Per watched PR it: reads `state / mergeable / mergeStateStatus / headRefOid /
author`; drops the PR from the watchlist if it is no longer `OPEN`; handles the
conflict case (below); otherwise classifies in-scope failed checks into
**real failing jobs** vs **gate-only** and stops with a TRIAGE request.

A PR swept in the last 30 minutes is skipped (so the daily track does not redo a
high sweep that just ran). `--force` or `--pr N` overrides.

### The `draft` track — watched, but not checked

A draft PR is still being written: its CI is the author's own scratchpad and its
code is not up for review. The sweep snapshots a draft and **stops there** — no
conflict notice, no CI triage, no re-run on the author's behalf. `pr-gate.yml`
agrees: it fails `Block draft PR` outright, so a watched draft would otherwise
show up as a mysterious all-red gate with nothing behind it.

The track moves itself. When a watched PR is a draft the sweep sets
`track: draft` and remembers the one it came from in `prev_track`; the first
sweep after it is marked Ready puts it straight back. You can also set it by
hand (`add --draft`, or the dropdown) — the sweep will correct it either way,
because the PR's own draft flag is the source of truth, not the watchlist.

A draft row renders as a single dim line and nothing else: no tally, no failure
block. Red gates that were never a verdict do not belong in a CI column.

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
failing jobs it judged. A later sweep that sees the same head SHA and the same
failures keeps the verdict instead of resetting to `awaiting-triage` — otherwise
a `code-fix` decision silently evaporates and the PR is re-triaged forever. A new
push, or a different set of failures, correctly re-opens triage.

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
bash ~/agent-box/skills/pr-ci-watch/serve_dashboard.sh [--port 8812]
```

Stdlib HTTP server, no dependencies, loopback only (both `127.0.0.1` and `::1`,
so an editor port-forwarder that resolves `localhost` to IPv6 still works). The
script prints the Remote-SSH / `ssh -L` instructions; `--stop` shuts it down.

Columns: Pri, PR + title + author, group, track, merge state, Verdict, Action,
Status, last swept (Taiwan time). Auto-refreshes every 60s.

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
load-bearing, not cosmetic: with auto layout the knobs in Pri / Group / Track
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
| `Monitoring ON/OFF` | Writes `_config.enabled`; every sweep, including a cron-fired one, exits immediately when off | no — takes effect instantly |
| `Refresh now` | Runs `sweep --track all --force` **without `--apply`** in a subprocess: re-reads merge state and red NVIDIA CI for every PR. Comments nothing, re-runs nothing | no |
| `Notify author` | Runs `watch.py notify --pr N --apply` — posts the conflict notice for that PR now instead of waiting for the next sweep. Shown **only** when the PR is conflicting and not yet notified for this head SHA; asks for confirmation first. The result (sent, with the comment URL — or why not) comes back as a banner, and the PR is re-swept so a button standing on stale state disappears | no |
| `⠿` grip | Drag a row by its grip to reorder it within its sort bucket. Manual order is a tiebreaker only, so a drop into another bucket is **refused** (the target row outlines in red) rather than accepted and sprung back on reload | no |
| `▸ history` (foot of Status) | Expands that PR's history as a full-width row below — every verdict, re-run, conflict notice and push, newest first. Fetched from `/api/history` on first open, so a page that redraws every 60s does not pay for panels nobody opened | no |
| Track dropdown | Sets `regular` or `high` explicitly, both directions | no |
| `Copy` (triage panel) | Copies `/pr-ci-watch triage <prs>` to paste into Claude | yes, to run it |
| `Copy` (status block) | Copies the `<P0><CI clear><PR…>` report as **rich text + plain text**, so `<PR41133>` stays a hyperlink when pasted into Teams | no |

Both `Copy` buttons degrade: rich clipboard → `navigator.clipboard.writeText`
→ `document.execCommand` → revealing the plain-text box with the content
selected. `navigator.clipboard` **does not exist outside a secure context**, so
reaching `localhost` by IP rather than through a port forward leaves only the
last two rungs; touching it unguarded throws and the button appears dead.

`Refresh now` is deliberately read-only. Deciding **re-run vs merge main vs real
bug** means reading job logs — that is `/ci-analysis`, which needs a Claude turn,
so no button can do it. Registering cron likewise needs a Claude turn.

**Verdict** is the in-scope CI state only — `Pass` / `Pending` (in flight,
first run or re-run) / `Fail`. A red *aggregation gate* never makes it `Fail`: a
gate is not a job, it is mirroring a vendor workflow this tool excludes, so
calling that Fail would report someone else's failure as this PR's NVIDIA
result. Rows are ordered **Pass first** (those are the ones you can go merge), then
P0 → P2, then **conflicts last within each priority** (nothing can progress on
them until the author rebases), then your manual drag order, then PR number. **Action** is what to do — `CI re-run` (one is in flight),
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

## Data

`$AGENT_SCRATCH_DIR/pr-ci-watch/` (override with `PR_CI_WATCH_DIR`):

| file | holds |
|---|---|
| `watchlist.json` | PR → track, added, note, repo |
| `state.json` | per PR: head SHA, `conflict_comment_sha`, `reruns{workflow:{sha,count}}`, last verdict/action, last sweep. Plus `_config`: `enabled`, `group_order` (the dragged tab order) |
| `sweeps/<ts>.json` | one record per sweep. Also the source of the `▸` history panel's push/outcome transitions — don't prune it without meaning to shorten that |
| `sweep.log` | append-only audit of every mutation. The other half of the history panel |

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
- **Draft PRs stay on the list** on the `draft` track — snapshotted every sweep
  so the move back happens by itself, but never triaged or re-run. CI on a draft
  is the author's scratchpad and is often intentionally red.
- **"4 pass, 4 fail" can mean CI never ran.** When a `pr-gate` blocks a
  workflow, every job under it is *skipped*, so the counts collapse to a handful
  of admin checks and look like a small test failure. Always read the gate
  reason before concluding anything about the tests.
