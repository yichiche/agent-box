---
name: pr-merge-triage
description: How I review a PR into sgl-project/sglang from the AMD side, plus a script that takes a PR number and prints a filled-in checklist — blast radius, is_hip/use_aiter guards, new flags, new-feature vs bug-fix, AMD-only vs shared bug, new kernel vs kernel upgrade — ending in a merge-ease verdict. Use when the user says '/pr-merge-triage', asks 'is this PR easy to merge', 'how hard is this to land upstream', 'review guidelines', or wants a shareable write-up of the review bar.
category: deliver
---

# How I review a PR into SGLang — and how to triage a new one fast

This document is two things on purpose:

1. **A write-up of the review bar**, meant to be read by other people —
   contributors who want their PR to land without three rounds of review, and
   reviewers who want to apply the same bar I do.
2. **A triage tool.** `triage.py <pr>` answers the mechanical half of that bar
   in one command and prints the checklist already ticked, so the first pass
   over a new PR takes a minute instead of twenty.

The context it is tuned for: **AMD-side changes going upstream into a codebase
that NVIDIA, CPU, NPU and others also ship from.** That shapes every rule below.
The cost of a mistake is asymmetric — a bug in AMD-only code costs AMD users; a
bug in shared code costs everybody and gets reverted.

---

## Quick start

```bash
python3 ~/agent-box/skills/pr-merge-triage/triage.py 41870
python3 ~/agent-box/skills/pr-merge-triage/triage.py https://github.com/sgl-project/sglang/pull/41870
python3 ~/agent-box/skills/pr-merge-triage/triage.py 41870 --json   # for scripting
```

Output is a markdown checklist you can paste straight into a review, plus a
verdict and a numbered "ask the author" list:

```
| | Check | Verdict | Risk | Evidence |
|---|---|---|---|---|
| - [ ] | Blast radius | CHECK | +1 | shared files, additive only: arg_groups/choices.py, … |
| - [x] | AMD guard | PASS | 0 | n/a — no existing shared behaviour rewritten |
| - [ ] | Guard choice **GATE** | CHECK | 0 | imports aiter; no is_hip/use_aiter token added |
…
| | **Total** | | **2** | band: LOW (LOW ≤3, MEDIUM ≤7, HIGH ≤12, SPLIT >12) |

**Verdict: BLOCKED ON AUTHOR** — open gate(s): Guard choice
```

It reads the PR with `gh` (prefixed `GH_TOKEN=""`, per `_shared/repo-config.md`)
and needs nothing else — no checkout, no build.

---

## The bar: four guidelines

These are the four things I actually check for, in the order they decide
whether a PR is cheap or expensive to land.

### 1. Minimize global variables and flags

If something should be enabled by default, **detect the hardware and enable it**
— `is_hip` / `use_aiter` — instead of shipping a knob the user has to find and
export. A feature behind a default-off env var is a feature almost nobody runs,
which means it is also a feature almost nobody tests.

`python/sglang/srt/environ.py` already carries **~650 registered knobs**. Every
new one is a permanent combination that someone, eventually, has to reason about
alongside all the others. The bar for adding one is: *the right value genuinely
depends on something the code cannot observe* — a user policy, a workload trade,
a risk the user accepts. "Which GPU am I on" is not that; the code can see it.

> If the flag exists because the path is not trusted yet, say so in the PR and
> give the condition for flipping the default. A knob with no plan to become the
> default is a knob forever.

### 2. Minimize code and interface changes unless necessary

Every signature you change is every caller you break, including callers in trees
you cannot see. Prefer:

- a keyword argument with a default that preserves today's behaviour, over a
  positional parameter inserted in the middle;
- a new function beside the old one, over a rewrite of the old one;
- leaving unrelated names alone — a rename that touches 200 lines buries the 20
  lines that are the actual change.

Diff noise is not free. It is paid by every reviewer, on every round.

### 3. Minimize changes to common code

This is the one that decides the review cost.

- **All changes behind `is_hip` / `use_aiter`, no common path affected** → an
  AMD-side review is sufficient. This is the cheap PR.
- **Change lands in code every vendor inherits** → you need **someone from the
  community to join the review**. That is not a formality; they are the ones who
  know what else depends on it. Plan for the extra round-trip, or find a way to
  keep the change out of the shared path.

Which guard to use:

| Guard | Use when |
|---|---|
| `is_hip()` / `_is_hip` | the kernel or path works on **all AMD GPUs** |
| `_use_aiter` (`get_bool_env_var("SGLANG_USE_AITER") and is_hip()`) | the code **imports from the AITER library** — `is_hip()` alone will run it on an AMD box that has no AITER installed, and crash |

The established idiom is a module-level constant, evaluated once:

```python
from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()
```

### 4. Split a complicated PR into several smaller PRs

One concern per PR. The usual split for AMD work:

1. the kernel (with its own correctness test and benchmark),
2. the model/backend wiring that calls it,
3. the default flip, once the first two have soaked.

A 1,500-line PR spanning a kernel, a model, a dispatcher and a flag does not get
a careful review — it gets a slow one, and then a shallow one. Three 300-line
PRs land faster in wall-clock time than one 900-line PR, because each one is
reviewable in a sitting.

---

## The classification: what the checklist asks

Ten axes. `triage.py` answers them from the diff; the ones it cannot decide are
marked and listed below.

| # | Axis | Why it matters |
|---|---|---|
| 1 | **Blast radius** — AMD-only paths / new files / existing shared files / hot common code | decides whether you need a community reviewer |
| 2 | **AMD guard** — is every *rewritten* hunk in shared code under `is_hip`/`use_aiter`? | an unguarded rewrite changes NVIDIA behaviour by accident |
| 3 | **Guard choice** — `is_hip` vs `use_aiter` | wrong one = crash on AMD boxes without AITER |
| 4 | **New flags** — new env vars, and their defaults | guideline 1; default-off AMD capability is the common smell |
| 5 | **New globals** in shared files | global state is the hardest thing to unpick later |
| 6 | **Interface churn** — signatures rewritten in shared code | guideline 2; every caller is a risk |
| 7 | **Kernel** — new kernel vs upgrade of an existing one | new ⇒ needs reference test + benchmark + fallback; upgrade ⇒ needs before/after on the same shapes |
| 8 | **Size** — lines and number of areas touched | guideline 4; the split trigger |
| 9 | **Evidence** — accuracy and perf numbers in the body | a numerics change with no GSM8K is unreviewable |
| 10 | **Tests** — is a test file in the diff? | AMD-only tests belong in `test/registered/amd/` |

**Hot common code** (auto-escalates): `layers/linear.py`, `layers/layernorm.py`,
`layers/activation.py`, `rotary_embedding.py`, `models/`, `managers/`,
`model_executor/`, `distributed/`, `mem_cache/`, `environ.py`, `arg_groups/`,
`server_args.py`, `entrypoints/`.

---

## What the script cannot decide

Four questions stay with the reviewer. They are also the four that most often
flip a verdict, so do not skip them.

**Is the guard the right one, and is it in the right place?**
The script matches guard tokens inside a hunk and its context window. It cannot
see a guard forty lines up, and it cannot tell a guard that covers the call from
one that covers only the import. `CHECK` means *go read it*.

**Is this one concern or three?**
Line count is a proxy, not the thing. A 600-line PR that adds one kernel and its
test is one concern. A 200-line PR that adds a kernel, rewires a dispatcher and
flips a default is three, and should be three PRs.

**New feature, or fix for an existing bug?**
This changes the bar, in both directions:

- *New feature* → must be opt-in-by-hardware, must not alter existing behaviour,
  needs tests. Risk is contained; it can land behind a guard and soak.
- *Existing bug* → **the regression window matters.** Find out when it broke and
  who is affected. A fix for a bug that is live on main is more urgent than a
  feature and deserves to skip the queue; it also deserves a regression test, or
  it comes back.

**Is the bug AMD-only, or does it hit both platforms?**
Decide this before you decide the guard:

- *AMD-only* → fix it in AMD-only code, guarded. Cheap, AMD-side review.
- *Both* → the fix belongs in common code **unguarded**, and now you need a
  community reviewer and a test that runs on NVIDIA CI. Wrapping a shared bug fix
  in `is_hip` to keep the review cheap is the wrong trade: it leaves the bug live
  for everyone else and leaves a confusing guard behind.

One more the script is deliberately harsh about: **additive-only common edits**
— adding a member to a choices enum, adding a branch keyed on a new backend
name — are flagged as common-code touches, because they are. But they are the
cheap kind: nothing existing changes behaviour. Say so in the review and move on.

---

## What must pass, and what merely adds up

Three mechanisms, deliberately kept apart. Collapsing them into one number is
the thing that makes a scorecard useless — a PR can be flawless on nine axes and
still be unmergeable because of the tenth.

### 1. Gates — binary, nothing buys them off

A gate marks something that is **a crash, a silent wrong answer, or a claim
nobody can check**. A PR with an open gate is not "higher risk"; it is
unreviewable until the gate is answered. There are only three, so that the word
keeps its force:

| Gate | Why it cannot be traded away |
|---|---|
| **Guard choice** | code importing AITER gated by `is_hip()` alone crashes on an AMD box without AITER. Not a style preference — a crash |
| **Evidence** | a numerics change with no accuracy number can only be believed, not reviewed |
| **Interface churn** | a rewritten shared signature must have every caller updated **in the same PR**, or main breaks for someone else |

### 2. Risk points — these add up

Everything else is a trade-off, scored 0–4:

| Axis | 0 | +1 | +2 | +3 | +4 |
|---|---|---|---|---|---|
| Blast radius | AMD-only paths | new file outside AMD path, **or shared files additive-only** | edits a shared file | | edits hot common code |
| AMD guard | all rewritten hunks guarded | guard not visible in hunk | | rewrites shared code unguarded | |
| New flags | none | new, default-on | | new, default-off | |
| New globals | none | | in shared code | | |
| Interface churn | none | | | signature rewritten | |
| Kernel | none touched | upgrade of existing | new kernel | | |
| Size | ≤300 lines | ≤800 lines | | >800 or >3 areas | |
| Evidence | numbers present | kernel without perf number | | | |
| Tests | test file in diff | | none in diff | | |

**Bands**: `LOW ≤3` · `MEDIUM ≤7` · `HIGH ≤12` · `SPLIT >12`

> **Additive-only is the discount that matters.** Adding an enum member, an
> accessor, or a validation branch keyed on a name nothing selects yet is a
> shared-code touch that changes nobody's behaviour. It scores +1, not +4. The
> script ignores removed *import* lines when deciding this — widening an import
> to pull in one more helper is not a behaviour rewrite.

### 3. Routing — who has to look at it

Independent of the score. **Any edit (not addition) to hot common code, or any
unguarded rewrite of shared code, needs a community reviewer** — even at 0
points. This is a schedule, not a rejection: name the owner of that code in the
PR description early, rather than discovering after two weeks that nobody with
merge rights has read it.

### Putting it together

| Verdict | Condition |
|---|---|
| **LOW RISK — MERGE** | no open gate, ≤3 points, no shared-code edit → AMD-side review is enough |
| **MEDIUM/HIGH RISK — REVIEW** | no open gate, 4–12 points → read the flagged rows before approving |
| **BLOCKED ON AUTHOR** | any open gate, at any score |
| **NEEDS COMMUNITY REVIEWER** | hot-common edit or unguarded shared rewrite, at any score |
| **SPLIT FIRST** | >12 points → ask for the split *before* reviewing; a review of the whole is wasted work |

**The short answer to "how many points is safe":** ≤3 **and** zero open gates
**and** no shared-code edit. Points alone never clear a PR — a 0-point PR with
an unanswered accuracy question is still blocked, and a 6-point PR that only
edits AMD files is fine.

---

## Ready-to-paste review comments

These are the four comments I write most often. Written to be neutral and
actionable — the author should know exactly what unblocks the PR.

**Unguarded common code**
> This hunk runs on every backend, not only AMD. Could you gate it with
> `_is_hip` (all AMD GPUs) or `_use_aiter` (if it needs the AITER library)? As
> written it changes the NVIDIA path too, which widens the review a lot.

**Default-off flag for a hardware capability**
> Rather than a new env var, could this default on when the hardware supports
> it — `_is_hip` / `_use_aiter` — so users don't have to know to export
> anything? `environ.py` is already at ~650 knobs; one that the code can decide
> for itself is worth not adding.

**Needs a split**
> This is doing N things: (1) …, (2) …, (3) …. Could we land them as separate
> PRs — kernel + test first, then the wiring, then the default flip? Each is
> reviewable in one sitting and the kernel can start soaking while the rest is
> still in review.

**Shared bug wrapped in an AMD guard**
> If this bug also reproduces on NVIDIA, the fix should go in the common path
> unguarded — the `is_hip` wrapper leaves it live for everyone else. Happy to
> help get a community reviewer on it.

---

## Related skills

| Skill | When |
|---|---|
| [`/sglang-pr-review`](../sglang-pr-review/SKILL.md) | after triage says it is mergeable — the deep correctness pass (weight loading, forward-path variants, scales, collectives) |
| [`/ci-analysis`](../ci-analysis/SKILL.md) | the PR's CI is red and you need to know whose fault it is |
| [`/pr-ci-watch`](../pr-ci-watch/SKILL.md) | keep watching a set of PRs and re-run CI that deserves it |
| [`/validate-pr`](../validate-pr/SKILL.md) | you need the accuracy/throughput numbers the Evidence row is asking for |

Typical flow for a PR landing from the AMD side:

```
/pr-merge-triage N      →  is it cheap to land? what do I ask for?
/sglang-pr-review N     →  is it correct?
/validate-pr            →  numbers, if the PR has none
/pr-ci-watch add N      →  babysit CI until it is green
```
