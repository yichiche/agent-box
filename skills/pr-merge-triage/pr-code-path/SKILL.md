---
name: pr-code-path
description: >-
  Visualize how an sglang PR changes the original code path and answer three
  merge questions: where it touches the common path, whether NVIDIA execution,
  internal interfaces, or numerical behavior change, and which AMD hardware
  the guard actually enables (all HIP, AITER, or gfx-specific). Use when the
  user says '/pr-code-path', asks whether a PR affects NVIDIA, whether an AMD
  guard contains the change, or which AMD GPUs are affected. When NVIDIA
  execution flow or the internal interface is not identical, prove numerical
  identity with a before/after value trace in code.
category: deliver
---

# /pr-code-path — where a PR touches the original path

Three questions, every time, in this order:

1. **Common Path** — which added or changed statements run for every vendor?
2. **Will this affect NVIDIA original behavior?** — answer at three separate layers:
   - Is the execution flow completely identical?
   - Is the internal interface completely identical?
   - Are NVIDIA numerical results and original behavior identical?
3. **Affected Hardware Scope** — all AMD (`is_hip`), AITER-enabled AMD
   (`use_aiter`), or one architecture such as gfx950 (`is_gfx95_supported`)?

Do not collapse these into “100% AMD-only.” A common-path instruction can be
new while NVIDIA numerical behavior remains identical. That distinction is the
main result.

`/pr-merge-triage` scores how expensive the PR is to land. This skill draws the
runtime path. Test-seam analysis lives in `/pr-test-seam`; do not include it
here. `/sglang-pr-review` reads the JSON this skill already wrote and does not
call this skill again.

## Run

```bash
python3 path_cover.py 39575
python3 path_cover.py https://github.com/sgl-project/sglang/pull/39575
```

Run these from the directory that contains this file. It writes `/tmp/pr-<number>-src/focus.md`: every changed
function at base and at head, with line numbers. A long function is the edited
branch plus the fallthrough after it. One level of functions that branch calls
is included. The stdout is only an index.

Read `focus.md` and stop. Do not open the source files under `base/` or `head/`.
Do not run `gh`, `git diff`, or `grep`. A `>` line was added. A `<` line was
removed. A `callee` section is one function the edited branch calls. Line numbers
in that file are the ones the value trace cites. A function marked `REALIGN`
in the index is already whole inside `focus.md`.

The script does not decide the three questions. Write the report from `focus.md`.

Do not re-read this skill. Do not open [examples.md](examples.md). The
calibration case is PR 39575.

## 1. Common Path

For each production function in the facts, name who calls it and what it calls. Draw the path the request already took, then hang each added or deleted executable statement on it.

When the facts say `REALIGN`, tag a line `# moved`, or a hunk moves a `return` or an `else`, open the function at the PR head and read it whole before classifying any line. Unified diffs attach an old `return` to a new function, and they reprint a moved statement as added.

Skip comments. A new function is its own node; its call site is a separate node. Both get a row.

A guard token in a hunk does not cover the lines around it. These are common
path:

- The line sits in the same block as code that already ran, not indented under the new `if`.
- The helper's body is guarded, but the caller invokes it unconditionally. The call is common-path; the body is AMD-only.
- A shared object is read (`forward_batch.extend_prefix_lens_cpu`) outside the guard, even if the value is only used inside it.
- A `return` was removed from a branch everyone takes, so later lines are now reachable there.
- An existing condition was edited (`else` became `if old or flag`).
- A shared struct, return tuple, or signature grew. A default keeps old callers working; the surface is still common.

For each common-path edit, classify what changed:

- extra read, call, branch, allocation, or tensor operation
- internal interface change (signature, tuple arity, struct field)
- changed error condition
- changed numerical result or externally visible behavior
- no-op off the guard

## 2. Will this affect NVIDIA original behavior?

Always report all three layers. Never infer one from another.

### A. Execution flow

Ask whether NVIDIA executes any new read, call, condition, allocation, or tensor
operation. If yes, answer **No, not completely identical**, even if the operation
is a no-op.

### B. Internal interface

Ask whether a private or public signature, return arity, metadata shape, required
attribute, or caller contract changed. If yes, answer **No, not completely
identical**. Say whether real production callers already satisfy the contract
and whether only fixtures or direct internal callers need updates.

### C. Numerical result and original behavior

If A or B is **No**, this layer is not a sentence. It is a concrete execution
trace. Pick one NVIDIA input and show before / after in code, with the value of
every variable the edit touches. Do this for every edit that changes execution
or an interface. PR 39575 has two: the `extend_prefix_lens_cpu` read, and
`_qsa_write_plan` growing from 4 return values to 7.

The trace has three code blocks:

1. **Before**, only the statements that ran, with values.
2. **After** on that same input, with values. Mark statements NVIDIA newly executes.
3. **Consumers.** Follow each new argument, return value, or field to the `if`

The first lines of every before, after, and consumer block name the file and
the line, taken from that side's revision (before = base, after = PR head):

```python
# qwen_sparse_attn_backend.py · _qsa_build_write_plan · line 562
# qsa_indexer.py · _forward_impl · line 691
```

The unit is one function. A scenario that crosses functions is split, and
each piece is headed by `file · function · line`. The canvas card title uses
the same `file · function`.
   that reads it. Show the NVIDIA values of that condition, then the variables
   the old path still uses (`group_locs`, `source_keys`, the original return
   slots). A new value proves “same result” only when one of these is visible
   in the code:

- the `if` is false, so the new value is not read
- the `if` is true but the default is an identity on the old variable
  (`maximum(group_locs, member_rows + 0) == group_locs`)

Then show the same `if`s with the hardware-specific values that make them true,
so the scope of the new behavior is the value of the guard, not a separate claim.

Answer **Yes, identical** only when that trace shows:

- the original outputs retain the same values
- new values are not read, or are an identity on the old inputs
- original assertions and errors still fire under the same NVIDIA conditions
- final stored tensors / model outputs are unchanged

Do not call execution “completely identical” when only the numerical result is
identical. Mention possible overhead from extra operations separately.

## 3. Affected Hardware Scope

Name the narrowest condition that enables the new behavior:

| Guard | Affected hardware scope |
|---|---|
| no platform guard | all backends that reach the path |
| `is_hip()` / `_is_hip` | all AMD GPUs using the HIP build |
| `_use_aiter` / `use_aiter` | AMD only when AITER is installed and enabled |
| `is_gfx95_supported()` | gfx950 / MI355 only; excludes MI300 and NVIDIA |
| `is_gfx94_supported()` or explicit gfx check | only the named architecture family |

If guards are nested, report the intersection. For example,
`is_gfx95_supported() and _use_aiter` means gfx950 with AITER, not all gfx950.
If a guard is inside a helper but its call is common, report both:
“common call; gfx950-only body.”

## Report

The chat reply is only these two parts. Do not print the path tree, evidence
tables, hardware table, or value-trace code in the chat.

1. **Final conclusion first**, then **Analysis**. No tables. Each analysis
   label is on its own line, and its answer is the bullet under it.

```
Final conclusion:
- Can merge. [One sentence: why NVIDIA is unaffected and where the new behavior is gated.]

Analysis:
Common path:
- Yes/No. [What every backend now executes.]
NVIDIA execution flow identical:
- Yes/No.
NVIDIA internal interface identical:
- Yes/No.
NVIDIA numerical results and original behavior identical:
- Yes/No/Unproven.
Affected hardware scope:
- [All backends / All AMD (`is_hip`) / AITER-enabled AMD (`use_aiter`) / gfx950 / MI355 only (`is_gfx95_supported()`)].
```

When the numerical layer is No or Unproven, or the new behavior is not
contained by the guard the PR claims, the verdict is **Cannot merge**, and the
reply must say what to change. Suggestions stay under Final conclusion:

```
Final conclusion:
- Cannot merge. [One sentence: what changes on NVIDIA or escapes the guard.]
- Suggestion: [the concrete code change, e.g. move `X` under `if is_gfx95_supported():`, keep the old return arity, restore the assert on non-gfx95]
- Suggestion: [the value trace that would prove it after the change]
```

Extra execution or a wider internal interface alone is not a reason for
Cannot merge. This verdict covers only the code-path question; CI, approvals,
and correctness of the AMD math are separate gates.

2. **Visualize link.** One canvas link, nothing else after it. The canvas
   is the path picture, the diffs, and the value traces. Follow **Canvas**
   below. Do not print those sections in the chat.


## When called from /pr-merge-triage

Do the same analysis, including the value trace when execution flow or the
internal interface is not identical. The only production code you read is
`/tmp/pr-<number>-src/focus.md`. Do not run `gh`, `git diff`, or `grep` to
find a line or a callee. Do not print the chat report. Do not read or edit
`triage.py`. Write the canvas by the **Canvas** section below. Do not open
`sdk/*.d.ts`, `examples.md`, or another PR's canvas. Do write
the canvas to `~/.cursor/projects/<workspace>/canvases/pr-<number>-code-path.canvas.tsx`
with the editor Write tool in this session, so the canvas host registers it.
A file that is only on disk, or a sidecar status of `canvas-missing`, opens
as canvas not found. If the user reports that, edit the `.canvas.tsx` again
in this session; repeating the link does not register it. Write `/tmp/pr-<number>-code-path.json` using the schema in
`/pr-merge-triage`. `nvidia_behavior_identical` is `true` only when that trace
shows the original outputs unchanged. Use `"unproven"` when the trace was not
done. `owner_action` is the Affected Scope fix, required only when
`nvidia_behavior_identical` is not `true`. Then also set `nvidia_diff_line`
to the PR-head `path:line` that changes NVIDIA behavior, and `nvidia_diff_why`
to one sentence on why that line differs on NVIDIA.

`guard_contains_new_behavior` is a separate boolean. Set it true only when the
new behavior runs inside `hardware_scope`. When it is false, set `guard_line`,
`guard_why`, and `guard_action`. Do not copy the NVIDIA line, reason, or fix
into those fields. Triage prints each failed row as its own Line, Why, and Fix.
`conclusion` does not decide either row.

## Canvas

Write `~/.cursor/projects/<workspace>/canvases/pr-<number>-code-path.canvas.tsx`
with the editor Write tool. Import only from `cursor/canvas`. Default-export
the top-level component. Read `~/.cursor/skills-cursor/canvas/SKILL.md` once
for host rules. Do not open `sdk/*.d.ts`, `examples.md`, or another PR's
canvas. This section is the layout.

Use `Stack`, `Grid`, `Row`, `H1`, `H2`, `Text`, `Stat`, `Callout`, `Table`,
`Card`, `CardHeader`, `CardBody`, `Divider`, `DiffView`, `DiffStats`, `Pill`,
`Swatch`, and `useHostTheme`. Colors come from `useHostTheme()`. No hardcoded hex.

Put the sections in this order:

1. `H1` `PR N code path`, then one `Text tone="secondary"` naming the PR and
   where the new statements live.
2. `Grid columns="2fr 1fr"` of two cards. The behavior answer and the
   hardware guard are separate questions, so they do not share a stat row.
   Left `CardHeader` is `Affected Scope`. One `Stat`, NVIDIA behavior
   identical. Yes only when original NVIDIA outputs are unchanged.
   `tone="success"` when Yes, `tone="danger"` when No or Unproven. That is
   the only colored stat on this card. The card shows the stat only. Then
   `Divider`, a `Text` reading `Trace`, and `Grid columns={3}` of `Stat`
   with no tone: Common path changed, NVIDIA execution identical, NVIDIA
   interface identical. A No in the trace stays uncolored.
   Right `CardHeader` is `AMD Guard`. One `Stat`, Hardware scope, a short
   token (`AITER`, `gfx950`, `All backends`). `tone="success"` when the
   guard contains the new behavior, `tone="danger"` when it does not. That
   tone does not follow the behavior stat. The card shows the stat only,
   centered horizontally and vertically in the card body.
3. Do not put a Can merge callout above the trace, and do not put a
   sentence under either stat. A Cannot merge callout still goes at the
   end, with the concrete code change.
4. `H2` `Who runs the new statements`. One svg, two columns, in the same
   shape as a caller graph. A `Row` legend uses `Swatch`: green
   `NVIDIA, unchanged`, orange `Common path NVIDIA also runs`, blue
   `New gfx950 path`. Column headers name the side. Each node is a rect
   with a title `function · line` and a one-line subtitle. Draw an edge
   only for a real caller-to-callee step. Do not connect unrelated chains.
   Unchanged NVIDIA nodes use `theme.category.green`. A common path NVIDIA
   also executes uses `theme.category.orange`, including a node whose
   original outputs stay the same while it computes new values. The subtitle
   names the added values first. The new guarded path uses
   `theme.accent.primary`. Fill is `theme.bg.elevated`. Text uses
   `theme.text.primary`, `theme.text.secondary`, and `theme.text.tertiary`.
   One `Text` under the svg: edges are real calls, which chains do not call
   each other, and what NVIDIA does instead of the right-hand path.
   Every node is clickable. Wrap its rect in a `<g>` with
   `style={{ cursor: "pointer" }}` and an `onClick` that calls
   `document.getElementById(id)?.scrollIntoView({ behavior: "smooth" })`.
   Each section in step 5 starts with `<div id={id} />` before its `H2`. A
   right-hand node jumps to the section of the function it lives in.
5. The behavior proof. Interface changes come first: a wider return, a new
   argument, a new field. For each added name, the next table follows it to
   the statement that reads it and names the `if`. If that `if` is false
   unless a gfx or AITER check is true, say so in the row. A caller that
   only forwards the value, or a helper that returns before the new work on
   NVIDIA, goes below that follow-through under `Does not change the NVIDIA
   store`. One `Text` states the single
   NVIDIA input. Each `H2` is the graph node's `function · line`. Its
   before/after is that node's own statements. Each `H2` starts with the
   code, then the tables. The code stays:
   a `Grid columns={2}` of two cards. The left card is `Pill` `before` and the
   base statements in a `DiffView` (`type: "unchanged"`). The right card is
   `Pill` `after` and the head statements; kept lines are `unchanged`, new
   lines are `added`. Show the same region on both sides so the old lines
   line up with the new ones. Then:
   - `Argument`, `Before`, `After`, `Read by the old store?` — one row per
     original argument and one row per new argument. Original rows show the
     same value on both sides. New rows say `absent` under Before.
   - `New argument`, `Value`, `The if`, `Why the old store is unchanged` —
     one row per new argument. Name the `if` and its NVIDIA value. Same
     result only when that `if` is false, or it is true and the value is an
     identity (`maximum(group_locs, member_rows + 0) == group_locs`).
   Then one table, `Same if on gfx950`, with columns `Argument`, `NVIDIA`,
   `gfx950`, using the values that make those `if`s true.
   Do not add a statement inventory. The caller graph, the side-by-side
   code, and the tables are the trace.
6. `H2` `Guard`. One paragraph: the narrowest condition, which hardware it
   includes, and any added statement outside it.
7. When the verdict is Cannot merge, a `Callout` with the concrete code
   change.
8. `Text tone="tertiary" size="small"` citing the PR-head paths and lines.

## Related

| Skill | Question it answers |
|---|---|
| [`/pr-merge-triage`](../SKILL.md) | Automated pre-review gate. This skill lives in that folder. |
| [`/pr-test-seam`](../pr-test-seam/SKILL.md) | Do the tests enter through the owning interface? |
| [`/sglang-pr-review`](../../sglang-pr-review/SKILL.md) | Is the AMD behavior correct, after pre-review passes? |
