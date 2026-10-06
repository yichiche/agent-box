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
3. **Consumers.** Follow each new name to the statement that loads the stored
   field and the statement that uses it. A line number in a table is not the
   consumer. Show those lines. A new value proves “same result” only when one
   of these is visible in the code:

- the `if` is false, so the new value is not read
- the `if` is true but the default is an identity on the old variable
  (`maximum(group_locs, member_rows + 0) == group_locs`)

The unit is one function, titled `function · line`, the same string as the
canvas node. Before lines are the base revision. After and consumer lines are
PR head.

When the read is gated, keep one NVIDIA input for the before/after. Show the
gate in order: the flag, the function that returns None while the flag is
false, then the load and use lines. Put the guarded input’s values in the
last column of that reader table. Do not add a second trace, and do not add
a closing Guard section.

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
`Card`, `CardHeader`, `CardBody`, `Divider`, `DiffView`, `Pill`, `Swatch`,
`CollapsibleSection`, and `useHostTheme`. Colors come from `useHostTheme()`.
No hardcoded hex. `Swatch` `color` is a palette name (`green`, `orange`,
`blue`), not a hex. `CardHeader` takes children, not `title`. Put `key` on
the `<g>`, not on a custom node component.

The canvas shows why NVIDIA behavior is identical: original outputs stay, and
each new name is either unread or an identity. Extra execution and a wider
interface are Trace stats. Hardware is the other card. One NVIDIA input for
the before/after. The guarded input appears only as the last column beside
the lines that use the new names.

1. `H1` `PR N code path`, then one `Text tone="secondary"` naming the PR and
   where the new statements live.
2. `Grid columns="2fr 1fr"`. Left card `Affected Scope`: one colored `Stat`,
   NVIDIA behavior identical (`success` on Yes, `danger` on No or Unproven).
   Then `Divider`, `Text` `Trace`, and three uncolored stats: Common path
   changed, NVIDIA execution identical, NVIDIA interface identical. Right
   card `AMD Guard`: one `Stat`, Hardware scope (`AITER`, `gfx950`, `All
   backends`). `success` when the guard contains the new behavior, `danger`
   when it does not. Center that stat in the card body, both axes. No
   sentence under either stat. No Can-merge callout here.
3. `H2` `Who runs the new statements`. Two-column caller svg. Legend:
   `NVIDIA, unchanged` / `Common path NVIDIA also runs` / `New gfx950 path`
   (rename the blue label to this PR’s guard). Edges are real calls only.
   Green (`theme.category.green`) only when the node computes nothing new.
   Orange (`theme.category.orange`) when NVIDIA runs it and it computes new
   values, even if the original outputs stay; the subtitle names the
   additions first. Blue (`theme.accent.primary`) is the guarded path.
   Fill `theme.bg.elevated`. Text from `theme.text`. One `Text` under the
   svg says which chains do not call each other. Each node is a `<g
   style={{ cursor: "pointer" }}>` whose `onClick` scrolls to
   `document.getElementById(id)`.
4. One `CollapsibleSection` per graph node, titled `function · line`, the
   same string as the node. `<div id={id} style={{ scrollMarginTop: 16 }} />`
   sits before the section, not inside it. A right-hand node jumps to the
   section of the function it lives in. The interface section (wider return,
   new argument, or new field) is first and `defaultOpen`. Every other node
   starts closed, so a closed header is the boundary of that node. Order
   after the interface: the function that stores the new names, then the
   function that reads them. A caller that only forwards the value, and a
   helper that returns before the new work on NVIDIA, go under `Does not
   change the NVIDIA store`.
5. Inside a node: side-by-side `DiffView`. Left is base, every line
   `unchanged`. Right is head, kept lines `unchanged` and new lines `added`,
   same region so the old lines line up. Then the argument table
   (`Argument`, `Before`, `After`, `Read by the old store?`). New rows say
   `absent` under Before. For each new name, the next section is the
   consumer, not another copy of the return. When the read is gated, show
   three steps as code before any summary table:
   - Step 1, the flag.
   - Step 2, the function that returns None while the flag is false. Name
     that return.
   - Step 3, one card per name, titled `name · loaded at LINE, used at
     LINE`, containing the load and the use. Then a table: the name, that
     use, `not executed` on NVIDIA, and the value on the guarded input.
   Same result only when that `if` is false, or it is true and the value is
   an identity.
6. Cannot merge: a `Callout` at the end with the concrete code change.
7. `Text tone="tertiary" size="small"` citing the PR-head paths and lines.

Do not add a statement inventory, a second-input comparison table, or a
closing Guard paragraph. Hardware scope stays in the AMD Guard card.

## Related

| Skill | Question it answers |
|---|---|
| [`/pr-merge-triage`](../SKILL.md) | Automated pre-review gate. This skill lives in that folder. |
| [`/pr-test-seam`](../pr-test-seam/SKILL.md) | Do the tests enter through the owning interface? |
| [`/sglang-pr-review`](../../sglang-pr-review/SKILL.md) | Is the AMD behavior correct, after pre-review passes? |
