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
runtime path. Test-seam analysis lives in `/sglang-pr-review`; do not include
it here.

## Run

```bash
python3 ~/agent-box/skills/pr-code-path/path_cover.py 39575
python3 ~/agent-box/skills/pr-code-path/path_cover.py https://github.com/sgl-project/sglang/pull/39575
```

The script prints production facts: added and removed executable lines, guard
tokens, moved statements, and removed bare returns. It does not decide the
three questions. Write the report from those facts plus a read of any function
the script marks `REALIGN` or `# moved`.

The calibration case is [examples.md](examples.md) (PR 39575).

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

2. **Visualize link.** One canvas link, nothing else after it. Put the path
   picture, guard quotes, and value traces in that canvas. When execution flow
   or the internal interface is **No**, the canvas must contain one three-block
   trace per such edit (before, after on the same input, consumers, then the
   same `if`s with the hardware-specific values that make them true). The
   calibration traces are in [examples.md](examples.md). Canvas file:
   `~/.cursor/projects/<workspace>/canvases/pr-<number>-code-path.canvas.tsx`.

## Related

| Skill | Question it answers |
|---|---|
| [`/pr-merge-triage`](../pr-merge-triage/SKILL.md) | How expensive is this to land? |
| [`/sglang-pr-review`](../sglang-pr-review/SKILL.md) | Is the behavior correct, and do tests enter through the changed interface? |
