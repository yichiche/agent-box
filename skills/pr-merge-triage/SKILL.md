---
name: pr-merge-triage
description: >-
  Automated pre-review gate for a PR into sgl-project/sglang. Use when the
  user says '/pr-merge-triage', asks whether a PR is ready for human review,
  or wants a standard checklist that tells the PR owner exactly which
  requirements failed. A pass means a human may start reading the code. It
  never approves a merge and never prints LGTM. This folder also contains
  /pr-code-path and /pr-test-seam, so installing this one skill is the whole
  pre-review.
category: deliver
---

# /pr-merge-triage — automated pre-review

`pr-code-path/` and `pr-test-seam/` in this folder are part of this skill.
Copy this folder and both analyses come with it. Run the scripts from the
directory that contains this file.

This is the first review pass. It replaces reading the diff by hand to decide
whether a PR is worth a human review. The user should be able to act on the
report without opening the code.

A failing row is a requirement the owner must fix. A passing report is not an
approval.

```text
BLOCKED — SPLIT FIRST
BLOCKED — REQUIREMENTS FAILED
BLOCKED — AUTOMATION INCOMPLETE
READY FOR HUMAN REVIEW
```

`READY FOR HUMAN REVIEW` means the checklist passed and a person may now read
the implementation. Do not write `MERGE`, `LGTM`, or `approve`.

## Procedure

The chat reply is the final script output and nothing else. Do not paste the
code-path report or the test-seam report. The script output includes the CI
links and the code-path canvas link; do not omit them.

```bash
# 1. Shape. Stop if it says SPLIT FIRST, unless the user explicitly asked
#    to continue.
python3 triage.py N --shape

# 2. Judgment files. Follow each skill's Run section once. If this session
#    already followed that skill, do not read it again.
#    path_cover writes /tmp/pr-N-src/focus.md. That is the only production
#    code you read. Do not run gh, git diff, or grep to find a line or a callee.
#    Run test_seam.py once, without --json. It prints the test body.
#    Do not fetch the test diff. Do not read or edit triage.py.
#    The table below is the score. A contribution-guide miss fails Unit Test
#    Quality. Coverage that is not complete does not.
#    Read ~/.cursor/skills-cursor/canvas/SKILL.md once. Do not open
#    sdk/*.d.ts, examples.md, or another PR's canvas.
#    Do not print those reports.
#    pr-code-path/SKILL.md
#    pr-test-seam/SKILL.md
#    python3 pr-code-path/path_cover.py N
#    python3 pr-test-seam/test_seam.py N
#    Canvas: ~/.cursor/projects/<workspace>/canvases/pr-N-code-path.canvas.tsx
#    Write that canvas with the editor in this session. A disk-only file, or
#    a sidecar status of canvas-missing, opens as canvas not found. If the
#    user reports that, edit the .canvas.tsx again; do not only repeat the link.

# 3. The only user-facing report. Pass the judgment files even when step 1
#    said SPLIT FIRST and the user asked to continue.
python3 triage.py N \
  --code-path /tmp/pr-N-code-path.json \
  --test-seam /tmp/pr-N-test-seam.json
```

`N` may be a PR number or a GitHub pull URL.

On `BLOCKED — SPLIT FIRST`, stop after step 1 unless the user explicitly asked
to continue. Continuing runs steps 2 and 3. The verdict stays
`BLOCKED — SPLIT FIRST` while One concern fails, and the other rows are part
of that report.

On `BLOCKED — AUTOMATION INCOMPLETE`, the judgment files are missing or do not
match the schema. Finish them and re-run step 3. Do not fill a blocked row by
hand and do not ack it.

## Who decides each row

| Check | Source | Fail means |
|---|---|---|
| Affected Scope | `/pr-code-path` JSON | NVIDIA numerical results or original behavior are not identical |
| AMD Guard | `/pr-code-path` JSON, plus an AITER import in the diff | the new behavior is outside the claimed guard, or AITER is imported under a scope other than `aiter` |
| Unit Test Quality | `/pr-test-seam` JSON | the contribution-guide test bar fails (`official_pass` is false). Coverage other than `complete` does not fail the row |
| Accuracy evidence | PR body, when attention, MoE, quantization, or a kernel changes | no accuracy number is present |
| Performance evidence | PR body, when a kernel file changes | no throughput, latency, or TTFT number is present |
| Flags | new `SGLANG_*` env bindings | a default-off binding whose comment is missing, incomplete, or only names the platform |

Unit Test Quality passes when the contribution-guide bar passes. That bar is `official_pass`. It passes when the PR follows the contribution guide and `test/README.md`:

- a bugfix or feature has a corresponding unittest
- the test uses the stdlib `unittest` framework
- one test function covers one scenario
- the name states the purpose
- asserts check the result. `return True`, or an assert that cannot fail, does not count
- tests clean up and do not affect one another
- small models, a reused server, and the file stays under 500 seconds

Coverage is recorded and does not decide the row. `mixed`, `past_the_seam`, `reimplements`, and `no_test` leave Unit Test Quality passing when `official_pass` is true. Still write `missing_test`. Do not read or edit `triage.py`. This table is the score.

A test that returns True is not a new check. It leaves the behavior untested, so `official_pass` is false and the row fails. A copied kernel is `reimplements` or `mixed`. That coverage result does not fail the row by itself.

A new `SGLANG_*` binding that defaults to `False` or is empty passes when the comment immediately above it states a user policy the code cannot infer: who sets it, when, and the choice. A comment that only names the platform, AITER, ROCm, CUDA, or a gfx check fails. So does a missing comment. The fix is then a hardware detection (`is_hip`, `use_aiter`, or a gfx check), or a comment that states the policy.

```python
# User policy: off keeps the accurate kernel. Set true only when the
# caller accepts a lower score for higher speed. The default stays off.
SGLANG_ALLOW_ACCURACY_LOSS = EnvBool(False)
```

```python
# Enable on MI350 when AITER is available.
SGLANG_USE_MXFP4_GEMM = EnvBool(False)  # fails: this is a platform check
```

A PR that is too large, or that spans more than three areas, still stops as `BLOCKED — SPLIT FIRST`. That row is shown only when it fails.

Hunk-level `is_hip` token matches are not a verdict. An import of `is_hip`
does not mean the behavior applies to all AMD GPUs. The hardware scope string
in the code-path JSON is the only guard result.

Affected Scope passes when NVIDIA numerical results and original behavior are
identical, including when execution flow or the internal interface is not
identical. Say that distinction in the evidence. Do not call the change
additive unless the code-path summary says so.

AMD Guard is a separate question. It passes when `guard_contains_new_behavior`
is true and, if the diff imports AITER, `hardware_scope` is `aiter`. It does
not read `conclusion` or `nvidia_behavior_identical`. A NVIDIA behavior change
does not fail AMD Guard. A guard that lets the wrong AMD GPU in does not fail
Affected Scope. When both fail, the owner comment gives each row its own line,
reason, and fix.

## Code-path JSON

Write `/tmp/pr-N-code-path.json` after the `/pr-code-path` analysis. Do not
guess `nvidia_behavior_identical`. Use `true` only when the value trace shows
the original outputs unchanged. Use `"unproven"` when that trace was not done;
that fails Affected Scope.

```json
{
  "conclusion": "can_merge",
  "common_path_changed": true,
  "nvidia_execution_identical": false,
  "nvidia_interface_identical": false,
  "nvidia_behavior_identical": true,
  "hardware_scope": "gfx950",
  "hardware_scope_detail": "gfx950 / MI355 only (is_gfx95_supported())",
  "guard_contains_new_behavior": true,
  "summary": "NVIDIA keeps its original outputs and alignment assertion.",
  "owner_action": ""
}
```

`conclusion` is `can_merge` or `cannot_merge`. Triage does not use it to pass
or fail Affected Scope or AMD Guard. `hardware_scope` is one of
`all_backends`, `all_amd`, `aiter`, `gfx950`, `gfx94`, `other`.

`guard_contains_new_behavior` is true only when every new behavior runs inside
`hardware_scope`. When it is false, AMD Guard fails. Also set:

- `guard_line`: PR-head `path:line` of the statement outside the guard.
- `guard_why`: one sentence on which hardware that line reaches.
- `guard_action`: the concrete change that puts the statement under the guard.

`owner_action` is the Affected Scope fix. It is required only when
`nvidia_behavior_identical` is not `true`.

When `nvidia_behavior_identical` is not `true`, Affected Scope fails. Also set:

- `nvidia_diff_line`: PR-head `path:line` of the statement that changes NVIDIA
  behavior. One line, the one a reviewer can open.
- `nvidia_diff_why`: one sentence on why that line behaves differently on
  NVIDIA than the code it replaced.

The owner comment prints each failed row as `Line`, `Why`, and `Fix`. The
two fixes stay different even when one edit would satisfy both.

## Test-seam JSON

Write `/tmp/pr-N-test-seam.json` after `/pr-test-seam`.

```json
{
  "official_pass": true,
  "official_evidence": "unittest covers the change; one scenario per function; the name states the purpose; asserts check the result; tests do not share a server; the file stays under 500s.",
  "official_action": "",
  "conclusion": "mixed",
  "changed_behavior": "For an unaligned prefix, snapshot ring keys before project_qk stores over them, then recompress the crossing group.",
  "correct_seam": "_forward_impl",
  "missing_test": "Call _forward_impl with an unaligned prefix and assert the stored compressed key, letting production snapshot, project, and overwrite."
}
```

`official_pass` is stage 1. Set it true only when every contribution-guide
item above holds. `no_test` means `official_pass` is false. `official_action`
is required when `official_pass` is false, and it is the concrete test change.
`official_evidence` is the sentence printed on the Unit Test Quality row.

`conclusion` is `complete`, `mixed`, `past_the_seam`, `reimplements`, or
`no_test`. Record it. It does not fail Unit Test Quality. Only `official_pass`
does. Every conclusion other than `complete` still requires `missing_test`,
naming the production interface, the trigger, and the observable result.

## Calibration

PR 39575 is the live calibration case. Shape, Affected Scope, and AMD Guard
pass. The guard is `gfx950`, not all AMD. Coverage is `mixed`: the recompress
tests step past `_forward_impl`. That does not fail Unit Test Quality, because
`official_pass` is true. The row stays passing.

`python3 triage.py --calibrate` locks the
two rows apart:

| Example | Affected Scope | AMD Guard |
|---|---|---|
| `examples/39575-code-path.json` | PASS | PASS |
| `examples/scope-only-code-path.json` | FAIL | PASS |
| `examples/guard-only-code-path.json` | PASS | FAIL |
| `examples/both-differ-code-path.json` | FAIL | FAIL, distinct line / reason / fix |
| `examples/aiter-import-code-path.json` | PASS | FAIL, AITER import only |

`examples/39575-test-seam.json` sits next to those files.

## What this skill does not do

It does not decide whether the AMD math is correct or whether every forward
path was updated. That is `/sglang-pr-review`, and only after this checklist
is `READY FOR HUMAN REVIEW` and `/pr-ci-watch` reports `CI clear`. This skill
does not judge CI.
