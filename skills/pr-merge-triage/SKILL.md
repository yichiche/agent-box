---
name: pr-merge-triage
description: >-
  Automated pre-review gate for a PR into sgl-project/sglang. Use when the
  user says '/pr-merge-triage', asks whether a PR is ready for human review,
  or wants a standard checklist that tells the PR owner exactly which
  requirements failed. A pass means a human may start reading the code. It
  never approves a merge and never prints LGTM.
category: deliver
---

# /pr-merge-triage — automated pre-review

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
code-path report, the test-seam report, or a canvas into the chat.

```bash
# 1. Shape. Stop if it says SPLIT FIRST.
python3 ~/agent-box/skills/pr-merge-triage/triage.py N --shape

# 2. Judgment files. Follow the two skills, including their value trace and
#    seam classification. Write only the JSON files; do not print their reports.
#    ~/agent-box/skills/pr-code-path/SKILL.md
#    ~/agent-box/skills/pr-test-seam/SKILL.md

# 3. The only user-facing report.
python3 ~/agent-box/skills/pr-merge-triage/triage.py N \
  --code-path /tmp/pr-N-code-path.json \
  --test-seam /tmp/pr-N-test-seam.json
```

`N` may be a PR number or a GitHub pull URL.

On `BLOCKED — SPLIT FIRST`, stop after step 1. Do not spend a code-path or
test-seam pass on a PR that has to be split.

On `BLOCKED — AUTOMATION INCOMPLETE`, the judgment files are missing or do not
match the schema. Finish them and re-run step 3. Do not fill a blocked row by
hand and do not ack it.

## Who decides each row

| Check | Source | Fail means |
|---|---|---|
| Affected Scope | `/pr-code-path` JSON | NVIDIA numerical results or original behavior are not identical |
| AMD Guard | `/pr-code-path` JSON, plus an AITER import in the diff | the new behavior is outside the claimed guard, or AITER is imported under a scope other than `aiter` |
| Unit Test Quality | `/pr-test-seam` JSON | the conclusion is not `complete` |
| Accuracy evidence | PR body, when attention, MoE, quantization, or a kernel changes | no accuracy number is present |
| Performance evidence | PR body, when a kernel file changes | no throughput, latency, or TTFT number is present |
| Flags | new `SGLANG_*` env bindings | a default-off hardware knob must become a hardware detection, or the owner must state the user policy it represents |

A PR that is too large, or that spans more than three areas, still stops as `BLOCKED — SPLIT FIRST`. That row is shown only when it fails.

Hunk-level `is_hip` token matches are not a verdict. An import of `is_hip`
does not mean the behavior applies to all AMD GPUs. The hardware scope string
in the code-path JSON is the only guard result.

Affected Scope passes when NVIDIA numerical results and original behavior are
identical, including when execution flow or the internal interface is not
identical. Say that distinction in the evidence. Do not call the change
additive unless the code-path summary says so.

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
  "summary": "NVIDIA keeps its original outputs and alignment assertion.",
  "owner_action": ""
}
```

`conclusion` is `can_merge` or `cannot_merge`. `hardware_scope` is one of
`all_backends`, `all_amd`, `aiter`, `gfx950`, `gfx94`, `other`.
`owner_action` is required when `conclusion` is `cannot_merge` or
`nvidia_behavior_identical` is not `true`. It is one concrete code change.

## Test-seam JSON

Write `/tmp/pr-N-test-seam.json` after `/pr-test-seam`.

```json
{
  "conclusion": "mixed",
  "changed_behavior": "For an unaligned prefix, snapshot ring keys before project_qk stores over them, then recompress the crossing group.",
  "correct_seam": "_forward_impl",
  "missing_test": "Call _forward_impl with an unaligned prefix and assert the stored compressed key, letting production snapshot, project, and overwrite."
}
```

`conclusion` is `complete`, `mixed`, `past_the_seam`, `reimplements`, or
`no_test`. Only `complete` passes. Every other conclusion requires
`missing_test`, naming the production interface, the trigger, and the
observable result.

## Calibration

PR 39575 is the calibration case. Shape, Affected Scope, and AMD Guard pass.
The guard is `gfx950`, not all AMD. Unit Test Quality fails because the
recompress tests step past `_forward_impl`. The verdict is
`BLOCKED — REQUIREMENTS FAILED`.

The sample judgment files are `examples/39575-code-path.json` and
`examples/39575-test-seam.json` next to `triage.py`.

## What this skill does not do

It does not decide whether the AMD math is correct, whether every forward path
was updated, or whether CI is green. Those start only after
`READY FOR HUMAN REVIEW`, in a human read or in `/sglang-pr-review`.
