---
name: pr-merge-triage
description: >-
  Automated pre-review gate for a PR into sgl-project/sglang. Use when the
  user says '/pr-merge-triage', asks whether a PR is ready for human review,
  or wants a standard checklist that tells the PR owner exactly which
  requirements failed. A pass means a human may start reading the code. It
  never approves a merge and never prints LGTM. One PR adds one feature and
  tests it on one model. A second feature or a second tested model is split
  out. This folder also contains /pr-code-path and /pr-test-seam, so
  installing this one skill is the whole pre-review.
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
# 1. Shape, then the feature and model judgment.
#    The chat reply is the --split output, not the --shape output.
#    Do not paste the manifest or the excerpt.
python3 triage.py N --shape
python3 triage.py N --manifest
python3 triage.py N --excerpt
# Write /tmp/pr-N-split.json from the Split section below.
# Use the manifest and the excerpt. Do not run gh, git diff, or a code-path review.
python3 triage.py N --shape --split /tmp/pr-N-split.json
# If the report says the split plan is not acceptable, fix the JSON and
# re-run this command.
# On BLOCKED — SPLIT FIRST, stop. Do not start step 2 unless the user
# asked to continue past the split.
# On PROCEED, continue to step 2.

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
#    python3 triage.py N --excerpt
#    That excerpt is the only PR text you score. Do not run gh for the body.
#    Write /tmp/pr-N-prose.json from the PR body section below.
#    Score Motivation from 1 to 10. Pass is 7 or higher.
#    The reader knows this is a GPU fix and does not know the kernel names.
#    On a fail, evidence is the reason and action is a Motivation the owner can paste.
#    Canvas: ~/.cursor/projects/<workspace>/canvases/pr-N-code-path.canvas.tsx
#    Write that canvas with the editor in this session. Layout is the Canvas
#    section of pr-code-path/SKILL.md: two cards, a clickable caller graph,
#    one foldable section per node titled function · line. Interface changes
#    open first. Follow each new name to the lines that load and use it,
#    through the gate that keeps NVIDIA from reading them.
#    A disk-only file, or a sidecar status of canvas-missing, opens as canvas
#    not found. If the user reports that, edit the .canvas.tsx again; do not
#    only repeat the link.

# 3. The only user-facing report. Pass the judgment files even when step 1
#    said SPLIT FIRST and the user asked to continue.
python3 triage.py N \
  --code-path /tmp/pr-N-code-path.json \
  --test-seam /tmp/pr-N-test-seam.json \
  --prose /tmp/pr-N-prose.json
```

`N` may be a PR number or a GitHub pull URL.

On `BLOCKED — SPLIT FIRST`, stop after the `--split` report unless the user
explicitly asked to continue. Continuing runs steps 2 and 3. The verdict stays
`BLOCKED — SPLIT FIRST` while One concern fails, and the other rows are part
of that report. The owner comment is the high-level suggestion from that report.

On `BLOCKED — AUTOMATION INCOMPLETE`, the judgment files are missing or do not
match the schema. Finish them and re-run step 3. Do not fill a blocked row by
hand and do not ack it.

## Who decides each row

| Check | Source | Fail means |
|---|---|---|
| One concern | split JSON, with size as a backstop | more than one feature, or tests on more than one model, or the diff is over 800 lines or 3 areas. The action is a high-level suggestion |
| Affected Scope | `/pr-code-path` JSON | NVIDIA numerical results or original behavior are not identical |
| AMD Guard | `/pr-code-path` JSON, plus an AITER import in the diff | the new behavior is outside the claimed guard, or AITER is imported under a scope other than `aiter` |
| Unit Test Quality | `/pr-test-seam` JSON | the contribution-guide test bar fails (`official_pass` is false). Coverage other than `complete` does not fail the row |
| PR body | prose judgment JSON | Motivation scores 6 or below. A reader cannot connect the machine and the user-visible result on one read, or the opening is a goal |
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

One concern fails when the PR adds more than one feature, or tests the change on more than one model. A diff over 800 lines or 3 areas still fails too, as a backstop, including when the feature and the model are already one each. The suggestion names the feature and the model. It does not list files.

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

## Split

Write `/tmp/pr-N-split.json` after `--manifest` and `--excerpt`, for every PR. The judgment is the feature and the model. Do not list files.

One PR adds one feature and tests it on one model.

A feature is a behavior a reviewer can accept or reject on its own. A second projection, a second hardware path, or a second kernel is another feature. Two files that implement the same behavior are one feature.

A tested model is one the PR reports accuracy, performance, or a model-level test for. The model named in that report is `model`. Every other tested model is `also_models`.

A model that is only guarded so the new behavior does not reach it is not a tested model. Put it in `simple_models` only when all three hold: it calls a helper the first model already uses, the model-file change is a few dozen lines, and the PR adds no test for it. Otherwise it is `also_models`.

`feature` is the one behavior the first PR should land. `also_features` is every other behavior in this diff.

`suggestions` are sentences, not paths. The first sentence is that one feature on that one model. Each extra feature or extra model gets its own later sentence. A second model that passes the three simple-model checks stays with the PR that owns the helper. Say so in one sentence. Do not give it a PR of its own.

The 800-line and 3-area gate is a backstop. One feature on one model that is still over that gate needs a suggestion that says what to cut so a person can review it. Do not satisfy the backstop with a file tree.

```json
{
  "feature": "Fuse the quant into the entry norm.",
  "model": "the model named in the benchmark",
  "also_features": ["The decode projection", "The prefill projection"],
  "also_models": ["A second model that has its own accuracy test"],
  "simple_models": [
    {
      "name": "A guarded model",
      "why": "It calls the same helper, the change is a few dozen lines, and it adds no test."
    }
  ],
  "suggestions": [
    "First PR: the entry fold on the benchmark model, with that path's check.",
    "Then: the other projections on that same model.",
    "Then: the second model on its own, with its own tests."
  ]
}
```

One feature on one model, inside the size gate:

```json
{
  "feature": "Fuse the quant into the entry norm.",
  "model": "the model named in the benchmark",
  "also_features": [],
  "also_models": [],
  "simple_models": [],
  "suggestions": []
}
```

The script rejects a second feature or a second tested model that has no suggestion, and a simple-model entry whose `why` is empty. Fix the JSON and re-run. Do not paste a plan the script rejected.

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


## PR body

Write `/tmp/pr-N-prose.json` after you read `python3 triage.py N --excerpt`.

Score Motivation only. Do not score tables, images, checklist items, or the merge-process boilerplate. Modifications do not change the score.

The reader knows this is a GPU fix. The reader does not know AOTriton, SDPA, `head_dim`, or the function names. Technical names may follow the first sentence. They must not be required to understand the first sentence.

The first sentence states both:

- what the user sees: wrong output, a crash, or a named error
- where: the GPU or the software version

One claim per sentence. The subject is the failure, not the author. Do not open with "This PR", "We", "in order to", or a benefit. Do not use "aims to", "leverage", "robust", "seamless", "comprehensive", "utilize", or "delve" as the point of a sentence.

| Score | Motivation |
|---|---|
| 10 | First sentence has the user-visible result and the place. Each later sentence is one technical cause. |
| 9 | Same as 10, with one extra name in the first sentence that the result does not depend on. |
| 8 | First sentence has the user-visible result and the GPU or version. One later sentence is long and still one claim. |
| 7 | The facts are present. The first sentence is only jargon. The wrong output or crash is in a later sentence. |
| 6 | The reader must re-read to connect the machine and the symptom. |
| 5 | One sentence stacks the cause, the design, and the benefit. |
| 4 | The opening is a goal. The fault comes later. |
| 3 | The opening is a goal. The fault is only a benefit word such as robustness or accuracy. |
| 2 | The point is an AI filler. No concrete fault. |
| 1 | Motivation is missing, or it does not say what failed. |

Pass is 7, 8, 9, or 10. Fail is 6 or below.

`score` is an integer from 1 to 10. `evidence` is the reason for that score, without the number. The report prints `Score N/10` in front of it. `action` is required below 7. It is a full Motivation the owner can paste. Keep the technical facts. Put the user-visible result in the first sentence.

```json
{
  "score": 9,
  "evidence": "The first sentence states the wrong decode and the crash, and it names the GPU and the ROCm version.",
  "action": ""
}
```

```json
{
  "score": 7,
  "evidence": "The first sentence names the backends and head_dim. The next sentence states the wrong values and the crash.",
  "action": ""
}
```

```json
{
  "score": 6,
  "evidence": "The GPU, the wrong output, and the crash are in the paragraph. A reader must re-read to connect them.",
  "action": "On gfx1250, image decode is wrong on ROCm 10.0 and crashes after denoise on ROCm 10.1.\nAOTriton's flash and mem_efficient SDPA backends fail when head_dim > 256."
}
```

```json
{
  "score": 3,
  "evidence": "The first sentence states a goal. It does not say what the user sees.",
  "action": "On gfx1250, image decode is wrong on ROCm 10.0 and crashes after denoise on ROCm 10.1.\nAOTriton's flash and mem_efficient SDPA backends fail when head_dim > 256."
}
```

Score 7, pass. The first sentence is jargon. The next sentence states the wrong values and the crash:

```text
On gfx1250-rocm10.1, AOTriton's flash and mem_efficient SDPA backends failed when head_dim > 256.
On ROCm 10.0 the VAE returns wrong values with no error. On ROCm 10.1 it raises hipErrorProfilerNotInitialized after the full denoise.
```

Score 3, fail. The first sentence is a goal:

```text
This PR aims to improve VAE robustness on next-generation AMD GPUs by leveraging the math SDPA backend.
```

Score 9, pass:

```text
On gfx1250, image decode is wrong on ROCm 10.0 and crashes after denoise on ROCm 10.1.
AOTriton's flash and mem_efficient SDPA backends fail when head_dim > 256.
ROCm 10.0 (AOTriton 0.13.50) returns wrong values and no error.
ROCm 10.1 (0.14.50) raises hipErrorProfilerNotInitialized.
```

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
