---
name: sglang-pr-review
description: >-
  Correctness review for an sgl-project/sglang PR. Use when the user says
  '/sglang-pr-review' and asks whether to approve. If the code-path and
  test-seam JSON are missing, follow /pr-merge-triage in this session first.
  Then read those files and the /pr-ci-watch CI token. Does not re-run a
  triage that already wrote them. Decides approve, comment, or
  request-changes from the AMD math, forward paths, weight loading,
  quantization, and kernels.
category: deliver
---

# /sglang-pr-review — correctness, then approve

This is the last step. `/pr-merge-triage` decides whether the PR is worth reading. `/pr-ci-watch` already decided whether NVIDIA CI is clear. This skill decides whether the new path's results are right. That decision is the only approve.

If `/tmp/pr-N-code-path.json` or `/tmp/pr-N-test-seam.json` is missing, follow `/pr-merge-triage` for this PR in this session before the checks below. Do not ask the user to run that command and come back. Those files live in `/tmp` and do not survive across days.

Do not re-score, re-derive, or reprint any of these. They already have an owner:

- shape, Affected Scope, AMD Guard, and the code-path canvas
- Unit Test Quality, the test-seam report, and the test entry map
- Accuracy evidence, Performance evidence, and Flags
- CI logs, root-cause jobs, and re-run versus code-fix

A `/pr-code-path` **Cannot merge** is already a failed triage row. Do not open a second Critical for it.

## Start

Stop unless all three are already true. A missing JSON file is the only miss you repair from here, by following `/pr-merge-triage` once and then re-reading Start. Do not repair a failed guard, a failed `official_pass`, or a CI token by re-running that skill. A test-seam `conclusion` other than `complete` does not stop this review.

```bash
# Written by /pr-merge-triage in this session or an earlier one.
# Do not run path_cover.py or test_seam.py yourself.
# A missing file is the only case that runs /pr-merge-triage from here.
# /tmp/pr-N-code-path.json
# /tmp/pr-N-test-seam.json

# CI token. Do not run /ci-analysis, gh pr checks, or job logs.
python3 ~/agent-box/skills/ci/pr-ci-watch/watch.py report
```

| What you see | What you do |
|---|---|
| Either JSON file is missing | Follow `/pr-merge-triage` for this PR now. It writes both JSON files. Then resume Start. Do not stop to ask the user. |
| `nvidia_behavior_identical` is not `true`, or `guard_contains_new_behavior` is not `true` | Stop. Triage is still blocked. |
| `official_pass` is not `true` | Stop. Triage is still blocked. A `conclusion` other than `complete` does not block. |
| This PR is absent from the report | Stop. Say to `/pr-ci-watch add` it first. |
| The token before `<PRN>` is not `CI clear` | Stop. Name that token. Approve waits. |

`CI clear` is the only token that means proceed. `CI running`, `CI red`, `CI fail`, `conflict`, `merge main`, `blocked`, `gated`, and `CI ?` all stop.

Read `hardware_scope` and `hardware_scope_detail` from the code-path JSON. That is the hardware this review is about. Do not invent a second scope.

## What to read

Triage does not read the implementation for correctness. This skill does:

```bash
export GH_PAGER=cat GIT_PAGER=cat
gh pr diff $N --repo sgl-project/sglang
gh pr view $N --repo sgl-project/sglang --json title,body,files
```

Use the PR body to understand the claimed behavior. Do not score it again for a GSM8K number, a throughput number, or a missing flag policy.

## Review

### Structural

For each changed file:

1. **What changed**: one sentence.
2. **Why it changed**: match it to the PR's claimed behavior.
3. **What else uses this code**: trace callers and callees. Check the paths the diff did not edit.
4. **Initialization order**: for `__init__`, lazy properties, and `set_*` methods, follow construction, then configuration, then first use. No path may read state before it is set.

### Deep

Apply [references/analysis-patterns.md](references/analysis-patterns.md) only for the sections the diff touches:

- **Weight loading** — checkpoint key to parameter name for every weight
- **Forward paths** — normal, dual-stream, DeepEP, CUDA graph, prefill, decode
- **Quantization** — scale and dtype travel with the tensor
- **Distributed** — TP, EP, and PP configurations the new code can run under
- **Fallback** — the fallback this PR ships returns the right dtype and shape. Triage already decided which hardware the guard enables.
- **Cache** — the cache key includes every input that changes the result

A new `SGLANG_*` binding was already scored by triage's Flags row. Do not re-decide whether it should be a hardware detection. If the diff changes what an existing env var does at runtime, check that the behavior matches the value the code reads.

### Related PRs

If the description names another PR, check ordering: does this diff depend on a behavior that other PR has not landed? Git conflicts are `/pr-ci-watch`'s job. Do not look for them here.

## SGLang Architecture Context

```
python/sglang/srt/
├── models/             # Model implementations (qwen2_moe.py, llama.py, etc.)
├── layers/
│   ├── moe/            # MoE layers: FusedMoE, token dispatchers (DeepEP, MoRI)
│   ├── quantization/   # FP8, GPTQ, AWQ, Quark quantization methods
│   ├── attention/      # Attention backends (fa3, triton, aiter, flashinfer)
│   ├── communicator/   # Fused comm+norm layer boundaries
│   ├── linear.py       # TP-aware linear layers
│   └── layernorm.py    # Fused layernorm variants
├── distributed/        # TP, PP, EP communication primitives
├── managers/           # Scheduler, request lifecycle, cache controller
├── mem_cache/          # KV-cache pools, allocators, radix/hierarchical cache
└── model_executor/     # Forward batch, CUDA graph runner

python/sglang/kernels/  # Custom CUDA/HIP/Triton kernels (jit/, aot/, ops/)
```

Key patterns:

- **MoE flow**: `gate(hidden) → TopK → dispatch → expert_forward → combine → add_shared_expert`
- **Weight loading**: `checkpoint_key → stacked_params_mapping / expert_params_mapping → param.weight_loader`
- **Fused kernels**: AllReduce+RMSNorm, RMSNorm+Quant, AllReduce+RMSNorm+Quant — each with fallback paths
- **Distributed**: TP (tensor parallel), EP (expert parallel via DeepEP/MoRI), PP (pipeline parallel)

## Checklist

Apply only the rows the diff can break.

### Memory & GPU Resources
- [ ] No GPU memory leaks (tensors held past use, missing `del`)
- [ ] KV-cache alloc/dealloc correct, no fragmentation
- [ ] `torch.inference_mode()` in inference paths
- [ ] Memory pool sizing considers OOM edge cases

### CUDA / Triton Kernels
- [ ] Grid/block dimensions correct for all input sizes
- [ ] Shared memory within hardware limits
- [ ] No race conditions, proper `__syncthreads__`
- [ ] Edge cases: zero-length sequences, non-power-of-2 batches
- [ ] dtype consistency (fp16/bf16/fp32 mixing intentional)
- [ ] Triton `tl.load`/`tl.store` have correct boundary masks

### Model & Weight Loading
- [ ] TP sharding correct for new/modified layers
- [ ] Weight loading handles all checkpoint formats (separate vs fused, per-expert vs bulk)
- [ ] Shared expert / fused expert weight remapping complete for ALL model variants
- [ ] Quantization compatibility (FP8, GPTQ, AWQ) maintained
- [ ] `num_experts` counts correct (base + fused_shared + redundant)

### Numerical Accuracy
- [ ] Scale tensors propagated end-to-end (not discarded via `_`)
- [ ] FP8/FP4 quantized outputs paired with correct scales downstream
- [ ] Fallback paths produce matching dtype (no silent bf16↔fp8 mismatch)
- [ ] Attention changes validated against reference

### Performance
- [ ] No unnecessary `torch.cuda.synchronize()` in hot path
- [ ] No per-forward tensor allocations that could be pre-allocated
- [ ] No redundant `.to()` / `.contiguous()` in hot loops

### Distributed Communication
- [ ] AllReduce/AllGather/ReduceScatter correct for all TP sizes
- [ ] EP dispatch/combine handles all expert configurations
- [ ] Fused communication kernels have proper fallback
- [ ] `except Exception: pass` never silently swallows errors — at minimum `logger.debug`

### Code Quality
- [ ] Follows existing codebase patterns
- [ ] No unnecessary variable renames that add diff noise
- [ ] Logging at appropriate levels (DEBUG for hot paths, INFO for lifecycle)
- [ ] No hardcoded model names, magic numbers without constants
- [ ] An existing configuration still means what it meant before this diff

## Output Format

```markdown
## PR #NNNNN Summary & Review

### Preconditions
- Scope: <hardware_scope_detail from the code-path JSON>
- Tests: official bar passed
- CI: CI clear

### Summary
- ≤8 bullets; each ≤120 chars; start with a verb
- Cover all changed files, one bullet per logical change

### SGLang-Specific Findings
- `[bug] CRITICAL — path:line — description`
- `[bug] High — path:line — description`
- `[memory] path:line — GPU memory concern`
- `[kernel] path:line — CUDA/Triton kernel issue`
- `[model] path:line — model loading/TP/PP/weight issue`
- `[perf] path:line — performance regression risk`
- `[numeric] path:line — numerical accuracy risk`
- `[api] path:line — API compatibility concern`
- `[sched] path:line — scheduling/batching concern`

### General Findings
- `[bug] path:line — correctness issue`
- `[security] path:line — risk & minimal fix`
- `[style] path:line — code quality issue`
- `[docs] path:line — missing documentation`
- `[question] path:line — needs clarification from author`

### Risk & Scope
| Risk | Detail |
|------|--------|
| Critical/High/Medium/Low | One-line description |

### Decision
**approve** | **comment** | **request-changes** — <one sentence>

Blocking lines:
- `path:line` — <one Critical finding>
```

**Preconditions** is three lines. Do not add the code-path analysis, the path tree, the value trace, the test entry map, or the accuracy and throughput numbers.

A **request-changes** decision lists only the Critical findings. Each Critical gets its own findings bullet, its own Risk row, and its own Blocking line. High, Medium, and Low stay out of Blocking lines.

Keep Risk rows labelled exactly `Critical`, `High`, `Medium`, or `Low`. A Critical bullet is `[bug] CRITICAL — path:line — …`. A High bullet is `[bug] High — path:line — …`.

Each `Critical` row blocks the merge on its own. Several Criticals are normal when several independent problems each block the merge. A note that only bounds the blast radius belongs in the `Low` row. It does not downgrade the Critical.

This report is the approve decision. Do not pipe it back into `triage.py`. That script has no review input.

### Severity

Critical means this problem blocks the merge.

- **CRITICAL**: A silent wrong answer, a hang, a crash, a `raise`, a startup failure, or any other bug on the path this PR ships. Always `request-changes`.
- **High**: Real damage on a specific configuration. It does not by itself refuse the merge. If you would not merge until it is fixed, label it Critical instead.
- **Medium**: An edge case or a fragile pattern in the implementation. `comment`.
- **Low**: Style, docs, minor cleanup. `comment` or `approve`.

## What this skill does not do

Passing the Start table is not an approval. Approve only when this read finds no Critical on the path the PR ships.

Do not run `/pr-code-path`, `/pr-test-seam`, or `/ci-analysis` on their own from this skill. The one exception is a missing `/tmp/pr-N-code-path.json` or `/tmp/pr-N-test-seam.json`: follow `/pr-merge-triage` once, which owns those steps, then resume Start. Do not run `sglang-pr-review/test_seam.py`.

A kernel change whose question is performance, not correctness, is [`/kernel-profile-triage`](../kernel-profile-triage/SKILL.md) and [`/validate-pr`](../validate-pr/SKILL.md).
