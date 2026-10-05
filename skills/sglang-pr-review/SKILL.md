---
name: sglang-pr-review
description: >-
  Specialized PR review for sgl-project/sglang, an LLM inference engine. Use
  when reviewing PRs in sgl-project/sglang or similar CUDA/Python ML inference
  codebases, or when asked whether a unit test calls the real interface.
  Covers Python runtime, CUDA/Triton kernels, scheduling, memory management,
  model serving, and MoE patterns. Confirms common path, NVIDIA impact, and
  hardware scope by following /pr-code-path, and judges whether tests enter
  through the changed interface, step past the seam, or reimplement production.
category: deliver
---

# SGLang PR Review

Systematic review process for [sgl-project/sglang](https://github.com/sgl-project/sglang).

## Review Process

### Phase 1: Data Gathering (parallel)

Fetch all data sources concurrently:

1. **PR metadata**: `WebFetch` the PR page — title, description, author, labels, review comments
2. **Full diff**: `WebFetch` the `.diff` URL (`https://patch-diff.githubusercontent.com/raw/sgl-project/sglang/pull/{N}.diff`)
3. **Base files**: For each significantly changed file, `WebFetch` the `main` branch version from `raw.githubusercontent.com` to understand pre-existing code around the changed regions

If `gh` CLI is available, prefer:
```bash
export GH_PAGER=cat GIT_PAGER=cat
gh pr view $N --repo sgl-project/sglang --json title,body,author,files,additions,deletions,commits,state
gh pr diff $N --repo sgl-project/sglang
```

### Confirm with `/pr-code-path`

Follow [`/pr-code-path`](../pr-code-path/SKILL.md) on this same PR before the
correctness verdict. Run its script, then write that skill's report (final
conclusion, analysis, and canvas):

```bash
python3 ~/agent-box/skills/pr-code-path/path_cover.py $N
```

Use that conclusion to confirm three things this review does not re-derive:

- what every backend now executes on the common path
- whether NVIDIA execution flow, internal interface, and numerical behavior stay identical
- which hardware the guard actually enables

A `/pr-code-path` **Cannot merge** is `[bug] CRITICAL` here and forces
**request-changes**. Quote its final conclusion under **Code path**. Do not
redraw its path tree or value traces in this report.

### Phase 2: Structural Analysis

For each changed file, answer:

1. **What changed**: Summarize the diff in ≤1 sentence per file
2. **Why it changed**: Match to the PR motivation
3. **What else uses this code**: Trace callers/callees of modified functions — check if other code paths break
4. **Initialization order**: For state changes (`__init__`, lazy properties, `set_*` methods), trace the lifecycle: construction → configuration → first use. Verify no path accesses state before it's set.

### Phase 3: Deep Analysis

Apply analysis patterns from [references/analysis-patterns.md](references/analysis-patterns.md) based on what the PR touches:

- **Weight loading changes** → Trace checkpoint key → parameter name mapping for every weight
- **Forward path changes** → Trace ALL forward paths (normal, dual-stream, DeepEP, etc.) to verify none are broken
- **Quantization changes** → Verify scale/dtype propagation end-to-end
- **Distributed/communication changes** → Check all parallel configurations (TP=1,2,4,8; EP; PP)
- **Fallback/guard changes** → Verify fallback produces correct dtype/shape, not silent degradation
- **Env var changes** → Check backward compatibility, default behavior, interaction with other env vars
- **Cache/memoization changes** → Verify cache key includes all relevant parameters

### Phase 4: Cross-cutting Concerns

- **Interaction with other PRs**: If the PR description mentions related PRs, check for merge conflicts and ordering dependencies
- **Platform guards**: Take the affected-hardware answer from the `/pr-code-path` conclusion. Do not invent a second scope.
- **Quantization compatibility**: If weights or activations change, verify FP8/FP4/INT4/BF16 paths

### Test seam

Follow [`/pr-test-seam`](../pr-test-seam/SKILL.md). Run
`python3 ~/agent-box/skills/pr-test-seam/test_seam.py $N`, classify the seam
there, and quote its conclusion under **Tests & Benchmarks**.

A conclusion other than `complete` is a Medium finding. It is Critical only
when that gap leaves a correctness bug on the shipped path able to stay green.
The facts script's old path, `sglang-pr-review/test_seam.py`, forwards to
`/pr-test-seam`.

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

## Review Checklist

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
- [ ] Fused kernels have benchmark data

### Distributed Communication
- [ ] AllReduce/AllGather/ReduceScatter correct for all TP sizes
- [ ] EP dispatch/combine handles all expert configurations
- [ ] Fused communication kernels have proper fallback
- [ ] `except Exception: pass` never silently swallows errors — at minimum `logger.debug`

### Tests
- [ ] Each changed behavior is tested through the production interface that owns it
- [ ] Expected values are literals, invariants, or another production implementation
- [ ] A test that orders the steps itself does not count as covering that ordering

### Code Quality
- [ ] Follows existing codebase patterns
- [ ] No unnecessary variable renames that add diff noise
- [ ] Logging at appropriate levels (DEBUG for hot paths, INFO for lifecycle)
- [ ] No hardcoded model names, magic numbers without constants
- [ ] Backward-compatible configuration changes

## Output Format

```markdown
## PR #NNNNN Summary & Review

### Summary
- ≤8 bullets; each ≤120 chars; start with a verb
- Cover all changed files, one bullet per logical change

### SGLang-Specific Findings
- `[bug] CRITICAL — path:line — description` (for correctness bugs that produce wrong results)
- `[bug] path:line — description` (for non-critical bugs)
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

### Code path
Quote the `/pr-code-path` final conclusion and its analysis bullets.
A Cannot merge is also a Critical finding above.

### Tests & Benchmarks
Test-seam report:

**Test-seam conclusion:** Complete | Mixed | Past the seam | Reimplements | No test
**Changed behavior:** [caller-visible regression]
**Correct seam:** [production interface callers use]

Test entry map: ASCII path from each test to production.

| Test | Seam it calls | Expected value source | Verdict |
|---|---|---|---|

**Missing interface test:** one concrete test, or None.

Then, when present: GSM8K / MMMU accuracy, throughput / latency / TTFT, and before/after regressions.

### Risk & Scope
| Risk | Detail |
|------|--------|
| Critical/High/Medium/Low | One-line description |

### Decision
**approve** | **comment** | **request-changes** — <one sentence>

**request-changes** — <which blocking finding, at which line>

Blocking lines:
- `path:line` — <one Critical finding>
- `path:line` — <another Critical finding, if there is one>
1. `path:line` — <the change required at that line>
```

A **request-changes** decision lists only the findings that block this merge.
Those are the Critical ones. There may be several, and each gets its own
`path:line` here, its own findings bullet, and its own Risk row. A High,
Medium, or Low finding stays out of **Blocking lines**.

**Keep the Risk & Scope rows labelled exactly `Critical` / `High` / `Medium` /
`Low`, and keep the severity word in the findings bullets.** A Critical bullet is `[bug] CRITICAL — path:line — …` and a High bullet is `[bug] High — path:line — …` (severity word, then `path:line`). Those shapes are what
[`/pr-merge-triage`](../pr-merge-triage/SKILL.md) parses when the
findings are folded back into the merge verdict:

```bash
/sglang-pr-review 41870 > /tmp/review.md
python3 ~/agent-box/skills/pr-merge-triage/triage.py 41870 --review /tmp/review.md
```

Each `Critical` row opens that skill's **Correctness gate** and blocks the merge
at any risk score. Several Criticals are normal when several independent
problems each block the merge; list every one, and do not let a narrower
finding cancel another. A row labelled `Critical` is a real decision, not an
emphasis. A note that only bounds the blast radius (`everything else still
falls back to the old kernel`) belongs in the `Low` row; it does not downgrade
the Critical.

### Severity Guidelines

The question for Critical is whether this problem blocks the merge. It is not
limited to silent wrong outputs, and a review can contain more than one.

- **CRITICAL**: Blocks the merge. Includes a silent wrong answer, a hang, a crash, a `raise`, a startup failure, or any other bug on the path this PR ships. Always `request-changes`. Each Critical is its own Risk row and its own Decision **Blocking lines** entry, as `[bug] CRITICAL — path:line — …`.
- **High**: Real damage on a specific configuration, but it does not by itself refuse the merge (triage scores it, it does not open the gate). If you would not merge until it is fixed, label it Critical instead.
- **Medium**: Edge cases, fragile patterns, missing tests, a test that steps past the seam or reimplements production. `comment`.
- **Low**: Style, docs, minor cleanup. `comment` or `approve`. Does not affect the merge.

## Additional Resources

- For detailed analysis patterns by change type, see [references/analysis-patterns.md](references/analysis-patterns.md)
- [`/pr-code-path`](../pr-code-path/SKILL.md) — required confirmation inside this review, not a substitute for it. Follow that skill and quote its conclusion under **Code path**
- [`/pr-merge-triage`](../pr-merge-triage/SKILL.md) — run it **before** this skill to see whether the PR should be split first (reviewing a PR that needs splitting is wasted work), and **after** to turn these findings into a merge verdict
- CI red on the PR you are reviewing? That is [`/ci-analysis`](../ci/ci-analysis/SKILL.md), not this skill — attributing a failure to the PR is a separate procedure
- Reviewing a kernel change for *performance* rather than correctness? Pair with
  [`/kernel-profile-triage`](../kernel-profile-triage/SKILL.md) and [`/validate-pr`](../validate-pr/SKILL.md)
