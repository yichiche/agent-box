---
name: sglang-pr-review
description: Specialized PR review for sgl-project/sglang, an LLM inference engine. Use when reviewing PRs in sgl-project/sglang or similar CUDA/Python ML inference codebases. Covers Python runtime, CUDA/Triton kernels, scheduling, memory management, model serving, and MoE patterns.
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
- **Platform guards**: If the change is platform-specific (AMD/HIP, CUDA, CPU), verify other platforms are unaffected
- **Quantization compatibility**: If weights or activations change, verify FP8/FP4/INT4/BF16 paths

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

### Tests & Benchmarks
- Test coverage assessment (what's tested, what's missing)
- GSM8K / MMMU accuracy results (if applicable)
- Throughput / latency / TTFT benchmarks (if applicable)
- Compare before/after numbers and flag regressions

### Risk & Scope
| Risk | Detail |
|------|--------|
| Critical/High/Medium/Low | One-line description |

### Decision
**approve** | **comment** | **request-changes** — rationale + numbered action items
```

**Keep the Risk & Scope rows labelled exactly `Critical` / `High` / `Medium` /
`Low`, and keep the severity word in the findings bullets.** Those two shapes
are what [`/pr-merge-triage`](../pr-merge-triage/SKILL.md) parses when the
findings are folded back into the merge verdict:

```bash
/sglang-pr-review 41870 > /tmp/review.md
python3 ~/agent-box/skills/pr-merge-triage/triage.py 41870 --review /tmp/review.md
```

A single `Critical` opens that skill's **Correctness gate** and blocks the merge
at any risk score — so a row labelled `Critical` is a real decision, not an
emphasis. A finding that bounds the blast radius of a Critical (`everything else
still falls back to the old kernel`) belongs in the `Low` row; it does not
downgrade the Critical.

### Severity Guidelines

- **CRITICAL**: Produces wrong model outputs silently (e.g., dropped scale, missing weight, lost expert). Always `request-changes`, and always a merge blocker.
- **High**: Can crash or degrade under specific configurations (e.g., dtype mismatch on fallback, DeepEP + fusion interaction). Likely `request-changes`.
- **Medium**: Edge cases, fragile patterns, missing tests. Can be `comment` with action items.
- **Low**: Style, docs, minor cleanup. `comment` or `approve`.

## Additional Resources

- For detailed analysis patterns by change type, see [references/analysis-patterns.md](references/analysis-patterns.md)
- [`/pr-merge-triage`](../pr-merge-triage/SKILL.md) — run it **before** this skill to see whether the PR should be split first (reviewing a PR that needs splitting is wasted work), and **after** to turn these findings into a merge verdict
- CI red on the PR you are reviewing? That is [`/ci-analysis`](../ci-analysis/SKILL.md), not this skill — attributing a failure to the PR is a separate procedure
- Reviewing a kernel change for *performance* rather than correctness? Pair with
  [`/kernel-profile-triage`](../kernel-profile-triage/SKILL.md) and [`/validate-pr`](../validate-pr/SKILL.md)
