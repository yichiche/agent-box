---
name: kl-consistency-test
description: Write, calibrate, and debug prefill-vs-decode logprob (KL) consistency tests in SGLang — batch-invariance vs cache/state bugs, helper selection, threshold calibration, operator localization. Use when adding a KL test to a model, picking a kl_div threshold, or investigating high KL after a kernel change. Complements /validate (gsm8k) which misses these bugs.
category: kernel-opt
---

# KL Consistency Tests

Adapted from upstream sglang skill. gsm8k accuracy is insensitive to kernel batch-invariance bugs; KL consistency is not.

## What the test is for

`kl_test_utils` scores the same token twice — once as a prefill input logprob, once as a decode output logprob — and compares. The two paths run different kernels over different shapes, so agreement is a statement about **state**, not answer quality: radix-cache prefix mismatch, stale conv/mamba checkpoint, SWA pool eviction.

**Use after kernel changes** alongside `/validate`. A kernel that passes gsm8k but fails KL is still broken.

## Two independent conditions for zero KL

1. **Every operator is batch-invariant.** A token's result must not depend on batch size. A kernel can be reproducible at M=1 and M=N separately but disagree between prefill (large M) and decode (M=1) — typical of tile-size switches or message-size-dependent reductions.

2. **Both paths compute the same function.** Decode state at a position must equal fresh prefill — same KV, sliding window, conv/mamba state, restored cache prefix.

**Debug order:** settle (1) first. Its noise hides (2) completely until fixed.

### Separating them

With (1) satisfied: `match` and `decode_cache_hit` read 0 while `prefill_cache_hit` stays nonzero → prefix restore bug (condition 2), not arithmetic.

## Three helpers

| Helper | Cache involvement |
|--------|-------------------|
| `..._match_helper` | Both sides flush; **no cache** |
| `..._match_prefill_cache_hit_helper` | Prompt prefilled to warm cache, generation prefill restores from it |
| `..._match_decode_cache_hit_helper` | Decode on warmed cache |

Pick deliberately — only the cache-hit pair exercises prefix reuse.

## Run it like CI

Defaults: `max_samples=32`, `max_new_tokens=512`. Do not characterize with fewer.

`avg_kl_div` uses k3 estimator `exp(logr) - 1 - logr`. At 4 samples the same config can spread 3× — invalid for A/B.

When characterizing (not gating), report tail stats (fraction past threshold, max) not just mean.

Generate past the sliding window so decode carries the window through prompt→generated handover.

## Condition 1: determinism ≠ batch-invariance

- **Deterministic:** same input + shape → same result every run
- **Batch-invariant:** token result independent of batch size

KL compares prefill (thousands of tokens) vs decode (one token) — it measures batch-invariance.

`--enable-deterministic-inference` helps but only covers aten ops in `batch_invariant_ops`. Custom kernels outside aten stay shape-dependent.

Nonzero KL under deterministic inference in **every helper** → batch-dependent kernel. Localize (below), don't widen threshold.

### MoE amplification

Top-k routing is discrete. A 1e-8 gate difference flips experts; 42 layers compound it. MoE KL in hundredths may be the **same bug class** as 1e-4 on dense — don't calibrate MoE threshold by dense analogy.

## Choosing a threshold

Once all kernels are batch-invariant, assert near-zero:

```python
KL_DIV_THRESHOLD = 1e-9   # measured 0; state bugs are orders above
```

Thresholds are per `(model, tp)`. tp=1 has no all-reduce — doesn't transfer to tp>1.

Prefer KL on a deterministic server; keep accuracy on production numerics.

## Localizing a divergence

### Forward-hook dumper

```bash
DUMPER_ENABLE=0 DUMPER_SERVER_PORT=reuse DUMPER_NON_INTRUSIVE_MODE=all \
DUMPER_DIR=/path/to/dumps python3 -m sglang.launch_server ... \
  --disable-cuda-graph --disable-prefill-cuda-graph
curl -X POST localhost:PORT/dumper/configure -d '{"enable": true, "exp_name": "dec"}'
```

Required settings (each fails silently if wrong):

- `DUMPER_ENABLE=0` + `DUMPER_SERVER_PORT=reuse` — hooks register but warmup doesn't dump
- `DUMPER_NON_INTRUSIVE_MODE=all` — default `core` writes no module tensors
- `--disable-prefill-cuda-graph` in addition to `--disable-cuda-graph`
- Prefer `dumper.py` over `--debug-tensor-dump-*` for multimodal wrappers

**Prove alignment first:** decode pass `k` and prefill row `plen + k` must have bit-identical embedding output.

First layer with identical inputs but different output = the operator.

### CUDA graph divergence

Hooks don't run during graph replay. Disabling graph to dump also removes the divergence. Probe **reused state outside the graph** instead — log slot id, claimed length, `abs().max()` at checkpoint donation. Diff slots with graph on vs off.

### Standalone repro

Ten-line script calling suspect op at M=1 and M=288 settles batch-invariance in seconds. Put in PR before end-to-end KL numbers.

## Integration with /validate-pr

After kernel change:

1. Run gsm8k gate (existing `/validate-pr` Step 2)
2. Add KL check for the target model if not already in CI
3. If KL fails: use helper table above to classify (1) vs (2), then dumper

For Qwen3.5 / hybrid models with Mamba state, always include `prefill_cache_hit` helper.

## Related skills

| Skill | Use when |
|-------|----------|
| `/validate`, `/validate-pr` | End-to-end perf + accuracy gate |
| `/debug-kernel-crash` | Crash/NaN during KL repro |
| `/implement-kernel` | Fixing the localized operator |

Upstream reference: https://github.com/sgl-project/sglang/tree/main/.claude/skills/kl-consistency-test
