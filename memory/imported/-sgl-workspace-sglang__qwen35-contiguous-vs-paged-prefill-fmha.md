---
name: qwen35-contiguous-vs-paged-prefill-fmha
description: Measured result — contiguous varlen FMHA is only ~3% faster than paged FMHA at matched problem size on MI355/gfx950 (Qwen3.5 GQA prefill)
metadata: 
  node_type: memory
  type: project
  originSessionId: fc6b9734-53f3-43c6-b11b-90ad47cae71f
---

Implemented + validated a contiguous-prefill FMHA fast-path in `python/sglang/srt/layers/attention/aiter_backend.py` `forward_extend`: for non-MLA GQA when `extend_no_prefix` (no cached prefix), route through `flash_attn_varlen_func` (FmhaFwdKernel varlen, contiguous K/V) instead of paged `mha_batch_prefill_func` (FmhaBatchPrefillWithPagedKVCache). Gated: BF16 only (flash_attn_varlen_func has no descale args — fp8 uses a separate path), window_size==(-1,-1), sinks None, logit_cap==0.0, not vectorized_5d, not draft_extend. ~30 added lines.

**Validated on Qwen3.5-397B-A17B-MXFP4, MI355 gfx950, TP2, IL8k/OL1k, BF16 kv, --disable-radix-cache** (model at /data/amd/Qwen3.5-397B-A17B-MXFP4; the configured `-7f34fa9` suffix path is stale). Results:
- Accuracy gsm8k=0.935 (max-new-tokens 8192, thinking model) — PASS, identical behavior.
- Kernel swap CONFIRMED in trace: baseline `FmhaBatchPrefillWithPagedKVCache` x60 → after `FmhaFwdKernel`/`kattr_no_packed_fp32_ops` x60.
- **Per-call prefill attn: 1759us → 1711us = only −48us (−2.7%)**, consistent across the bimodal distribution (not noise).

**Why:** At MATCHED problem size the paged-gather overhead vs contiguous is only ~3% on this gfx950 build — NOT the 1.3-1.5x I estimated. The ATOM(785us) vs flydsl(2607us) 3.3x gap was almost entirely the ~2.5x batch-size difference (ATOM ran fewer prefill tokens; its GEMMs were ~2.5x smaller), confirming the apples-to-oranges caveat. The benchmark TTFT delta (−27-35%) was run-to-run noise at n=8; real kernel saving (~1.4ms/prefill) is <0.5% of TTFT. **How to apply:** Don't trust cross-run kernel-time gaps without normalizing for token/batch count first (GEMM durations are the cheapest proxy for token count). This optimization is correct + low-risk but marginal; not clearly worth a PR on its own. Relates to [[qwen35-mxfp4-fp8-prefill-attn-config]] (FP8 KV is the bigger prefill-attn lever) and [[qwen35-mxfp4-flydsl-fully-fused]].
