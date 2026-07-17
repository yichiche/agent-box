---
name: attn-decode-fused-reduce
target: Attention decode (unified_attention + reduce_segments)
phase: decode
source: B200-vs-MI355 conc4 IL8k/OL1k trace
s: 0.063
speedup: 6.5
confidence: 0.6
status: queued
gate: 0
priority: 3.2
---

# Attention decode — fuse split-KV reduction into the epilogue

- **AMD now:** `unified_attention_3d` 84.9µs + separate `reduce_segments` 6.6µs.
- **B200:** `fmhaSm100f …PagedKvCausal MultiCtasKv` 13.9µs — the **split-KV
  partial-softmax reduction is FUSED in-kernel** (no separate reduce launch).
- **Borrow:** fold `reduce_segments` into the `unified_attention` epilogue
  (MultiCtasKv-style in-kernel reduction), eliminating the separate launch.
- **Confidence 0.6:** clear fusion target; needs a Triton/kernel epilogue change.
- Note: `reduce_segments` alone micro-tuned +21.7% but is only ~0.5% share — the win
  is the *fusion*, not tuning the separate kernel (Gate 2 = filter, see [[../workflows/gates]]).

Source: [[qwen35-decode-b200-vs-mi355x]].
