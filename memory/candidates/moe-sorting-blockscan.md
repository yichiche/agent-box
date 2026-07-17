---
name: moe-sorting-blockscan
target: MoE routing / token sort (decode)
phase: decode
source: B200-vs-MI355 conc4 IL8k/OL1k trace
s: 0.13
speedup: 2.6
confidence: 0.6
status: queued
gate: 0
priority: 4.7
---

# MoE sorting — block-scan routing, drop decode padding

**#2 in the queue.** Feasible mid-size win with good share.

- **AMD now:** `moe_sorting` (opus) ≈ 13µs/call; plus `topkGatingSoftmax` 574µs and
  `fused_mx_quant_moe_sort` 493µs in the routing region.
- **B200:** `moe::dev::routing::routingIndicesBlockKernel` ≈ 5.0µs — **block-scan
  routing, no 16K-row padded writes** for a tiny decode token count.
- **Borrow:** block-scan routing + bound the work to actual `token*topk` at decode
  (drop the 16K-row padding that only makes sense at large batch).
- **Confidence 0.6:** algorithmic but well-scoped; the padding waste is clear at
  conc4 decode.

Source: [[qwen35-decode-b200-vs-mi355x]]. Land per [[../workflows/sglang-integration]].
