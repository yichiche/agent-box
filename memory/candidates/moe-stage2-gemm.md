---
name: moe-stage2-gemm
target: MoE stage-2 GEMM (mfma_moe2, flydsl fp4)
phase: decode
source: B200-vs-MI355 conc4 IL8k/OL1k trace
s: 0.118
speedup: 3.7
confidence: 0.15
status: queued
gate: 0
priority: 1.3
---

# MoE stage-2 — fp4 GEMM (at HW floor, lowest confidence)

**Bottom of the queue.** Real share but the win requires HW-parity we don't have.

- **AMD now:** `mfma_moe2` ≈ 43.0µs/call.
- **B200:** `bmm_Bfloat16_E2m1` grouped fp4 GEMM ≈ 11.5µs.
- **What's reachable:** little in-scope — same fp4-GEMM HW-floor story as
  [[moe-stage1-finalize-fusion]] (split-K dead for fp4, MFMA M-floor). No combine
  slice to borrow here.
- **Confidence 0.15:** essentially blocked without a kernel-math breakthrough.

Keep queued for visibility, but do not prioritize over feasible wins (GDN, MoE-sort).
Source: [[qwen35-decode-b200-vs-mi355x]], [[qwen35-moe-decode-roundzero]].
