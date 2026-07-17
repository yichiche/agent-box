---
name: moe-stage1-finalize-fusion
target: MoE stage-1 GEMM (mfma_moe1_silu_mul, flydsl fp4)
phase: decode
source: B200-vs-MI355 conc4 IL8k/OL1k trace
s: 0.21
speedup: 4.2
confidence: 0.2
status: queued
gate: 0
priority: 3.2
---

# MoE stage-1 — finalize/combine fusion (GEMM itself is at HW floor)

**Highest raw headroom (16%) but low confidence** — the fp4 GEMM math is at the AMD
HW floor, so the big number is mostly unreachable in-scope.

- **AMD now:** `mfma_moe1_silu_mul afp4_wfp4` ≈ 76.6µs/call.
- **B200:** `bmm_E2m1` grouped fp4 GEMM (splitK) + tiny fused `finalizeKernel` (4.8µs)
  for the combine — total ≈ 18.4µs.
- **What's reachable:** the **finalize/combine fusion**, NOT the GEMM. The fp4 GEMM is
  at the HW floor (MFMA M-floor=16, scale-group=32, split-K dead — fp4 reduction
  coupling). Micro-tuning exhausted (best +0.9%). See [[qwen35-moe-decode-roundzero]].
- **Confidence 0.2:** reflects that only the combine-fusion slice is achievable; the
  headline 4.2× requires HW parity we don't have.

Do NOT chase the GEMM tiles (dead end, journal-confirmed). Source:
[[qwen35-decode-b200-vs-mi355x]].
