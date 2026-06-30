---
name: flydsl-moe-reduction-grid-collapse-e2e-win
description: "flydsl MoE stage2 topk-reduction grid-collapse = +5.16% e2e throughput, invisible to microbench — validates e2e gating"
metadata: 
  node_type: memory
  type: project
  originSessionId: b6e22005-3025-4468-b717-ae6f9573c11b
---

implement-e2e run on Qwen3.5-397B-A17B-MoE-MXFP4 (MI355 tp8, IL8192/OL1024) found a SHIPPABLE win on the flydsl 2-stage MoE: in `aiter/ops/flydsl/kernels/moe_gemm_2stage.py` `compile_moe_reduction`, collapse the stage2 topk-reduction grid from `(m_tokens, num_col_tiles)` (~32768 CTAs) to `(m_tokens, 1)` (8192 CTAs) — each CTA owns a full token row and walks all model_dim column tiles via an in-kernel grid-stride loop, reusing its per-token i64 buffer-resource descriptors across every tile. Result: **e2e total_throughput +5.16% median (+4.16% min, no per-conc regression)**, gsm8k 0.715→0.76 (no drop), allclose pass. Patch: `~/.kernel-fusion-pipeline/e2e_1_flydsl_fused_MoE_GEMM_kernels_mfma_moe1_silu.patch`.

**Why it matters:** the change is **latency-NEUTRAL at the isolated GEMM microbench (+0.5%)** — the gain is reduced CTA-launch + redundant descriptor-setup overhead that only appears end-to-end. The old microbench-gated `implement-deep` would have DISCARDED this real +5.16% win. This is the concrete proof that kernel work must be gated on e2e, not microbench. Same run: attention `increase_gemm0_k_unroll` (bk0/bk1 32→64 for hd=256 bf16 CK FMHA batch_prefill) got +4.43% e2e from +1.5% microbench — also e2e>>microbench. See [[implement-e2e-vs-microbench-gate]].

**Non-wins same run:** decode quant→gemm fusion (author) never produced a correct first impl within the 3-noimprove convergence budget (deep change, needs more rounds / decisive first version); moe_sort replace was a mis-specified target (opus_moe_sorting_entry is HIP not CK, no faster asm) and is a ~7-10us kernel with tiny e2e share — all variants regressed/noise.
