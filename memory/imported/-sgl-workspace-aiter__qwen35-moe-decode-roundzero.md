---
name: qwen35-moe-decode-roundzero
description: "Qwen3.5-397B MXFP4 MoE decode profile — where decode TPOT goes, which levers are dead"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6c212835-0a65-4c7b-b99f-65624f138e40
---

Round-0 profile of Qwen3.5-397B-A17B-MoE-MXFP4 decode (TP2, aiter, IL=70k), goal=+5% geomean e2e.

**conc=4 decode trace (1 step, per TP rank; kernel-sum=11.4ms but TPOT=38ms → kernels only ~30% of
TPOT, rest is memory-stall/gap at low batch).** Top kernels (% of kernel-sum):
- mfma_moe1 silu_mul afp4_wfp4 (MoE gemm1) 1179us 10% ; mfma_moe2 652us 6% — aiter MXFP4 fused_moe
- `Cijk_*` rocBLAS bf16 gemms (QKV/o_proj/etc) ~3000us **26% — LARGEST category**
- opus_moe_sorting 856us 7.5% ; vllm topkGatingSoftmax 574us 5% ; fused_mx_quant_moe_sort 493us 4% — routing
- allreduce_fusion 853us 7.5% (already fused via --enable-aiter-allreduce-fusion)
- unified_attention 738us + gated_delta 377us = ~10% (OUT OF SCOPE; hybrid: 15/60 full-attn, 45 linear)

**DEAD ENDS (verified):**
- Dense bf16 gemm falls back to `torch solution:0` (tuned bf16 csv has N=9216 not model's N=8704), BUT
  microbench (GPU7) shows aiter `gemm_a16w16_opus`/`_asm` are 0.37–1.10x vs torch — i.e. **torch/rocBLAS
  already optimal, aiter no better**. Tuning bf16 gemm won't help. (opus output verified max_diff=0.)
- FMOE tuned config (`qwen3_5_397b_fp4_tuned_fmoe.csv`) is **already auto-merged** by
  `get_config_file` (globs `*tuned_fmoe*.csv` under model_configs when AITER_CONFIG_FMOE unset). No free win.
- allreduce already fused.

**Round-0 gate verdict:** no single kernel >5% TPOT at the measurable point; largest addressable
category (dense gemm) already optimal; MoE path already tuned. 5% geomean needs architectural kernel
work (fuse MoE routing/sort/quant, or new MXFP4 MoE gemm) — not in-session-feasible.

**Why verification is infeasible here:** profiling at saturated conc (64×70k + torch profiler) **OOM-kills
the host scheduler (exit -9)**. Full 7-conc sweep at IL=70k ~3hr/run (conc=256 = 2560×70k prefill);
×base/opt×3 rounds ≈ 18hr on a shared, unstable node (VRAM reclaim lag, disk fills). See
[[unified-attn-decode-perf]].
