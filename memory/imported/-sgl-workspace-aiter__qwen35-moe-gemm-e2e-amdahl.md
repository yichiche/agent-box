---
name: qwen35-moe-gemm-e2e-amdahl
description: Qwen3.5 MXFP4 MoE GEMM prefill optimization is Amdahl-blocked for e2el (IL8192/OL1024 is 96%+ decode)
metadata: 
  node_type: memory
  type: project
  originSessionId: 2f6e9e9c-cdf1-43d6-bec6-b03959a8da5d
---

Qwen3.5-397B-A17B MXFP4 (TP2, MI355X gfx950): optimizing the prefill MoE GEMM
(flydsl `mfma_moe1_silu_mul` + `mfma_moe2` in mixed_moe_gemm_2stage.py) CANNOT move
e2e median e2el by 5% for the IL=8192/OL=1024 workload.

**Measured (2026-06-27, base vs s_setprio-opt, GPUs 6,7):**
- e2el is **96–98% DECODE**: median TTFT(prefill)≈350ms vs decode≈10–17s. Prefill share = 2.1–3.5% across conc 4/8/16.
- e2el opt/base = 0.995 / 0.950 / 1.006 (conc 4/8/16). conc8 0.950 was decode-TPOT shared-node variance (TTFT identical 354↔354), not the kernel. Gate (≤0.95 all) FAILS.
- TTFT (where prefill MoE kernels run) unchanged: 1.003/0.999/0.989.

**Why no lever:**
1. Prefill governs <3.5% of e2el → even infinite prefill MoE speedup gives <3.5%; MoE is only part of prefill.
2. Decode (the 96%) runs MoE at token≈conc(4..16): micro fused_moe opt/base 0.985–1.012 with s_setprio — BW/latency bound, no scheduling lever. Matches [[qwen35-moe-decode-roundzero]].
3. Kernel-internal (micro, tuned csv): at token=8192 moe1+moe2 = 92% of fused_moe (moe1=202us, moe2=342us, overhead=48us) but only ~10–20% of MXFP4 peak → BW/scatter/reduce-bound. 30% cut needs structural rework (expert grouping / reduce path), not low-risk scheduling.

**s_setprio experiment:** wrapping gemm2 MFMA k_idx loop in `rocdl.s_setprio(1/0)`+sched_barrier (mirrors gemm1) is accuracy-safe but gives only ~2–3.5% at very-large-M (≥16384), 0% elsewhere. Reverted (no goal value).

**Gotchas for this workload:**
- microbench: `op_tests/test_moe_2stage.py --no-legacy` with `AITER_CONFIG_FMOE=<csv>` runs `fail_on_aot_cache_miss` — after editing a flydsl kernel you MUST rebuild AOT cache: `python -m aiter.aot.flydsl.moe --csv <csv>` (cache auto-dir = aiter/jit/flydsl_cache via aiter/__init__.py). A "miss" = had to compile; first run populates disk, re-run passes.
- bench client: /home/yichiche/run_qwen3.5_mxfp4_inferencemax_client.sh points at a BROKEN symlink (./bench_serving → /sgl-workspace/bench_serving missing). Use `python -m sglang.bench_serving` directly: ignore-eos is default-on, save via `--output-file`, no `--percentile-metrics/--result-dir/--ignore-eos`. Working wrapper saved at e2e_runs/moe_gemm_goal/client2.sh.
- Recommendation: to move e2el here, target DECODE (attention over 8k+ KV, decode MoE BW). For prefill goals, use OL≤1 and measure TTFT, not e2el.
