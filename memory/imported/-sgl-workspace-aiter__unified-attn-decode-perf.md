---
name: unified-attn-decode-perf
description: unified_attention Triton decode (sq=1) perf regimes & where speedups live
metadata: 
  node_type: memory
  type: project
  originSessionId: 6c212835-0a65-4c7b-b99f-65624f138e40
---

unified_attention decode (sq=1, d=256, block_size=16, hq=16, hk=1, sk=8192) on MI355X:
all 8 batch sizes (1..128) take the **3D split-KV path** (num_2d_prgms=b ≤ 128 < target=cu*4=1024).

Two distinct regimes:
- **Small batch (1–8): overhead-bound.** The split-KV `reduce_segments` kernel is a ~7us fixed
  cost (b=1: 54% of total) because its grid is only (num_tokens·num_query_heads) WGs doing a heavy
  `[NUM_SEGMENTS, HEAD]` reduction → poor occupancy. Fix: **head-block the reduce** (3rd grid dim
  `NUM_HEAD_BLOCKS`) + **cap decode segments at 64** (oversubscription with short segments hurts
  both attn HBM efficiency and reduce). Gave 21–28% speedup on b=1,2,4,8. Done in
  `aiter/ops/triton/attention/unified_attention.py` (select_3d_config `all_decode` seg cap, reduce
  launch head-blocking) and `_triton_kernels/attention/unified_attention.py` (reduce_segments
  `NUM_HEAD_BLOCKS`). `UA_BASELINE=1` env restores original for A/B.
- **Mid/large batch (16–128): HBM-bandwidth-bound**, ~75–85% of peak. attn reads all K+V (1GB at
  b=128). Config sweeps (TILE/warps/stages/waves/segments) and cache-modifier (.cg is best) give
  ~0%. TILE>16 hurts (random block tables → gather). **20% is not achievable here without reducing
  KV bytes (e.g. fp8 KV)** — a different feature. My changes are neutral (no regression) there.

**E2E validation (Qwen3.5-397B-A17B-MoE-MXFP4, tp=2 on GPU5,6, conc=4, in=70k/out=300):**
- Model is HYBRID: only 15/60 layers use full `unified_attention`; other 45 use
  `fused_recurrent_gated_delta_rule` (linear attn). head_dim=256, 32 q / 2 kv heads (16 q/kv, =micro).
- **A fixed seg cap of 64 REGRESSED long context** (sk~70k → 4375 tiles; 64 segs under-parallelizes
  → unified_attention_3d 49→66us/call in profile). Micro (sk=8192) overfit. Fix: **tile-aware cap**
  `seg_cap = min(128, max(8, total_tiles//8))` — sk=8192→64 (micro win kept), sk≥~16k→128 (==baseline,
  no regression). Head-blocked reduce is the universal safe win (9.27→4.63us/call at sk70k).
- Clean conc=4 e2e (2 reps each, fresh servers): baseline TPOT 38.4ms, fixed-opt TPOT 38.05ms → ~1%,
  **e2e-NEUTRAL**. Because for this MoE-A17B model attention is only ~2% of decode TPOT (735us/38ms);
  MoE/gemm dominate. An attention kernel win can't move e2e TPOT here. (Earlier "36% e2e win" was a
  contention-inflated baseline run — single full runs on shared node are unreliable; take medians.)
- gsm8k accuracy (optimized) = **0.9718** (1319 ex) — change preserves correctness.
- Server VRAM reclaim after kill -9 lags ~60-90s; relaunching before reclaim → "memory capacity
  unbalanced" crash. Wait until BOTH tp GPUs <25GB before relaunch. Client bench: `bench_serving`
  symlink broken; use `python3 -m sglang.bench_serving` (args: --output-file not --save-result;
  ignore-eos is default). Profiling: launch server w/ SGLANG_TORCH_PROFILER_DIR, bench `--profile
  --profile-by-stage` → traces in /tmp/<ts>/*-DECODE/EXTEND.trace.json.gz. Single-step decode
  profile is noisy (variable seqlens at range-ratio 0.8); full runs average better. Root disk fills.

**IL=8192/OL=1024 conc-sweep gate (2026-06-27, tp2 GPU6,7, fresh A/B via UA_BASELINE):** confirms
e2e-neutral at a 2nd operating point. Median TPOT opt/base: conc4 10.34/10.28=1.006, conc8
13.43/13.49=0.996, conc16 18.32/17.99=1.018 — all within ±2% noise, all FAIL a ≤0.98 gate.
Profile decomposition (TP-0 DECODE, 15 steps): **ATTN+REDUCE = 2.6% of conc4 decode kernel time,
5.0% at conc64.** Per-call opt/base: conc4 ATTN=1.01 (unchanged!), REDUCE=**0.54** (head-block win
holds); conc64 both ~1.01. KEY: the microbench 0.73–0.78 win is **almost entirely the reduce
kernel**, not the attention 3d kernel — at sk=8192 the seg cap (512 tiles→cap 64 vs 128) barely
changes attn kernel time in-situ. Amdahl ceiling: even halving all attention → <2% TPOT, so the
≤0.98 e2e gate is **structurally unachievable** by any unified_attention change. Decided STOP (no
commit): kernel opt is correct + neutral, but not an e2e lever. Don't loop further on this.

**Bench gotchas:** node is SHARED — all 8 GPUs 80–99% used; GPU0-3 VRAM full (OOM). Use
`HIP_VISIBLE_DEVICES=6` (or 4/5/7, have free VRAM). Absolute timings are contention-throttled and
unreliable (a pure triton 1GB read measured 4TB/s while attn "measured" 6TB/s — impossible). Only
**back-to-back ratios** (interleaved fresh processes, median of reps) are trustworthy. Do NOT switch
configs in-process (breaks cudagraph/JIT caching → 50x garbage numbers); use one config per process.
See [[aiter-jit-deadlock-gpu-reclaim]].
