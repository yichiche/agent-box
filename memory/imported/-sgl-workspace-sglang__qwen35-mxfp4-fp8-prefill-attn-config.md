---
name: qwen35-mxfp4-fp8-prefill-attn-config
description: "Qwen3.5-MXFP4 MI355 prefill-attention gap vs B200 is mostly a KV-cache-dtype config difference, not a kernel/fusion gap"
metadata: 
  node_type: memory
  type: project
  originSessionId: fc6b9734-53f3-43c6-b11b-90ad47cae71f
---

In the Qwen3.5-MXFP4 prefill comparison (MI355 flydsl all_kernel_fusion vs B200, IL8k TP2 cc4), the full-attention core gap (MI355 ck_tile FmhaBatchPrefillWithPagedKVCache BF16 = 2619us/layer vs B200 fmhaSm100 QkvE4m3 FP8 = 848us, 3.1x) is **mostly a launch-config difference**: the B200 run used FP8 KV cache (`_fused_fp8_set_kv_buffer_kernel`, `QkvE4m3`), the MI355 run used BF16 (`store_kvcache` BF16). 

The MI355 aiter MHA path already supports FP8 — gated on `kv_cache_dtype == fp8_dtype` at `python/sglang/srt/layers/attention/aiter_backend.py:2310-2312` (casts q→fp8, sets q_descale; `mha_batch_prefill_func` in aiter `ops/mha.py:2971` takes q/k/v_descale). gfx950 confirmed `is_gfx95_supported()==True`. NOTE: `_use_fp8_prefill_attn` / `mla_fp8_prefill_attn` in that file are **MLA-only** (DeepSeek), NOT used by Qwen3.5 GQA.

**Why:** A fair MI355-vs-B200 prefill-attn comparison must match `--kv-cache-dtype fp8_e4m3`; otherwise ~half the "3.1x" is apples-to-oranges. **How to apply:** Re-run MI355 with `--kv-cache-dtype fp8_e4m3` first (validate accuracy — thinking model, max_tokens=8192 per [[qwen35-thinking-eval-max-tokens]]) before any CK kernel tuning. Residual after FP8 is Blackwell HW + CK tile tuning (head-dim 256). The bigger overall gap in this trace is AllReduce (quickreduce twoshot ~1860us/op vs NCCL NVLink-multicast ~485us, ~2.7-2.9ms/layer). Relates to [[qwen35-mxfp4-flydsl-fully-fused]] (no Tier-1 fusions left; gaps are HW/kernel/config).
