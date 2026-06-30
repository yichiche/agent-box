---
name: qwen35-gluon-5d
description: How Qwen3.5-MoE was enabled for the pa_decode_gluon / SHUFFLE 5D KV path (PR
metadata: 
  node_type: memory
  type: project
  originSessionId: dfd60cbd-28e0-4750-aaaf-f85fb960aa2b
---

Enabling AITER `pa_decode_gluon` + `SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d` (SHUFFLE 5D) for **Qwen3.5-397B-A17B-MoE-MXFP4** (`Qwen3_5MoeForConditionalGeneration`, model file `models/qwen3_5.py`, head_dim=256, GQA 32/2, gated attn, hybrid GDN+full-attn, `full_attention_interval:4`).

**Why it needed code (gpt-oss in PR #27063 did not):** the gluon path is model-agnostic at the attention-backend/pool layer, BUT Qwen3.5 is a **hybrid** model → full-attn KV uses `HybridLinearKVPool` (wraps an inner `MHATokenToKVPool` as `full_kv_pool`). Two gaps:
1. `AiterAttnBackend._pool_is_vec5d` did `getattr(pool, "kv_cache_layout")` — the wrapper lacked the attr → always "nhd" → gluon never enabled.
2. `aiter_utils.forward_extend_vectorized_5d` resolved sub-pool via `pool.layers_mapping` + `pool.k_buffer` — wrapper has neither.

**Fix (2 files):**
- `mem_cache/memory_pool.py`: added `kv_cache_layout` property on `HybridLinearKVPool` delegating to `full_kv_pool`.
- `layers/attention/aiter_utils.py`: added hybrid branch in `forward_extend_vectorized_5d` (`sub_pool = pool.full_kv_pool`, `sub_layer_id = pool._transfer_full_attention_id(layer.layer_id)`).

Decode works unchanged via `get_kv_buffer` delegation; writes via `set_kv_buffer(layer_id_override=...)` + existing 5D writer. head_dim=256 is pow2 (kernel uses HEAD_SIZE_POW2) — aiter tests only cover 128, so RUNTIME-VERIFY on hardware. NOT done (optional perf follow-up): fused RoPE+KV write (`enable_fused_set_kv_buffer`) on qwen3_5.

**Decode PERF bug found & fixed (3rd change, aiter_utils.py):** `forward_decode_vectorized_5d` set `max_context_partition_num = get_recommended_splits(bs, num_kv_heads)` (capped at `min(...,8)`). The gluon decode kernel actually used (`paged_attention_decode_sliding_window_head_1`, the sliding-window variant, used even for full attn with window=0) HAS a grid-stride loop: `sequence_split_count = num_programs(2)//CONTEXT_PARTITION_SIZE_PER_BLOCK` and each CTA loops over `cdiv(total_partitions, splits)` context partitions of 256 tokens. So it is CORRECT for any max_part_num (NO truncation bug — earlier worry was wrong) but with too few splits each CTA serializes more partitions → slow. With max_part_num=8 on 8K ctx each CTA loops 4× serially. Fix: `max_part_num = max(1, min(64, parts_needed, sm_splits))` where `parts_needed = cdiv(block_tables_pa.shape[1]*backend.page_size, 256)` (page-table WIDTH, graph-safe), `sm_splits = cdiv(props.multi_processor_count*2, bs*num_kv_heads)` (occupancy, scales down at high bs), and **64 = FLYDSL_WARP_SIZE HARD cap**. Removed unused `get_recommended_splits` import. NOTE: autotune in pa_decode_gluon.py is all commented out — not a tuning problem.

**CRASH found during testing (why the 64 cap):** an earlier version used uncapped `cdiv(page_width*page_size,256)`. At CUDA-graph capture bs=512 the page-table buffer width gave max_part_num>64, hitting the flydsl ps-reduce kernel's else-branch (`max_context_partition_num > FLYDSL_WARP_SIZE`, FLYDSL_WARP_SIZE=64 at pa_decode_gluon.py:4524) which fails to compile: `arith.constant(chunk_size) -> TypeError std::bad_cast`. So max_part_num MUST stay <=64. sm_splits also caps it low at high bs (bs=512 -> sm_splits=1) avoiding both the crash and per-call temp-buffer blowup (exp_sums/max_logits/temporary_output scale with max_part_num).

**TRACE RESULTS (MI355X, TP2, IL8k/OL1k). "cc"=decode batch (cc4=bs4, cc64=bs64).** Same kernel `paged_attention_decode_sliding_window_head_1`:
- bs4: gluon before(max_part=8)=37.6us → after(max_part=64)=**22.8us** = ATOM 22.1us PARITY ✓
- bs64: before=125.7us → after=**120.8us** (sm_splits correctly keeps max_part=8 since 64 seqs already saturate GPU; compute-bound, fix can't help here — this is correct)

**HONEST CONCLUSION: gluon is NOT a perf win over unified_attention for Qwen3.5's shape.** unified `kernel_unified_attention_3d` (occupancy-scaled NUM_SEGMENTS_PER_SEQ, 128@bs4/16@bs64) = **11.3us@bs4, 88us@bs64** — beats gluon at BOTH batches. Reasons (structural, not tunable in sglang): (1) unified splits up to 128, gluon capped at 64 by flydsl reduce bug → half the parallelism at low bs; (2) gluon is 2-pass (compute+reduce) vs unified 1 kernel; (3) pa_decode_gluon tuned for gpt-oss (head64/8kv), Qwen3.5 is head256/1kv-per-rank(TP2) — ATOM hits same ~22us ceiling. To beat unified needs aiter-side work (lift flydsl 64 cap / retune gluon for head256). unified num_segments heuristic is at aiter `ops/triton/attention/unified_attention.py:108-154`.

**PR: https://github.com/sgl-project/sglang/pull/29009** (branch `yichiche/qwen3_5-moe-gluon-decode` on yichiche fork → sgl-project:main). Framed as enablement(hybrid) + ATOM-parity + CUDA-graph-capture crash fix, NOT a perf win vs unified. Commit 785aa65c52.

Launch: `SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d ... --prefill-attention-backend aiter --decode-attention-backend aiter --page-size 64 --disable-radix-cache`.
