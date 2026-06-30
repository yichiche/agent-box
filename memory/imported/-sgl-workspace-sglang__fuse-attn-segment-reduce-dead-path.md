# fuse_attn_segment_reduce: wrong file + single-split never fires on this workload

Fusion slug `fuse_attn_segment_reduce` (fold reduce_segments into kernel_unified_attention_3d).

## Two independent reasons it cannot be validated on the qwen3.5_mxfp4 perf bench

1. WRONG FILE. Worktree edits `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`
   (sglang-native triton `decode_attention_fwd_grouped`). The perf script runs
   `--attention-backend aiter` + `SGLANG_USE_AITER_UNIFIED_ATTN=1`, so the live decode path is
   `aiter.ops.triton.attention.unified_attention.unified_attention` (imported at
   aiter_backend.py line 49). decode_attention.py is never imported -> dead code.

2. SINGLE-SPLIT FAST PATH NEVER FIRES. The worktree fusion only folds the merge when
   MAX_KV_SPLITS == 1 (single segment). Live decode ALWAYS takes the 3D multi-segment path
   (use_2d=False) with NUM_SEGMENTS in {16..128}, never 1. Confirmed empirically via a temp
   FUSE_ATTN_PROBE print in unified_attention(): decode shows
   ALL_DECODE=True max_k=262144 hq=16 hk=1 hd=256 use_2d=False NUM_SEGMENTS=16..128
   (model: num_attention_heads=32, num_key_value_heads=2, head_dim=256, TP=2 -> hq16/hk1).
   select_3d_config floors at MIN_SEGMENTS=8 and only reaches 1 if max_seqlen_k<=512 (which routes
   to 2D anyway). So single-split fold is structurally unreachable for large-context decode.

## The only fusion that WOULD touch the live kernels is already known net-negative
Folding the multi-segment merge into kernel_unified_attention_3d = the atomic-flag approach from
[fused-attn-num-segments] which is CAPPED ABOVE the two-launch baseline (register/LDS poisoning;
+85% at seg128, best seg32 still 1.8x slower than two-launch). No positive before/after region exists.

## Verdict
status=fail, kernel_time_improvement_pct=null. To make this fusion meaningful you must (a) target the
aiter package live path, and (b) pick a fusion that helps the multi-segment case, which prior work shows
single-kernel merge cannot. GPUs 1,2 free; server boots in ~90s; probe-via-/generate long prompt works.
