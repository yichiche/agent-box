# fuse_attn_segment_reduce targeted the wrong code path

## Finding
The fusion `fuse_attn_segment_reduce` modified:
`python/sglang/srt/layers/attention/triton_ops/decode_attention.py`
folding the stage-2 segment reduce into `_fwd_grouped_kernel_stage1` for the
single-split (max_kv_splits==1) case, via `decode_attention_fwd_grouped` ->
`_decode_grouped_att_m_fwd`.

## Why it cannot be validated e2e
- Benchmark = run_qwen3.5_mxfp4_perf_agent.sh -> `--attention-backend aiter`
  with `SGLANG_USE_AITER_UNIFIED_ATTN=1`.
- That dispatches decode through
  `aiter.ops.triton.attention.unified_attention.unified_attention`
  (kernels `kernel_unified_attention_3d` + `reduce_segments`), located in
  `/sgl-workspace/aiter/aiter/ops/triton/`.
- `decode_attention_fwd_grouped` (the modified fn) is imported only by
  `triton_backend.py` and `wave_backend.py`. `aiter_backend.py` never calls it.
- So the edited code is dead under this benchmark; the named target kernels
  (kernel_unified_attention_3d, reduce_segments) were never touched.

## Lesson
The config "kernels" field named aiter kernels but the patch edited the
sglang-native Triton path. To actually fuse reduce_segments into
kernel_unified_attention_3d you must edit the aiter package's
unified_attention.py, not sglang's triton_ops/decode_attention.py.
PERF GATE: report kernel_time_improvement_pct = null (no before/after possible).
