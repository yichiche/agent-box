# flydsl MoE fp8 reduce-intermediate (accurate re-impl)

Variant `fp8_intermediate_scalar_unpack`: store the reduce-mode stage2
[tokens*topk, model_dim] intermediate as fp8 e4m3 (1 byte) instead of bf16
(2 bytes), halving the dominant HBM traffic (GEMM2 write + reduction read).

- Files: aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py (compile_mixed_moe_gemm2
  gains reduce_intermediate_fp8 flag; no-atomic store_pair extf bf16 frag -> f32 ->
  cvt_pk_fp8_f32 packed i32 words, 1-byte addressed via out_elem_bytes=1),
  moe_gemm_2stage.py (compile_moe_reduction gains in_dtype_str='f8': loads i32 words,
  cvt_pk_f32_fp8 unpack to f32, scalar f32 accumulators, writes bf16),
  moe_kernels.py (allocate target as float8_e4m3fn, thread flag + in_dtype_str).
  Env gate FLYDSL_MOE_REDUCE_FP8_INTERMEDIATE (default 1); off reverts to baseline exactly.

- MEASURED (vs baseline 286.44/373.76/520.96 us): t2048 266.2 (+7.1%), t4096 329.0
  (+12.0%), t8192 437.8 (+16.0%); agg +11.67%, min +7.06%. Reproducible; flag-off
  matches baseline (286.08/519.60).

- KEY: accuracy max_delta=0.0352 (vs prior unscaled-fp8 BEST's 1.15) at 100% close.
  This UNSCALED e4m3 cast is far more accurate than the historical fp8 variant claimed,
  because the GEMM2 frag is already routing-weight-scaled and lands in e4m3 range here.
  So this is near-best perf (~12% vs 13.16% BEST) but SHIPPABLE accuracy — preferred for e2e.

- Why slightly below 13.16% BEST: kept VEC_WIDTH=8 for fp8 reduction (BEST may have
  vectorized the fp8 unpack / wider loads). NEXT: vectorize the cvt_pk_f32_fp8 unpack
  into vec accumulators + widen fp8 reduction reads to dwordx4 (load 4 i32 words/slot),
  should recover the ~1.5% gap. The GEMM2 fp8 write side is now the larger remaining term.
