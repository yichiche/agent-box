---
name: qwen35-mxfp4-flydsl-fully-fused
description: "Qwen3.5-MXFP4 MI355 \"all_kernel_fusion_flydsl\" decode trace is already fully fused; B200 gap is hardware not fusion"
metadata: 
  node_type: memory
  type: project
  originSessionId: 303152cc-487d-40e5-a56e-cc53f549ca01
---

The Qwen3.5-MXFP4 MI355 decode traces under `.../all_kernel_fusion_flydsl_*/` are the POST-fusion state — they already fuse every elementwise/norm pattern (`_fused_qk_gemma_rmsnorm_gate_kernel`, `_triton_mrope_forward_fused`, `_sigmoid_gate_mul[_broadcast]_kernel`, `act_and_mul`, aiter `allreduce_fusion_kernel_1stage`).

Comparing against the B200 reference (IL8k decode), B200 actually runs MORE separate small kernels: split `sigmoid_kernel`+`MulFunctor` (vs MI355's fused sigmoid_gate_mul), dual separate `RMSNormKernel` (vs MI355's fused qk_gemma_rmsnorm), and cublas `dot_kernel`/`reduce_1Block` gemv overhead MI355 lacks.

**Why:** Means a kernel-fusion-pipeline run on these two traces finds ZERO Tier-1 fusions (no op B200 fuses is left unfused on MI355). The remaining MI355 vs B200 gaps are NOT fusions: `reduce_segments` (~9.4us, decode attention split-K reduction — attention-backend/HIP, Tier-2/3) and GEMM efficiency (Tensile `Cijk` vs `nvjet`). The residual `CUDAFunctor_add` is separate on BOTH platforms, so not a fusion gap.

**How to apply:** Don't fabricate a Tier-1 fusion for these traces — terminate at compare step and report Tier-2/3 (attention reduce_segments, GEMM). Real wins here are hardware/library, not [[kernel-fusion-pipeline]] elementwise fusions. Note `post_residual_addition` in GemmaRMSNorm (layernorm.py) is only non-None for VL deepstack models, not text Qwen3.5.
