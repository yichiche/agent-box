---
name: qwen35-decode-layer17-optimization
description: "Qwen3.5 decode LinearDecoderLayer optimization — shared-gate fusion win, <100us infeasible under constraints on MI355X"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2d9d6c67-249b-4705-9357-fbf5baf6d4b1
---

Goal was to get `Qwen3_5LinearDecoderLayer_17 [decode]` 16-kernel block <100µs without fused_moe / aiter recompile.

**Environment (this box):** MI355X (gfx950), TP2, checkpoint `/raid/models/Qwen3.5-397B-A17B-MXFP4-d1db9a1`. Baseline reference (114.5µs) was MI300X + checkpoint `96f60ef` — different HW + checkpoint, not directly comparable.

**Profiling method:** must launch with `--disable-cuda-graph` to get module-level decode breakdown (graph-on traces attribute all decode kernels to CudaGraphReplay roots → no module split). Profile via `python3 -m sglang.bench_serving ... --profile --profile-start-step N --profile-steps 24`; trace lands in server's `SGLANG_TORCH_PROFILER_DIR` (or /tmp). Analyze: `trace_module_analyzer.py <trace> -o out.xlsx --detail-module Qwen3_5LinearDecoderLayer --max-detail-modules 20` (NO `--phase-index`; roots are `Qwen3_5MoeForCausalLM_0`, decode instances auto-tagged `[decode]`). Current SGLang splits sub-modules (GatedDeltaNet/FusedMoE/RadixLinearAttention) so full layer = 26 kernels; the baseline-comparable 16-kernel block = exclude GDN-attention internals + norms + in_proj.

**Result:** pristine 132.4µs → candidate 117.0µs (−15.4µs). Win = shared-expert-gate fusion (`_fused_gate_sigmoid_mul_kernel` in [[qwen35-shared-expert-fusion-online-requant]] area): collapses gate gemm 9.4 + sigmoid 4.1 + mul 5.4 = 18.9µs → 4.0µs. The **in_proj fusion is dead code on d1db9a1** (gates on UnquantizedLinearMethod, but in_proj is quantized here) — dropped from the PR.

**<100µs infeasible** under constraints: remaining 117µs = aiter/vllm MoE path 58µs (topk/sorting/quant/mfma1/mfma2) + quantized hipBLASLt gemms 40µs + eltwise/allreduce. Crossing 100µs needs fused_moe or aiter recompile (both forbidden). User accepted 117µs as best-achievable-under-constraints and chose to PR it.
