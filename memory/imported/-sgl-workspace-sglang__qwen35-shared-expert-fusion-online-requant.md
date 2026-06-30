---
name: qwen35-shared-expert-fusion-online-requant
description: How BF16 shared expert + FP4 routed expert fusion works for Qwen3.5 MXFP4 via online BF16->FP4 requant at load time
metadata: 
  node_type: memory
  type: project
  originSessionId: d7ab85c4-8dae-4065-b42f-b2e2aaed7bf1
---

Qwen3.5-397B MXFP4 checkpoints (e.g. `-d1db9a1`, quant_method=`quark`) exclude the shared expert from FP4 quantization (~244 `mlp.shared_expert.*` entries in `quantization_config.exclude`), so shared experts are BF16 while routed experts are FP4. To still fuse the shared expert into the MoE kernel (aiter path), SGLang online-quantizes the BF16 shared-expert weights to MXFP4 at load time.

**Implementation (in working tree, not yet committed as of 2026-06-11):**
- `qwen2_moe.py::can_fuse_shared_expert(..., allow_online_requant=True)` — no longer hard-disables fusion when shared expert is excluded from quant; `Qwen2MoeSparseMoeBlock` sets `_online_requant=True` for quant names `mxfp4`/`quark`/`quark_mxfp4`.
- `fused_moe_triton/layer.py::FusedMoE._buffer_bf16_for_online_requant` buffers BF16 shared-expert shards (TP-narrowed) during weight load; `_online_requant_fused_shared_experts_mxfp4` quantizes them (`aiter.ops.triton.quant.dynamic_mxfp4_quant` for quark, `aiter.ops.quant.per_1x32_f4_quant` for mxfp4) into expert slot `_num_local_routed` of `w13_weight`/`w2_weight` + e8m0 scales.
- Called from `Mxfp4MoEMethod.process_weights_after_loading` and `QuarkW4A4MXFp4MoE.process_weights_after_loading` (BEFORE scale pre-shuffle).
- Shared expert global id = `config.num_experts` → maps to local slot `_num_local_routed`. MX scale column mapping: packed uint8 weight cols // 16 = hidden//32 scale cols.

**Verified mechanically:** server log shows "Shared expert fusion enabled" + "Online-quantized 3 BF16 shared expert weight shard(s) to MXFP4" per layer. Reference: ATOM commit 02d13056 (layerwise shared expert fusion). Eval gotcha: [[qwen35-thinking-eval-max-tokens]].

**ACCURACY DOES NOT HOLD (re-measured 2026-06-12, MI355X/gfx950, ckpt -d1db9a1, tp2).** The earlier "fused 0.975" claim does NOT reproduce. Current numbers (gsm8k, benchmark/gsm8k/bench_sglang.py):
- Unfused (`--disable-shared-experts-fusion`): **0.904** (Invalid 0.005) — matches the ~0.907 baseline; the non-fusion path is healthy.
- Fused (online requant): **~0.64-0.67** (Invalid ~0.24). Consistent across: flydsl on (0.651) / off (0.637); aiter `/sgl-workspace/aiter`@7a8ff7dd4 vs user's `/home/yichiche/aiter`@3a9ed5dd6 (0.669); max_tokens 512/2048/8192; and **thinking mode** (`--enable-thinking`, 8192 tok: 0.607, Invalid 0.007 — so it's wrong reasoning, NOT just format/truncation).

**Root cause = inherent, not a code bug.** The requant CODE is correct, proven 3 ways: (1) quantizer matches checkpoint method — `dynamic_mxfp4_quant` scaling_mode="even", standalone shared-expert MLP-output cosine vs BF16 = 0.997; (2) placement/scale layout matches the proven `_load_w13`/`_load_w2` + e8m0/weight shuffle (bias-127 e8m0, magnitudes match a real routed expert); (3) zeroing the slot → 0.000/Invalid 1.0, so the placed weights ARE read and load-bearing (0.00→0.64). The shared expert is **dense (every token) and dominant**, so per-layer 8% MXFP4 error compounds across ~48 layers (0.997^48≈0.87) and wrecks multi-step reasoning. This is exactly why the checkpoint EXCLUDES the shared expert from FP4. **Conclusion: keep the shared expert unfused (BF16) for accuracy; fusing it into the MXFP4 kernel is a bad accuracy/perf trade for this ckpt.** Whether the prior 0.975 was a different sglang base, different ckpt, or a measurement error is unresolved.
