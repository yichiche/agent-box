---
name: aiter-fmoe-untuned-csv-shape-rule
description: "How to derive expert/topk/inter_dim columns for aiter untuned_fmoe.csv (shared expert folded in, inter_dim is TP-sharded)"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 07bbdb90-7c00-484a-9eed-7484ebcc5034
---

aiter fused-MoE tuner (`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`) reads shapes from an untuned_fmoe.csv. Two columns are non-obvious:

- **`expert`/`topk` shared-expert folding is PATH-DEPENDENT — do not assume.** The FP8 a8w8 path folds the shared expert into the fmoe key: Qwen3.5 a8w8 uses `513`/`11` (512 routed+1, topk 10+1), DSv3 fp4 uses 257/9. BUT the **MXFP4 path does NOT fold it** — the shipped, runtime-tuned `qwen3_5_397b_fp4_*_fmoe.csv` uses `512`/`10`. So for Qwen3.5 MXFP4 use **512/10**, for Qwen3.5 FP8 use 513/11. When unsure, let the runtime auto-dump (see below) rather than guessing.
- **`inter_dim` = moe_intermediate_size / TP** (per-GPU sharded), NOT the full intermediate size. `model_dim` stays full (only intermediate is sharded). Qwen3.5 moe_intermediate_size=1024 → TP2 gives 512, TP4 gives 256. DSv3 2048/TP8 = 256. NOTE: shipped `qwen3_5_397b_fp4_*` files are tuned for TP4 (inter_dim=256) — for TP2 you must retune with inter_dim=512.
- **Authoritative way to get exact shapes:** aiter auto-writes `untuned_fmoe.csv` from real runtime fmoe calls (`aiter/fused_moe.py:1065`). Move any existing tuned csv aside, start the server with your real TP, send a few requests, and it dumps the true shapes (correct expert/topk/inter_dim for that exact config). Then tune that file.

Other Qwen3.5-397B MXFP4 cols: model_dim=4096, act=ActivationType.Silu, dtype=torch.bfloat16, q_dtype_a/w=torch.float4_e2m1fn_x2, q_type=QuantType.per_1x32, use_g1u1=1, doweight_stage1=0. token rows should be powers of 2 (tuner keys on nextPow2(token)).

Cross-check against shipped CSVs in `aiter/configs/model_configs/` (e.g. a8w8_blockscale_untuned_fmoe_qwen3_5_397b.csv already uses 513/11) before authoring a new one.

Caveat: for MXFP4 the served path may select kernel structurally and ignore the tuned CSV — see [[mxfp4-moe-not-csv-tunable]]. Gate wins on e2e per [[implement-e2e-vs-microbench-gate]].
