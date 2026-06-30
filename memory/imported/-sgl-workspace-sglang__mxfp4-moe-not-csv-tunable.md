---
name: mxfp4-moe-not-csv-tunable
description: Qwen3.5 MXFP4 fused-MoE 2-stage path cannot be tuned via tuned_fmoe.csv — dispatch is structural
metadata: 
  node_type: memory
  type: project
  originSessionId: 5398c61a-c651-46c8-b372-8b4f53625571
---

The Qwen3.5-397B-A17B-MXFP4 fused-MoE 2-stage path is NOT tunable via `aiter/configs/tuned_fmoe.csv`.
For its signature (`q_type=per_1x32`, `q_dtype_a/w=fp4x2`, `act=Silu`) the kernel is chosen STRUCTURALLY in
`aiter/fused_moe.py` get_2stage_cfgs dispatch: `ksplit>1 & is_shuffled` → `cktile_moe_stage1/2`
(`aiter.moe_cktile2stages_gemm1/2` = `ck_tile::MoeFlatmmKernel`, fused_moe.py:1047), else `ck_moe_stage1`
(`kernel_moe_mxgemm_2lds`, fused_moe.py:1073). Neither reads `kernelName1/2` from the CSV, so the official
`gemm_moe_tune.py` (emits `moe_ck2stages_gemm`/`asm`/`flydsl_*`) is a no-op → runtime logs `[fused_moe] using 2stage default`.

Two compounding traps when targeting this with an autotuner:
1. **Structural dispatch** — tuned CSV kernel names are ignored for this signature (the real cause).
2. **Exact key on padded token** — CSV lookup keys on `token = get_padded_M(token_num) = nextPow2(token_num)`
   (fused_moe.py:614). A prefill bucket of 7237 tokens keys on **8192**; tuning at raw 7237 never matches.

**Why:** the kernel-fusion-pipeline's aiter-tune track assumed "slow MoE GEMM ⇒ has an official CSV tuner whose
output the runtime reads." False for MXFP4 here.

**How to apply:** Before proposing/running an aiter MoE/GEMM CSV tune, confirm (a) the runtime selects the kernel
from the matched CSV row (not a structural branch) for the deployed signature, (b) the tuner's kernel family
matches the kernels in the baseline trace, and (c) the tune shape uses the `nextPow2(token)` bucket. The pipeline
now enforces this via a pre-tune gate (`TUNE_GATE_SCHEMA`). Real perf levers for this path are `ksplit`/`block_m`
heuristics (`get_ksplit`/`get_block_size_M`) and the cktile-vs-mxgemm choice — an aiter code change, not a CSV.
Related: [[qwen35-mxfp4-flydsl-fully-fused]], [[implement-e2e-vs-microbench-gate]].
