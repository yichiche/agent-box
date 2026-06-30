# MoE decode prelude (CK-opus vs HIP index-sort) — E2E gate ceiling

E2E gate (sglang.bench_serving IL=8192/OL=1024 TP2 on the MXFP4 fused_moe decode path)
CANNOT run in this env: no servable MXFP4 MoE weights on disk. The only candidate
(amd/Qwen3.5-397B-A17B-MXFP4) is a ref pointer only — no blobs/safetensors, ~12K, and a
397B model needs >>TP2. No other model has weights locally.

FALLBACK (per plan): keep the standalone HIP index-sort (exp `decode_specialized_grid`)
wired into _moe_sorting_impl behind AITER_USE_HIP_INDEX_SORT (default OFF, opus fallback
for mask/padding/local-id). It is the minimal CORRECT improvement.

Closest honest proxy measured = PRODUCTION dispatch path `aiter.moe_sorting` toggling the
flag (run_perftest, HIP_VISIBLE_DEVICES=0, MI355X, E256 topk8 md4096 block32):
  M=16:  CK 10.39/10.44  -> HIP 6.20/6.28  (+40%)
  M=64:  CK  8.37/9.06   -> HIP 6.77/6.72  (+19%)
  M=128: CK  8.23/9.20   -> HIP 7.50/7.57  (+9-19%)
HIP is M-FLAT ~6.2-7.6us; CK fluctuates 8.2-10.4us (more launch-bound jitter).
Worst-case shape (M=128) still +9% min. Correctness: test_moe_sorting + test_moe_sorting_mxfp4
all pass.

The fused index+scale kernel (moe_index_scale_sort_hip, exp fuse_index_and_scale_sort) is
even faster on the COMBINED prelude (single launch ~8.5-9.8us vs two-kernel 12.6-14.5us,
+30-35%) but is NOT wired into production — reachable only via direct binding, because
fused_dynamic_mx_quant_moe_sort calls moe_sorting and the scale-sort as two separate ops.
Productionizing the FUSED kernel requires merging those two call sites in fused_moe.py.

CEILING NOTE: the prelude is a tiny fraction of decode e2e (a ~3-4us absolute saving per
decode step against full MoE GEMM+attn). Without a real e2e run we cannot claim a serving
total_throughput delta; expect it to be well under 5% and likely in the noise. Promote the
HIP index-sort as a clean micro-win, do NOT claim an e2e throughput number.
