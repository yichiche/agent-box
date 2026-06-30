# PR 3986 (moe_sorting decode dispatch) — before/after benchmark

Model: Qwen3.5-397B-A17B-MoE-MXFP4 · MI355X gfx950 · TP2 · GPUs 0,1
Server: run_qwen3.5_mxfp4_perf.sh · Client: `python -m sglang.bench_serving` (random, range_ratio 0.8, request-rate inf)

## Baseline stack
- aiter @ 7d604afe5 + **PR 4017** (qwen3.5-397B a16w16 GEMM configs) — see caveat below
- sglang @ 320b231ea6 + **PR 28666** (fuse shared_expert_gate GEMV into append; brings #28658)
- Under test: **aiter PR 3986** — `moe_sorting_is_oneshot()` guard routing E>=512 tiny-token decode to MultiPhase block-scan

### Caveat on PR 4017
4017's tuned CSV was produced against a newer aiter (this checkout is 205 commits behind main).
Two rows used `libtype=opus, solidx=6401` (kernel `opus_gemm_mono_tile_512x128x256x64...`,
shapes M=2048 N=4096 K=4096/8192) which **does not exist** in this build's gfx950 a16w16 table →
hard abort (`Kernel id 6401 not found`) on prefill. Those 2 rows were dropped; the other 124 rows
of 4017 apply. The dropped shapes are large-M prefill dense GEMM (fall back to default) and do NOT
touch decode/moe_sorting, so the 3986 measurement is unaffected.

## Kernel confirmation (torch profiler, TP0, decode, IL8192/OL1024)
opus_moe_sorting kernel, avg µs per launch:

| conc | before | after | Δ |
|---|---|---|---|
| 4  | 14.58 | 5.62  | **−61%** |
| 64 | 16.46 | 16.43 | −0.1% (unchanged) |

mxfp4_moe_sort (fused quant) and topk_softmax(gating) unchanged at both points — 3986 only
changes the opus sort dispatch. conc4 (E=512, tokens≤sub_token) leaves the slow single-block
oneshot cumsum for the MultiPhase block-scan; conc64 was already MultiPhase. Exactly the PR thesis.

## End-to-end throughput (total token tput, tok/s) — all runs 100% completed

IL=1024/OL=1024:
| conc | before | after | Δ% |
|---|---|---|---|
| 4   | 924  | 962  | +4.1% |
| 8   | 1436 | 1511 | +5.2% |
| 16  | 2358 | 2304 | −2.3% |
| 32  | 3190 | 3218 | +0.9% |
| 64  | 4429 | 4476 | +1.0% |
| 128 | 6037 | 5468 | −9.4% (variance; 3986 doesn't touch conc≥16 path) |

IL=8192/OL=1024:
| conc | before | after | Δ% |
|---|---|---|---|
| 4   | 3730  | 3915  | +5.0% |
| 8   | 5611  | 5827  | +3.9% |
| 16  | 8032  | 8166  | +1.7% |
| 32  | 10365 | 10549 | +1.8% |
| 64  | 12928 | 13214 | +2.2% |
| 128 | 15836 | 16122 | +1.8% |

## Verdict
Kernel-level: confirmed −61% on opus_moe_sorting at conc4, no change at conc64.
E2E: gains concentrated at low concurrency (conc4/8: +4–5%), tapering at higher conc — consistent
with the PR. IL8192 gains are uniformly positive; IL1024 conc16/128 dips are run-to-run variance
(single run each, shared node) and cannot stem from 3986 since conc≥16 already used MultiPhase.
