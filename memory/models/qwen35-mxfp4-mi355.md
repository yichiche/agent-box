---
model: Qwen3.5-397B-A17B-MoE-MXFP4
hardware: MI355 (gfx95)
aliases: [qwen35, qwen3.5 mxfp4, qwen moe mxfp4]
---

# Qwen3.5 397B MoE MXFP4 on MI355

## Canonical scripts

| Role | Path |
|---|---|
| Server | `$HOME/run_qwen3.5_mxfp4_perf.sh` |
| Client (InferenceX-style sweep) | `$HOME/run_qwen3.5_mxfp4_inferencemax_client.sh` |
| Agent-tuned client | `$HOME/run_qwen3.5_mxfp4_perf_agent.sh` |
| ATOM variant | `$HOME/run_qwen3.5_mxfp4_perf_atom.sh` |

## Server flags (from `run_qwen3.5_mxfp4_perf.sh`)

```bash
AITER_FLYDSL_FORCE=1 \
SGLANG_USE_AITER_UNIFIED_ATTN=1 SGLANG_USE_AITER=1 \
python3 -m sglang.launch_server \
  --model-path /data/amd/Qwen3.5-397B-A17B-MoE-MXFP4 --tp 2 \
  --attention-backend aiter --trust-remote-code \
  --chunked-prefill-size 32768 \
  --model-loader-extra-config '{"enable_multithread_load": true}' \
  --watchdog-timeout 1200 --mem-fraction-static 0.9 \
  --host 0.0.0.0 --port 9000 --disable-radix-cache \
  --enable-aiter-allreduce-fusion --max-running-requests 512 \
  --page-size 16
```

## Benchmark defaults (client)

- `PORT=9000`, `INPUT_LEN=70000`, `OUTPUT_LEN=300` (override with env)
- Concurrency sweep: `4 8 16 32 64 128 256`
- `num_prompts = concurrency × 10` (benchmark) — matches
  `run_qwen3.5_mxfp4_inferencemax_client.sh`; profiling capture uses × 2 (see [[../workflows/workloads]]).
- **CWD:** launch server AND client from `/tmp` or other neutral dir — see [[../gotchas/bench-cwd-shadow]]

## Reference table (the default output compares against this)

Every benchmark of this model presents its results **side by side with this reference
table** and shows the per-cell delta (`(measured − ref)/ref`). Machine-readable copy:
[`qwen35-mxfp4-mi355-reference.csv`](qwen35-mxfp4-mi355-reference.csv). Both shapes are
part of the benchmark set ([[../workflows/workloads]]); perf **claims** are still made
only on `canonical-8k`.

Columns: **Median E2E (ms) · total tok/s · total tok/s/gpu · Median TTFT (ms) ·
Median TPOT (ms)**, per concurrency.

### diag-1k — ISL 1024 / OSL 1024

| conc | Median E2E (ms) | total tok/s | tok/s/gpu | TTFT (ms) | TPOT (ms) |
|---:|---:|---:|---:|---:|---:|
| 4 | 7,415.5 | 957.0 | 478.5 | 118.6 | 8.0 |
| 8 | 9,222.9 | 1,580.2 | 790.1 | 121.4 | 9.8 |
| 16 | 13,244.6 | 2,470.6 | 1,235.3 | 125.9 | 12.8 |
| 32 | 16,626.4 | 3,561.3 | 1,780.7 | 137.4 | 17.6 |
| 64 | 23,922.4 | 5,038.2 | 2,519.1 | 137.0 | 25.2 |
| 128 | 33,678.6 | 6,840.3 | 3,420.2 | 203.0 | 37.5 |
| 256 | — | — | — | — | — |

### canonical-8k — ISL 8192 / OSL 1024 (claim shape)

| conc | Median E2E (ms) | total tok/s | tok/s/gpu | TTFT (ms) | TPOT (ms) |
|---:|---:|---:|---:|---:|---:|
| 4 | 8,584.4 | 3,772.2 | 1,886.1 | 372.6 | 8.8 |
| 8 | 11,006.0 | 5,939.7 | 2,969.8 | 377.2 | 11.4 |
| 16 | 15,023.2 | 8,567.6 | 4,283.8 | 380.8 | 16.0 |
| 32 | 24,241.7 | 10,691.7 | 5,345.9 | 468.3 | 25.3 |
| 64 | 38,581.7 | 13,822.5 | 6,911.3 | 523.5 | 41.1 |
| 128 | 61,862.7 | 17,290.7 | 8,645.3 | 633.3 | 66.4 |
| 256 | — | — | — | — | — |

> conc 256 rows are intentionally blank in the reference (not yet measured). Fill them
> in here once a trusted 256 run lands; until then, treat 256 as no-reference.

## Accuracy

- Dataset: GSM8K. **Thinking model — eval protocol is mandatory** (see [[../workflows/accuracy]]):
  ```bash
  python3 benchmark/gsm8k/bench_sglang.py --num-questions 200 --parallel 100 --num-shots 5 \
    --enable-thinking --tokenizer-path /data/amd/Qwen3.5-397B-A17B-MoE-MXFP4 \
    --max-new-tokens 8192 --port <PORT>
  ```
- **Ship threshold: 0.92 — but only for an accuracy-VALID config.** Read a low
  score by invalid rate first:
  - **invalid high (>~0.05)** → eval artifact (token budget / thinking off / bare
    prompt). Fix the eval, re-run. NOT a weights problem.
  - **invalid low but acc low** → real degradation. Don't ship; don't relax the bar.
- **Tier 1 (valid, target ≥0.92):** proper eval 0.985; shared expert unfused BF16
  0.904; a16w4+instruction prompt 0.970; GOOD ckpt (`Qwen3.5-397B-A17B-MXFP4`) 0.968.
- **Tier 2 (KNOWN-BAD, NEVER SHIP):** this MoE-MXFP4 ckpt's **fp4 shared expert**
  caps accuracy — fused fp4 shared ~0.61–0.67, a16w4 bare ~0.46, a4w4 ~0.62. Fix =
  give the shared expert BF16 precision (unfused). Confirmed 2026-07-16: the live
  `run_qwen3.5_mxfp4_perf.sh` interleave path (`AITER_FP4BF16_USE_ITLV=1` +
  `SGLANG_USE_AITER_MOE_GU_ITLV=true`) measured **0.625 / invalid 0.010** = Tier 2.
- **a16w4 INTERLEAVE is a *second, independent* degrader (2026-07-12/13,
  [[a16w4-gsm8k-accuracy-regression]]).** The a16w4 routed-expert path (bf16 act +
  gui_fp4 FlyDSL kernel) **~doubles GSM8K invalid rate**: a4w4 0.617/inv 0.211 →
  a16w4 0.456/inv 0.525 (correct-when-valid stays ~0.9+). It is weight-value-dependent
  and NOT the shared-expert fusion: `--disable-shared-experts-fusion` (E=512) did not
  help (0.417/0.519), overturning the earlier shared-fusion hypothesis. **Dead ends —
  do NOT re-chase:** (1) NOT the fp8-activation branch (forcing bf16 via
  `AITER_BF16_FP8_MOE_BOUND=1000000` gave identical 0.456/0.525; the fp8 branch only
  *crashes* at cudagraph capture for M≥256); (2) NOT truncation (max_new_tokens
  512→2048 didn't lower invalid); (3) NOT shared fusion (above). **Practical:** serve
  with **a4w4** (`SGLANG_USE_AITER_MOE_GU_ITLV=0`) — a16w4 is a net accuracy loss here,
  its tuning is perf-only. Root fix = tighten the op-level test vs fp32 ref (current
  `op_tests/test_flydsl_moe_a16w4.py` atol=1.0 is too loose: passes 100% while a small
  systematic error derails long generation).
- The old "verified 0.945 @ 2026-06" number predates this checkpoint drift; treat
  0.92 as the bar under the protocol above, not as a standing measurement.

## Profiling

- Use `/generate-profile` or `python profile/trace_module_analyzer.py`
- Decode-heavy @ `canonical-8k` (IL8192/OL1024): prefill optimizations have low
  Amdahl leverage (~2–3.5%)
- High-concurrency profiling can OOM host — profile at conc 4 for decode deep-dives

### `trace_module_analyzer.py` params for Qwen3.5 (NOT DSv4 defaults)

Qwen3.5 MoE is a **hybrid** decoder: `Qwen3_5LinearDecoderLayer` and
`Qwen3_5AttentionDecoderLayer` appear in the module tree. The parser's default
detail modules (`DeepseekV4DecoderLayer` / `Layer` at instances 31/32 or 59-62)
are **wrong** for this model and produce empty detail sheets.

**Critical: module-level detail needs a COMBINED trace, not a stage-separated one.**
Verified 2026-07-16 on real traces:

- A stage-separated **`*-DECODE.trace.json.gz`** (what `--profile-by-stage`
  produces) is CUDA-graph-replayed → its module tree collapses to a single
  `CudaGraphReplay` node (~92% of time, only ~16 nn.Module events).
  `Qwen3_5LinearDecoderLayer` is **not found** → you get only the kernel-category
  breakdown, no per-module detail. So for Qwen module analysis, capture a
  **combined** trace (omit `--profile-by-stage`).
- On a **combined** trace, this reproduces the reference prefill detail sheets:

  ```bash
  python3 $AGENT_BOX_DIR/profile/trace_module_analyzer.py \
    <combined-TP-0.trace.json.gz> -o analysis.xlsx \
    --detail-module Qwen3_5LinearDecoderLayer --detail-instance 0 1
  # → sheets: Qwen3_5LinearDecoderLayer_0 (pre), _1 (pre)  [VERIFIED]
  ```

- **OPEN / unverified:** the exact recipe for the *decode* per-layer detail sheets
  seen in `~/qwen3.5-mxfp4/agent_round1/analysis_decode_pristine.xlsx` (the `(dec)`
  tabs) is NOT reproduced by plain `--detail-instance 0 1` nor `--phase-index Decode`
  on the combined trace. Confirm the correct instances/phase against the **Module
  Tree** sheet on a fresh combined capture before trusting decode detail, and update
  this card once nailed down.
- `--detail-instance` IDs can shift with server config — always sanity-check against
  the Module Tree sheet. To also detail attention layers, add
  `Qwen3_5AttentionDecoderLayer` as a second `--detail-module`.

### Output directory convention

**Do NOT save Qwen3.5 artifacts under `~/dsv4/`** (the DSv4 default some skills
still hardcode). Use `~/qwen3.5-mxfp4/<label>/`. Mixing models in `~/dsv4/`
already happened (`~/dsv4/qwen35_mxfp4_validation/`) and makes traces hard to find.

## Related memory

- [[../gotchas/bench-cwd-shadow]]
- [[../gotchas/container-bench-flags]]
- Imported: `~/.claude/.../qwen35-moe-decode-roundzero.md`, `qwen35-moe-gemm-e2e-amdahl.md`
