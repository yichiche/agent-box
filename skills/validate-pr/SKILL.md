---
name: validate-pr
description: "Validate an aiter and/or sglang PR end-to-end for a served model (default Qwen3.5-397B-A17B-MXFP4): (1) conc4 kernel validation before/after via profiling + trace_module_analyzer to confirm the target kernel time changed or a new kernel replaced the old one, (2) gsm8k accuracy on the PR (after), (3) benchmark before (conc4~256), (4) benchmark after (conc4~256), (5) a per-concurrency performance table (ISL/OSL/conc/Median E2E/total tok-s/tok-s-per-gpu/TTFT/TPOT). Use when asked to 'validate a PR', '/validate-pr', or do before/after kernel+accuracy+throughput review of a PR."
category: measure
---

# validate-pr — end-to-end PR validation (kernel → accuracy → before/after benchmark → table)

Validate a performance/correctness PR (aiter, sglang, or both together) on a served model.
Produces four artifacts: a **kernel before/after** (profiling proof the target kernel changed),
an **accuracy** number (gsm8k), and **before/after throughput sweeps** rendered as one **perf table**.

Default model: **Qwen3.5-397B-A17B-MXFP4** (TP2, MI355/gfx950). Generalize by swapping the model
card, run script, and the `trace_module_analyzer` `--detail-module/--detail-instance` values.

Honor the global conventions in `~/agent-box/CLAUDE.md`:
- **num_prompts** — profiling capture `conc × 2`; benchmark/sweep `conc × 10`.
- **Profiling/kernel anchors** — conc4 (and conc64 if time). **Benchmark** — conc 4,8,16,32,64,128,(256).
- **Shapes** — if none given, run both `diag-1k` (IL1024/OL1024) and `canonical-8k` (IL8192/OL1024); **claim perf on canonical-8k**.
- **Model card** — `memory/models/qwen35-mxfp4-mi355.md` (accuracy threshold, run script, profiling module/instances).

---

## Step 0 — Setup (before/after mechanism, GPUs, server)

1. **Identify the PR(s) and the before/after toggle.**
   - aiter PR: usually already applied as an uncommitted working tree under `/sgl-workspace/aiter`
     (`git stash` = **before**, restore = **after**). Confirm with `git -C /sgl-workspace/aiter status -s`.
     aiter is an editable/JIT install → **no rebuild** for pure-Python/flydsl changes; `.so`/csrc changes need `make build`.
   - sglang PR: apply the diff (`gh pr diff <N> | git -C /sgl-workspace/sglang apply`) or hand-edit;
     back up originals to `/tmp` first. **Do NOT `gh pr checkout`** — it moves to the PR's *old* base;
     you want *current main + the PR diff*.
   - **before = feature reverted, after = feature applied.** Keep everything else identical between the two.

2. **Pick free GPUs** — `python ~/agent-box/skills/gpu-status/gpu_status.py` → use the printed
   `CUDA_VISIBLE_DEVICES` (SGLang uses **CUDA_VISIBLE_DEVICES**, not HIP; rocm-smi index ≠ CUDA index).
   Verify the target GPUs read ~0.3 GiB before launching.

3. **Server launch** — reference `~/run_qwen3.5_mxfp4_perf.sh`; client `~/bench_serving/benchmark_serving.py`
   and `~/run_qwen3.5_mxfp4_inferencemax_client.sh`. Launch each server with `setsid` so it is its own
   process group (clean, targeted teardown). Wait for `/get_model_info` to return 200 (up to ~30 min for 397B).

> **Load-bearing gotchas** (learned the hard way — do not skip):
> - **Kill with SIGTERM to the server's process group, never a broad `pkill -9`.** `SIGKILL` orphans the
>   TP-worker/scheduler children → **~200 GB/shard VRAM leak that is unreclaimable from inside the container**
>   (needs host-side kill). Allow >120 s for graceful SIGTERM; a Bash-tool timeout mid-wait can force a bad SIGKILL.
> - **A stray `sweep.sh` reparents to init and keeps spawning benchmark clients** after you kill its parent —
>   kill the whole tree (`pkill -f sweep.sh; pkill -f benchmark_serving.py`).
> - **Never `pkill -f sglang.launch_server` broadly** — it kills other tenants' servers on other GPUs.
> - **Kill and relaunch the server between before/after** (and between configs); do not hot-swap.

---

## Step 1 — Kernel validation (conc4, before / after, with profiling)

Goal: prove the PR's target kernel time changed, **or** that a new kernel replaced the old one.

For **each** variant (before, after):
1. Launch the server with `SGLANG_TORCH_PROFILER_DIR=<profdir>` set.
2. Capture a conc4 profile (num_prompts = conc×2 = **8**):
   ```bash
   python3 ~/bench_serving/benchmark_serving.py --model="$MODEL" --backend=sglang --port=8000 \
     --dataset-name=random --random-input-len=8192 --random-output-len=1024 --random-range-ratio=0.8 \
     --num-prompts=8 --max-concurrency=4 --request-rate=inf --ignore-eos --profile \
     --save-result --result-dir=<profdir> --result-filename=prof_conc4.json
   ```
3. Kill the server.

Parse each trace with **trace_module_analyzer** (Qwen3.5-MXFP4 values shown; adjust per model card):
```bash
python3 /home/yichiche/agent-box/profile/trace_module_analyzer.py \
    <profdir>/<...>-TP-0.trace.json.gz \
    -o analysis_decode.xlsx \
    --phase-index CudaGraphReplay_0 \
    --detail-module Qwen3_5LinearDecoderLayer Qwen3_5LinearDecoderLayer Qwen3_5LinearDecoderLayer Qwen3_5AttentionDecoderLayer \
    --detail-instance 15 16 17 5
```
Then compare **before vs after**: read the target kernel's summed/mean µs from each Excel (or
`/compare-kernels before.xlsx after.xlsx`). Report either **Δµs on the same kernel** or **old kernel → new kernel name**.

> **Profiling gotchas:**
> - `--phase-index CudaGraphReplay_0` requires the decode ran **under CUDA graph** (backend `full`).
>   If the config is **eager** (`--disable-decode-cuda-graph`), the CudaGraphReplay phase won't exist —
>   use the eager decode phase / omit `--phase-index`, and expect a much larger trace.
> - **Profiling eager decode is dangerous**: even conc4/np8 produced a **>1 GB** trace that took ~40 min to
>   flush and **hung the scheduler** (GPU 0%, no forward passes). Keep profiling num_prompts small; prefer
>   profiling a config that can use CUDA graph; if you must profile eager, expect huge traces and verify the
>   server recovers before running accuracy/benchmark on it.
> - The heavy TP rank (usually TP-1) holds the MoE-GEMM kernels; the other rank's trace may be near-empty.
> - The aiter **dispatch log** (`[aiter][fused_moe] using 2stage (kernelName1=...)`) is itself conclusive
>   proof of which kernel/dtype ran — grep the server log for it as a fast confirmation.

---

## Step 2 — Accuracy check (gsm8k, AFTER)

On the **after** (PR applied) server:
```bash
python3 -m sglang.test.few_shot_gsm8k --num-questions 200 --num-shots 5 \
  --max-new-tokens 8192 --parallel 128 --port 8000
```
- Qwen3.5 is a **thinking model** → use `--max-new-tokens 8192` (small values tank the score to ~0.55).
- Pass threshold from the model card (Qwen3.5-MXFP4 ≈ **0.92**). Report `Accuracy` and `Invalid`.
- Optionally run on **before** too for a correctness delta.

**Gate:** if accuracy collapses, stop and debug before spending hours on the benchmark sweeps.

---

## Step 3 & 4 — Benchmark before / after (conc 4 → 256)

For each variant, sweep concurrencies with num_prompts = conc×10:
```bash
for c in 4 8 16 32 64 128 256; do
  python3 ~/bench_serving/benchmark_serving.py --model="$MODEL" --backend=sglang --port=8000 \
    --dataset-name=random --random-input-len="$ISL" --random-output-len="$OSL" --random-range-ratio=0.8 \
    --num-prompts=$((c*10)) --max-concurrency=$c --request-rate=inf --ignore-eos \
    --percentile-metrics=ttft,tpot,itl,e2el --save-result --result-dir=<vdir> \
    --result-filename=bench_${TAG}_il${ISL}_ol${OSL}_conc${c}.json
done
```
(`~/agent-box/skills/validate-pr/` may keep a `sweep.sh` helper; or reuse `/perf-sweep`.)

> **Fairness — configs MUST match between before/after** (this is the #1 mistake):
> compare like-for-like. A `decode=full` (CUDA graph) run vs a `decode=disabled` (eager) run differs by
> **~7× on TPOT** and will completely mask/invert the real kernel effect. Confirm both servers logged the
> same `decode=PhaseConfig(backend=...)`. If the PR path can't use CUDA graph, run **both** sides eager for
> the apples-to-apples number *and* note the production (cuda-graph) reference separately.
> **Empty results** (`completed=0`) usually mean the server crashed on the first forward, or a tuned-CSV
> **duplicate-shape merge error** ("Found N duplicate shape entries... Please re-run") — check the server log.

---

## Step 5 — Performance table

Render before and after side by side, per shape, in this exact column format (one block per ISL/OSL;
`total tok/s/gpu` = total tok/s ÷ TP size):

```
ISL    OSL    concurrency   Median E2E   total tok/s   total tok/s/gpu   Median TTFT (ms)   Median TPOT (ms)
1024   1024   4
1024   1024   8
1024   1024   16
1024   1024   32
1024   1024   64
1024   1024   128
1024   1024   256
```
Pull fields from each result JSON: `median_e2el_ms`, `total_token_throughput`, `median_ttft_ms`, `median_tpot_ms`
(`completed` must equal num_prompts). Emit both **before** and **after** tables (and a Δ% column if useful).
**Claim the headline number on canonical-8k**, and state the caveats (eager vs cuda-graph, tuned vs borrowed configs).

---

## Deliverable summary

1. **Kernel** — target kernel Δµs (or old→new kernel), from conc4 before/after traces.
2. **Accuracy** — gsm8k after (vs threshold; vs before if run).
3. **Before/after perf tables** — conc4→256 per shape, in the Step-5 format, headline on canonical-8k.
4. **Caveats** — config parity (cuda-graph vs eager), tuned vs borrowed kernel configs, any shippability blocker.
