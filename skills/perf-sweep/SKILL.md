---
name: perf-sweep
description: "Accuracy-gated concurrency-sweep benchmark for an SGLang server, end to end. Launches the server once, runs a GSM8K accuracy gate (must pass a threshold before benchmarking), then sweeps concurrency (e.g. 4,8,…,256) at a fixed IL/OL, with optional per-concurrency profiling, and emits a summary CSV + table. Model-agnostic; can inherit server flags from a reference bash script under /home/yichiche. Use when the user asks to benchmark a model across concurrencies, run a perf sweep, or 'accuracy then throughput/latency sweep'."
category: measure
---

# perf-sweep — accuracy gate → concurrency sweep (→ optional profiling)

Chains the whole flow against ONE server load: **GSM8K accuracy gate → concurrency
sweep → optional profiling → summary**. The heavy lifting is in `perf_sweep.sh`, which
runs *inside the target container*. This skill reports/relays; the user picks the
model, GPUs, and knobs.

## When to use
- "benchmark Qwen3.5 mxfp4 TP2 IL8k/OL1k, conc 4…256, accuracy first"
- any "run a concurrency sweep / perf sweep" where you want an accuracy gate up front.
For a single before/after change benchmark use `/benchmark`; for just a profile use
`/generate-profile`; for code-change validation use `/validate`.

## How to drive it (Claude)

1. **Pick the container** on the user's image (`docker ps`), and the **free GPUs**.
   Get free CUDA indices from the **gpu-status** skill (cached map) —
   `python3 ~/.claude/skills/gpu-status/gpu_status.py`. Use those indices as `GPUS`
   (they are passed as `HIP_VISIBLE_DEVICES`, which on ROCm uses the **same enumeration
   torch uses**, i.e. the CUDA index — NOT the rocm-smi index). Need `TP` free GPUs.
2. **Find a reference script** under `/home/yichiche` for server flags / model path if
   the user didn't give them: `ls /home/yichiche/run_*{perf,mxfp4,client,agent}*.sh`.
   Parse its `python3 -m sglang.launch_server` flags into `SERVER_ARGS`, and its model
   path into `MODEL`. (For Qwen3.5 MXFP4 the reference is
   `run_qwen3.5_mxfp4_perf_agent.sh`.)
3. **Run in background**, logging to a mounted path so you can follow it:
   ```bash
   docker exec -e MODEL=/data/amd/<model> -e GPUS=6,7 -e TP=2 \
     -e INPUT_LEN=8192 -e OUTPUT_LEN=1024 \
     -e CONCURRENCIES="4 8 16 32 64 128 256" -e NUM_PROMPTS_MULT=10 \
     -e ACCURACY=1 -e ACC_THRESHOLD=0.92 \
     -e PROFILE_CONCS="4" \
     -e RESULT_DIR=/home/yichiche/<run_dir> \
     <container> bash /home/yichiche/.claude/skills/perf-sweep/perf_sweep.sh \
     > /home/yichiche/<run_dir>/driver.log 2>&1
   ```
   (The skill dir is under the mounted home, so the container sees the script.)
4. **Monitor** `server.log` (load), `gsm8k_accuracy.log` (gate), and `bench_conc*.log`.
   If the accuracy gate fails, the script aborts WITHOUT running the sweep (by design).
5. **Report** the `summary.csv` table and accuracy. Confirm the server was torn down
   (the script pkills it on exit unless `KEEP_SERVER=1`).

## Key knobs (env vars)

| Var | Default | Meaning |
|---|---|---|
| `MODEL` | — (required) | model path |
| `GPUS` | — (required) | **CUDA/torch** indices, e.g. `6,7` (→ HIP_VISIBLE_DEVICES). Never empty. |
| `TP` | 2 | tensor parallel |
| `INPUT_LEN`/`OUTPUT_LEN` | 8192/1024 | random dataset IL/OL |
| `RANGE_RATIO` | 0.8 | random range ratio |
| `CONCURRENCIES` | `4 8 16 32 64 128 256` | sweep points |
| `NUM_PROMPTS_MULT` | 10 | benchmark: `num_prompts = conc * 10` (measurement) |
| `PROFILE_NUM_PROMPTS_MULT` | 2 | profiling capture: `num_prompts = conc * 2` (trace only) |
| `NUM_PROMPTS_CAP` | 0 | cap num_prompts (0 = uncapped) |
| `ACCURACY` | 1 | run GSM8K gate first |
| `ACC_THRESHOLD` | 0.92 | min accuracy to proceed |
| `ACC_NUM_Q`/`ACC_PARALLEL`/`ACC_NUM_SHOTS` | 200/2000/5 | GSM8K params |
| `PROFILE_CONCS` | "" | concurrencies to profile, e.g. `"4"` (sets `SGLANG_TORCH_PROFILER_DIR` + `--profile`) |
| `SERVER_ARGS` | Qwen3.5-MXFP4 aiter set | server launch flags (override per model) |
| `SERVER_ENV` | aiter env | extra env for the server process |
| `LAUNCH_SERVER`/`KEEP_SERVER` | 1/0 | reuse an existing server / leave it running |
| `BENCH_SERVING_DIR` | autodetect | dir holding `benchmark_serving.py` |
| `RESULT_DIR` | `/tmp/perf_sweep_<ts>` | outputs (use a mounted path to read from host) |
| `BASELINE_CSV` | "" | a **previous run's `summary.csv`** → auto-verdict (WIN/REGRESSION/INCONCLUSIVE) |
| `SHIP_THRESHOLD_PCT` | 5 | primary metric (`total_throughput`) must gain ≥ this to WIN |
| `REGRESSION_THRESHOLD_PCT` | 2 | any monitored metric worse beyond this = REGRESSION (mirrors `debug/perf-regression/config.py`) |
| `ACC_REGRESSION_THRESHOLD_PP` | 2 | accuracy drop (abs pp) beyond this = REGRESSION |
| `VERDICT_EXIT` | 0 | `1` = exit non-zero (rc 2) on REGRESSION, for CI/pipeline |

## Gate K keep-decision (`keep_decision.py`)

For per-op KEEP decisions (bank a real kernel win even when its Amdahl share makes
e2e move little), use `skills/perf-sweep/keep_decision.py` — it encodes Gate K from
[[../../memory/workflows/gates]]: served-trace per-op time, run ×2 for consistency,
`≥30%` faster + e2e **not regressed** → KEEP into a cumulative ledger. Get the per-op
µs from `/compare-kernels --budget` on the served baseline/after traces.

```bash
python3 skills/perf-sweep/keep_decision.py --target "moe stage-1" \
  --baseline 76.6 77.1 --after 52.0 51.4 \
  --e2e-verdict RESULT_DIR/verdict.json --ledger ~/qwen3.5-mxfp4/keep_ledger.json --e2e-share 0.21
python3 skills/perf-sweep/keep_decision.py --ledger ~/qwen3.5-mxfp4/keep_ledger.json --summary
```

The stacked ship claim (Gate S) is a separate `/perf-sweep` on the all-kept build,
judged against the cumulative goal — the ledger's summed estimate is only a guide.

## Auto-verdict (A/B vs a baseline)

Set `BASELINE_CSV` to a prior run's `summary.csv` and the sweep finishes with a
**verdict computed from the raw csv** (never recalled numbers), written to
`verdict.json` + printed. It mirrors the `debug/perf-regression` logic
([[../../memory/workflows/gates]] Gate 4):

- **Config exact-match gate:** compares this run's `run_meta.env` (MODEL/TP/IL/OL +
  a `SERVER_SIG` hash of server args) against the baseline's. A shape/config mismatch
  is **refused → INCONCLUSIVE**, never reported as a win.
- **REGRESSION** if any monitored metric (`total_throughput` higher-better,
  `median_e2e_latency_ms` lower-better) worsens beyond `REGRESSION_THRESHOLD_PCT`,
  OR accuracy drops beyond `ACC_REGRESSION_THRESHOLD_PP` — even if throughput improved
  (accuracy-correct-but-net-loss still counts as a regression).
- **WIN** if `total_throughput` improves ≥ `SHIP_THRESHOLD_PCT` at ≥ half the shared
  concurrencies and nothing regressed.
- **INCONCLUSIVE** otherwise (within the noise band, or no/again mismatched baseline).

## Gotchas this skill already handles (learned on this box)

- **GPU index permutation & empty VISIBLE vars.** rocm-smi index ≠ CUDA/HIP index.
  Use the gpu-status cached map for free **CUDA** indices and pass them as `GPUS`.
  An **empty** `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` means *no GPUs visible*
  (torch shows `device_count 0`); the script refuses an empty `GPUS`.
- **benchmark_serving.py arg drift.** The newer sglang variant in these images does
  NOT accept `--ignore-eos` (it's the default; only `--disable-ignore-eos` exists),
  `--save-result`, `--percentile-metrics`, `--result-dir`, or `--result-filename`; it
  saves via `--output-file` and prints percentiles by default. The reference
  `run_qwen3.5_mxfp4_perf_agent.sh` client step crashes on these images for that
  reason. `perf_sweep.sh` **probes `--help` and builds a compatible arg list**, so it
  works on both old and new clients. (If you must use the perf-agent script directly,
  set `SKIP_CLIENT=1 KEEP_SERVER=1` and run the client yourself with `--output-file`.)
- **Containers report "No HIP GPUs" until env is right.** A freshly-started container
  can still drive GPUs; if torch says `device_count 0`, check you didn't export an
  empty VISIBLE var, and that `/dev/kfd` + `/dev/dri` are mapped.
- **One model load.** Server is launched once and kept alive across the accuracy gate
  and every concurrency point; only torn down at the end.

## Output
`RESULT_DIR/` contains: `server.log`, `gsm8k_accuracy.log`, `accuracy.txt`,
`result_conc<N>.json` + `bench_conc<N>.log` per point, optional profiler traces in
`PROFILE_DIR`, and `summary.csv` (+ a printed table: conc, out tok/s, total tok/s,
median TTFT/TPOT/ITL/E2E).
