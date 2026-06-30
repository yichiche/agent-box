---
type: gotcha
---

# Container bench_serving CLI drift

On rocm/sgl-dev images, `benchmark_serving.py` is a **newer** variant than scripts written for older sglang.

| Old (broken on new images) | New (correct) |
|---|---|
| `--save-result` | `--output-file` |
| `--result-dir` | (use `--output-file` path) |
| default respect eos | `--disable-ignore-eos` to match perf scripts |
| `--percentile-metrics` | not available |

`run_qwen3.5_mxfp4_perf_agent.sh` client step may crash for this reason.

**Fix:** use `perf_sweep.sh` (probes `--help` and adapts) or `python -m sglang.bench_serving` directly.

JSON keys: `total_throughput`, `median_e2e_latency_ms` (not `total_token_throughput` / `median_e2el_ms`).
