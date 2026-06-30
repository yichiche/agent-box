---
name: perf-sweep-skill
description: The perf-sweep skill runs accuracy-gated concurrency-sweep benchmarks for SGLang
metadata: 
  node_type: memory
  type: project
  originSessionId: 8d336da0-5c21-4099-ae78-5bfcfc312a1d
---

`~/.claude/skills/perf-sweep/` (SKILL.md + `perf_sweep.sh`) — built 2026-06-16. Runs
inside a container: GSM8K accuracy gate (default threshold 0.92) → one server load →
concurrency sweep (e.g. 4,8,…,256 at fixed IL/OL) → optional profiling pass → CSV +
table. Model-agnostic via env; can inherit server flags from a `/home/yichiche/run_*.sh`
reference script.

Encodes gotchas on this box (MI35x, rocm/sgl-dev images):
- The container's `benchmark_serving.py` is a NEWER sglang variant: ignore-eos is
  default (only `--disable-ignore-eos`), saving via `--output-file` (NOT
  `--save-result`/`--result-dir`/`--result-filename`), no `--percentile-metrics`. The
  reference `run_qwen3.5_mxfp4_perf_agent.sh` client step crashes on these images for
  that reason. perf_sweep.sh probes `--help` and adapts.
- JSON result keys: `total_throughput`, `median_e2e_latency_ms` (NOT
  `total_token_throughput`/`median_e2el_ms`).
- GPU selection: pass free CUDA indices (from [[gpu-status-cached-map]]) as
  `HIP_VISIBLE_DEVICES`; never an empty VISIBLE var (= hides all GPUs).
- Profiling distorts the profiled conc's throughput, so it's a SEPARATE pass with
  `--profile-output-dir`, not mixed into the perf table.

Verified run: Qwen3.5-397B-A17B-MXFP4 TP2 IL8k/OL1k, GSM8K 0.945, out throughput
356→1971 tok/s for conc 4→256. Related: [[no-edit-running-bash-script]].
