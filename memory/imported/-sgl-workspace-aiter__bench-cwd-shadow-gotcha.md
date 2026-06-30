---
name: bench-cwd-shadow-gotcha
description: Running aiter/sglang from /home/yichiche silently imports STALE shadow copies; use neutral cwd
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c7d532b0-939f-4f26-b4a0-d82282226d4d
---

When benchmarking unified_attention E2E, the canonical packages are the editable installs: `aiter` → `/sgl-workspace/aiter/aiter` (pip editable, version matches git HEAD) and `sglang` → `/sgl-workspace/sglang/python/sglang`.

BUT `/home/yichiche/` contains stale shadow dirs `aiter/` (a different fork at commit 3a9ed5d, lacks the seg_cap/head-blocked-reduce optimization) and `sglang/` (lacks `sglang.benchmark.datasets`). Because Python puts cwd at `sys.path[0]`, launching the server or `run_qwen3.5_mxfp4_*` scripts from `/home/yichiche` imports those stale copies — so the A/B benchmark would silently test the WRONG aiter and the bench client crashes with `ModuleNotFoundError: No module named 'sglang.benchmark.datasets'`.

**Why:** the goal's verify scripts say `cd /home/yichiche/aiter`, but that's a stale template path; the real working copy is `/sgl-workspace/aiter`.

**How to apply:** run BOTH the sglang server and `python -m sglang.bench_serving` client from a neutral cwd (e.g. `/tmp`) where no `aiter/` or `sglang/` subdir exists. Verify via server log line `[aiter] import ... under /sgl-workspace/aiter/...`. A/B is via `UA_BASELINE=1` env at server launch (same checkout, no second clone needed). The client's `./bench_serving/benchmark_serving.py` symlink is broken; use `python -m sglang.bench_serving` directly. Related: [[aiter-jit-deadlock-gpu-reclaim]], [[unified-attn-decode-perf]].
