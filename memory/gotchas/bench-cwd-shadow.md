---
type: gotcha
severity: high
---

# Bench CWD shadow (stale aiter/sglang)

When benchmarking, canonical packages are editable installs under `/sgl-workspace/aiter` and `/sgl-workspace/sglang`.

**Problem:** `$HOME/aiter/` and `$HOME/sglang/` are **stale shadows**. Python puts cwd at `sys.path[0]`, so launching `run_qwen3.5_mxfp4_*.sh` from `$HOME` imports wrong code silently.

**Symptoms:**
- Server log shows aiter import under `/home/yichiche/aiter/...` (wrong)
- Client crashes: `ModuleNotFoundError: No module named 'sglang.benchmark.datasets'`

**Fix:**
1. `cd /tmp` (or any dir without `aiter/` / `sglang/` subdirs)
2. Launch server + client from there
3. Verify server log: `[aiter] import ... under /sgl-workspace/aiter/...`
4. A/B via env flags (e.g. `UA_BASELINE=1`), not second clone

Related: [[../workflows/benchmark]], [[container-bench-flags]]
