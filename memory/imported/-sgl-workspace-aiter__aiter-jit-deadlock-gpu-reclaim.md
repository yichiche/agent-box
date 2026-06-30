---
name: aiter-jit-deadlock-gpu-reclaim
description: "Two operational gotchas when running sglang+aiter servers on this MI355X box (JIT baton deadlock, slow GPU memory reclaim after kill -9)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9d66bd43-50c3-434e-8e01-9cc2b2be4c1f
---

Running the sglang launch_server with aiter (`SGLANG_USE_AITER=1 SGLANG_USE_AITER_UNIFIED_ATTN=1 AITER_FLYDSL_FORCE=1`, tp2) on this box has two non-obvious failure modes:

1. **JIT baton deadlock.** aiter JIT builds under `/sgl-workspace/aiter/aiter/jit/build/` use baton lock files. If a server is `kill -9`'d mid-build, stale locks remain and the next run's forward pass hangs forever (0% GPU util, `/generate` never returns, log shows `waiting for baton release`). There are TWO lock name patterns: outer `lock_*` at build/ root AND an inner file named exactly `lock` inside `<kernel>/build/`. Clear BOTH before launch:
   `find /sgl-workspace/aiter/aiter/jit/build \( -name "lock" -o -name "lock_*" \) -delete`

2. **Slow GPU memory reclaim.** After `kill -9` of TP workers, the KFD driver takes ~60–90s to actually free VRAM (rocm-smi shows ~440GB held by dead PIDs, then drops). Launching the next server too soon OOMs. Poll `rocm-smi --showmeminfo vram` until total used < ~15GB before relaunching.

**Why:** Cost me ~1h of debugging a "hung server" that was actually a stale-lock deadlock plus a too-fast variant restart.
**How to apply:** The sweep orchestrator at `/home/yichiche/gemmtune_sweep.sh` already bakes in `clear_stale_locks` + `wait_gpu_free` before each server launch. Reuse that pattern for any back-to-back server runs.

Also: device numbering differs — `HIP_VISIBLE_DEVICES=6,7` maps to rocm-smi physical GPU5+GPU6. The client `sglang.bench_serving` must run from a neutral cwd (e.g. /tmp); `/home/yichiche/sglang/` shadows the editable install. Use `--output-file` (not `--save-result/--result-dir`); `--ignore-eos` is the default.
