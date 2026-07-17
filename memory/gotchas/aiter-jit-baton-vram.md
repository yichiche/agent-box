# aiter JIT baton deadlock + slow VRAM reclaim

Two operational failure modes when running sglang+aiter (`SGLANG_USE_AITER=1
SGLANG_USE_AITER_UNIFIED_ATTN=1 AITER_FLYDSL_FORCE=1`, TP2) back-to-back on the
MI355X box. Both look like "the server is hung/OOM" but are neither.

## 1. JIT baton deadlock

aiter JIT builds under `/sgl-workspace/aiter/aiter/jit/build/` use **baton lock
files**. If a server is `kill -9`'d mid-build, stale locks remain and the next run's
first forward pass **hangs forever** (0% GPU util, `/generate` never returns, log
shows `waiting for baton release`).

There are **two** lock name patterns — an outer `lock_*` at `build/` root AND an
inner file named exactly `lock` inside `<kernel>/build/`. Clear **both** before launch:

```bash
find /sgl-workspace/aiter/aiter/jit/build \( -name "lock" -o -name "lock_*" \) -delete
```

## 2. Slow GPU memory reclaim after kill -9

After `kill -9` of TP workers, the KFD driver takes **~60–90s** to actually free
VRAM (rocm-smi shows hundreds of GB still held by dead PIDs, then drops). Launching
the next server too soon **OOMs**. Poll before relaunching:

```bash
# wait until VRAM is actually free (not just "process killed")
rocm-smi --showmeminfo vram   # relaunch only when total used < ~15GB
```

**How to apply:** for any back-to-back server runs, clear stale locks + wait for GPU
free before each launch (the sweep orchestrator `~/gemmtune_sweep.sh` bakes in
`clear_stale_locks` + `wait_gpu_free` — reuse that pattern). This is the first thing
to check when a server "hangs" or a fresh launch OOMs on a box you just used.

Source: [[../journal/2026-06/-sgl-workspace-aiter__aiter-jit-deadlock-gpu-reclaim]].
Related: [[aiter-version-skew]], [[bench-cwd-shadow]].
