---
name: add-jit-kernel
description: Step-by-step guide for adding a lightweight JIT kernel to SGLang — CUDA path via python/sglang/kernels/jit, ROCm/MI355 path via Triton or aiter. Use when adding a new kernel from scratch, choosing JIT vs AOT, or wiring a lightweight op into SGLang. Complements /implement-kernel (optimization of existing paths).
category: kernel-opt
---

# Add a JIT / Lightweight Kernel to SGLang

Tutorial for **adding a new kernel from scratch**. For optimizing an existing hot path on MI355, use `/implement-kernel` instead.

## Decision tree

| Platform | Kernel weight | Path | Skill |
|----------|--------------|------|-------|
| CUDA (B200/H100) | Lightweight, no CUTLASS | `python/sglang/kernels/jit/` | **This skill** (CUDA section) |
| CUDA | Heavyweight / CUTLASS | `python/sglang/kernels/aot/` (sgl-kernel) | `/add-sgl-kernel` |
| ROCm (MI355) | Triton, no aiter dep | `sglang/jit_kernel/triton/` or layer-local Triton | `/implement-kernel` Tier 2 |
| ROCm (MI355) | aiter Triton wrapper | `aiter/ops/triton/` + SGLang dispatch | `/implement-kernel` Tier 2 |
| ROCm (MI355) | CK / HIP / assembly | `/sgl-workspace/aiter/csrc/` | `/add-sgl-kernel` (ROCm section) |

**Exception:** kernels depending on flashinfer (or CUTLASS via flashinfer) can stay in `jit_kernel` even on CUDA.

Upstream canonical CUDA tutorial (705 lines, file map, conventions, tests):
https://github.com/sgl-project/sglang/tree/main/.claude/skills/add-jit-kernel

---

## Step 0: Detect active SGLang root

```bash
SGLANG_ROOT=$(python3 -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).resolve().parents[2])")
echo "Active SGLang root: $SGLANG_ROOT"
```

Use `$SGLANG_ROOT` for all paths below.

---

## CUDA path — `jit_kernel` module

### When to use

- Element-wise, small fused ops, lightweight custom CUDA
- No CUTLASS / large C++ project dependency
- Benefits from first-use compilation and rapid iteration

### Repository map

| Area | Path |
|------|------|
| CUDA source | `$SGLANG_ROOT/python/sglang/kernels/jit/csrc/` |
| Shared headers | `$SGLANG_ROOT/python/sglang/kernels/jit/include/sgl_kernel/` |
| Python loader | `$SGLANG_ROOT/python/sglang/kernels/jit/` |
| Tests | `$SGLANG_ROOT/test/registered/jit/**/test_*.py` |
| Benchmarks | `$SGLANG_ROOT/test/registered/jit/benchmark/**/bench_*.py` |

### Key conventions (from upstream)

1. **`namespace sglang`** wraps all JIT device + host code.
2. **Validation hierarchy:** `static_assert` > C++ `TensorMatcher` > cached Python > per-call Python.
3. **Use project abstractions** — never raw CUDA when an abstraction exists:
   - `TensorMatcher`, `SymbolicSize`, `SymbolicDevice` (`tensor.h`)
   - `LaunchKernel`, `PDLWaitPrimary`/`PDLTriggerSecondary` (`utils.cuh`)
   - `AlignedVector`, `tile::Memory`, `warp::reduce` (`vec.cuh`, `tile.cuh`, `warp.cuh`)
4. **`const T* __restrict__`** for read-only pointers; target ~64 registers for memory-bound kernels.
5. **ASCII only** in C++/CUDA sources — no Unicode in comments.
6. **Fixed-width integers** (`int32_t`, `int64_t`, `uint32_t`) across FFI boundary.

### Workflow checklist

1. Implement kernel in `kernels/jit/csrc/<category>/`
2. Add Python binding via `load_jit` / module factory with `@cache_once`
3. Register in the appropriate Python dispatch layer
4. Add pytest under `test/registered/jit/`
5. Add benchmark under `test/registered/jit/benchmark/`
6. Run CI suite: `base-b-kernel-unit-test-*` (see `/write-sglang-test` upstream skill)

### IDE support

```bash
python -m sglang.kernels.jit
python -m sglang.kernels.jit --dep cutlass flashinfer  # if needed
```

---

## ROCm / MI355 path — Triton (Tier 2)

On MI355, lightweight new kernels usually go through **Triton**, not `jit_kernel` (CUDA-only).

### Where to put code

| Location | When |
|----------|------|
| `$SGLANG_ROOT/python/sglang/jit_kernel/triton/` | SGLang-owned Triton, works on HIP |
| Layer file (e.g. `layers/attention/dsv4/tilelang_kernel.py`) | Model-specific fused kernel |
| `/sgl-workspace/aiter/aiter/ops/triton/` | Reusable across models, wired via aiter |

### Gating (mandatory)

Follow Backend-Gated Changes from `/implement-kernel`:

```python
from sglang.srt.utils.common import get_bool_env_var, is_hip

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.ops.triton.some_kernel import some_op
```

- **aiter imports** → inside `if _use_aiter:`
- **Non-aiter Triton** → inside `if _is_hip:` (or unconditional if HIP+CUDA compatible)
- Never break the CUDA path in the `else:` branch

### Validation

After implementation, run `/validate` or `/validate-pr` — profile both **decode (M=1–8)** and **extend (M=hundreds)**.

---

## JIT vs AOT quick reference

```
Need CUTLASS or wheel-build AOT?  → /add-sgl-kernel
Lightweight, rapid iteration?
  CUDA                           → jit_kernel (this skill, CUDA section)
  ROCm                           → Triton (this skill, ROCm section) or /implement-kernel Tier 2
Optimizing existing dispatch?    → /implement-kernel Tier 1 first
```

---

## Related skills

| Skill | Use when |
|-------|----------|
| `/implement-kernel` | Optimize existing kernel path end-to-end |
| `/add-sgl-kernel` | Heavyweight AOT (CUDA sgl-kernel or aiter C++) |
| `/kl-consistency-test` | Verify prefill/decode correctness after kernel change |
| `/debug-kernel-crash` | Illegal memory access, NaN, device assert |
| `/validate-pr` | Before/after benchmark + profiling gate |
