---
name: add-sgl-kernel
description: Step-by-step guide for adding a heavyweight AOT kernel — CUDA path via sgl-kernel (CUTLASS), ROCm/MI355 path via aiter C++/HIP (CK tile, assembly). Includes tests and benchmarks. Use when the kernel depends on CUTLASS or needs wheel-build registration. Complements /add-jit-kernel (lightweight) and /implement-kernel (optimization).
category: kernel-opt
---

# Add a Heavyweight AOT Kernel

Tutorial for **adding a new production kernel** that belongs in the AOT build, not JIT. For lightweight ops, use `/add-jit-kernel`. For optimizing existing dispatch, use `/implement-kernel`.

## Decision tree

| Platform | Dependency | Path |
|----------|-----------|------|
| CUDA (B200/H100) | CUTLASS, large C++ | `$SGLANG_ROOT/python/sglang/kernels/aot/` (sgl-kernel) |
| CUDA | flashinfer-provided CUTLASS | Can stay in `jit_kernel` — see `/add-jit-kernel` |
| ROCm (MI355) | CK tile, HIP, assembly | `/sgl-workspace/aiter/csrc/` + Python wrapper in `aiter/ops/` |

Upstream canonical CUDA AOT tutorial:
https://github.com/sgl-project/sglang/tree/main/.claude/skills/add-sgl-kernel

---

## Step 0: Detect roots

```bash
SGLANG_ROOT=$(python3 -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).resolve().parents[2])")
AITER_ROOT=/sgl-workspace/aiter   # or wherever aiter is installed
echo "SGLang: $SGLANG_ROOT"
echo "aiter:  $AITER_ROOT"
```

---

## CUDA path — `sgl-kernel` (AOT)

### Repository map

| Area | Path |
|------|------|
| Implementation | `$SGLANG_ROOT/python/sglang/kernels/aot/csrc/<category>/` |
| Public declarations | `$SGLANG_ROOT/python/sglang/kernels/aot/include/sgl_kernel_ops.h` |
| Torch extension | `$SGLANG_ROOT/python/sglang/kernels/aot/csrc/common_extension.cc` |
| Build | `$SGLANG_ROOT/python/sglang/kernels/aot/CMakeLists.txt` |
| Python API | `$SGLANG_ROOT/python/sglang/kernels/aot/python/sgl_kernel/` |
| Tests | `$SGLANG_ROOT/python/sglang/kernels/aot/tests/test_*.py` |
| Benchmarks | `$SGLANG_ROOT/python/sglang/kernels/aot/benchmark/bench_*.py` |

### Subdirectory guide

- `csrc/elementwise/` — element-wise ops
- `csrc/gemm/` — GEMM kernels
- `csrc/attention/` — attention kernels
- `csrc/moe/` — MoE kernels

### Every new kernel must ship with

1. **pytest** correctness test
2. **Benchmark script** (triton.testing or equivalent)
3. **Torch op registration** in `common_extension.cc`
4. **CMakeLists.txt** entry in `SOURCES`

### Build & test

```bash
cd "$SGLANG_ROOT/python/sglang/kernels/aot"
pip install -e .
python -m pytest tests/test_<kernel>.py -v
```

CI suites: `base-b-kernel-unit-test-*`, `base-b-kernel-benchmark-test-*`

---

## ROCm / MI355 path — aiter C++/HIP (Tier 3)

On MI355, heavyweight kernels live in **aiter**, not sgl-kernel.

### Repository map

| Area | Path |
|------|------|
| CK GEMM | `$AITER_ROOT/csrc/ck_gemm_a8w8_blockscale/`, `csrc/ck_tile_gemm_moe_2stages/` |
| Other HIP kernels | `$AITER_ROOT/csrc/<category>/` |
| Python wrappers | `$AITER_ROOT/aiter/ops/*.py` |
| Triton (lighter) | `$AITER_ROOT/aiter/ops/triton/` |
| Unit tests | `$AITER_ROOT/op_tests/test_*.py` |
| GEMM configs | `$AITER_ROOT/aiter/configs/` |
| Pre-compiled .so | `$AITER_ROOT/aiter/jit/module_*.so` |

### Workflow checklist

1. Implement kernel in `$AITER_ROOT/csrc/`
2. Add Python wrapper in `aiter/ops/<name>.py`
3. Add unit test in `op_tests/test_<name>.py`
4. Wire into SGLang dispatch (guard with `_use_aiter` — see `/implement-kernel`)
5. Rebuild aiter JIT modules if needed:
   ```bash
   cd "$AITER_ROOT" && python setup.py develop
   ```
6. Validate with `/validate-pr` — kernel time change must show in trace analysis

### SGLang wiring pattern

In the SGLang layer file:

```python
if _use_aiter:
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale as _aiter_gemm
    output = _aiter_gemm(input, weight, ...)
else:
    output = _cuda_fallback(input, weight, ...)
```

Never import aiter at module top level. See Backend-Gated Changes in `/implement-kernel`.

### aiter PR + SGLang PR

Tier 3 changes often need **two PRs**:
1. aiter: kernel + wrapper + op_test
2. sglang: dispatch wiring + `_use_aiter` branch

Use `/pr-ab-benchmark` for before/after on the combined stack.

---

## Test placement (both platforms)

| Platform | Test location | CI suite |
|----------|--------------|----------|
| CUDA sgl-kernel | `test/registered/jit/` or `kernels/aot/tests/` | `base-b-kernel-unit-test-*` |
| aiter | `$AITER_ROOT/op_tests/` | aiter CI + `/validate-pr` |

Do **not** put `register_*_ci(...)` under `python/sglang/` — pre-commit hook rejects it.

---

## Related skills

| Skill | Use when |
|-------|----------|
| `/add-jit-kernel` | Lightweight kernel (no CUTLASS) |
| `/implement-kernel` | Full optimize → validate → commit pipeline |
| `/validate-pr` | Confirm kernel time changed in trace |
| `/kl-consistency-test` | Correctness beyond gsm8k accuracy |
| `/pr-ab-benchmark` | aiter + sglang combined A/B |
