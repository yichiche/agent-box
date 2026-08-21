---
name: debug-kernel-crash
description: Debug GPU kernel crashes in SGLang (CUDA and ROCm) using @debug_kernel_api logging — illegal memory access, device assert, NaN/Inf, wrong shapes. Use when a server or kernel test crashes before normal debug output flushes.
category: kernel-opt
---

# Debug Kernel Crashes with API Logging

Adapted from upstream `debug-cuda-crash` skill. Works on both CUDA and ROCm (ROCm reports device as `cuda:N` in PyTorch).

## Goal

When code crashes with illegal memory access, device-side assert, OOB, or NaN/Inf:

- Capture input tensors **before** the crash
- Track shapes, dtypes, values at the kernel boundary that triggered it
- Detect NaN/Inf or obviously wrong shapes

## Why kernel API logging?

CUDA/HIP errors often abort before stdout flushes. `@debug_kernel_api` logs inputs **before** execution.

## What is covered

- Custom ops via `register_custom_op(...)` / `register_custom_op_from_extern(...)`
- LLM attention, linear, quantization, multi-platform wrappers
- Selected `torch.ops.sglang.*` hotspots

Does **not** cover every pure PyTorch call — only decorated kernel boundaries.

## Step 1: Enable logging

### Level 1 — function names only

```bash
export SGLANG_KERNEL_API_LOGLEVEL=1
export SGLANG_KERNEL_API_LOGDEST=stdout

# launch server or run test
bash run_qwen3.5_mxfp4.sh 2>&1 | tee /tmp/kernel_api.log
```

### Level 3 — inputs with metadata (shapes, dtypes)

```bash
export SGLANG_KERNEL_API_LOGLEVEL=3
export SGLANG_KERNEL_API_LOGDEST=/tmp/debug.log
```

### Level 5 — tensor statistics (min/max/mean, nan_count, inf_count)

```bash
export SGLANG_KERNEL_API_LOGLEVEL=5
export SGLANG_KERNEL_API_LOGDEST=/tmp/debug.log
```

Start at level 3. Escalate to 5 only on the last few calls before crash (level 5 is slow).

## Step 2: Reproduce minimally

1. Reduce to smallest repro: single request, conc=1, short prompt
2. Disable cuda graph if crash is graph-specific:
   ```bash
   --disable-cuda-graph --disable-prefill-cuda-graph
   ```
3. On MI355, note which backend is active (`SGLANG_USE_AITER`, `_use_aiter` path)

## Step 3: Read the log

Find the **last successful kernel call** before crash. Check:

| Signal | Likely cause |
|--------|-------------|
| Wrong shape on last call | Dispatch bug, incorrect M/N/K |
| `nan_count > 0` or `inf_count > 0` | Numerical instability upstream |
| Shape jumps between calls | Batch composition change mid-forward |
| Missing expected call | Wrong code path taken (check `_use_aiter` / `_is_hip` gates) |

## Step 4: Bisect the call chain

If the crash is deep in aiter:

```bash
# Run aiter op_test in isolation
cd /sgl-workspace/aiter
python op_tests/test_<op>.py -v
```

If in SGLang Triton:

```python
# Minimal script at the failing M
import torch
# call kernel directly at M=1 and M=8
```

Use `/kl-consistency-test` standalone repro pattern (M=1 vs M=288) for batch-invariance crashes.

## Step 5: ROCm-specific notes

- HIP errors surface as `CUDA error` in PyTorch — same logging env vars work
- `rocgdb` / `HSA_ENABLE_DEBUG=1` for native HIP crashes outside Python
- Check `dmesg` for GPU page fault after hard crash
- Container OOM vs GPU OOM: `rocm-smi` shows VRAM; host shows if process was killed

## Common crash patterns on MI355

| Pattern | Check |
|---------|-------|
| MoE dispatch shape mismatch | `topk_ids` shape vs expert count, EP size |
| FP8 block scale wrong dims | `scale` tensor M/N alignment with CK kernel |
| Attention workspace too small | `max_seq_len`, page size, batch size interaction |
| aiter .so stale after rebuild | Re-run `python setup.py develop` in aiter |

## Related skills

| Skill | Use when |
|-------|----------|
| `/kl-consistency-test` | Crash during KL repro; batch-invariance |
| `/implement-kernel` | Fix identified after bisect |
| `/validate-pr` | Re-validate after fix |

Upstream reference: https://github.com/sgl-project/sglang/tree/main/.claude/skills/debug-cuda-crash
