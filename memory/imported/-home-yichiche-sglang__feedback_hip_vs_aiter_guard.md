---
name: hip-vs-aiter-guard
description: Use _is_hip (not _use_aiter) to guard non-aiter HIP kernels like custom Triton kernels
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e4fd557e-c11d-4c56-a4b0-f67447f330be
---

Gate HIP code paths with the correct flag based on whether they depend on the aiter library:
- `_use_aiter`: for kernels imported from `aiter.ops.*` — disabled by `SGLANG_USE_AITER=0`
- `_is_hip`: for non-aiter kernels (custom Triton in `sglang.jit_kernel.triton.*`, `sgl_kernel.*`) — always active on HIP

**Why:** Galin (sogalin) review on PR #27636. The sigmoid_gate_mul Triton kernel doesn't import anything from aiter, so gating it with `_use_aiter` would unnecessarily disable it when `SGLANG_USE_AITER=0`. Related: [[backend-gated-skill]].

**How to apply:** When adding a new kernel dispatch branch, check if the kernel comes from aiter or not. If not from aiter, use `elif _is_hip:`. Updated in implement-kernel SKILL.md.
