---
name: gpu-pinning-cuda-not-hip
description: "On this MI355 box sglang pins GPUs via CUDA_VISIBLE_DEVICES (not HIP), rocm-smi index != CUDA index; use gpu_status.py"
metadata: 
  node_type: memory
  type: reference
  originSessionId: b6e22005-3025-4468-b717-ae6f9573c11b
---

GPU selection on this box (MI355, the container the agent runs in):
- **sglang reads `CUDA_VISIBLE_DEVICES`, NOT `HIP_VISIBLE_DEVICES`.** Setting HIP does nothing or double-remaps → server lands on the WRONG card. This caused a kill incident (a workflow agent killed a user's server after misreading indices) and bogus GPU pinning.
- **rocm-smi GPU index != CUDA/torch index** — they are permuted; only the PCI bus id bridges them. e.g. observed CUDA 4 == rocm-smi 7 == PCI F5.
- **Authoritative tool: `python3 /home/yichiche/agent-box/skills/gpu-status/gpu_status.py`** — joins rocm-smi+CUDA via cached PCI map, prints a "Free CUDA indices: [...]" line that is copy-paste ready for `CUDA_VISIBLE_DEVICES`. Trust ONLY this for free/occupied; do NOT trust rocm-smi index equality or plan_gpu_slots.py.

Applied fixes: `perf_sweep.sh` now launches the server with `CUDA_VISIBLE_DEVICES="$GPUS"` (GPUS = CUDA indices), not HIP. `validate_patches.js` / `implement_e2e.js` GPU args are CUDA indices; validate_patches has a Schedule phase (`gpus:"auto"`) that runs gpu_status.py and picks the first tp free CUDA indices, and every sweep agent re-verifies the assigned CUDA indices are 🟢FREE via gpu_status.py before launching and never kills a server it did not start. See [[implement-e2e-vs-microbench-gate]].
