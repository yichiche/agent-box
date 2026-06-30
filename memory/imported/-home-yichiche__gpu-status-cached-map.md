---
name: gpu-status-cached-map
description: gpu-status skill caches the CUDA<->PCI map (keyed by boot_id) so normal runs skip torch
metadata: 
  node_type: memory
  type: project
  originSessionId: 8d336da0-5c21-4099-ae78-5bfcfc312a1d
---

`~/.claude/skills/gpu-status/` caches the CUDA-index↔PCI-bus map in `pci_cuda_map.json`,
keyed by kernel `boot_id`. Normal runs reuse the cache and only refresh live VRAM via
`rocm-smi` (no torch probe) — fast, works even on the host where torch can't see GPUs.
On reboot (boot_id changes) the map must be re-seeded with `gpu_status.py --remap` on a
GPU-healthy container.

Key facts for launching SGLang here (MI35x): rocm-smi index ≠ CUDA/torch index (they're
permuted). `HIP_VISIBLE_DEVICES` uses the SAME enumeration as the CUDA column, so the
free CUDA indices double as launch indices. Never export an empty
`CUDA/HIP/ROCR_VISIBLE_DEVICES` — empty = hide ALL GPUs (`device_count 0`,
"No HIP GPUs"). Related: [[perf-sweep-skill]].
