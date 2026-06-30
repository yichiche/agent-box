---
type: gotcha
---

# GPU pinning on MI355

- `rocm-smi` GPU index ≠ PyTorch CUDA index. Use `/gpu-status` + `pci_cuda_map.json`.
- **Never** set `HIP_VISIBLE_DEVICES=` or `CUDA_VISIBLE_DEVICES=` to empty string — hides all GPUs.
- `gpu-status` caches PCI map by `boot_id`; stale after reboot → re-run `/gpu-status`.
