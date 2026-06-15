---
name: gpu-status
description: "Show which AMD GPUs are free and the exact CUDA_VISIBLE_DEVICES to use for SGLang. Joins rocm-smi to torch's CUDA index via PCI bus (the numberings differ!). Use when the user says '/gpu-status', '/gpu', asks which GPUs are free, or which CUDA_VISIBLE_DEVICES / device to launch on."
---

# GPU Status (which AMD GPUs are free + the CUDA index to use)

Report each AMD GPU's free/occupied state AND the **CUDA index** to actually pass to
SGLang. The user decides which GPUs to use; this skill only reports — it does **not**
launch anything or set any environment variable.

## Two facts this skill exists to handle (do not forget)

1. **SGLang on ROCm uses `CUDA_VISIBLE_DEVICES`, never `HIP_VISIBLE_DEVICES`.**
   It selects GPUs via `CUDA_VISIBLE_DEVICES` + `--base-gpu-id`. Setting
   `HIP_VISIBLE_DEVICES` does nothing useful and can double-remap you onto a busy
   card. Never tell the user to set `HIP_VISIBLE_DEVICES`, and never set both.

2. **rocm-smi's GPU index ≠ CUDA/torch index.** They are permuted on this hardware
   (e.g. rocm-smi GPU0 = CUDA index 1; CUDA index 0 = rocm-smi GPU3). The only
   reliable bridge is the **PCI bus id**. Never assume `rocm-smi GPU N == CUDA N`.

The script handles both: it joins `rocm-smi --json` (VRAM/bus per physical GPU) to
`torch.cuda` (CUDA index → PCI bus, probed with a clean env) on the PCI bus, so the
`CUDA` column it prints is copy-paste ready as a `CUDA_VISIBLE_DEVICES` value.

## How to run

```bash
python3 ~/.claude/skills/gpu-status/gpu_status.py
```

Run it from a dir where `import torch` works (e.g. `/sgl-workspace/sglang/python`).

Options:
- `--threshold G` — a GPU counts as FREE when used VRAM < G GiB (default `5`).
- `--json` — emit `{gpus: [...], free_cuda_indices: [...]}` instead of the table.

## Output

```
CUDA rocm-smi  PCI Bus        Status           VRAM used  GPU%
   0        3  0000:75:00.0   🔴OCCUPIED    220.0/288 GiB    0%
   2        2  0000:65:00.0   🟢FREE          0.3/288 GiB    0%
...
Free CUDA indices: [2, 6, 7]
==> CUDA_VISIBLE_DEVICES=2,6,7
    e.g.  CUDA_VISIBLE_DEVICES=2,6 python -m sglang.launch_server --tp 2 ...
```

## What to tell the user

1. Run the script and show the table.
2. Report the **CUDA indices** that are free (the `CUDA` column), not the rocm-smi
   indices. The ready-to-use line is `CUDA_VISIBLE_DEVICES=<free cuda indices>`.
3. Let the user pick how many / which. To launch on N of them, take the first N free
   CUDA indices: `CUDA_VISIBLE_DEVICES=<picked> python -m sglang.launch_server --tp N ...`.
4. Do not choose for them or run any workload.

To verify a chosen launch lands on idle cards, the printed bus should match a FREE
row in the table:
```bash
CUDA_VISIBLE_DEVICES=2,6 python3 -c "import torch; [print(i, torch.cuda.get_device_properties(i).pci_bus_id) for i in range(torch.cuda.device_count())]"
```

## Notes

- Free/occupied is judged by **used VRAM** (`--threshold`), not GPU%: a card can sit
  at 0% util while still holding a model in VRAM (it's still in use).
- If torch can't be probed, the `CUDA` column shows `?` and a warning is printed —
  do NOT fall back to the rocm-smi index, it is not a safe substitute.
- Requires `rocm-smi` (at `/opt/rocm/bin`) and an importable `torch`.
