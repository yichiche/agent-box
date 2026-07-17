---
name: gpu-status
description: "Show which AMD GPUs are free and the exact CUDA_VISIBLE_DEVICES to use for SGLang. Joins rocm-smi to torch's CUDA index via PCI bus (the numberings differ!). Use when the user says '/gpu-status', '/gpu', asks which GPUs are free, or which CUDA_VISIBLE_DEVICES / device to launch on."
category: infra
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
   (current map: CUDA 0→rocm 3, 1→0, 2→2, 3→1, 4→7, 5→4, 6→6, 7→5). The only reliable
   bridge is the **PCI bus id**. Never assume `rocm-smi GPU N == CUDA N`. This map is
   cached (see below) and only changes on reboot.

The script handles both: it joins `rocm-smi --json` (VRAM/bus per physical GPU) to the
CUDA index → PCI bus map on the PCI bus, so the `CUDA` column it prints is copy-paste
ready as a `CUDA_VISIBLE_DEVICES` value.

## The CUDA↔PCI map is CACHED — normally no torch probe (this is the fast path)

The CUDA↔PCI mapping only changes when the **host reboots**, and probing torch is slow
and frequently impossible here (the host and freshly-started/empty containers report
`No HIP GPUs`; only a GPU-healthy container can probe). So the map is **cached** in
`pci_cuda_map.json` next to the script, keyed by the kernel **boot_id**:

- **Normal run (boot_id unchanged):** the script reuses the cached map and only runs
  `rocm-smi` to refresh **current GPU usage** (VRAM/util). No torch probe. Fast. The
  footer prints `(CUDA<->PCI map source: cache; ...)`. **Just run it — do not re-derive
  the mapping.**
- **After a reboot (boot_id changed) — must re-confirm:** the script detects this and
  tries a fresh torch probe; if it succeeds it rewrites the cache automatically. If
  torch can't probe where you ran it, it prints a loud `WARNING: reboot detected ...
  CUDA indices may be WRONG` and falls back to the stale map. In that case **re-seed
  the map** (see below) before trusting the CUDA column.

**To re-seed / re-confirm the map after a reboot:** run on a GPU-healthy container
(one that has loaded a model / can `import torch` and see all GPUs):
```bash
docker exec <healthy-container> python3 ~/.claude/skills/gpu-status/gpu_status.py --remap
```
`--remap` forces a torch probe and rewrites `pci_cuda_map.json` with the new boot_id.
(The home dir is mounted into the containers, so the cache it writes is shared back to
the host and other containers.)

## How to run

```bash
python3 ~/.claude/skills/gpu-status/gpu_status.py
```

Normal runs work **anywhere** (host included) thanks to the cache — torch is not
needed unless re-seeding after a reboot.

Options:
- `--threshold G` — a GPU counts as FREE when used VRAM < G GiB (default `5`).
- `--json` — emit `{gpus: [...], free_cuda_indices: [...], map_source: ...}`.
- `--remap` — force a fresh torch probe and rewrite the cache (use after a reboot, on
  a GPU-healthy container).

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
- The CUDA column comes from the **cache** on normal runs (see the caching section).
  `map_source` in the footer / JSON tells you which map was used: `cache` (normal),
  `probe (cache rewritten)` (re-seeded), or `STALE cache` (reboot, re-confirm needed).
- If there's no cache **and** torch can't be probed, the `CUDA` column shows `?` and a
  warning is printed — do NOT fall back to the rocm-smi index, it is not a safe
  substitute; re-seed with `--remap` on a GPU-healthy container.
- `rocm-smi` (at `/opt/rocm/bin`) is always required. `torch` is required **only** for
  `--remap` / first-time seeding, not for normal cached runs.
- **Never export an *empty* `CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES`.**
  An empty value means *no GPUs visible* — torch then reports `device_count 0` and
  servers fail with "No HIP GPUs". To use the free cards, pass the actual indices
  (e.g. `HIP_VISIBLE_DEVICES=6,7`); to see all, leave the var **unset**, don't set `=`.
  On ROCm, `HIP_VISIBLE_DEVICES` indexes the **same enumeration as the CUDA column**
  here, so the free CUDA indices double as the HIP indices for launching.
