#!/usr/bin/env python3
"""
Suggest CUDA_VISIBLE_DEVICES (HIP/ROCm) from rocm-smi + optional PyTorch PCI mapping.

Busy = GPU use > FREE_CUDA_MAX_GPU_UTIL (default 5) OR VRAM% > FREE_CUDA_MAX_VRAM_PCT (default 15).

Use the same Python interpreter as your workload (venv / container) so PyTorch can map
logical cuda:0..N-1 to the correct physical GPU rows from rocm-smi.

  python3 .../free_cuda_devices.py --table
    Print HIP / rocm-smi Device mapping table (no PyTorch, one rocm-smi call).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


MAX_GPU_UTIL = _env_int("FREE_CUDA_MAX_GPU_UTIL", 5)
MAX_VRAM_PCT = _env_int("FREE_CUDA_MAX_VRAM_PCT", 15)


@dataclass(frozen=True)
class GpuBus:
    rocm_gpu_index: int
    pci_bus_id: str  # normalized e.g. 0000:75:00.0


def normalize_pci(bus: str) -> str:
    b = bus.strip().lower().replace("0x", "")
    parts = b.split(":")
    if len(parts) == 3:
        return "0000:" + ":".join(parts)
    return b


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        sys.stderr.write(out)
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")
    return out


def parse_rocm_pci_buses(text: str) -> list[GpuBus]:
    """Parse `rocm-smi --showbus` for GPU[n] : PCI Bus: ..."""
    found: list[GpuBus] = []
    for m in re.finditer(
        r"GPU\[(\d+)\]\s*:\s*PCI Bus:\s*([0-9a-fA-F:]+)", text, re.MULTILINE
    ):
        idx = int(m.group(1))
        found.append(GpuBus(rocm_gpu_index=idx, pci_bus_id=normalize_pci(m.group(2))))
    found.sort(key=lambda g: g.rocm_gpu_index)
    return found


def rocm_json_stats() -> tuple[dict[int, int], dict[int, int]]:
    """Returns (gpu_use_pct, vram_pct) keyed by rocm GPU index (cardN <-> GPU[N])."""
    raw = _run(["rocm-smi", "-u", "--showmemuse", "--json"])
    blob = None
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                blob = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if blob is None:
        raise RuntimeError("could not parse rocm-smi -u --json output")

    gpu_use: dict[int, int] = {}
    vram_pct: dict[int, int] = {}
    for key, stats in blob.items():
        m = re.match(r"card(\d+)$", key, re.I)
        if not m:
            continue
        i = int(m.group(1))
        gu = stats.get("GPU use (%)", "0")
        vm = stats.get(
            "GPU Memory Allocated (VRAM%)", stats.get("VRAM use (%)", "0")
        )
        try:
            gpu_use[i] = int(str(gu).replace("%", "").strip())
        except ValueError:
            gpu_use[i] = 0
        try:
            vram_pct[i] = int(str(vm).replace("%", "").strip())
        except ValueError:
            vram_pct[i] = 0
    return gpu_use, vram_pct


def torch_cuda_to_rocm(pci_by_rocm: dict[str, int]) -> list[int] | None:
    """For each torch cuda index i, return matching rocm GPU index, or None if unavailable."""
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    n = torch.cuda.device_count()
    out: list[int] = []
    for cuda_i in range(n):
        p = torch.cuda.get_device_properties(cuda_i)
        raw = str(getattr(p, "pci_bus_id", "") or "").strip()
        if not raw:
            return None
        bus = normalize_pci(raw)
        rocm = pci_by_rocm.get(bus)
        if rocm is None:
            return None
        out.append(rocm)
    return out


def is_idle(rocm_idx: int, gpu_use: dict[int, int], vram: dict[int, int]) -> bool:
    gu = gpu_use.get(rocm_idx, 0)
    vp = vram.get(rocm_idx, 0)
    return gu <= MAX_GPU_UTIL and vp <= MAX_VRAM_PCT


def parse_rocm_concise(text: str) -> list[tuple[int, int, int, int]]:
    """Parse default `rocm-smi` concise table: Device, Node, …, VRAM%%, GPU%%."""
    rows: list[tuple[int, int, int, int]] = []
    for line in text.splitlines():
        s = line.strip()
        if len(s) < 8 or not s[0].isdigit():
            continue
        m = re.match(r"^(\d+)\s+(\d+)\s+0x[0-9a-f]+", s, re.I)
        if not m:
            continue
        dev, node = int(m.group(1)), int(m.group(2))
        parts = s.split()
        pcts = [p for p in parts if p.endswith("%")]
        if len(pcts) < 2:
            continue
        try:
            vram = int(pcts[-2].rstrip("%"))
            gpu = int(pcts[-1].rstrip("%"))
        except ValueError:
            continue
        rows.append((dev, node, vram, gpu))
    rows.sort(key=lambda r: r[0])
    return rows


def label_row(vram: int, gpu: int) -> str:
    if gpu > MAX_GPU_UTIL or vram > MAX_VRAM_PCT:
        return "occupied"
    return "free"


def emit_hip_table(rows: list[tuple[int, int, int, int]]) -> None:
    print(f"# thresholds: FREE_CUDA_MAX_GPU_UTIL={MAX_GPU_UTIL} FREE_CUDA_MAX_VRAM_PCT={MAX_VRAM_PCT}")
    print(
        "# HIP index: on typical MI300/MI355 nodes this matches rocm-smi Concise column "
        '"Device" (same as one number in HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES lists).'
    )
    print("# After HIP_VISIBLE_DEVICES=a,b,c logical cuda:0 is the first listed physical GPU.")
    print()
    print("| HIP (Device) | Node | VRAM% | GPU% | suggested |")
    print("|:---:|:---:|:---:|:---:|:---|")
    for dev, node, vram, gpu in rows:
        print(f"| {dev} | {node} | {vram} | {gpu} | **{label_row(vram, gpu)}** |")
    print()
    for dev, node, vram, gpu in rows:
        tag = label_row(vram, gpu)
        print(
            f"HIP_VISIBLE_DEVICES={dev}  →  rocm Device {dev} (Node {node})  →  **{tag}**"
        )


def cmd_table() -> int:
    text = _run(["rocm-smi"])
    rows = parse_rocm_concise(text)
    if not rows:
        print(
            "ERROR: could not parse rocm-smi concise lines (need English concise output).",
            file=sys.stderr,
        )
        return 2
    emit_hip_table(rows)
    return 0


def main() -> int:
    if "--table" in sys.argv or "-t" in sys.argv:
        return cmd_table()

    alt = os.environ.get("FREE_CUDA_PYTHON", "").strip()
    if alt and os.path.realpath(alt) != os.path.realpath(sys.executable):
        os.execv(alt, [alt, *sys.argv])

    bus_text = _run(["rocm-smi", "--showbus"])
    gpus = parse_rocm_pci_buses(bus_text)
    if not gpus:
        print(
            "ERROR: no GPU[n] PCI Bus lines found in rocm-smi --showbus",
            file=sys.stderr,
        )
        return 2

    pci_to_rocm: dict[str, int] = {g.pci_bus_id: g.rocm_gpu_index for g in gpus}
    gpu_use, vram_pct = rocm_json_stats()

    cuda_to_rocm = torch_cuda_to_rocm(pci_to_rocm)

    all_rocm = sorted({g.rocm_gpu_index for g in gpus})
    idle_rocm = [r for r in all_rocm if is_idle(r, gpu_use, vram_pct)]
    busy_rocm = [r for r in all_rocm if not is_idle(r, gpu_use, vram_pct)]

    print(f"# FREE_CUDA_MAX_GPU_UTIL={MAX_GPU_UTIL} FREE_CUDA_MAX_VRAM_PCT={MAX_VRAM_PCT}")
    print(f"# idle rocm-smi GPU[n] / cardN indices: {','.join(map(str, idle_rocm))}")
    if busy_rocm:
        print(f"# busy rocm-smi GPU[n] indices: {','.join(map(str, busy_rocm))}")

    if cuda_to_rocm is None:
        sys.stderr.write(
            "WARN: PyTorch not available or PCI match failed — cannot compute logical CUDA order.\n"
            "      Re-run with FREE_CUDA_PYTHON pointing at the same interpreter as sglang/torch.\n"
        )
        print("# (no CUDA_VISIBLE_DEVICES line: mapping unknown without PyTorch)")
        return 0

    free_cuda: list[int] = []
    busy_cuda: list[int] = []
    for cuda_i, rocm in enumerate(cuda_to_rocm):
        if is_idle(rocm, gpu_use, vram_pct):
            free_cuda.append(cuda_i)
        else:
            busy_cuda.append(cuda_i)

    free_s = ",".join(str(i) for i in free_cuda)
    print(f"# mapping: torch_pci_per_cuda_index")
    if busy_cuda:
        print(f"# busy logical cuda indices: {','.join(str(i) for i in busy_cuda)}")
    print(f"CUDA_VISIBLE_DEVICES={free_s}")
    print(f"export CUDA_VISIBLE_DEVICES={free_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
