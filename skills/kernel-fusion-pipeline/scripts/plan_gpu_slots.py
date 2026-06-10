#!/usr/bin/env python3
"""One-shot GPU inventory and slot planning for kernel-fusion-pipeline.

Run once at pipeline start (Step 0). Parses rocm-smi, marks free GPUs,
builds TP-sized slots with ports. Plan is fixed upfront — no runtime locking.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

STATE_DIR = Path.home() / ".kernel-fusion-pipeline"
DEFAULT_STATE = STATE_DIR / "state.json"
# Loaded 397B TP2 uses >>10GB; idle MI355 shows ~300MB
VRAM_FREE_BYTES = 10 * 1024**3


def run_cmd(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr.strip()}"
        )
    return result.stdout


def parse_vram_used() -> dict[int, int]:
    out = run_cmd(["rocm-smi", "--showmeminfo", "vram"])
    used: dict[int, int] = {}
    for line in out.splitlines():
        if "Used Memory" not in line:
            continue
        m_gpu = re.search(r"GPU\[(\d+)\]", line)
        m_bytes = re.search(r"(\d+)\s*$", line.strip())
        if m_gpu and m_bytes:
            used[int(m_gpu.group(1))] = int(m_bytes.group(1))
    if not used:
        raise RuntimeError("rocm-smi returned no GPU VRAM data")
    return used


def parse_gpu_busy() -> dict[int, int]:
    out = run_cmd(["rocm-smi", "--showuse"])
    busy: dict[int, int] = {}
    for line in out.splitlines():
        m = re.search(r"GPU\[(\d+)\].*GPU use \(%\):\s*(\d+)", line)
        if m:
            busy[int(m.group(1))] = int(m.group(2))
    return busy


def sglang_occupied_gpus() -> set[int]:
    occupied: set[int] = set()
    ps = subprocess.run(["ps", "aux"], capture_output=True, text=True, check=False)
    for line in ps.stdout.splitlines():
        if "sglang.launch_server" not in line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        pid = parts[1]
        env_path = Path(f"/proc/{pid}/environ")
        if not env_path.exists():
            continue
        raw = env_path.read_bytes()
        for item in raw.split(b"\0"):
            if item.startswith(b"HIP_VISIBLE_DEVICES="):
                val = item.decode().split("=", 1)[1]
                for g in val.split(","):
                    g = g.strip()
                    if g.isdigit():
                        occupied.add(int(g))
    return occupied


def free_gpus(
    vram_used: dict[int, int],
    gpu_busy: dict[int, int],
    sglang_gpus: set[int],
) -> list[int]:
    free: list[int] = []
    for gpu_id in sorted(vram_used):
        if gpu_id in sglang_gpus:
            continue
        if vram_used[gpu_id] >= VRAM_FREE_BYTES:
            continue
        if gpu_busy.get(gpu_id, 0) > 5:
            continue
        free.append(gpu_id)
    return free


def plan_contiguous_groups(free: list[int], tp: int) -> list[list[int]]:
    """Prefer contiguous TP groups: 0,1 / 2,3 / 4,5 / 6,7."""
    free_set = set(free)
    groups: list[list[int]] = []
    used: set[int] = set()

    if not free:
        return groups

    max_id = max(free)
    start = 0
    while start <= max_id:
        group = list(range(start, start + tp))
        if all(g in free_set and g not in used for g in group):
            groups.append(group)
            used.update(group)
        start += tp

    remaining = [g for g in sorted(free_set) if g not in used]
    while len(remaining) >= tp:
        groups.append(remaining[:tp])
        remaining = remaining[tp:]
    return groups


def build_slots(free: list[int], tp: int, port_base: int) -> tuple[list[dict], list[int]]:
    groups = plan_contiguous_groups(free, tp)
    slots: list[dict] = []
    for i, gpus in enumerate(groups):
        slots.append(
            {
                "slot_id": i,
                "gpus": ",".join(str(g) for g in gpus),
                "port": port_base + i,
                "fusion_slug": None,
                "status": "idle",
            }
        )
    assigned = {g for grp in groups for g in grp}
    leftover = [g for g in free if g not in assigned]
    return slots, leftover


def assign_fusions(slots: list[dict], slugs: list[str]) -> list[dict]:
    if not slots:
        return []
    plan: list[dict] = []
    for idx, slug in enumerate(slugs):
        slot = slots[idx % len(slots)]
        wave = idx // len(slots)
        plan.append(
            {
                "fusion_slug": slug,
                "slot_id": slot["slot_id"],
                "gpus": slot["gpus"],
                "port": slot["port"],
                "wave": wave,
            }
        )
    return plan


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--port-base", type=int, default=8000)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument(
        "--assign",
        nargs="*",
        default=None,
        help="Fusion slugs → slot/wave plan (written to state fusion_plan)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    vram = parse_vram_used()
    busy = parse_gpu_busy()
    sglang = sglang_occupied_gpus()
    free = free_gpus(vram, busy, sglang)
    slots, leftover = build_slots(free, args.tp, args.port_base)

    gpu_plan = {
        "tp": args.tp,
        "port_base": args.port_base,
        "total_gpus": len(vram),
        "free_gpus": free,
        "occupied_gpus": sorted(sglang),
        "leftover_gpus": leftover,
        "max_parallel": len(slots),
        "slots": slots,
    }

    state = load_state(args.state)
    state["gpu_plan"] = gpu_plan
    if args.assign is not None:
        state["fusion_plan"] = assign_fusions(slots, args.assign)

    save_state(args.state, state)

    if args.json:
        print(json.dumps(gpu_plan, indent=2))
        return 0

    print("=== GPU Plan (upfront) ===")
    print(f"  Total GPUs:        {gpu_plan['total_gpus']}")
    print(f"  Free GPUs:         {free or '(none)'}")
    print(f"  Occupied (sglang): {gpu_plan['occupied_gpus'] or '(none)'}")
    print(f"  TP:                {args.tp}")
    print(f"  Max parallel:      {len(slots)} validate job(s) at once")
    if leftover:
        print(f"  Leftover GPUs:     {leftover} (not enough for another TP group)")
    print()
    for slot in slots:
        print(
            f"  Slot {slot['slot_id']}: HIP_VISIBLE_DEVICES={slot['gpus']}  port={slot['port']}"
        )
    if args.assign:
        print()
        print("=== Fusion → slot (wave) plan ===")
        for item in state.get("fusion_plan", []):
            print(
                f"  wave {item['wave']}  {item['fusion_slug']}  →  "
                f"GPUs {item['gpus']}  port {item['port']}"
            )
    print()
    print(f"State: {args.state}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
