#!/usr/bin/env python3
"""Per physical GPU (rocm-smi Device / amd-smi gpu id): VRAM%, GPU%, top VRAM PID -> docker name."""
import json
import re
import subprocess
from typing import Dict, List, Tuple


def run(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)


def pid_container(pid: str) -> str:
    try:
        raw = open(f"/proc/{pid}/cgroup", "rb").read().decode(errors="replace")
    except OSError:
        return "(no access)"
    m = re.search(r"docker-([a-f0-9]{64})", raw)
    if not m:
        return "(host/other)"
    cid = m.group(1)
    try:
        out = subprocess.check_output(
            ["docker", "ps", "--no-trunc", "--filter", f"id={cid}", "--format", "{{.Names}}"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or cid[:12] + "…"
    except OSError:
        return cid[:12] + "…"


def concise_vram_gpu_pct() -> Dict[int, Tuple[str, str]]:
    out = {}
    for line in run("rocm-smi").splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        dev = int(parts[0])
        perc = re.findall(r"(\d+)%", line)
        if len(perc) >= 2:
            out[dev] = (perc[-2], perc[-1])
    return out


def fmt_mem(b: int) -> str:
    if b >= 1 << 30:
        return f"{b / (1 << 30):.1f} GiB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.0f} MiB"
    return f"{b} B"


def main() -> None:
    concise = concise_vram_gpu_pct()
    raw = subprocess.check_output(["amd-smi", "process", "--json", "-G"], text=True)
    data = json.loads(raw)

    print("GPU\tVRAM%\tGPU%\tamd-smi VRAM (max PID)\tcontainer")
    for block in data:
        g = int(block["gpu"])
        plist = block.get("process_list") or []
        best = (0, None)  # bytes, pid
        for item in plist:
            pi = item.get("process_info") or {}
            pid = pi.get("pid")
            mu = pi.get("mem_usage") or {}
            b = int(mu.get("value") or 0)
            if pid is not None and b > best[0]:
                best = (b, int(pid))
        mem_b, pid = best
        vp, gp = concise.get(g, ("?", "?"))
        if mem_b == 0:
            extra = "(no VRAM bytes on this GPU in amd-smi)"
            cont = "—"
        else:
            extra = f"PID {pid} {fmt_mem(mem_b)}"
            cont = pid_container(str(pid))
        print(f"{g}\t{vp}%\t{gp}%\t{extra}\t{cont}")


if __name__ == "__main__":
    main()
