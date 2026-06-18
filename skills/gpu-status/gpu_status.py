#!/usr/bin/env python3
"""Show which AMD GPUs are free and the CUDA index to actually use.

IMPORTANT mapping facts on this box (learned the hard way):
  * SGLang on ROCm selects GPUs via CUDA_VISIBLE_DEVICES + --base-gpu-id. It does
    NOT read HIP_VISIBLE_DEVICES, so setting HIP_VISIBLE_DEVICES does nothing (or
    double-remaps and lands you on the wrong card).
  * rocm-smi's GPU index != CUDA/torch index. They are permuted. The only reliable
    bridge between them is the PCI bus id.

So this script does NOT trust index equality. It joins two sources on PCI bus:
  1. rocm-smi --json  -> per physical GPU: VRAM used, util, PCI bus.
  2. CUDA index i -> PCI bus, from a CACHED map (see below); falls back to a live
     torch probe when the cache is missing/stale.
torch's default enumeration (no *_VISIBLE_DEVICES set) is exactly what
CUDA_VISIBLE_DEVICES=i indexes into, so the printed CUDA index is copy-paste ready.

CACHING (why this is fast):
  The CUDA<->PCI map is a property of the host's GPU enumeration and only changes on
  REBOOT. Re-probing torch every run is slow and often impossible here (the host and
  freshly-started/empty containers report "No HIP GPUs"; only a GPU-healthy container
  can probe). So the map is cached in pci_cuda_map.json next to this script, keyed by
  the kernel boot_id (/proc/sys/kernel/random/boot_id):
    * boot_id matches cache  -> reuse cached map, skip torch entirely. Only rocm-smi
      runs, to refresh live VRAM/usage. This is the normal, fast path.
    * boot_id changed (reboot) or no cache -> the map MUST be re-confirmed: try a live
      torch probe, and if it succeeds rewrite the cache. If torch can't probe here,
      the script says so explicitly and tells you to re-seed from a GPU-healthy
      container (gpu_status.py --remap there, or copy its map into pci_cuda_map.json).
  Force a re-probe + cache rewrite any time with  --remap.

A GPU is FREE when used VRAM < --threshold GiB (default 5). We judge on VRAM, not
util%: a card can sit at 0% util while still holding a model in VRAM.
"""
import argparse
import json
import os
import re
import subprocess
import sys

GIB = 1024 ** 3
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "pci_cuda_map.json")


def boot_id():
    """Kernel boot id; changes on every reboot. '' if unreadable."""
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            return f.read().strip()
    except OSError:
        return ""


def load_cache():
    """Return (map {bus_key: cuda_index}, cached_boot_id) or ({}, '')."""
    try:
        with open(CACHE_PATH) as f:
            c = json.load(f)
        return {k: int(v) for k, v in c.get("map", {}).items()}, c.get("boot_id", "")
    except (OSError, ValueError):
        return {}, ""


def save_cache(cuda_of, bid):
    """Persist {bus_key: cuda_index} keyed by boot_id. Best-effort."""
    import datetime
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump({
                "_comment": "Cached CUDA<->PCI map; reused until boot_id changes. "
                            "Re-seed with gpu_status.py --remap on a GPU-healthy host.",
                "boot_id": bid,
                "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                "map": {k: cuda_of[k] for k in cuda_of},
            }, f, indent=2)
    except OSError as e:  # noqa: BLE001
        print(f"WARNING: could not write cache {CACHE_PATH}: {e}", file=sys.stderr)


def run(cmd, env=None):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                              env=env).stdout
    except Exception as e:  # noqa: BLE001
        print(f"failed to run {' '.join(cmd)}: {e}", file=sys.stderr)
        return ""


def bus_key(domain, bus, device):
    """Canonical PCI key, ints in -> 'DDDD:BB:dd'."""
    return f"{int(domain):04x}:{int(bus):02x}:{int(device):02x}"


def rocm_gpus():
    """Physical GPUs keyed by PCI bus key. rocm-smi reports bus as hex string."""
    out = run(["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showbus", "--json"])
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        print("could not parse rocm-smi JSON (is rocm-smi on PATH?)", file=sys.stderr)
        sys.exit(1)
    gpus = {}
    for card, v in data.items():
        m = re.match(r"card(\d+)", card)
        if not m:
            continue
        bus_str = v.get("PCI Bus", "")  # e.g. '0000:05:00.0'
        parts = bus_str.split(":")
        if len(parts) >= 3:
            domain = int(parts[0], 16)
            bus = int(parts[1], 16)
            device = int(parts[2].split(".")[0], 16)
            key = bus_key(domain, bus, device)
        else:
            key = bus_str
        gpus[key] = {
            "rocm_idx": int(m.group(1)),
            "bus": bus_str,
            "use": int(v.get("GPU use (%)", "0") or 0),
            "vram_used": int(v.get("VRAM Total Used Memory (B)", "0") or 0),
            "vram_total": int(v.get("VRAM Total Memory (B)", "0") or 0),
        }
    return gpus


def torch_map():
    """CUDA index -> PCI bus key, probed with a clean env so the enumeration is the
    canonical one CUDA_VISIBLE_DEVICES indexes into. Returns {} if torch is absent."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                        "ROCR_VISIBLE_DEVICES")}
    probe = (
        "import json,torch;"
        "print(json.dumps([[i,"
        "getattr(torch.cuda.get_device_properties(i),'pci_domain_id',0),"
        "getattr(torch.cuda.get_device_properties(i),'pci_bus_id',0),"
        "getattr(torch.cuda.get_device_properties(i),'pci_device_id',0)]"
        "for i in range(torch.cuda.device_count())]))"
    )
    out = run([sys.executable, "-c", probe], env=env)
    line = next((l for l in out.splitlines() if l.strip().startswith("[")), "")
    try:
        rows = json.loads(line)
    except json.JSONDecodeError:
        return {}
    # torch reports pci ids as decimal ints.
    return {bus_key(d, b, dev): idx for idx, d, b, dev in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=5.0,
                    help="free if used VRAM < this many GiB (default 5)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument("--remap", action="store_true",
                    help="force a fresh torch probe of the CUDA<->PCI map and rewrite "
                         "the cache (use after a reboot, or on a GPU-healthy container)")
    args = ap.parse_args()

    gpus = rocm_gpus()

    # CUDA<->PCI map: use the cache unless it's stale (reboot) or --remap is given.
    bid = boot_id()
    cached_map, cached_bid = load_cache()
    map_source = None
    if not args.remap and cached_map and cached_bid == bid:
        cuda_of = cached_map           # fast path: no torch probe
        map_source = "cache"
    else:
        cuda_of = torch_map()          # reboot / no cache / forced: must re-confirm
        if cuda_of:
            save_cache(cuda_of, bid)
            map_source = "probe (cache rewritten)"
        elif cached_map:
            # torch can't probe here; fall back to the stale cache but be loud about it.
            cuda_of = cached_map
            map_source = "STALE cache"

    if map_source == "STALE cache":
        print("WARNING: reboot detected (boot_id changed) but torch could not be "
              "probed here to re-confirm the CUDA<->PCI map. Showing the OLD cached "
              "map -- CUDA indices may be WRONG. Re-seed from a GPU-healthy container: "
              "run `gpu_status.py --remap` there (or copy its map into "
              f"{CACHE_PATH}).", file=sys.stderr)
    elif not cuda_of:
        print("WARNING: no cached map and torch could not be probed; CUDA index column "
              "will be blank. The rocm-smi index is NOT a safe substitute. Seed the "
              "cache from a GPU-healthy container with `gpu_status.py --remap`.",
              file=sys.stderr)

    rows = []
    for key, g in gpus.items():
        used = g["vram_used"] / GIB
        total = g["vram_total"] / GIB
        rows.append({
            "cuda_index": cuda_of.get(key),
            "rocm_smi_index": g["rocm_idx"],
            "pci_bus": g["bus"],
            "status": "FREE" if used < args.threshold else "OCCUPIED",
            "vram_used_gib": round(used, 1),
            "vram_total_gib": round(total, 1),
            "gpu_use_pct": g["use"],
        })
    # sort by CUDA index when known (that's the axis the user acts on), else rocm idx
    rows.sort(key=lambda r: (r["cuda_index"] is None,
                             r["cuda_index"] if r["cuda_index"] is not None
                             else r["rocm_smi_index"]))

    free = [r["cuda_index"] for r in rows
            if r["status"] == "FREE" and r["cuda_index"] is not None]

    if args.json:
        print(json.dumps({"gpus": rows, "free_cuda_indices": free,
                          "map_source": map_source}, indent=2))
        return

    print(f"{'CUDA':>4} {'rocm-smi':>8}  {'PCI Bus':<14} {'Status':<9} "
          f"{'VRAM used':>16} {'GPU%':>5}")
    print("-" * 70)
    for r in rows:
        cu = "?" if r["cuda_index"] is None else str(r["cuda_index"])
        vram = f"{r['vram_used_gib']:.1f}/{r['vram_total_gib']:.0f} GiB"
        mark = "🟢" if r["status"] == "FREE" else "🔴"
        print(f"{cu:>4} {r['rocm_smi_index']:>8}  {r['pci_bus']:<14} "
              f"{mark}{r['status']:<8} {vram:>16} {r['gpu_use_pct']:>4}%")
    print("-" * 70)
    print(f"Free CUDA indices: {free or 'none'}")
    if free:
        print(f"==> CUDA_VISIBLE_DEVICES={','.join(map(str, free))}")
        two = ",".join(map(str, free[:2]))
        print(f"    e.g.  CUDA_VISIBLE_DEVICES={two} "
              f"python -m sglang.launch_server --tp 2 ...")
    print("NOTE: SGLang uses CUDA_VISIBLE_DEVICES, not HIP_VISIBLE_DEVICES. "
          "Do not set both.")
    if map_source:
        print(f"(CUDA<->PCI map source: {map_source}; rebuild with --remap)")


if __name__ == "__main__":
    main()
