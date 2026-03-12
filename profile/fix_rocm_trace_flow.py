#!/usr/bin/env python3
"""Fix missing CUDA-graph flow START events in ROCm (MI3xx) torch profiler traces.

ROCm's roctracer/kineto backend omits ac2g flow START events for
hipGraphLaunch, breaking the visual link between graph launches (CPU)
and their replayed GPU kernels in Chrome/Perfetto trace viewers.

This script injects the missing flow START events so the trace viewer
can draw CPU-to-GPU arrows for CUDA graph replays, matching NVIDIA/CUPTI
behaviour.

Root cause: pytorch/kineto  libkineto/src/RoctracerActivity_inl.h
    RuntimeActivity<T>::flowStart() does not include HIP_API_ID_hipGraphLaunch
    in its whitelist, so no flow start event is emitted for graph launches.

Usage:
    python fix_rocm_trace_flow.py  input.trace.json.gz  [-o output.trace.json.gz]
    python fix_rocm_trace_flow.py  input.trace.json.gz  --in-place
"""

import argparse
import gzip
import json
import os
import sys
from typing import Any, Dict, List, Tuple


def load_trace(path: str) -> Dict[str, Any]:
    """Load a Chrome trace JSON file (plain or gzipped)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_trace(data: Dict[str, Any], path: str) -> None:
    """Write a Chrome trace JSON file (plain or gzipped)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)


def detect_graph_launches(events: List[Dict]) -> List[Dict]:
    """Find all hipGraphLaunch runtime events."""
    return [
        e for e in events
        if e.get("cat") == "cuda_runtime"
        and e.get("name", "").startswith("hipGraphLaunch")
    ]


def find_existing_flow_start_ids(events: List[Dict]) -> set:
    """Collect IDs of all existing flow START events."""
    return {
        e["id"] for e in events
        if e.get("ph") == "s" and e.get("cat") == "ac2g"
    }


def build_missing_flow_starts(
    graph_launches: List[Dict],
    existing_flow_ids: set,
) -> List[Dict]:
    """Create synthetic flow START events for graph launches missing them."""
    missing = []
    for gl in graph_launches:
        corr = gl.get("args", {}).get("correlation")
        if corr is None:
            continue
        if corr in existing_flow_ids:
            continue
        missing.append({
            "ph": "s",
            "id": corr,
            "pid": gl["pid"],
            "tid": gl["tid"],
            "ts": gl["ts"],
            "cat": "ac2g",
            "name": "ac2g",
        })
    return missing


def fix_trace(data: Dict[str, Any]) -> Tuple[Dict[str, Any], int, int]:
    """Inject missing flow START events for hipGraphLaunch.

    Returns (fixed_data, num_graph_launches, num_injected).
    """
    if isinstance(data, list):
        events = data
    else:
        events = data.get("traceEvents", [])

    graph_launches = detect_graph_launches(events)
    if not graph_launches:
        return data, 0, 0

    existing_ids = find_existing_flow_start_ids(events)
    new_flows = build_missing_flow_starts(graph_launches, existing_ids)

    if new_flows:
        events.extend(new_flows)

    return data, len(graph_launches), len(new_flows)


def main():
    parser = argparse.ArgumentParser(
        description="Fix missing CUDA-graph flow events in ROCm traces."
    )
    parser.add_argument(
        "input",
        help="Input trace file (.json or .json.gz)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: <input>_fixed.<ext>)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.in_place:
        output_path = args.input
    elif args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        if base.endswith(".trace.json"):
            base = base[: -len(".trace.json")]
            ext = ".trace.json" + ext
        output_path = base + "_fixed" + ext

    print(f"Loading {args.input} ...")
    data = load_trace(args.input)

    data, n_launches, n_injected = fix_trace(data)

    if n_launches == 0:
        print("No hipGraphLaunch events found — nothing to fix.")
        return

    already = n_launches - n_injected
    print(f"hipGraphLaunch events:  {n_launches}")
    print(f"Already had flow start: {already}")
    print(f"Injected flow starts:   {n_injected}")

    if n_injected == 0:
        print("Trace already correct — no changes written.")
        return

    print(f"Writing {output_path} ...")
    save_trace(data, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
