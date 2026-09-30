#!/usr/bin/env python3
"""Render an InferenceMax run directory as the house results table.

  render_table.py --run-dir <dir> --mode fixed|agent [--reference <csv>]

fixed  reads each conc*/<RESULT_FILENAME>.json written by bench_serving
       (millisecond latency fields) and, when a reference CSV matches the shape,
       shows the per-cell delta against it.
agent  reads each conc*/<RESULT_FILENAME>.json written by
       utils/agentic/aggregation/process_agentic_result.py, whose latency fields
       are SECONDS (same rule the inferencex-table skill documents).
"""

import argparse
import csv
import glob
import json
import os
import sys

REFERENCE_DEFAULT = "/home/yichiche/agent-box/memory/models/qwen35-mxfp4-mi355-reference.csv"


# Everything the recipes drop next to the result that is NOT the result.
SIDECAR_PREFIXES = ("agg_", "gpu_metrics", "power_validation", "agentic_power")


def _result_json(d):
    """The one result file in a conc dir: fixed -> bench_serving output (ms
    fields), agent -> process_agentic_result aggregate (second fields)."""
    return [p for p in sorted(glob.glob(os.path.join(d, "*.json")))
            if not os.path.basename(p).startswith(SIDECAR_PREFIXES)]


def load_conc_results(run_dir, mode):
    """Return [(conc, payload, agg_or_None)] sorted by conc."""
    out = []
    for d in sorted(glob.glob(os.path.join(run_dir, "conc*"))):
        try:
            conc = int(os.path.basename(d)[4:])
        except ValueError:
            continue
        cands = _result_json(d)
        if not cands:
            continue
        try:
            with open(cands[0]) as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        agg = None
        for p in glob.glob(os.path.join(d, "agg_*.json")):
            try:
                with open(p) as f:
                    agg = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        out.append((conc, payload, agg))
    return sorted(out, key=lambda r: r[0])


def load_reference(path, isl, osl):
    if not path or not os.path.isfile(path):
        return {}
    ref = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row["ISL"]) == isl and int(row["OSL"]) == osl:
                if row["total_tok_s"]:
                    ref[int(row["concurrency"])] = row
    return ref


def delta(measured, reference):
    if reference in (None, "", 0) or measured is None:
        return ""
    try:
        return f" ({(measured - float(reference)) / float(reference) * 100:+.1f}%)"
    except (TypeError, ValueError):
        return ""


def render_fixed(rows, tp, reference_path):
    if not rows:
        return "no fixed-length results found"
    agg0 = rows[0][2] or {}
    isl = agg0.get("isl") or rows[0][1].get("random_input_len") or 8192
    osl = agg0.get("osl") or rows[0][1].get("random_output_len") or 1024
    ref = load_reference(reference_path, int(isl), int(osl))

    lines = []
    if ref:
        lines.append(f"Reference: {os.path.basename(reference_path)} "
                     f"(delta = (measured - ref)/ref)")
    lines.append("")
    lines.append("| ISL | OSL | conc | completed | Median E2E (ms) | total tok/s | tok/s/gpu | Median TTFT (ms) | Median TPOT (ms) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for conc, r, agg in rows:
        tot = r.get("total_token_throughput")
        # tput_per_gpu is what InferenceX itself publishes; prefer it over tot/tp
        per_gpu = (agg or {}).get("tput_per_gpu")
        if per_gpu is None and tot and tp:
            per_gpu = tot / tp
        r_ref = ref.get(conc, {})
        lines.append(
            f"| {isl} | {osl} | {conc} | {r.get('completed')} "
            f"| {fmt(r.get('median_e2el_ms'))}{delta(r.get('median_e2el_ms'), r_ref.get('median_e2e_ms'))} "
            f"| {fmt(tot)}{delta(tot, r_ref.get('total_tok_s'))} "
            f"| {fmt(per_gpu)}{delta(per_gpu, r_ref.get('total_tok_s_gpu'))} "
            f"| {fmt(r.get('median_ttft_ms'))}{delta(r.get('median_ttft_ms'), r_ref.get('median_ttft_ms'))} "
            f"| {fmt(r.get('median_tpot_ms'))}{delta(r.get('median_tpot_ms'), r_ref.get('median_tpot_ms'))} |"
        )
        if r.get("completed") is not None and r.get("completed") != conc * 10:
            lines.append(f"|   |   |   | **completed != conc*10 -- run is incomplete** | | | | | |")
    return "\n".join(lines)


def render_agent(rows, tp):
    if not rows:
        return "no agentic results found"
    lines = [
        "",
        "| conc | requests ok | TPOT mean (ms) | Interactivity mean (tok/s/user) | E2E p90 (s) | TTFT p50 (s) | total tok/s | tok/s/gpu | cache hit |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for conc, r, _agg in rows:
        m = r.get("request_metrics", {})
        lat = m.get("latency", {})
        tput = m.get("throughput", {})
        # latency fields are seconds -> ms for TPOT, keep seconds for E2E/TTFT
        tpot = lat.get("tpot", {}).get("mean")
        lines.append(
            f"| {conc} | {r.get('num_requests_successful')}/{r.get('num_requests_total')} "
            f"| {fmt(tpot * 1000 if tpot else None)} "
            f"| {fmt(lat.get('intvty', {}).get('mean'))} "
            f"| {fmt(lat.get('e2el', {}).get('p90'))} "
            f"| {fmt(lat.get('ttft', {}).get('p50'))} "
            f"| {fmt(tput.get('total', {}).get('tokens_per_second'))} "
            f"| {fmt(tput.get('per_gpu', {}).get('total_tput_tps'))} "
            f"| {fmt(m.get('cache', {}).get('theoretical_cache_hit_rate'))} |"
        )
        for w in r.get("warnings") or []:
            lines.append(f"|   | **warning:** {w} | | | | | | | |")
    return "\n".join(lines)


def fmt(v):
    if v is None:
        return "-"
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return str(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--mode", required=True, choices=["fixed", "agent"])
    p.add_argument("--tp", type=int, default=None)
    p.add_argument("--reference", default=REFERENCE_DEFAULT)
    args = p.parse_args()

    rows = load_conc_results(args.run_dir, args.mode)
    tp = args.tp
    if tp is None and rows:
        # the aggregate records it; the run-dir name may not carry it
        tp = (rows[0][2] or {}).get("tp") or rows[0][1].get("tp")
    if tp is None:
        for part in os.path.basename(args.run_dir.rstrip("/")).split("_"):
            if part.startswith("tp") and part[2:].isdigit():
                tp = int(part[2:])
    tp = tp or 1

    print(f"# InferenceMax {args.mode} results — {os.path.basename(args.run_dir.rstrip('/'))}  (TP{tp})")
    if args.mode == "fixed":
        print(render_fixed(rows, tp, args.reference))
    else:
        print(render_agent(rows, tp))
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
