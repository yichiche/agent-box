#!/usr/bin/env python3
"""Convert local benchmark output into an InferenceX Curve import CSV.

Target site: https://duyi-wang.github.io/InferenceXCurve/
Contracts:   reference/import-csv.md (InferenceX Curve editor CSV)
             reference/import-plot-tool-csv.md (Plot Tool generic Pareto CSV)

The two contracts are NOT interchangeable. Default target is `inferencex`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import glob as _glob
import io
import json
import math
import os
import re
import sys

# ── Canonical headers ────────────────────────────────────────────────────────
# Exact order from docs/import-csv.md. Do not reorder; the app matches by name
# but Download CSV round-trips in this order and diffs stay readable.
IX_HEADER = [
    "Line ID", "Line Name", "Title", "Line Note", "Model", "Scenario",
    "Precision", "MTP", "HW Key", "Color Mode", "Resolved Color", "Line Type",
    "Line Marker", "Layer", "Included in Chart", "Active Line", "Point Index",
    "Roofline Point", "Point Marker",
    "Interactivity (tok/s/user)", "Throughput/GPU (tok/s/gpu)", "TTFT (s)",
    "End-to-end (s)",
    "P50 Interactivity (tok/s/user)", "P75 Interactivity (tok/s/user)",
    "P90 Interactivity (tok/s/user)", "P95 Interactivity (tok/s/user)",
    "P50 TTFT (s)", "P75 TTFT (s)", "P90 TTFT (s)", "P95 TTFT (s)",
    "P50 End-to-end (s)", "P75 End-to-end (s)", "P90 End-to-end (s)",
    "P95 End-to-end (s)",
    "P75 E2E Normalized Interactivity (tok/s/user)",
    "P90 E2E Normalized Interactivity (tok/s/user)",
    "Prefill GPUs", "Decode GPUs", "Total GPUs", "Prefill TP", "Prefill EP",
    "Prefill DCP", "Prefill DPA", "Prefill Workers", "Decode TP", "Decode EP",
    "Decode DCP", "Decode DPA", "Decode Workers", "DPA", "Disagg",
    "Multi-node", "KV Offload", "Chip Cache Hit Rate",
    "External Cache Hit Rate", "CPU Cache Hit Rate",
    "Theoretical Cache Hit Rate", "Concurrency", "Strategy", "Note",
]

PT_HEADER = ["Line ID", "Line Name", "X", "Y", "Color", "Line Type",
             "Line Marker", "Layer"]

# Ignored on import (derived) — we emit them empty rather than guessing.
IX_DERIVED = {"HW Key", "Included in Chart", "Active Line", "Point Index",
              "Roofline Point", "Total GPUs"}

# Fields that must be identical across every row sharing a Line ID.
IX_LINE_FIELDS = ["Line Name", "Title", "Line Note", "Model", "Scenario",
                  "Precision", "MTP", "Color Mode", "Resolved Color",
                  "Line Type", "Line Marker", "Layer"]
PT_LINE_FIELDS = ["Line Name", "Color", "Line Type", "Line Marker", "Layer"]

# At least one of these must be numeric on every point row.
IX_X_METRICS = ["Interactivity (tok/s/user)", "TTFT (s)", "End-to-end (s)",
                "P75 E2E Normalized Interactivity (tok/s/user)",
                "P90 E2E Normalized Interactivity (tok/s/user)"]

VALID_LINE_TYPES = {"solid", "dashed", "dotted", "dashdot", "long-dash"}
VALID_MARKERS = {"circle", "square", "triangle", "diamond", "star", "plus",
                 "cross"}

SITE = "https://duyi-wang.github.io/InferenceXCurve/"


class SpecError(Exception):
    """A problem with the input the user/agent supplied."""


# ── small helpers ────────────────────────────────────────────────────────────
def num(v):
    """Parse to float, or None if absent/blank/non-numeric/non-finite."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(float(v)) else None
    s = str(v).strip()
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


def fmt(v, sig=6):
    """Format a number compactly without scientific notation for normal ranges."""
    if v is None:
        return ""
    f = float(v)
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return f"{f:.{sig}g}"


def as_bool(v):
    """Return 'true'/'false'/'' using the app's accepted boolean spellings."""
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return "true"
    if s in ("false", "0", "no", "n"):
        return "false"
    return ""


def slug(s, default="inferencex"):
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(s or "")).strip("-").lower()
    return s or default


def _first(d, *keys):
    """First present-and-numeric value among keys."""
    for k in keys:
        v = num(d.get(k))
        if v is not None:
            return v
    return None


# ── spec → rows ──────────────────────────────────────────────────────────────
def line_meta_row(line, layer):
    """Build the line-level half of a row (repeated on every point row)."""
    color = (line.get("color") or "").strip()
    line_type = (line.get("line_type") or "solid").strip()
    marker = (line.get("line_marker") or "precision").strip()

    if line_type and line_type not in VALID_LINE_TYPES:
        raise SpecError(
            f"line {line.get('line_id')!r}: line_type {line_type!r} is not one of "
            f"{sorted(VALID_LINE_TYPES)}")
    if marker and marker not in VALID_MARKERS | {"precision", "default", "auto"}:
        raise SpecError(
            f"line {line.get('line_id')!r}: line_marker {marker!r} is not one of "
            f"{sorted(VALID_MARKERS)} (or precision/default/auto)")

    return {
        "Line ID": str(line.get("line_id", "")).strip(),
        "Line Name": str(line.get("line_name", "")).strip(),
        "Title": str(line.get("title", "") or "").strip(),
        "Line Note": str(line.get("line_note", "") or "").strip(),
        "Model": str(line.get("model", "")).strip(),
        "Scenario": str(line.get("scenario", "")).strip(),
        "Precision": str(line.get("precision", "")).strip(),
        # The app infers empty MTP from ids/names; the contract says generated
        # files should be explicit instead.
        "MTP": str(line.get("mtp", "") or "Non-MTP").strip(),
        "Color Mode": "Custom" if color else "Auto",
        "Resolved Color": color,
        "Line Type": line_type,
        "Line Marker": marker,
        "Layer": fmt(num(line.get("layer")) if line.get("layer") is not None
                     else layer),
    }


def point_metrics(pt, line, y_metric):
    """Convert one benchmark point into InferenceX metric columns.

    Y  = <total|output>_throughput / gpus     (tok/s/gpu)
    X  = 1000 / median_tpot_ms                (tok/s/user), ITL as fallback
    ms → s for TTFT and end-to-end.
    """
    gpus = num(pt.get("gpus")) or num(line.get("gpus"))
    out = {}

    # ── Y: throughput per GPU ────────────────────────────────────────────
    tput_gpu = _first(pt, "throughput_per_gpu", "tput_per_gpu")
    if tput_gpu is None:
        if y_metric == "output":
            total = _first(pt, "output_throughput")
            src = "output_throughput"
        else:
            total = _first(pt, "total_throughput")
            src = "total_throughput"
        if total is None:
            raise SpecError(
                f"line {line.get('line_id')!r} conc={pt.get('concurrency')}: "
                f"missing {src} (and no throughput_per_gpu override)")
        if gpus is None:
            raise SpecError(
                f"line {line.get('line_id')!r}: 'gpus' is required to compute "
                "Throughput/GPU. Set it on the line (or per point). Refusing to "
                "guess — a wrong GPU count rescales the whole Y axis.")
        if gpus <= 0:
            raise SpecError(f"line {line.get('line_id')!r}: gpus must be > 0")
        tput_gpu = total / gpus
    out["Throughput/GPU (tok/s/gpu)"] = tput_gpu

    # ── X: interactivity ─────────────────────────────────────────────────
    intvty = _first(pt, "interactivity", "intvty")
    if intvty is None:
        per_tok_ms = _first(pt, "median_tpot_ms", "median_itl_ms")
        if per_tok_ms is not None and per_tok_ms > 0:
            intvty = 1000.0 / per_tok_ms
    out["Interactivity (tok/s/user)"] = intvty

    # ── latencies: ms → s ────────────────────────────────────────────────
    ttft_s = _first(pt, "ttft_s")
    if ttft_s is None:
        ms = _first(pt, "median_ttft_ms")
        ttft_s = ms / 1000.0 if ms is not None else None
    out["TTFT (s)"] = ttft_s

    e2e_s = _first(pt, "e2e_s")
    if e2e_s is None:
        ms = _first(pt, "median_e2e_latency_ms", "median_e2el_ms")
        e2e_s = ms / 1000.0 if ms is not None else None
    out["End-to-end (s)"] = e2e_s

    if all(out.get(k) is None for k in
           ("Interactivity (tok/s/user)", "TTFT (s)", "End-to-end (s)")):
        raise SpecError(
            f"line {line.get('line_id')!r} conc={pt.get('concurrency')}: no "
            "X-axis metric. Need at least one of median_tpot_ms / median_itl_ms "
            "(interactivity), median_ttft_ms, or median_e2e_latency_ms.")
    return out


def spec_to_ix_rows(spec, y_metric):
    rows = []
    for i, line in enumerate(spec.get("lines", []), start=1):
        for field in ("line_id", "line_name", "model", "scenario", "precision"):
            if not str(line.get(field, "") or "").strip():
                raise SpecError(
                    f"line #{i}: {field!r} is required and must be non-empty "
                    "(the app's fallback for it depends on live app state and is "
                    "not suitable for generated CSV)")
        meta = line_meta_row(line, i)
        pts = line.get("points") or []
        if not pts:
            raise SpecError(f"line {line['line_id']!r}: no points")

        for pt in pts:
            row = {c: "" for c in IX_HEADER}
            row.update(meta)
            for k, v in point_metrics(pt, line, y_metric).items():
                row[k] = fmt(v)

            pm = (pt.get("point_marker") or "").strip()
            if pm and pm not in VALID_MARKERS | {"default", "auto", "precision"}:
                raise SpecError(f"invalid point_marker {pm!r}")
            row["Point Marker"] = pm

            # Deployment metadata: point-level wins, else inherit from line.
            def g(key):
                return pt.get(key, line.get(key))

            row["Prefill GPUs"] = fmt(num(g("prefill_gpus")))
            row["Decode GPUs"] = fmt(num(g("decode_gpus")))
            row["Prefill TP"] = fmt(num(g("prefill_tp")))
            row["Prefill EP"] = fmt(num(g("prefill_ep")))
            row["Prefill DCP"] = fmt(num(g("prefill_dcp")))
            row["Prefill Workers"] = fmt(num(g("prefill_workers")))
            row["Decode TP"] = fmt(num(g("decode_tp")))
            row["Decode EP"] = fmt(num(g("decode_ep")))
            row["Decode DCP"] = fmt(num(g("decode_dcp")))
            row["Decode Workers"] = fmt(num(g("decode_workers")))
            row["Prefill DPA"] = as_bool(g("prefill_dpa"))
            row["Decode DPA"] = as_bool(g("decode_dpa"))
            row["DPA"] = as_bool(g("dpa"))
            row["Disagg"] = as_bool(g("disagg"))
            row["Multi-node"] = as_bool(g("multi_node"))
            row["KV Offload"] = str(g("kv_offload") or "")
            row["Strategy"] = str(g("strategy") or "")

            for col, key in (("Chip Cache Hit Rate", "chip_cache_hit_rate"),
                             ("External Cache Hit Rate", "external_cache_hit_rate"),
                             ("CPU Cache Hit Rate", "cpu_cache_hit_rate"),
                             ("Theoretical Cache Hit Rate",
                              "theoretical_cache_hit_rate")):
                r = num(g(key))
                if r is not None and not (0.0 <= r <= 1.0):
                    raise SpecError(
                        f"{col} must be a raw ratio in 0..1, got {r} "
                        "(the tooltip does the percent formatting)")
                row[col] = fmt(r)

            row["Concurrency"] = fmt(num(pt.get("concurrency")))
            row["Note"] = str(pt.get("note") or "")
            rows.append(row)
    return rows


def spec_to_pt_rows(spec, y_metric):
    """Plot Tool: X = interactivity, Y = throughput/GPU, style only."""
    rows = []
    for i, line in enumerate(spec.get("lines", []), start=1):
        for field in ("line_id", "line_name"):
            if not str(line.get(field, "") or "").strip():
                raise SpecError(f"line #{i}: {field!r} is required")
        color = (line.get("color") or "").strip()
        line_type = (line.get("line_type") or "solid").strip()
        marker = (line.get("line_marker") or "circle").strip()
        if marker in ("precision", "default", "auto"):
            marker = "circle"  # not valid in the Plot Tool contract
        layer = fmt(num(line.get("layer")) if line.get("layer") is not None else i)

        for pt in line.get("points") or []:
            m = point_metrics(pt, line, y_metric)
            x = m.get("Interactivity (tok/s/user)")
            if x is None:
                raise SpecError(
                    f"line {line['line_id']!r} conc={pt.get('concurrency')}: "
                    "Plot Tool requires a numeric X; supply median_tpot_ms / "
                    "median_itl_ms / interactivity")
            rows.append({
                "Line ID": str(line["line_id"]).strip(),
                "Line Name": str(line["line_name"]).strip(),
                "X": fmt(x),
                "Y": fmt(m["Throughput/GPU (tok/s/gpu)"]),
                "Color": color,
                "Line Type": line_type,
                "Line Marker": marker,
                "Layer": layer,
            })
    return rows


# ── Pareto frontier ──────────────────────────────────────────────────────────
def pareto_filter(rows, target):
    """Keep only non-dominated points per Line ID, sorted by X descending.

    A point is dominated when another point of the same line is at least as good
    on BOTH axes (X = interactivity, Y = throughput/GPU) and strictly better on
    one. This is the default because a raw sweep runs past the knee: the
    post-saturation points sit far inside the frontier and drag the curve into a
    tail that reads as a broken chart rather than an optimization result.

    Returns (kept_rows, dropped) where dropped maps Line ID -> [concurrency, ...].
    """
    xcol = "Interactivity (tok/s/user)" if target == "inferencex" else "X"
    ycol = "Throughput/GPU (tok/s/gpu)" if target == "inferencex" else "Y"
    ccol = "Concurrency" if target == "inferencex" else None

    order, by_line = [], {}
    for r in rows:
        lid = (r.get("Line ID") or "").strip()
        if lid not in by_line:
            by_line[lid] = []
            order.append(lid)
        by_line[lid].append(r)

    kept_rows, dropped = [], {}
    for lid in order:
        group = by_line[lid]
        xy = [(num(r.get(xcol)), num(r.get(ycol))) for r in group]
        if any(x is None or y is None for x, y in xy):
            kept_rows.extend(group)          # cannot rank: keep the line intact
            continue
        keep = []
        for i, (x, y) in enumerate(xy):
            if any(j != i and xj >= x and yj >= y
                   and (xj > x or yj > y or (xj == x and yj == y and j < i))
                   for j, (xj, yj) in enumerate(xy)):
                continue
            keep.append(i)
        gone = [group[i].get(ccol, "") if ccol else "" for i in range(len(group))
                if i not in keep]
        if gone:
            dropped[lid] = gone
        keep.sort(key=lambda i: xy[i][0], reverse=True)
        kept_rows.extend(group[i] for i in keep)
    return kept_rows, dropped


# ── validation ───────────────────────────────────────────────────────────────
def validate(header, rows, target):
    """Return (errors, warnings) against the published import contract."""
    errs, warns = [], []
    canonical = IX_HEADER if target == "inferencex" else PT_HEADER
    line_fields = IX_LINE_FIELDS if target == "inferencex" else PT_LINE_FIELDS

    missing = [c for c in canonical if c not in header]
    if missing:
        errs.append(f"missing required header column(s): {missing}")
        return errs, warns
    if header != canonical:
        warns.append("header is not in canonical order (import still works)")
    if not rows:
        errs.append("file has no point rows")
        return errs, warns

    seen = {}  # line id -> (row number, line-field values)
    for n, row in enumerate(rows, start=2):  # row 1 is the header
        lid = (row.get("Line ID") or "").strip()
        if not lid:
            errs.append(f"row {n}: empty 'Line ID'")
            continue
        if not (row.get("Line Name") or "").strip():
            errs.append(f"row {n}: empty 'Line Name'")

        if target == "inferencex":
            for f in ("Model", "Scenario", "Precision"):
                if not (row.get(f) or "").strip():
                    errs.append(f"row {n}: empty {f!r}")

            y = num(row.get("Throughput/GPU (tok/s/gpu)"))
            xs = [num(row.get(c)) for c in IX_X_METRICS]
            has_point_data = y is not None or any(v is not None for v in xs)
            if not has_point_data:
                continue  # all-empty point rows are skipped by the importer
            if y is None:
                errs.append(f"row {n}: non-numeric 'Throughput/GPU (tok/s/gpu)'")
            if not any(v is not None for v in xs):
                errs.append(f"row {n}: no numeric X-axis metric "
                            f"(need one of {IX_X_METRICS})")
            if y is not None and y <= 0:
                warns.append(f"row {n}: Throughput/GPU <= 0 — the app refuses "
                             "Log Scale while such a value is visible")
            for c in IX_X_METRICS:
                v = num(row.get(c))
                if v is not None and v <= 0:
                    warns.append(f"row {n}: {c} <= 0 — blocks Log Scale on X")
            lt = (row.get("Line Type") or "").strip()
            if lt and lt not in VALID_LINE_TYPES and not re.fullmatch(
                    r"[\d\s,]+", lt):
                warns.append(f"row {n}: unrecognized 'Line Type' {lt!r}")
        else:
            for c in ("X", "Y"):
                if num(row.get(c)) is None:
                    errs.append(f"row {n}: {c!r} must be a finite number, got "
                                f"{row.get(c)!r}")
            lay = row.get("Layer")
            if (lay or "").strip() and num(lay) is None:
                errs.append(f"row {n}: non-numeric 'Layer' {lay!r}")

        vals = {f: (row.get(f) or "") for f in line_fields}
        if lid in seen:
            first_n, first_vals = seen[lid]
            for f in line_fields:
                if vals[f] != first_vals[f]:
                    errs.append(
                        f"row {n}: line field {f!r} = {vals[f]!r} differs from "
                        f"row {first_n} ({first_vals[f]!r}) for Line ID {lid!r} "
                        "— the importer rejects the whole file on this")
        else:
            seen[lid] = (n, vals)
    return errs, warns


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        r = csv.DictReader(fh)
        return (r.fieldnames or []), [dict(row) for row in r]


# ── loaders for our own benchmark output ─────────────────────────────────────
def load_bench_json(pattern):
    """sglang benchmark_serving.py result JSONs (perf-sweep result_conc*.json)."""
    files = sorted(_glob.glob(pattern))
    if not files:
        raise SpecError(f"no files matched {pattern!r}")
    pts = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (OSError, ValueError) as e:
            print(f"warn: skipping {f}: {e}", file=sys.stderr)
            continue
        if isinstance(d, list):  # some runs emit a list of result objects
            for item in d:
                if isinstance(item, dict):
                    pts.append(_bench_point(item, f))
        elif isinstance(d, dict):
            pts.append(_bench_point(d, f))
    if not pts:
        raise SpecError(f"no usable result objects in {pattern!r}")
    pts.sort(key=lambda p: p.get("concurrency") or 0)
    return pts


def _bench_point(d, src):
    keep = ("output_throughput", "total_throughput", "median_tpot_ms",
            "median_itl_ms", "median_ttft_ms", "median_e2e_latency_ms")
    pt = {k: d[k] for k in keep if k in d}
    conc = d.get("max_concurrency", d.get("concurrency"))
    pt["concurrency"] = conc
    pt["note"] = f"conc{conc}" if conc not in (None, "") else os.path.basename(src)
    return pt


def load_sweep_csv(path):
    """perf-sweep summary.csv (columns from perf_sweep.sh summarize())."""
    _, rows = read_csv(path)
    if not rows:
        raise SpecError(f"{path} has no data rows")
    pts = [_bench_point(r, path) for r in rows]
    pts.sort(key=lambda p: num(p.get("concurrency")) or 0)
    return pts


LINE_FLAGS = ("line_id", "line_name", "model", "scenario", "precision", "mtp",
              "gpus", "title", "line_note", "color", "line_type", "line_marker",
              "layer", "prefill_gpus", "decode_gpus", "prefill_tp", "prefill_ep",
              "decode_tp", "decode_ep", "decode_workers", "prefill_workers",
              "dpa", "disagg", "multi_node", "strategy", "kv_offload", "hw")


def spec_from_flags(args, points):
    line = {k: getattr(args, k) for k in LINE_FLAGS
            if getattr(args, k, None) is not None}
    line["points"] = points
    return {"lines": [line]}


# ── output ───────────────────────────────────────────────────────────────────
def render(header, rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header, extrasaction="ignore",
                       lineterminator="\n")
    w.writeheader()
    for row in rows:
        w.writerow({c: row.get(c, "") for c in header})
    return buf.getvalue()


def default_out(spec, target, source_hint):
    first = (spec.get("lines") or [{}])[0]
    base = slug(f"{first.get('model', 'plot')}-{first.get('hw', '')}")
    date = _dt.date.today().isoformat()
    name = f"{base}-{date}.csv" if target == "inferencex" else \
           f"{base}-plot-tool-{date}.csv"
    if source_hint:
        d = source_hint if os.path.isdir(source_hint) \
            else os.path.dirname(os.path.abspath(source_hint))
        if d and os.path.isdir(d) and os.access(d, os.W_OK):
            return os.path.join(d, name)
    return os.path.join(os.path.expanduser("~/inferencex-plots"), name)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Build an InferenceX Curve import CSV from benchmark output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Import at {SITE}#/inferencex via 'Import File'.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--spec", help="normalized JSON spec ('-' for stdin)")
    src.add_argument("--from-sweep", help="perf-sweep summary.csv")
    src.add_argument("--from-bench-json",
                     help="glob for benchmark_serving result JSONs, e.g. "
                          "'RUNDIR/result_conc*.json'")
    src.add_argument("--check", help="validate an existing CSV and exit")

    p.add_argument("--target", choices=("inferencex", "plot-tool"),
                   default="inferencex")
    p.add_argument("--y-metric", choices=("total", "output"), default="total",
                   help="throughput basis for tok/s/gpu (default: total)")
    p.add_argument("--out", help="output CSV path")
    p.add_argument("--append", action="store_true",
                   help="merge into --out if it already exists")
    p.add_argument("--stdout", action="store_true",
                   help="print the CSV only; write no file")
    p.add_argument("--no-echo", action="store_true",
                   help="suppress the copy/paste CSV dump")
    p.add_argument("--all-points", action="store_true",
                   help="keep every point; default is the Pareto frontier only")
    p.add_argument("--pareto", action="store_true",
                   help="explicit no-op: Pareto filtering is the default")
    for f in LINE_FLAGS:
        p.add_argument(f"--{f.replace('_', '-')}",
                       help=f"line metadata: {f}")
    args = p.parse_args(argv)

    # ── check mode ───────────────────────────────────────────────────────
    if args.check:
        header, rows = read_csv(args.check)
        target = args.target
        if "X" in (header or []) and "Y" in (header or []) \
                and "Throughput/GPU (tok/s/gpu)" not in (header or []):
            target = "plot-tool"
        errs, warns = validate(header, rows, target)
        for w in warns:
            print(f"warn: {w}")
        for e in errs:
            print(f"ERROR: {e}", file=sys.stderr)
        n_lines = len({(r.get('Line ID') or '').strip() for r in rows})
        if errs:
            print(f"\nFAIL: {args.check} ({len(errs)} error(s)) [{target}]",
                  file=sys.stderr)
            return 1
        print(f"OK: {args.check} — {len(rows)} point(s), {n_lines} line(s) "
              f"[{target} contract]")
        return 0

    # ── build the spec ───────────────────────────────────────────────────
    source_hint = None
    if args.spec:
        text = sys.stdin.read() if args.spec == "-" else open(args.spec).read()
        spec = json.loads(text)
        if "lines" not in spec:
            raise SpecError("spec JSON must have a top-level 'lines' array")
        if args.spec != "-":
            source_hint = args.spec
    elif args.from_sweep:
        spec = spec_from_flags(args, load_sweep_csv(args.from_sweep))
        source_hint = args.from_sweep
    else:
        spec = spec_from_flags(args, load_bench_json(args.from_bench_json))
        source_hint = args.from_bench_json

    header = IX_HEADER if args.target == "inferencex" else PT_HEADER
    rows = (spec_to_ix_rows(spec, args.y_metric) if args.target == "inferencex"
            else spec_to_pt_rows(spec, args.y_metric))

    # Pareto frontier is the DEFAULT; --all-points opts out.
    if not args.all_points:
        n_before = len(rows)
        rows, dropped = pareto_filter(rows, args.target)
        if dropped:
            for lid, concs in dropped.items():
                shown = ", ".join(str(c) for c in concs if str(c).strip())
                print(f"pareto: {lid}: dropped {len(concs)} dominated point(s)"
                      + (f" (conc {shown})" if shown else ""))
        print(f"pareto: kept {len(rows)}/{n_before} point(s) "
              f"(pass --all-points to keep every point)")

    out_path = args.out or default_out(spec, args.target, source_hint)

    # ── append: prepend the existing rows, then validate the union ───────
    if args.append and not args.stdout and os.path.exists(out_path):
        old_header, old_rows = read_csv(out_path)
        if [c for c in header if c not in (old_header or [])]:
            raise SpecError(
                f"--append: {out_path} does not match the {args.target} contract "
                "(different columns). Use a different --out.")
        new_ids = {(r.get("Line ID") or "").strip() for r in rows}
        kept = [r for r in old_rows
                if (r.get("Line ID") or "").strip() not in new_ids]
        replaced = len(old_rows) - len(kept)
        if replaced:
            print(f"note: replacing {replaced} existing row(s) for Line ID(s) "
                  f"{sorted(new_ids)}")
        rows = kept + rows

    errs, warns = validate(header, rows, args.target)
    for w in warns:
        print(f"warn: {w}")
    if errs:
        for e in errs:
            print(f"ERROR: {e}", file=sys.stderr)
        print("\nRefusing to write — the site would reject this file.",
              file=sys.stderr)
        return 1

    text = render(header, rows)
    n_lines = len({(r.get("Line ID") or "").strip() for r in rows})

    if args.stdout:
        sys.stdout.write(text)
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)

    ws = "#/inferencex" if args.target == "inferencex" else "#/plot-tool"
    button = "Import File" if args.target == "inferencex" else "Import CSV"
    print(f"\nWrote {len(rows)} point(s) / {n_lines} line(s) → {out_path}")
    print(f"Import: open {SITE}{ws} → {button}")
    if not args.no_echo:
        # The benchmark runs in a container; the browser is usually elsewhere.
        # Echo the file so copy/paste works when the path isn't reachable.
        print("\n--- CSV (copy/paste fallback) ---")
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SpecError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    except BrokenPipeError:
        sys.exit(0)
