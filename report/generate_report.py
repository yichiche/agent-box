#!/usr/bin/env python3
"""
Kernel Regression Report Generator

Scans benchmark profile runs, extracts per-layer time breakdowns,
detects significant changes, computes kernel-level diffs, and dumps
results as CSV files for the dashboard to consume.

Usage:
    # Directory scan mode (writes CSV to data/):
    python -m report.generate_report /home/yichiche/benchmark_runs/

    # Filter by config:
    python -m report.generate_report /home/yichiche/benchmark_runs/ --config-filter rocm720

    # Custom threshold:
    python -m report.generate_report /home/yichiche/benchmark_runs/ --threshold 0.05

    # Custom output directory:
    python -m report.generate_report /home/yichiche/benchmark_runs/ -o /tmp/report_data

    # Direct file list:
    python -m report.generate_report a.xlsx b.xlsx c.xlsx
"""

import argparse
import collections
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openpyxl

from report.compare_traces import parse_excel, match_layers, diff_pair, group_and_select


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class RunInfo:
    dir_name: str
    xlsx_path: str
    date_str: str
    label: str
    rocm_version: str
    tp_size: int
    model_short: str

    @property
    def config_key(self) -> Tuple[str, int, str]:
        return (self.rocm_version, self.tp_size, self.model_short)


@dataclass
class GroupTimeSummary:
    layer_type: str
    stage: str
    layer_count: int
    avg_total_us: float
    avg_attention_us: float
    avg_moe_us: float
    avg_linear_us: float
    avg_comm_us: float
    avg_quant_us: float
    kernel_count_mode: int


@dataclass
class ChangeEvent:
    layer_type: str
    stage: str
    run_old: RunInfo
    run_new: RunInfo
    old_time_us: float
    new_time_us: float
    pct_change: float
    significant: bool


@dataclass
class KernelDiffSummary:
    change_event: ChangeEvent
    similarity: float
    n_replaced: int
    n_added: int
    n_removed: int
    diff_entries: List  # non-SAME DiffEntry items


# ── Stage 1: Discover runs ──────────────────────────────────────────────────

_DIR_RE = re.compile(
    r"v(?P<ver>[^-]+)-rocm(?P<rocm>\d+)-(?P<gpu>[^-]+)-(?P<date>\d{8})_TP(?P<tp>\d+)_profile"
)

def discover_runs(base_dir: str, config_filter: Optional[str] = None) -> List[RunInfo]:
    """Scan base_dir for *_profile directories, return sorted list of RunInfo."""
    runs = []

    for entry in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, entry)
        if not os.path.isdir(full):
            continue

        xlsx = os.path.join(full, "trace_analysis", "profile.csv.xlsx")
        if not os.path.isfile(xlsx):
            continue

        m = _DIR_RE.match(entry)
        if not m:
            continue

        rocm = m.group("rocm")
        if len(rocm) == 3:
            rocm_label = f"{rocm[0]}.{rocm[1]}.{rocm[2]}"
        else:
            rocm_label = rocm

        if config_filter and config_filter.lower() not in entry.lower():
            continue

        runs.append(RunInfo(
            dir_name=entry,
            xlsx_path=xlsx,
            date_str=m.group("date"),
            label=m.group("date"),
            rocm_version=rocm_label,
            tp_size=int(m.group("tp")),
            model_short=m.group("gpu"),
        ))

    runs.sort(key=lambda r: (r.config_key, r.date_str))
    return runs


# ── Stage 2: Extract metrics ────────────────────────────────────────────────

_TRACKED_LAYER_TYPES = {"MLA+FC", "MLA+MoE"}
_TRACKED_STAGES = {"prefill", "decode"}

def extract_metrics_from_xlsx(xlsx_path: str) -> List[GroupTimeSummary]:
    """Read profile.csv.xlsx Summary sheet layer table, aggregate by (Type, Stage)."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb["Summary"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    header_idx = None
    for i, row in enumerate(rows):
        if row and row[0] == "Layer" and row[1] == "Type" and row[2] == "Stage":
            header_idx = i
            break

    if header_idx is None:
        return []

    groups: Dict[Tuple[str, str], list] = collections.defaultdict(list)
    for row in rows[header_idx + 1:]:
        if row[0] is None:
            continue
        layer_type = str(row[1]) if row[1] else ""
        stage = str(row[2]) if row[2] else ""
        if not layer_type or not stage:
            continue
        if layer_type not in _TRACKED_LAYER_TYPES or stage not in _TRACKED_STAGES:
            continue

        def _float(v):
            try:
                return float(v) if v is not None else 0.0
            except (ValueError, TypeError):
                return 0.0

        def _int(v):
            try:
                return int(v) if v is not None else 0
            except (ValueError, TypeError):
                return 0

        groups[(layer_type, stage)].append({
            "total": _float(row[3]),
            "attention": _float(row[4]),
            "moe": _float(row[5]),
            "linear": _float(row[6]),
            "comm": _float(row[7]),
            "quant": _float(row[8]),
            "kernels": _int(row[9]),
        })

    summaries = []
    for (lt, st), records in sorted(groups.items()):
        n = len(records)
        if n == 0:
            continue
        kernel_counts = [r["kernels"] for r in records]
        mode_count = collections.Counter(kernel_counts).most_common(1)[0][0] if kernel_counts else 0
        summaries.append(GroupTimeSummary(
            layer_type=lt, stage=st, layer_count=n,
            avg_total_us=sum(r["total"] for r in records) / n,
            avg_attention_us=sum(r["attention"] for r in records) / n,
            avg_moe_us=sum(r["moe"] for r in records) / n,
            avg_linear_us=sum(r["linear"] for r in records) / n,
            avg_comm_us=sum(r["comm"] for r in records) / n,
            avg_quant_us=sum(r["quant"] for r in records) / n,
            kernel_count_mode=mode_count,
        ))
    return summaries


def extract_metrics(runs: List[RunInfo]) -> Dict[str, List[GroupTimeSummary]]:
    """Extract metrics for all runs. Keyed by dir_name."""
    result = {}
    for run in runs:
        try:
            result[run.dir_name] = extract_metrics_from_xlsx(run.xlsx_path)
        except Exception as e:
            print(f"  Warning: Failed to read {run.xlsx_path}: {e}", file=sys.stderr)
            result[run.dir_name] = []
    return result


# ── Stage 3: Detect changes ─────────────────────────────────────────────────

def detect_changes(
    runs: List[RunInfo],
    metrics: Dict[str, List[GroupTimeSummary]],
    threshold: float = 0.10,
) -> List[ChangeEvent]:
    """Compare consecutive runs in same config group, flag significant changes."""
    changes = []
    config_groups: Dict[Tuple, List[RunInfo]] = collections.defaultdict(list)
    for run in runs:
        config_groups[run.config_key].append(run)

    for _key, group_runs in config_groups.items():
        group_runs.sort(key=lambda r: r.date_str)
        for i in range(1, len(group_runs)):
            old_run = group_runs[i - 1]
            new_run = group_runs[i]
            old_metrics = {(g.layer_type, g.stage): g for g in metrics.get(old_run.dir_name, [])}
            new_metrics = {(g.layer_type, g.stage): g for g in metrics.get(new_run.dir_name, [])}

            all_keys = set(old_metrics.keys()) | set(new_metrics.keys())
            for lt_st in sorted(all_keys):
                old_g = old_metrics.get(lt_st)
                new_g = new_metrics.get(lt_st)
                if not old_g or not new_g:
                    continue
                if old_g.avg_total_us == 0:
                    continue
                pct = (new_g.avg_total_us - old_g.avg_total_us) / old_g.avg_total_us
                sig = abs(pct) > threshold
                changes.append(ChangeEvent(
                    layer_type=lt_st[0], stage=lt_st[1],
                    run_old=old_run, run_new=new_run,
                    old_time_us=old_g.avg_total_us, new_time_us=new_g.avg_total_us,
                    pct_change=pct, significant=sig,
                ))
    return changes


# ── Stage 4: Kernel diffs ───────────────────────────────────────────────────

def get_kernel_diffs(significant_changes: List[ChangeEvent]) -> List[KernelDiffSummary]:
    """For significant changes, compute kernel-level diffs."""
    diffs = []
    parsed_cache: Dict[str, dict] = {}

    for change in significant_changes:
        if not change.significant:
            continue
        try:
            if change.run_old.xlsx_path not in parsed_cache:
                parsed_cache[change.run_old.xlsx_path] = parse_excel(change.run_old.xlsx_path)
            if change.run_new.xlsx_path not in parsed_cache:
                parsed_cache[change.run_new.xlsx_path] = parse_excel(change.run_new.xlsx_path)

            layers_a = parsed_cache[change.run_old.xlsx_path]
            layers_b = parsed_cache[change.run_new.xlsx_path]

            matched, only_a, only_b = match_layers(layers_a, layers_b)
            pair_results = [diff_pair(a, b, method) for a, b, method in matched]
            group_results = group_and_select(pair_results, only_a, only_b)

            target_group = None
            for gr in group_results:
                if gr.layer_type == change.layer_type and gr.stage == change.stage:
                    target_group = gr
                    break

            if target_group and target_group.representative:
                rep = target_group.representative
                non_same = [e for e in rep.entries if e.status != "SAME"]
                counts = collections.Counter(e.status for e in non_same)
                diffs.append(KernelDiffSummary(
                    change_event=change, similarity=rep.similarity,
                    n_replaced=counts.get("REPLACED", 0),
                    n_added=counts.get("ADDED", 0),
                    n_removed=counts.get("REMOVED", 0),
                    diff_entries=non_same,
                ))
            else:
                diffs.append(KernelDiffSummary(
                    change_event=change, similarity=0.0,
                    n_replaced=0, n_added=0, n_removed=0, diff_entries=[],
                ))
        except Exception as e:
            print(f"  Warning: Kernel diff failed for {change.layer_type}/{change.stage}: {e}",
                  file=sys.stderr)
    return diffs


# ── Stage 5: Dump CSV ───────────────────────────────────────────────────────

def _config_folder_name(rocm_version: str, tp_size: int) -> str:
    """e.g. '7.0.0' / 8 → 'rocm700_TP8'."""
    rocm_compact = rocm_version.replace(".", "")
    return f"rocm{rocm_compact}_TP{tp_size}"


def dump_csv(
    runs: List[RunInfo],
    metrics: Dict[str, List[GroupTimeSummary]],
    changes: List[ChangeEvent],
    diffs: List[KernelDiffSummary],
    threshold: float,
    output_dir: str,
) -> None:
    """Write metrics, changes, and kernel diffs to CSV files, one subfolder per config group."""
    from datetime import datetime

    # Group everything by (rocm_version, tp_size)
    config_groups: Dict[Tuple[str, int], List[RunInfo]] = collections.defaultdict(list)
    for run in runs:
        config_groups[(run.rocm_version, run.tp_size)].append(run)

    group_dirs = []

    for (rocm, tp), group_runs in sorted(config_groups.items()):
        folder = _config_folder_name(rocm, tp)
        group_dir = os.path.join(output_dir, folder)
        os.makedirs(group_dir, exist_ok=True)
        group_dirs.append(folder)

        dir_names = {r.dir_name for r in group_runs}
        group_changes = [c for c in changes if c.run_old.dir_name in dir_names]
        group_diffs = [d for d in diffs if d.change_event.run_old.dir_name in dir_names]

        print(f"  {folder}/")

        # runs.csv
        with open(os.path.join(group_dir, "runs.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dir_name", "date", "label", "rocm_version", "tp_size", "model"])
            for r in group_runs:
                w.writerow([r.dir_name, r.date_str, r.label, r.rocm_version, r.tp_size, r.model_short])

        # metrics.csv
        with open(os.path.join(group_dir, "metrics.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "dir_name", "date", "rocm_version", "tp_size", "model",
                "layer_type", "stage", "layer_count",
                "avg_total_us", "avg_attention_us", "avg_moe_us",
                "avg_linear_us", "avg_comm_us", "avg_quant_us",
                "kernel_count_mode",
            ])
            for run in group_runs:
                for g in metrics.get(run.dir_name, []):
                    w.writerow([
                        run.dir_name, run.date_str, run.rocm_version, run.tp_size, run.model_short,
                        g.layer_type, g.stage, g.layer_count,
                        f"{g.avg_total_us:.2f}", f"{g.avg_attention_us:.2f}", f"{g.avg_moe_us:.2f}",
                        f"{g.avg_linear_us:.2f}", f"{g.avg_comm_us:.2f}", f"{g.avg_quant_us:.2f}",
                        g.kernel_count_mode,
                    ])

        # changes.csv
        with open(os.path.join(group_dir, "changes.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "layer_type", "stage",
                "old_dir", "old_date", "new_dir", "new_date",
                "rocm_version", "tp_size",
                "old_time_us", "new_time_us", "pct_change", "significant",
            ])
            for c in group_changes:
                w.writerow([
                    c.layer_type, c.stage,
                    c.run_old.dir_name, c.run_old.date_str,
                    c.run_new.dir_name, c.run_new.date_str,
                    c.run_old.rocm_version, c.run_old.tp_size,
                    f"{c.old_time_us:.2f}", f"{c.new_time_us:.2f}",
                    f"{c.pct_change:.6f}", c.significant,
                ])

        # kernel_diffs.json
        diffs_data = []
        for d in group_diffs:
            ce = d.change_event
            entries = []
            for e in d.diff_entries:
                entries.append({
                    "pos": e.pos,
                    "old_short": e.file_a_short,
                    "old_kernel": e.file_a_kernel,
                    "new_short": e.file_b_short,
                    "new_kernel": e.file_b_kernel,
                    "status": e.status,
                })
            diffs_data.append({
                "layer_type": ce.layer_type, "stage": ce.stage,
                "old_dir": ce.run_old.dir_name, "new_dir": ce.run_new.dir_name,
                "old_date": ce.run_old.date_str, "new_date": ce.run_new.date_str,
                "pct_change": round(ce.pct_change, 6),
                "old_time_us": round(ce.old_time_us, 2),
                "new_time_us": round(ce.new_time_us, 2),
                "similarity": round(d.similarity, 4),
                "n_replaced": d.n_replaced, "n_added": d.n_added, "n_removed": d.n_removed,
                "entries": entries,
            })
        with open(os.path.join(group_dir, "kernel_diffs.json"), "w") as f:
            json.dump(diffs_data, f, indent=2)

        # meta.json
        n_sig = sum(1 for c in group_changes if c.significant)
        with open(os.path.join(group_dir, "meta.json"), "w") as f:
            json.dump({
                "config": folder,
                "rocm_version": rocm,
                "tp_size": tp,
                "threshold": threshold,
                "generated_at": datetime.now().isoformat(),
                "n_runs": len(group_runs),
                "n_changes": len(group_changes),
                "n_significant": n_sig,
                "n_kernel_diffs": len(group_diffs),
            }, f, indent=2)

    # Top-level index listing all config groups
    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "threshold": threshold,
            "groups": group_dirs,
        }, f, indent=2)
    print(f"  index.json")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract kernel profile metrics and dump as CSV for the dashboard."
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Base directory to scan, or individual .xlsx files",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output directory for CSV/JSON files (default: <benchmark_runs>/data/kernel_profile)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.10,
        help="Significance threshold for flagging changes (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--config-filter", default=None,
        help="Only include runs matching this substring (e.g. 'rocm720')",
    )
    args = parser.parse_args()

    # Default output: <home>/report/<timestamp>/
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        report_root = args.output
    elif len(args.inputs) == 1 and os.path.isdir(args.inputs[0]):
        report_root = str(Path(args.inputs[0]).parent / "report")
    else:
        _agent_box = Path(__file__).resolve().parent.parent
        report_root = str(_agent_box.parent / "report")
    output_dir = os.path.join(report_root, timestamp)

    # Determine if inputs are a directory or file list
    if len(args.inputs) == 1 and os.path.isdir(args.inputs[0]):
        print(f"Scanning {args.inputs[0]} for profile runs...")
        runs = discover_runs(args.inputs[0], config_filter=args.config_filter)
    else:
        runs = []
        for i, path in enumerate(args.inputs):
            if not os.path.isfile(path):
                print(f"Warning: {path} not found, skipping", file=sys.stderr)
                continue
            basename = os.path.basename(os.path.dirname(os.path.dirname(path)))
            runs.append(RunInfo(
                dir_name=basename or f"file_{i}",
                xlsx_path=path, date_str=f"{i:04d}",
                label=os.path.basename(path),
                rocm_version="direct", tp_size=0, model_short="direct",
            ))

    if not runs:
        print("No runs found. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(runs)} run(s):")
    for r in runs:
        print(f"  {r.label} ({r.rocm_version}, TP{r.tp_size})")

    print("Extracting metrics...")
    metrics = extract_metrics(runs)

    print("Detecting changes...")
    changes = detect_changes(runs, metrics, threshold=args.threshold)
    sig = [c for c in changes if c.significant]
    print(f"  {len(sig)} significant change(s) found")

    if sig:
        print("Computing kernel diffs for significant changes...")
        diffs = get_kernel_diffs(sig)
    else:
        diffs = []

    print(f"Writing CSV/JSON to {output_dir}...")
    dump_csv(runs, metrics, changes, diffs, args.threshold, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
