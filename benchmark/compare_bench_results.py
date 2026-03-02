#!/usr/bin/env python3
"""Compare two benchmark summary CSV files side-by-side.

Reads two bench_summary.csv files produced by run-local-benchmark-e2e.sh
and prints a comparison table for each concurrency level.

Usage:
  python3 compare_bench_results.py \
      --csv-a results_a/bench_summary.csv \
      --csv-b results_b/bench_summary.csv \
      --label-a "v0.5.7" --label-b "v0.5.8" \
      [--output-csv comparison.csv]
"""

import argparse
import csv
import sys

# Metrics where a higher value is better
HIGHER_IS_BETTER = {
    "request_throughput_req_s",
    "input_throughput_tok_s",
    "output_throughput_tok_s",
    "total_throughput_tok_s",
    "completed_requests",
}

# Metrics where a lower value is better
LOWER_IS_BETTER = {
    "benchmark_duration_s",
    "median_e2e_latency_ms",
    "median_ttft_ms",
    "median_itl_ms",
}

METRIC_ORDER = [
    "completed_requests",
    "benchmark_duration_s",
    "request_throughput_req_s",
    "input_throughput_tok_s",
    "output_throughput_tok_s",
    "total_throughput_tok_s",
    "median_e2e_latency_ms",
    "median_ttft_ms",
    "median_itl_ms",
]


def read_csv(path):
    """Read a bench_summary.csv and return {concurrency: {metric: value}}."""
    data = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conc = row.get("concurrency", "").strip()
            if not conc:
                continue
            metrics = {}
            for key, val in row.items():
                if key == "concurrency":
                    continue
                try:
                    metrics[key] = float(val)
                except (ValueError, TypeError):
                    metrics[key] = None
            data[conc] = metrics
    return data


def verdict_label(metric, pct_change):
    """Return a clear verdict: BETTER / WORSE / same."""
    if pct_change is None:
        return ""
    if abs(pct_change) < 0.01:
        return "  same"
    if metric in HIGHER_IS_BETTER:
        return "  ✅ BETTER" if pct_change > 0 else "  ❌ WORSE"
    if metric in LOWER_IS_BETTER:
        return "  ✅ BETTER" if pct_change < 0 else "  ❌ WORSE"
    return ""


def format_value(val):
    """Format a numeric value for display."""
    if val is None:
        return "N/A"
    if abs(val) >= 1000:
        return f"{val:,.2f}"
    if abs(val) >= 1:
        return f"{val:.4f}"
    return f"{val:.6f}"


def print_comparison(data_a, data_b, label_a, label_b, quiet=False):
    """Build comparison data and optionally print tables. Returns rows."""
    all_concurrencies = sorted(
        set(list(data_a.keys()) + list(data_b.keys())),
        key=lambda x: int(x),
    )

    comparison_rows = []
    total_wins_a = 0
    total_wins_b = 0
    total_ties = 0

    for conc in all_concurrencies:
        metrics_a = data_a.get(conc, {})
        metrics_b = data_b.get(conc, {})
        wins_a = 0
        wins_b = 0
        ties = 0

        print(f"\n{'=' * 100}")
        print(f"  Concurrency: {conc}")
        print(f"{'=' * 100}")

        header = f"{'Metric':<32} {label_a:>14} {label_b:>14} {'Delta':>14} {'% Change':>10} {'Verdict':>12}"
        print(header)
        print("-" * 100)

        for metric in METRIC_ORDER:
            val_a = metrics_a.get(metric)
            val_b = metrics_b.get(metric)

            if val_a is not None and val_b is not None:
                delta = val_b - val_a
                if val_a != 0:
                    pct = (delta / abs(val_a)) * 100
                else:
                    pct = None
            else:
                delta = None
                pct = None

            verdict = verdict_label(metric, pct)
            pct_str = f"{pct:+.2f}%" if pct is not None else "N/A"
            delta_str = f"{delta:+.4f}" if delta is not None else "N/A"

            print(
                f"{metric:<32} {format_value(val_a):>14} {format_value(val_b):>14} "
                f"{delta_str:>14} {pct_str:>10} {verdict}"
            )

            comparison_rows.append({
                "concurrency": conc,
                "metric": metric,
                "value_a": val_a,
                "value_b": val_b,
                "delta": delta,
                "pct_change": pct,
            })

            # Tally wins
            if delta is not None and abs(delta) > 1e-9:
                if metric in HIGHER_IS_BETTER:
                    if delta > 0:
                        wins_b += 1
                    else:
                        wins_a += 1
                elif metric in LOWER_IS_BETTER:
                    if delta < 0:
                        wins_b += 1
                    else:
                        wins_a += 1
            elif delta is not None:
                ties += 1

        total_wins_a += wins_a
        total_wins_b += wins_b
        total_ties += ties

        print()
        print(f"  Concurrency {conc} summary: {label_a} wins {wins_a}, "
              f"{label_b} wins {wins_b}, ties {ties}")

    print(f"\n{'=' * 100}")
    print(f"  OVERALL SUMMARY")
    print(f"{'=' * 100}")
    print(f"  {label_a} wins: {total_wins_a}")
    print(f"  {label_b} wins: {total_wins_b}")
    print(f"  Ties:          {total_ties}")

    if total_wins_a > total_wins_b:
        print(f"\n  >> {label_a} is the overall winner <<")
    elif total_wins_b > total_wins_a:
        print(f"\n  >> {label_b} is the overall winner <<")
    else:
        print(f"\n  >> It's a tie <<")
    print()

    return comparison_rows


SUMMARY_METRICS = [
    ("median_e2e_latency_ms", "Latency (ms)"),
    ("total_throughput_tok_s", "Throughput (tok/s)"),
    ("median_ttft_ms", "TTFT (ms)"),
    ("median_itl_ms", "ITL (ms)"),
]


def summary_pct(metric_key, va, vb):
    """Compute B-vs-A percentage so that >100% always means B is better.

    - LOWER_IS_BETTER (latency): A/B  (B smaller => ratio > 1 => better)
    - HIGHER_IS_BETTER (throughput): B/A  (B larger => ratio > 1 => better)
    """
    if va is None or vb is None:
        return None
    if metric_key in LOWER_IS_BETTER:
        return (va / vb * 100) if vb != 0 else None
    else:
        return (vb / va * 100) if va != 0 else None


def write_summary_csv(data_a, data_b, path, label_a, label_b):
    """Write a compact summary CSV: absolute values + B/A percentages."""
    all_concurrencies = sorted(
        set(list(data_a.keys()) + list(data_b.keys())),
        key=lambda x: int(x),
    )

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)

        # Header row 1: metric group names
        header1 = [""]
        for _, display_name in SUMMARY_METRICS:
            header1.extend([display_name, "", ""])
        w.writerow(header1)

        # Header row 2: A / B / B vs A %
        header2 = ["Concurrency"]
        for _ in SUMMARY_METRICS:
            header2.extend([label_a, label_b, "B vs A %"])
        w.writerow(header2)

        # Data rows: absolute values
        pct_rows = []
        for conc in all_concurrencies:
            ma = data_a.get(conc, {})
            mb = data_b.get(conc, {})
            row = [conc]
            pct_row = [conc]
            for metric_key, _ in SUMMARY_METRICS:
                va = ma.get(metric_key)
                vb = mb.get(metric_key)
                row.append(f"{va:.2f}" if va is not None else "")
                row.append(f"{vb:.2f}" if vb is not None else "")
                pct = summary_pct(metric_key, va, vb)
                if pct is not None:
                    row.append(f"{pct:.1f}%")
                    pct_row.append(pct)
                else:
                    row.append("")
                    pct_row.append(None)
            w.writerow(row)
            pct_rows.append(pct_row)

        # Average row
        avg_row = ["Average"]
        for col_idx in range(len(SUMMARY_METRICS)):
            pct_vals = [r[col_idx + 1] for r in pct_rows if r[col_idx + 1] is not None]
            avg_row.append("")  # no absolute avg for A
            avg_row.append("")  # no absolute avg for B
            if pct_vals:
                avg_pct = sum(pct_vals) / len(pct_vals)
                avg_row.append(f"{avg_pct:.1f}%")
            else:
                avg_row.append("")
        w.writerow(avg_row)


def print_summary_table(data_a, data_b, label_a, label_b):
    """Print the compact summary table to stdout."""
    import io
    buf = io.StringIO()
    writer = csv.writer(buf)

    all_concurrencies = sorted(
        set(list(data_a.keys()) + list(data_b.keys())),
        key=lambda x: int(x),
    )

    # Header row 1
    header1 = [""]
    for _, display_name in SUMMARY_METRICS:
        header1.extend([display_name, "", ""])
    writer.writerow(header1)

    # Header row 2
    header2 = ["Concurrency"]
    for _ in SUMMARY_METRICS:
        header2.extend([label_a, label_b, "B vs A %"])
    writer.writerow(header2)

    # Data rows
    pct_rows = []
    for conc in all_concurrencies:
        ma = data_a.get(conc, {})
        mb = data_b.get(conc, {})
        row = [conc]
        pct_row = [conc]
        for metric_key, _ in SUMMARY_METRICS:
            va = ma.get(metric_key)
            vb = mb.get(metric_key)
            row.append(f"{va:.2f}" if va is not None else "")
            row.append(f"{vb:.2f}" if vb is not None else "")
            pct = summary_pct(metric_key, va, vb)
            if pct is not None:
                row.append(f"{pct:.1f}%")
                pct_row.append(pct)
            else:
                row.append("")
                pct_row.append(None)
        writer.writerow(row)
        pct_rows.append(pct_row)

    # Average row
    avg_row = ["Average"]
    for col_idx in range(len(SUMMARY_METRICS)):
        pct_vals = [r[col_idx + 1] for r in pct_rows if r[col_idx + 1] is not None]
        avg_row.append("")
        avg_row.append("")
        if pct_vals:
            avg_row.append(f"{sum(pct_vals) / len(pct_vals):.1f}%")
        else:
            avg_row.append("")
    writer.writerow(avg_row)

    # Pretty-print the CSV
    buf.seek(0)
    rows_csv = list(csv.reader(buf))
    if rows_csv:
        ncols = max(len(r) for r in rows_csv)
        widths = [max((len(r[i]) if i < len(r) else 0) for r in rows_csv) for i in range(ncols)]
        for r in rows_csv:
            print("  ".join((r[i] if i < len(r) else "").rjust(w) for i, w in enumerate(widths)))


def write_comparison_csv(rows, path, label_a, label_b):
    """Write comparison results to a CSV file."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "concurrency",
                "metric",
                f"value_{label_a}",
                f"value_{label_b}",
                "delta",
                "pct_change",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "concurrency": row["concurrency"],
                "metric": row["metric"],
                f"value_{label_a}": row["value_a"],
                f"value_{label_b}": row["value_b"],
                "delta": row["delta"],
                "pct_change": row["pct_change"],
            })


def main():
    parser = argparse.ArgumentParser(
        description="Compare two benchmark summary CSV files.",
    )
    parser.add_argument("--csv-a", required=True, help="Path to first bench_summary.csv")
    parser.add_argument("--csv-b", required=True, help="Path to second bench_summary.csv")
    parser.add_argument("--label-a", default="Image A", help="Label for first image")
    parser.add_argument("--label-b", default="Image B", help="Label for second image")
    parser.add_argument("--output-csv", default=None, help="Write comparison to CSV file")
    parser.add_argument("--summary-csv", default=None, help="Write compact summary CSV (for spreadsheets)")
    args = parser.parse_args()

    data_a = read_csv(args.csv_a)
    data_b = read_csv(args.csv_b)

    if not data_a:
        print(f"ERROR: No data found in {args.csv_a}", file=sys.stderr)
        sys.exit(1)
    if not data_b:
        print(f"ERROR: No data found in {args.csv_b}", file=sys.stderr)
        sys.exit(1)

    # Always print the compact summary table
    print_summary_table(data_a, data_b, args.label_a, args.label_b)

    if args.summary_csv:
        write_summary_csv(data_a, data_b, args.summary_csv, args.label_a, args.label_b)

    if args.output_csv:
        # Generate detailed rows for CSV without printing to stdout
        import os
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            rows = print_comparison(data_a, data_b, args.label_a, args.label_b)
            sys.stdout = old_stdout
        write_comparison_csv(rows, args.output_csv, args.label_a, args.label_b)


if __name__ == "__main__":
    main()
