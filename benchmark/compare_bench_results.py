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


def direction_symbol(metric, pct_change):
    """Return a symbol indicating whether a change is good or bad."""
    if pct_change is None:
        return ""
    if abs(pct_change) < 0.01:
        return "  ~"
    if metric in HIGHER_IS_BETTER:
        return " (+)" if pct_change > 0 else " (-)"
    if metric in LOWER_IS_BETTER:
        return " (+)" if pct_change < 0 else " (-)"
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


def print_comparison(data_a, data_b, label_a, label_b):
    """Print comparison tables and return summary stats."""
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

        print(f"\n{'=' * 90}")
        print(f"  Concurrency: {conc}")
        print(f"{'=' * 90}")

        header = f"{'Metric':<32} {label_a:>14} {label_b:>14} {'Delta':>14} {'% Change':>12}"
        print(header)
        print("-" * 90)

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

            sym = direction_symbol(metric, pct)
            pct_str = f"{pct:+.2f}%{sym}" if pct is not None else "N/A"
            delta_str = f"{delta:+.4f}" if delta is not None else "N/A"

            print(
                f"{metric:<32} {format_value(val_a):>14} {format_value(val_b):>14} "
                f"{delta_str:>14} {pct_str:>12}"
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

    print(f"\n{'=' * 90}")
    print(f"  OVERALL SUMMARY")
    print(f"{'=' * 90}")
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
    args = parser.parse_args()

    data_a = read_csv(args.csv_a)
    data_b = read_csv(args.csv_b)

    if not data_a:
        print(f"ERROR: No data found in {args.csv_a}", file=sys.stderr)
        sys.exit(1)
    if not data_b:
        print(f"ERROR: No data found in {args.csv_b}", file=sys.stderr)
        sys.exit(1)

    rows = print_comparison(data_a, data_b, args.label_a, args.label_b)

    if args.output_csv:
        write_comparison_csv(rows, args.output_csv, args.label_a, args.label_b)
        print(f"Comparison CSV written to: {args.output_csv}")


if __name__ == "__main__":
    main()
