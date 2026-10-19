#!/usr/bin/env python3
"""Rank one rocprofv3 kernel_stats.csv by accumulated GPU time (stdlib only)."""
import argparse
import csv
import math
import sys


def summarize(path):
    totals = {}
    with open(path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'Name', 'Calls', 'TotalDurationNs'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('expected Name, Calls, and TotalDurationNs columns; inspect the CSV schema and units')
        for line, row in enumerate(reader, 2):
            name = row['Name'].strip()
            try:
                calls = int(row['Calls'])
                duration = float(row['TotalDurationNs'])
            except (TypeError, ValueError) as exc:
                raise ValueError(f'invalid numeric value on CSV line {line}') from exc
            if not name or calls <= 0 or not math.isfinite(duration) or duration < 0:
                raise ValueError(f'invalid kernel name, call count, or duration on CSV line {line}')
            previous = totals.setdefault(name, [0, 0.0])
            previous[0] += calls
            previous[1] += duration
    if not totals:
        raise ValueError('no kernel rows in the summary')
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', help='one kernel_stats.csv from one run/device')
    parser.add_argument('--top', type=int, default=10)
    args = parser.parse_args()
    if args.top <= 0:
        parser.error('--top must be positive')
    try:
        totals = summarize(args.csv)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    duration_sum = sum(value[1] for value in totals.values())
    writer = csv.writer(sys.stdout)
    writer.writerow(['Rank', 'Kernel', 'Calls', 'TotalUs', 'MeanUs', 'SharePct'])
    for rank, (name, (calls, duration)) in enumerate(sorted(totals.items(), key=lambda item: item[1][1], reverse=True)[:args.top], 1):
        share = 100 * duration / duration_sum if duration_sum else 0.0
        writer.writerow([rank, name, calls, f'{duration / 1000:.3f}', f'{duration / calls / 1000:.3f}', f'{share:.2f}'])


if __name__ == '__main__':
    main()
