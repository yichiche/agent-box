#!/usr/bin/env python3
"""Pull a one-line performance claim out of an SGLang PR body.

These PRs report results as markdown tables, e.g.

    | TPOT median (ms/token) | 2.60 | 2.48 | -4.6% |
    | Total throughput | 23928 tok/s | 24617 tok/s | +2.88% | 0.54% |

so the delta is a signed percentage in the row for a named metric. Two things
make this less trivial than it looks:

  * A row can carry a second, unsigned percentage that is a standard error
    (`| -2.84% | 0.30% |`). Only signed cells are deltas.
  * Latency and throughput move in opposite directions — -4.6% TPOT is a win,
    -2.88% throughput is a loss — so the sign alone does not say "improvement".

Which metrics get reported follows the rule: TTFT/TPOT if the PR states them
(both), otherwise fall back to e2e latency and total throughput.
"""

from __future__ import annotations

import re
import statistics

# U+2212 MINUS SIGN shows up in these tables as often as ASCII '-'.
PCT = re.compile(r"([+\-−])\s*(\d+(?:\.\d+)?)\s*%")

# (key, label, regex, lower_is_better)
METRICS = [
    ("tpot", "TPOT", re.compile(r"\bTPOT\b", re.I), True),
    ("ttft", "TTFT", re.compile(r"\bTTFT\b", re.I), True),
    ("e2e", "E2E", re.compile(r"\bE2E\b", re.I), True),
    ("throughput", "throughput",
     re.compile(r"\b(?:total\s+)?throughput\b", re.I), False),
]
PREFERRED = re.compile(r"\b(median|p50|mean|avg)\b", re.I)
# Rows that are not a speed delta in their own right:
#   * accuracy tables also contain percentages;
#   * "Interactivity (1 / TPOT)" and friends are the *reciprocal* of a latency
#     already listed, so counting them flips the sign and turns one result into
#     both an improvement and a regression.
EXCLUDE = re.compile(
    r"\b(gsm8k|accuracy|acc|pass@|agree)\b|interactivity|1\s*/\s*(tpot|ttft)",
    re.I,
)


SEP = re.compile(r"^[\s|:-]+$")
UNIT_METRIC = [(re.compile(r"tok/s|tokens?/s|throughput", re.I), "throughput")]


def _tables(body: str):
    """Yield (header_cells, data_rows). A header is the row before a |---|---| separator."""
    lines = [ln.strip() for ln in body.splitlines()]
    i = 0
    while i < len(lines) - 1:
        if (lines[i].startswith("|") and SEP.match(lines[i + 1] or "")
                and lines[i + 1].startswith("|")):
            header = [c.strip() for c in lines[i].strip("|").split("|")]
            rows, j = [], i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append([c.strip() for c in lines[j].strip("|").split("|")])
                j += 1
            yield header, rows
            i = j
        else:
            i += 1


def _metric_of(text: str) -> str | None:
    for key, _, pattern, _ in METRICS:
        if pattern.search(text):
            return key
    for pattern, key in UNIT_METRIC:
        if pattern.search(text):
            return key
    return None


def _signed(cell: str) -> float | None:
    m = PCT.search(cell)
    if not m:
        return None
    return (-1 if m.group(1) in "-\u2212" else 1) * float(m.group(2))


def _row_delta(cells: list[str]) -> float | None:
    """First *signed* percentage in the row — unsigned ones are error bars."""
    for cell in cells[1:]:
        d = _signed(cell)
        if d is not None:
            return d
    return None


def _collect(body: str) -> dict[str, list[tuple[float, bool]]]:
    """metric -> [(delta, came_from_a_median/p50_row), ...]"""
    out: dict[str, list[tuple[float, bool]]] = {}
    for header, rows in _tables(body):
        # Map columns whose header names a metric. A bare delta column ("Δ",
        # "Delta", "%") inherits the metric from the nearest named column to its
        # left, which is how these tables pair "This PR tok/s/GPU" with "Δ".
        col, current = {}, None
        for idx, cell in enumerate(header):
            m = _metric_of(cell)
            if m:
                current = m
            if current and (PCT.search(cell) or re.search(r"Δ|delta|%", cell, re.I)
                            or m):
                col[idx] = current
        header_mapped = any(
            re.search(r"Δ|delta|%", header[i], re.I) for i in col
        )
        for cells in rows:
            label = cells[0] if cells else ""
            if EXCLUDE.search(label):
                continue
            pref = bool(PREFERRED.search(label))
            if header_mapped:
                for idx, key in col.items():
                    if idx < len(cells):
                        d = _signed(cells[idx])
                        if d is not None:
                            out.setdefault(key, []).append((d, pref))
            else:
                key = _metric_of(label)
                if key:
                    d = _row_delta(cells)
                    if d is not None:
                        out.setdefault(key, []).append((d, pref))
    return out


def extract(body: str) -> str:
    """-> 'TPOT 4.6% improvement, TTFT 5.2% regression', or '' if none found."""
    if not body:
        return ""
    collected = _collect(body)
    found = {}
    for key, vals in collected.items():
        # Prefer median/p50 rows when the PR marks them; otherwise take the
        # median across rows so a multi-shape sweep is not cherry-picked.
        chosen = [d for d, pref in vals if pref] or [d for d, _ in vals]
        found[key] = statistics.median(chosen)

    # Both of TTFT/TPOT stated -> that pair is the story. Otherwise fill up to
    # two metrics from the default order, so a PR that quotes only TTFT still
    # shows its e2e/throughput headline instead of a lone number.
    if "tpot" in found and "ttft" in found:
        order = ["tpot", "ttft"]
    else:
        order = [k for k in ("tpot", "ttft", "e2e", "throughput") if k in found][:2]
    parts = []
    for key in order:
        delta = found[key]
        label = next(lb for k, lb, _, _ in METRICS if k == key)
        lower_better = next(lo for k, _, _, lo in METRICS if k == key)
        better = (delta < 0) if lower_better else (delta > 0)
        if abs(delta) < 0.05:
            parts.append(f"{label} unchanged")
        else:
            parts.append(f"{label} {abs(delta):.1f}% "
                         f"{'improvement' if better else 'regression'}")
    return ", ".join(parts)


if __name__ == "__main__":
    import subprocess
    import sys

    for pr in sys.argv[1:]:
        body = subprocess.run(
            ["gh", "pr", "view", pr, "--repo", "sgl-project/sglang",
             "--json", "body", "--jq", ".body"],
            capture_output=True, text=True, env={"GH_TOKEN": "", "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "/home/yichiche"},
        ).stdout
        print(f"#{pr}: {extract(body) or '(no perf numbers found)'}")
