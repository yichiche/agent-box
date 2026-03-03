"""Trace kernel diff: compare two profile trace xlsx files and return JSON-serializable results.

Wraps the core logic from agent-box/benchmark/compare_traces.py for use by
the orchestrator (auto-compare after profiling) and the dashboard API
(on-demand comparison).
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Add the benchmark directory to sys.path so we can import compare_traces
_BENCHMARK_DIR = str(Path(__file__).resolve().parent.parent.parent / "benchmark")
if _BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, _BENCHMARK_DIR)


def _serialize_group(g) -> dict:
    """Convert a GroupResult to a JSON-serializable dict."""
    result = {
        "layer_type": g.layer_type,
        "stage": g.stage,
        "status": g.status,
        "n_pairs": g.n_pairs,
        "n_identical": g.n_identical,
        "n_changed": g.n_changed,
        "n_skipped": g.n_skipped,
        "change_summary": g.change_summary,
    }

    if g.representative and g.representative.entries:
        rep = g.representative
        result["representative"] = {
            "layer_a_idx": rep.layer_a.layer_idx,
            "layer_b_idx": rep.layer_b.layer_idx,
            "similarity": round(rep.similarity, 4),
            "match_method": rep.match_method,
            "entries": [
                {
                    "pos": e.pos,
                    "file_a_short": e.file_a_short,
                    "file_a_kernel": e.file_a_kernel,
                    "file_b_short": e.file_b_short,
                    "file_b_kernel": e.file_b_kernel,
                    "status": e.status,
                }
                for e in rep.entries
            ],
        }

    if g.only_layers:
        result["only_layers"] = [
            {"layer_idx": l.layer_idx, "n_kernels": len(l.kernels)}
            for l in g.only_layers
        ]

    return result


def compare_trace_files(file_a: str, file_b: str) -> Optional[dict]:
    """Compare two trace analysis xlsx files and return a JSON-serializable diff dict.

    Returns None if the comparison cannot be performed (missing files, import errors, etc.).
    """
    try:
        from compare_traces import parse_excel, match_layers, diff_pair, group_and_select
    except ImportError as e:
        logger.warning("Cannot import compare_traces: %s", e)
        return None

    path_a = Path(file_a)
    path_b = Path(file_b)

    if not path_a.exists():
        logger.warning("Trace file A not found: %s", file_a)
        return None
    if not path_b.exists():
        logger.warning("Trace file B not found: %s", file_b)
        return None

    try:
        layers_a = parse_excel(str(path_a))
        layers_b = parse_excel(str(path_b))

        if not layers_a or not layers_b:
            logger.warning("Empty layer data: A=%d layers, B=%d layers", len(layers_a), len(layers_b))
            return None

        matched, only_a, only_b = match_layers(layers_a, layers_b)

        pair_results = [diff_pair(a, b, method) for a, b, method in matched]

        group_results = group_and_select(pair_results, only_a, only_b)

        n_identical = sum(1 for g in group_results if g.status == "IDENTICAL")
        n_changed = sum(1 for g in group_results if g.status == "CHANGED")
        n_only_a = sum(1 for g in group_results if g.status == "ONLY_IN_A")
        n_only_b = sum(1 for g in group_results if g.status == "ONLY_IN_B")

        has_changes = n_changed > 0 or n_only_a > 0 or n_only_b > 0

        return {
            "file_a": str(path_a),
            "file_b": str(path_b),
            "layers_a": len(layers_a),
            "layers_b": len(layers_b),
            "matched_pairs": len(matched),
            "has_changes": has_changes,
            "summary": {
                "n_groups": len(group_results),
                "n_identical": n_identical,
                "n_changed": n_changed,
                "n_only_a": n_only_a,
                "n_only_b": n_only_b,
            },
            "groups": [_serialize_group(g) for g in group_results],
        }

    except Exception as e:
        logger.warning("Trace comparison failed: %s", e, exc_info=True)
        return None


def compare_and_save(file_a: str, file_b: str, output_path: str,
                     tag_a: str = "", tag_b: str = "") -> Optional[dict]:
    """Compare trace files and save the result as JSON.

    Returns the diff dict, or None on failure.
    """
    diff = compare_trace_files(file_a, file_b)
    if diff is None:
        return None

    diff["tag_a"] = tag_a
    diff["tag_b"] = tag_b

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(diff, indent=2))
        logger.info("Saved kernel diff to %s", output_path)
    except Exception as e:
        logger.warning("Failed to save kernel diff: %s", e)

    return diff
