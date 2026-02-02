"""Torch profiler trace analyzer for kernel execution analysis.

Analyzes torch profiler trace files (.trace.json.gz) and provides breakdowns by:
- Stage: prefill vs decode
- Kernel type: attention, MoE, quantization, communication, linear, memory, other
"""

import argparse
import csv
import gzip
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Stage(Enum):
    """Execution stage for a kernel."""

    PREFILL = "prefill"
    DECODE = "decode"
    UNKNOWN = "unknown"


class KernelType(Enum):
    """Classification of kernel by operation type."""

    ATTENTION = "attention"
    MOE = "moe"
    QUANTIZATION = "quantization"
    COMMUNICATION = "communication"
    LINEAR = "linear"
    MEMORY = "memory"
    OTHER = "other"


@dataclass
class KernelEvent:
    """Single kernel execution event from the trace."""

    name: str
    timestamp_us: float  # microseconds
    duration_us: float  # microseconds
    stage: Stage
    kernel_type: KernelType
    grid: Optional[Tuple[int, int, int]] = None
    block: Optional[Tuple[int, int, int]] = None

    @property
    def duration_ms(self) -> float:
        return self.duration_us / 1000.0


@dataclass
class KernelStats:
    """Aggregated statistics for a group of kernels."""

    count: int = 0
    total_time_us: float = 0.0
    min_time_us: float = float("inf")
    max_time_us: float = 0.0

    def add(self, duration_us: float) -> None:
        self.count += 1
        self.total_time_us += duration_us
        self.min_time_us = min(self.min_time_us, duration_us)
        self.max_time_us = max(self.max_time_us, duration_us)

    @property
    def avg_time_us(self) -> float:
        return self.total_time_us / self.count if self.count > 0 else 0.0

    @property
    def total_time_ms(self) -> float:
        return self.total_time_us / 1000.0

    @property
    def avg_time_ms(self) -> float:
        return self.avg_time_us / 1000.0

    @property
    def min_time_ms(self) -> float:
        return self.min_time_us / 1000.0 if self.min_time_us != float("inf") else 0.0

    @property
    def max_time_ms(self) -> float:
        return self.max_time_us / 1000.0


@dataclass
class AnalysisResult:
    """Complete analysis result with all breakdowns."""

    # All kernel events
    events: List[KernelEvent] = field(default_factory=list)

    # Per-kernel stats (keyed by kernel name)
    per_kernel_stats: Dict[str, KernelStats] = field(default_factory=dict)

    # Per-stage stats
    per_stage_stats: Dict[Stage, KernelStats] = field(default_factory=dict)

    # Per-type stats
    per_type_stats: Dict[KernelType, KernelStats] = field(default_factory=dict)

    # Per-stage per-type stats
    per_stage_type_stats: Dict[Tuple[Stage, KernelType], KernelStats] = field(
        default_factory=dict
    )

    # Per-stage per-kernel stats
    per_stage_kernel_stats: Dict[Tuple[Stage, str], KernelStats] = field(
        default_factory=dict
    )

    @property
    def total_time_us(self) -> float:
        return sum(e.duration_us for e in self.events)

    @property
    def total_time_ms(self) -> float:
        return self.total_time_us / 1000.0

    @property
    def total_time_s(self) -> float:
        return self.total_time_us / 1_000_000.0


class KernelClassifier:
    """Classifies kernels by stage and type using pattern matching."""

    def __init__(self, grid_threshold: int = 10000):
        """
        Args:
            grid_threshold: Grid[0] threshold for decode heuristic.
                           Grid[0] > threshold suggests decode (large batch).
        """
        self.grid_threshold = grid_threshold

        # Stage detection patterns (priority order)
        # qseqlen1 -> decode (single query token)
        # qseqlen[2-9] or qseqlen\d{2,} -> prefill (multi-token)
        self._stage_patterns = [
            (re.compile(r"qseqlen1[^0-9]|qseqlen1$", re.IGNORECASE), Stage.DECODE),
            (re.compile(r"qseqlen[2-9]|qseqlen\d{2,}", re.IGNORECASE), Stage.PREFILL),
            (re.compile(r"_decode_|decode_attention", re.IGNORECASE), Stage.DECODE),
            (
                re.compile(r"_prefill_|flash_attn|fmha_fwd", re.IGNORECASE),
                Stage.PREFILL,
            ),
            (
                re.compile(r"generate_draft_decode|create_extend_after_decode"),
                Stage.DECODE,
            ),
        ]

        # Kernel type patterns
        self._type_patterns = [
            # Attention patterns
            (
                re.compile(
                    r"aiter::mla_|decode_attention|flash_attn|attention|softmax|"
                    r"fmha_|mla_reduce|kv_cache|flashinfer|set_mla_kv|"
                    r"kn_entry_2c_sbhd|kn_get_mla_metadata|qk_rope",
                    re.IGNORECASE,
                ),
                KernelType.ATTENTION,
            ),
            # MoE patterns
            (
                re.compile(
                    r"fused_moe|moe_align|topk|expert|MoeFlatmm|MoeSorting|"
                    r"moe_gemm|shared_experts|grouped_topk",
                    re.IGNORECASE,
                ),
                KernelType.MOE,
            ),
            # Quantization patterns
            (
                re.compile(
                    r"mxfp4|fp8|quant|_gemm_afp4wfp4|_fused_rms_mxfp4|"
                    r"_batched_gemm_a16wfp4|dynamic_per_group_scaled_quant|"
                    r"_dynamic_mxfp4|_fused_flatten_mxfp4|fp4x2",
                    re.IGNORECASE,
                ),
                KernelType.QUANTIZATION,
            ),
            # Communication patterns
            (
                re.compile(
                    r"all_reduce|cross_device_reduce|nccl|rccl|broadcast|"
                    r"allgather|reduce_scatter|rcclGenericKernel",
                    re.IGNORECASE,
                ),
                KernelType.COMMUNICATION,
            ),
            # Linear/GEMM patterns (after quant to avoid overlap)
            (
                re.compile(
                    r"gemm|gemv|matmul|cublas|cutlass|hipblas|rocblas|"
                    r"Cijk_Alik_Bljk|_gemm_a16_w16",
                    re.IGNORECASE,
                ),
                KernelType.LINEAR,
            ),
            # Memory patterns
            (
                re.compile(
                    r"memcpy|memset|copy|__amd_rocclr_copyBuffer|"
                    r"vectorized_elementwise_kernel.*copy",
                    re.IGNORECASE,
                ),
                KernelType.MEMORY,
            ),
        ]

    def classify_stage(
        self, name: str, grid: Optional[Tuple[int, int, int]] = None
    ) -> Stage:
        """Classify kernel stage based on name and grid dimensions.

        Priority:
        1. Kernel name patterns (most reliable)
        2. Grid dimension heuristics (fallback)
        """
        # Try name patterns first
        for pattern, stage in self._stage_patterns:
            if pattern.search(name):
                return stage

        # Fallback to grid heuristics if available
        if grid is not None and len(grid) >= 1:
            grid_0 = grid[0]
            # Large grid[0] suggests decode (many sequences in batch)
            if grid_0 > self.grid_threshold:
                return Stage.DECODE
            # Small grid[0] suggests prefill (fewer sequences, longer context)
            elif grid_0 < 200:
                return Stage.PREFILL

        return Stage.UNKNOWN

    def classify_type(self, name: str) -> KernelType:
        """Classify kernel type based on name patterns."""
        for pattern, kernel_type in self._type_patterns:
            if pattern.search(name):
                return kernel_type
        return KernelType.OTHER


class TraceAnalyzer:
    """Main analyzer that loads and processes trace files."""

    def __init__(self, grid_threshold: int = 10000):
        self.classifier = KernelClassifier(grid_threshold=grid_threshold)

    def load_trace(self, path: str) -> Dict[str, Any]:
        """Load trace file (supports .json.gz and .json)."""
        logger.info(f"Loading trace file: {path}")

        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    def _parse_grid(self, grid_data: Any) -> Optional[Tuple[int, int, int]]:
        """Parse grid dimensions from trace data."""
        if grid_data is None:
            return None

        # Handle list format [x, y, z]
        if isinstance(grid_data, list) and len(grid_data) >= 3:
            return (int(grid_data[0]), int(grid_data[1]), int(grid_data[2]))

        # Handle string format "[x, y, z]"
        if isinstance(grid_data, str):
            try:
                parsed = json.loads(grid_data)
                if isinstance(parsed, list) and len(parsed) >= 3:
                    return (int(parsed[0]), int(parsed[1]), int(parsed[2]))
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    def analyze(self, trace_data: Dict[str, Any]) -> AnalysisResult:
        """Analyze trace data and compute statistics."""
        result = AnalysisResult()

        events = trace_data.get("traceEvents", [])
        logger.info(f"Total events in trace: {len(events)}")

        # Filter kernel events (ph="X" means complete event, cat="kernel" means GPU kernel)
        kernel_events = [
            e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"
        ]
        logger.info(f"Kernel events found: {len(kernel_events)}")

        # Process each kernel event
        for event in kernel_events:
            name = event.get("name", "")
            # Timestamps in trace are in microseconds (or nanoseconds, need to check)
            # Looking at the sample data, ts and dur appear to be in nanoseconds actually
            # ts: 2845311300552.151, dur: 54.679
            # The dur values are small, suggesting milliseconds or microseconds
            # Let's treat dur as microseconds based on the scale
            timestamp_us = event.get("ts", 0)
            duration_us = event.get("dur", 0)

            args = event.get("args", {})
            grid = self._parse_grid(args.get("grid"))
            block = self._parse_grid(args.get("block"))

            # Classify
            stage = self.classifier.classify_stage(name, grid)
            kernel_type = self.classifier.classify_type(name)

            # Create event
            kernel_event = KernelEvent(
                name=name,
                timestamp_us=timestamp_us,
                duration_us=duration_us,
                stage=stage,
                kernel_type=kernel_type,
                grid=grid,
                block=block,
            )
            result.events.append(kernel_event)

            # Update per-kernel stats
            if name not in result.per_kernel_stats:
                result.per_kernel_stats[name] = KernelStats()
            result.per_kernel_stats[name].add(duration_us)

            # Update per-stage stats
            if stage not in result.per_stage_stats:
                result.per_stage_stats[stage] = KernelStats()
            result.per_stage_stats[stage].add(duration_us)

            # Update per-type stats
            if kernel_type not in result.per_type_stats:
                result.per_type_stats[kernel_type] = KernelStats()
            result.per_type_stats[kernel_type].add(duration_us)

            # Update per-stage-type stats
            stage_type_key = (stage, kernel_type)
            if stage_type_key not in result.per_stage_type_stats:
                result.per_stage_type_stats[stage_type_key] = KernelStats()
            result.per_stage_type_stats[stage_type_key].add(duration_us)

            # Update per-stage-kernel stats
            stage_kernel_key = (stage, name)
            if stage_kernel_key not in result.per_stage_kernel_stats:
                result.per_stage_kernel_stats[stage_kernel_key] = KernelStats()
            result.per_stage_kernel_stats[stage_kernel_key].add(duration_us)

        return result


class ReportFormatter:
    """Formats analysis results for console output."""

    def __init__(self, result: AnalysisResult):
        self.result = result

    def format_time(self, time_ms: float) -> str:
        """Format time value with appropriate unit."""
        if time_ms >= 1000:
            return f"{time_ms / 1000:.3f} s"
        elif time_ms >= 1:
            return f"{time_ms:.3f} ms"
        else:
            return f"{time_ms * 1000:.3f} us"

    def format_percentage(self, part: float, total: float) -> str:
        """Format percentage."""
        if total == 0:
            return "0.0%"
        return f"{100.0 * part / total:.1f}%"

    def print_summary(self) -> None:
        """Print overall summary."""
        print("=" * 80)
        print("TRACE ANALYSIS SUMMARY")
        print("=" * 80)
        print()

        total_time_ms = self.result.total_time_ms
        print(f"Total Kernel Time:   {self.format_time(total_time_ms)}")

        # Stage breakdown
        for stage in [Stage.PREFILL, Stage.DECODE, Stage.UNKNOWN]:
            stats = self.result.per_stage_stats.get(stage, KernelStats())
            stage_time_ms = stats.total_time_ms
            pct = self.format_percentage(stage_time_ms, total_time_ms)
            print(
                f"  - {stage.value.capitalize():12s} {self.format_time(stage_time_ms):>12s} ({pct})"
            )

        print()

    def print_type_breakdown(self) -> None:
        """Print breakdown by kernel type."""
        print("-" * 80)
        print("BREAKDOWN BY KERNEL TYPE")
        print("-" * 80)
        print()
        print(f"{'Type':<16} {'Count':>10} {'Total':>14} {'Avg':>12} {'%':>8}")
        print("-" * 64)

        total_time_ms = self.result.total_time_ms

        # Sort by total time descending
        sorted_types = sorted(
            self.result.per_type_stats.items(),
            key=lambda x: x[1].total_time_us,
            reverse=True,
        )

        for kernel_type, stats in sorted_types:
            pct = self.format_percentage(stats.total_time_ms, total_time_ms)
            print(
                f"{kernel_type.value:<16} {stats.count:>10,} {self.format_time(stats.total_time_ms):>14} "
                f"{self.format_time(stats.avg_time_ms):>12} {pct:>8}"
            )

        print()

    def print_stage_type_breakdown(self, stage: Stage) -> None:
        """Print type breakdown for a specific stage."""
        print("-" * 80)
        print(f"BREAKDOWN BY TYPE ({stage.value.upper()})")
        print("-" * 80)
        print()
        print(f"{'Type':<16} {'Count':>10} {'Total':>14} {'Avg':>12} {'%':>8}")
        print("-" * 64)

        stage_stats = self.result.per_stage_stats.get(stage, KernelStats())
        stage_total_ms = stage_stats.total_time_ms

        # Filter and sort by total time descending
        stage_type_items = [
            (kt, stats)
            for (s, kt), stats in self.result.per_stage_type_stats.items()
            if s == stage
        ]
        sorted_items = sorted(
            stage_type_items, key=lambda x: x[1].total_time_us, reverse=True
        )

        for kernel_type, stats in sorted_items:
            pct = self.format_percentage(stats.total_time_ms, stage_total_ms)
            print(
                f"{kernel_type.value:<16} {stats.count:>10,} {self.format_time(stats.total_time_ms):>14} "
                f"{self.format_time(stats.avg_time_ms):>12} {pct:>8}"
            )

        print()

    def print_top_kernels(self, stage: Optional[Stage] = None, top_n: int = 20) -> None:
        """Print top kernels by total time."""
        if stage is not None:
            title = f"TOP {top_n} KERNELS ({stage.value.upper()})"
            # Filter by stage
            kernel_items = [
                (name, stats)
                for (s, name), stats in self.result.per_stage_kernel_stats.items()
                if s == stage
            ]
            stage_stats = self.result.per_stage_stats.get(stage, KernelStats())
            total_time_ms = stage_stats.total_time_ms
        else:
            title = f"TOP {top_n} KERNELS (ALL)"
            kernel_items = list(self.result.per_kernel_stats.items())
            total_time_ms = self.result.total_time_ms

        print("-" * 80)
        print(title)
        print("-" * 80)
        print()
        print(
            f"{'#':<4} {'Count':>10} {'Total':>14} {'Avg':>12} {'%':>8}  {'Kernel Name'}"
        )
        print("-" * 80)

        # Sort by total time descending
        sorted_kernels = sorted(
            kernel_items, key=lambda x: x[1].total_time_us, reverse=True
        )

        for i, (name, stats) in enumerate(sorted_kernels[:top_n], 1):
            pct = self.format_percentage(stats.total_time_ms, total_time_ms)
            # Truncate name if too long
            display_name = name[:60] + "..." if len(name) > 63 else name
            print(
                f"{i:<4} {stats.count:>10,} {self.format_time(stats.total_time_ms):>14} "
                f"{self.format_time(stats.avg_time_ms):>12} {pct:>8}  {display_name}"
            )

        print()

    def print_full_report(
        self, top_n: int = 20, stage_filter: Optional[str] = None
    ) -> None:
        """Print full analysis report."""
        self.print_summary()
        self.print_type_breakdown()

        if stage_filter is None or stage_filter == "all":
            # Print for each stage
            for stage in [Stage.PREFILL, Stage.DECODE]:
                if stage in self.result.per_stage_stats:
                    self.print_stage_type_breakdown(stage)
                    self.print_top_kernels(stage=stage, top_n=top_n)
        elif stage_filter == "prefill":
            self.print_stage_type_breakdown(Stage.PREFILL)
            self.print_top_kernels(stage=Stage.PREFILL, top_n=top_n)
        elif stage_filter == "decode":
            self.print_stage_type_breakdown(Stage.DECODE)
            self.print_top_kernels(stage=Stage.DECODE, top_n=top_n)


class CSVExporter:
    """Exports analysis results to CSV files."""

    def __init__(self, result: AnalysisResult):
        self.result = result

    def export_kernel_stats(self, path: str) -> None:
        """Export per-kernel statistics to CSV."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "kernel_name",
                    "count",
                    "total_time_us",
                    "avg_time_us",
                    "min_time_us",
                    "max_time_us",
                ]
            )

            for name, stats in sorted(
                self.result.per_kernel_stats.items(),
                key=lambda x: x[1].total_time_us,
                reverse=True,
            ):
                writer.writerow(
                    [
                        name,
                        stats.count,
                        f"{stats.total_time_us:.3f}",
                        f"{stats.avg_time_us:.3f}",
                        (
                            f"{stats.min_time_us:.3f}"
                            if stats.min_time_us != float("inf")
                            else "0"
                        ),
                        f"{stats.max_time_us:.3f}",
                    ]
                )

        logger.info(f"Kernel stats exported to: {path}")

    def export_events(self, path: str) -> None:
        """Export all kernel events to CSV."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp_us",
                    "duration_us",
                    "stage",
                    "kernel_type",
                    "grid",
                    "block",
                    "kernel_name",
                ]
            )

            for event in self.result.events:
                writer.writerow(
                    [
                        f"{event.timestamp_us:.3f}",
                        f"{event.duration_us:.3f}",
                        event.stage.value,
                        event.kernel_type.value,
                        str(event.grid) if event.grid else "",
                        str(event.block) if event.block else "",
                        event.name,
                    ]
                )

        logger.info(f"Events exported to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze torch profiler trace files for kernel execution times.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m sglang.srt.utils.trace_analyzer /path/to/trace.json.gz

  # Show top 30 kernels, decode stage only
  python -m sglang.srt.utils.trace_analyzer trace.json.gz --top-n 30 --stage decode

  # Export to CSV
  python -m sglang.srt.utils.trace_analyzer trace.json.gz --csv-stats stats.csv --csv-events events.csv
""",
    )

    parser.add_argument("trace_file", help="Path to trace file (.json.gz or .json)")
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top kernels to show (default: 20)",
    )
    parser.add_argument(
        "--stage",
        choices=["prefill", "decode", "all"],
        default="all",
        help="Filter by stage (default: all)",
    )
    parser.add_argument(
        "--csv-stats",
        metavar="FILE",
        help="Export kernel stats to CSV file",
    )
    parser.add_argument(
        "--csv-events",
        metavar="FILE",
        help="Export all events to CSV file",
    )
    parser.add_argument(
        "--grid-threshold",
        type=int,
        default=10000,
        help="Grid[0] threshold for decode heuristic (default: 10000)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Run analysis
    analyzer = TraceAnalyzer(grid_threshold=args.grid_threshold)

    try:
        trace_data = analyzer.load_trace(args.trace_file)
    except FileNotFoundError:
        logger.error(f"Trace file not found: {args.trace_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in trace file: {e}")
        sys.exit(1)

    result = analyzer.analyze(trace_data)

    # Print report
    formatter = ReportFormatter(result)
    formatter.print_full_report(top_n=args.top_n, stage_filter=args.stage)

    # Export CSVs if requested
    if args.csv_stats or args.csv_events:
        exporter = CSVExporter(result)
        if args.csv_stats:
            exporter.export_kernel_stats(args.csv_stats)
        if args.csv_events:
            exporter.export_events(args.csv_events)


if __name__ == "__main__":
    main()
