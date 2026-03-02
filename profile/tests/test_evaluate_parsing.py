"""Unit tests for evaluate_parsing.py scoring and evaluation functions."""

import json
import sys
import os

import pytest

# Add parent directory to path so we can import evaluate_parsing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate_parsing import (
    EvaluationReport,
    GroupKey,
    GroupMetrics,
    LayerKernelPattern,
    LayerSummaryRecord,
    ModelArchitectureConfig,
    OverallStats,
    ParsingEvaluator,
    StructuralMetrics,
    _grade,
    _median,
    _run_length_encode,
    _score_mode_match,
    _score_outlier_pct,
    _score_stage_coverage,
    _score_time_cv,
    _segment_into_rounds,
    _trimmed_cv,
)


# ── Scoring function tests ──────────────────────────────────────────────────


class TestScoreStageCoverage:
    def test_perfect_coverage(self):
        assert _score_stage_coverage(100.0) == 100.0

    def test_high_coverage(self):
        assert _score_stage_coverage(99.0) == 80.0
        assert _score_stage_coverage(98.0) == 80.0

    def test_medium_coverage(self):
        assert _score_stage_coverage(95.0) == 60.0
        assert _score_stage_coverage(90.0) == 60.0

    def test_low_coverage(self):
        score = _score_stage_coverage(45.0)
        assert 0 < score < 60
        assert _score_stage_coverage(0.0) == 0.0

    def test_boundary_98(self):
        assert _score_stage_coverage(97.9) == 60.0


class TestScoreTimeCV:
    def test_very_low_cv(self):
        assert _score_time_cv(0.01) == 100.0
        assert _score_time_cv(0.05) == 100.0

    def test_low_cv(self):
        assert _score_time_cv(0.07) == 80.0
        assert _score_time_cv(0.10) == 80.0

    def test_medium_cv(self):
        assert _score_time_cv(0.15) == 60.0
        assert _score_time_cv(0.20) == 60.0

    def test_high_cv(self):
        score = _score_time_cv(0.50)
        assert 0 < score < 60
        assert _score_time_cv(1.0) == 0.0

    def test_zero_cv(self):
        assert _score_time_cv(0.0) == 100.0


class TestScoreModeMatch:
    def test_perfect_match(self):
        assert _score_mode_match(100.0) == 100.0

    def test_high_match(self):
        assert _score_mode_match(97.0) == 100.0
        assert _score_mode_match(95.0) == 100.0

    def test_medium_match(self):
        assert _score_mode_match(85.0) == 80.0
        assert _score_mode_match(80.0) == 80.0

    def test_low_match(self):
        score = _score_mode_match(40.0)
        assert 0 < score < 60
        assert _score_mode_match(0.0) == 0.0


class TestScoreOutlierPct:
    def test_very_few_outliers(self):
        assert _score_outlier_pct(0.0) == 100.0
        assert _score_outlier_pct(1.5) == 100.0

    def test_some_outliers(self):
        assert _score_outlier_pct(3.0) == 80.0
        assert _score_outlier_pct(4.9) == 80.0

    def test_moderate_outliers(self):
        assert _score_outlier_pct(7.0) == 60.0
        assert _score_outlier_pct(9.9) == 60.0

    def test_many_outliers(self):
        score = _score_outlier_pct(50.0)
        assert 0 < score < 60
        assert _score_outlier_pct(100.0) == 0.0


class TestGrade:
    def test_grades(self):
        assert _grade(95) == "A"
        assert _grade(90) == "A"
        assert _grade(85) == "B"
        assert _grade(75) == "B"
        assert _grade(70) == "C"
        assert _grade(60) == "C"
        assert _grade(59) == "F"
        assert _grade(0) == "F"


# ── Trimmed CV tests ────────────────────────────────────────────────────────


class TestTrimmedCV:
    def test_identical_values(self):
        cv, outliers = _trimmed_cv([100.0] * 10)
        assert cv == 0.0
        assert outliers == []

    def test_single_value(self):
        cv, outliers = _trimmed_cv([42.0])
        assert cv == 0.0
        assert outliers == []

    def test_two_values_same(self):
        cv, outliers = _trimmed_cv([5.0, 5.0])
        assert cv == 0.0

    def test_low_variation(self):
        values = [100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7, 100.8, 100.9]
        cv, outliers = _trimmed_cv(values)
        assert cv < 0.01
        assert len(outliers) == 0

    def test_with_outliers(self):
        # 8 normal values around 100, plus 2 extreme outliers
        values = [100.0, 101.0, 99.0, 100.5, 99.5, 100.2, 99.8, 100.1, 500.0, 10.0]
        cv, outliers = _trimmed_cv(values)
        assert len(outliers) >= 1
        assert cv < 0.05

    def test_empty_list(self):
        cv, outliers = _trimmed_cv([])
        assert cv == 0.0


# ── Helper function tests ──────────────────────────────────────────────────


class TestMedian:
    def test_odd_count(self):
        assert _median([1.0, 2.0, 3.0]) == 2.0

    def test_even_count(self):
        assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_single(self):
        assert _median([5.0]) == 5.0

    def test_empty(self):
        assert _median([]) == 0.0

    def test_unsorted(self):
        assert _median([3.0, 1.0, 2.0]) == 2.0


class TestSegmentIntoRounds:
    def test_empty(self):
        assert _segment_into_rounds([]) == []

    def test_single_stage(self):
        records = _make_records("decode", "MLA+MoE", 5)
        rounds = _segment_into_rounds(records)
        assert len(rounds) == 1
        assert rounds[0][0] == "decode"
        assert len(rounds[0][1]) == 5

    def test_two_stages(self):
        records = (
            _make_records("prefill", "MLA+MoE", 3, start_idx=0)
            + _make_records("decode", "MLA+MoE", 5, start_idx=3)
        )
        rounds = _segment_into_rounds(records)
        assert len(rounds) == 2
        assert rounds[0][0] == "prefill"
        assert len(rounds[0][1]) == 3
        assert rounds[1][0] == "decode"
        assert len(rounds[1][1]) == 5

    def test_alternating_stages(self):
        records = (
            _make_records("prefill", "MLA+MoE", 2, start_idx=0)
            + _make_records("decode", "MLA+MoE", 3, start_idx=2)
            + _make_records("prefill", "MLA+MoE", 2, start_idx=5)
            + _make_records("decode", "MLA+MoE", 3, start_idx=7)
        )
        rounds = _segment_into_rounds(records)
        assert len(rounds) == 4
        assert [s for s, _ in rounds] == ["prefill", "decode", "prefill", "decode"]


class TestRunLengthEncode:
    def test_empty(self):
        assert _run_length_encode([]) == []

    def test_single(self):
        assert _run_length_encode(["A"]) == [(1, "A")]

    def test_all_same(self):
        assert _run_length_encode(["A", "A", "A"]) == [(3, "A")]

    def test_mixed(self):
        result = _run_length_encode(["A", "B", "B", "C", "C", "C"])
        assert result == [(1, "A"), (2, "B"), (3, "C")]

    def test_alternating(self):
        result = _run_length_encode(["A", "B", "A", "B"])
        assert result == [(1, "A"), (1, "B"), (1, "A"), (1, "B")]


# ── Evaluator component tests ───────────────────────────────────────────────


def _make_records(
    stage, layer_type, n, time_us=100.0, kernel_count=27, start_idx=0,
    attention_time_us=0.0, moe_time_us=0.0, linear_time_us=0.0,
    comm_time_us=0.0, quant_time_us=0.0,
):
    """Helper to create n LayerSummaryRecord objects."""
    return [
        LayerSummaryRecord(
            layer_idx=start_idx + i,
            layer_type=layer_type,
            stage=stage,
            total_time_us=time_us,
            kernel_count=kernel_count,
            attention_time_us=attention_time_us,
            moe_time_us=moe_time_us,
            linear_time_us=linear_time_us,
            comm_time_us=comm_time_us,
            quant_time_us=quant_time_us,
        )
        for i in range(n)
    ]


def _make_patterns(stage, layer_type, n, short_names=("GEMM", "FMHA", "ALLREDUCE"), start_idx=0):
    """Helper to create n LayerKernelPattern objects."""
    return [
        LayerKernelPattern(
            layer_idx=start_idx + i,
            layer_type=layer_type,
            stage=stage,
            short_names=short_names,
        )
        for i in range(n)
    ]


# ── S1: Prefill Ordering Tests ─────────────────────────────────────────────


class TestPrefillOrdering:
    def test_correct_order(self):
        """Prefill before decode should score 100."""
        ev = ParsingEvaluator()
        records = (
            _make_records("prefill", "MLA+MoE", 5, start_idx=0)
            + _make_records("decode", "MLA+MoE", 10, start_idx=5)
        )
        score, details = ev.compute_prefill_ordering(records)
        assert score == 100.0
        assert details["violations"] == 0

    def test_decode_before_prefill(self):
        """Decode appearing before first prefill should be penalized."""
        ev = ParsingEvaluator()
        records = (
            _make_records("decode", "MLA+MoE", 2, start_idx=0)
            + _make_records("prefill", "MLA+MoE", 5, start_idx=2)
            + _make_records("decode", "MLA+MoE", 10, start_idx=7)
        )
        score, details = ev.compute_prefill_ordering(records)
        assert score < 100.0
        assert details["violations"] == 2

    def test_decode_only(self):
        """Decode-only trace should score 100 (no prefill to violate)."""
        ev = ParsingEvaluator()
        records = _make_records("decode", "MLA+MoE", 10)
        score, details = ev.compute_prefill_ordering(records)
        assert score == 100.0

    def test_empty_records(self):
        ev = ParsingEvaluator()
        score, details = ev.compute_prefill_ordering([])
        assert score == 100.0


# ── S2: Architecture Signature Tests ────────────────────────────────────────


class TestArchitectureSignature:
    def test_auto_detect_regular_pattern(self):
        """Regular prefill pattern should auto-detect and score 100."""
        ev = ParsingEvaluator()
        records = (
            _make_records("prefill", "MLA+FC", 3, start_idx=0)
            + _make_records("prefill", "MLA+MoE", 57, start_idx=3)
        )
        score, details = ev.compute_architecture_signature(records)
        assert score == 100.0
        assert details["auto_detected"] is True

    def test_matching_config(self):
        """When config matches, should score 100."""
        config = ModelArchitectureConfig(
            expected_prefill_pattern=[(3, "MLA+FC"), (57, "MLA+MoE")]
        )
        ev = ParsingEvaluator(config=config)
        records = (
            _make_records("prefill", "MLA+FC", 3, start_idx=0)
            + _make_records("prefill", "MLA+MoE", 57, start_idx=3)
        )
        score, details = ev.compute_architecture_signature(records, config)
        assert score == 100.0
        assert details["matched"] is True

    def test_mismatching_config(self):
        """When config doesn't match, should score low."""
        config = ModelArchitectureConfig(
            expected_prefill_pattern=[(5, "MLA+FC"), (55, "MLA+MoE")]
        )
        ev = ParsingEvaluator(config=config)
        records = (
            _make_records("prefill", "MLA+FC", 3, start_idx=0)
            + _make_records("prefill", "MLA+MoE", 57, start_idx=3)
        )
        score, details = ev.compute_architecture_signature(records, config)
        assert score == 30.0
        assert details["matched"] is False

    def test_no_prefill(self):
        """No prefill round should score 0."""
        ev = ParsingEvaluator()
        records = _make_records("decode", "MLA+MoE", 10)
        score, details = ev.compute_architecture_signature(records)
        assert score == 0.0


# ── S3: Round Consistency Tests ─────────────────────────────────────────────


class TestRoundConsistency:
    def test_uniform_decode_rounds(self):
        """All decode rounds same size should score 100."""
        ev = ParsingEvaluator()
        records = (
            _make_records("prefill", "MLA+MoE", 60, start_idx=0)
            + _make_records("decode", "MLA+MoE", 60, start_idx=60)
            + _make_records("prefill", "MLA+MoE", 60, start_idx=120)
            + _make_records("decode", "MLA+MoE", 60, start_idx=180)
        )
        score, details = ev.compute_round_consistency(records)
        assert score == 100.0
        assert details["mode_size"] == 60

    def test_variable_decode_rounds(self):
        """Mixed decode round sizes should lower score."""
        ev = ParsingEvaluator()
        records = (
            _make_records("prefill", "MLA+MoE", 60, start_idx=0)
            + _make_records("decode", "MLA+MoE", 60, start_idx=60)
            + _make_records("prefill", "MLA+MoE", 60, start_idx=120)
            + _make_records("decode", "MLA+MoE", 30, start_idx=180)  # Different size
        )
        score, details = ev.compute_round_consistency(records)
        assert score < 100.0
        assert details["distinct_sizes"] == 2

    def test_no_decode(self):
        """No decode rounds should score 100."""
        ev = ParsingEvaluator()
        records = _make_records("prefill", "MLA+MoE", 60)
        score, details = ev.compute_round_consistency(records)
        assert score == 100.0


# ── S4: Type Sequence Repeatability Tests ───────────────────────────────────


class TestTypeSequenceRepeatability:
    def test_identical_rounds(self):
        """Identical type patterns in all rounds should score 100."""
        ev = ParsingEvaluator()
        records = (
            _make_records("decode", "MLA+FC", 3, start_idx=0)
            + _make_records("decode", "MLA+MoE", 57, start_idx=3)
        )
        score, details = ev.compute_type_sequence_repeatability(records)
        assert score == 100.0

    def test_empty_records(self):
        ev = ParsingEvaluator()
        score, details = ev.compute_type_sequence_repeatability([])
        assert score == 100.0


# ── S6: Transition Regularity Tests ─────────────────────────────────────────


class TestTransitionRegularity:
    def test_clean_transition(self):
        """Clean P->D transition should score 100."""
        ev = ParsingEvaluator()
        records = (
            _make_records("prefill", "MLA+MoE", 60, start_idx=0)
            + _make_records("decode", "MLA+MoE", 180, start_idx=60)
        )
        score, details = ev.compute_transition_regularity(records)
        assert score == 100.0
        assert details["back_transitions"] == 0

    def test_single_stage(self):
        """Single stage should score 100."""
        ev = ParsingEvaluator()
        records = _make_records("decode", "MLA+MoE", 60)
        score, details = ev.compute_transition_regularity(records)
        assert score == 100.0

    def test_interleaved(self):
        """Heavy interleaving should reduce score."""
        ev = ParsingEvaluator()
        records = []
        for i in range(10):
            stage = "prefill" if i % 2 == 0 else "decode"
            records.extend(_make_records(stage, "MLA+MoE", 5, start_idx=i * 5))
        score, details = ev.compute_transition_regularity(records)
        assert details["back_transitions"] > 0


# ── Legacy component tests (diagnostic) ──────────────────────────────────


class TestTimeConsistency:
    def test_identical_times(self):
        ev = ParsingEvaluator()
        records = _make_records("decode", "MLA+MoE", 10, time_us=100.0)
        score, cv, outliers = ev.compute_time_consistency(records)
        assert score == 100.0
        assert cv == 0.0
        assert outliers == []

    def test_varied_times(self):
        ev = ParsingEvaluator()
        records = _make_records("decode", "MLA+MoE", 10, time_us=100.0)
        for i, r in enumerate(records):
            r.total_time_us = 100.0 + i * 0.5
        score, cv, outliers = ev.compute_time_consistency(records)
        assert score > 80.0


class TestKernelCountConsistency:
    def test_all_same_count(self):
        ev = ParsingEvaluator()
        records = _make_records("decode", "MLA+MoE", 10, kernel_count=27)
        score, mode, pct, distinct = ev.compute_kernel_count_consistency(records)
        assert score == 100.0
        assert mode == 27
        assert pct == 100.0
        assert distinct == 1

    def test_mostly_same(self):
        ev = ParsingEvaluator()
        records = _make_records("decode", "MLA+MoE", 20, kernel_count=27)
        records[0].kernel_count = 28
        score, mode, pct, distinct = ev.compute_kernel_count_consistency(records)
        assert score == 100.0  # 95% match -> full score with loosened threshold
        assert mode == 27
        assert distinct == 2


class TestPatternConsistency:
    def test_all_same_pattern(self):
        ev = ParsingEvaluator()
        patterns = _make_patterns("decode", "MLA+MoE", 10)
        score, pct = ev.compute_pattern_consistency(patterns)
        assert score == 100.0
        assert pct == 100.0

    def test_no_patterns(self):
        ev = ParsingEvaluator()
        score, pct = ev.compute_pattern_consistency([])
        assert score == 0.0
        assert pct == 0.0

    def test_mostly_same_pattern(self):
        ev = ParsingEvaluator()
        patterns = _make_patterns("decode", "MLA+MoE", 20)
        patterns[0] = LayerKernelPattern(0, "MLA+MoE", "decode", ("GEMM", "FMHA"))
        score, pct = ev.compute_pattern_consistency(patterns)
        assert score == 100.0  # 95% match -> full score with loosened threshold


class TestGrouping:
    def test_basic_grouping(self):
        ev = ParsingEvaluator()
        records = (
            _make_records("decode", "MLA+MoE", 5, start_idx=0)
            + _make_records("prefill", "MLA+FC", 3, start_idx=5)
        )
        patterns = _make_patterns("decode", "MLA+MoE", 3, start_idx=0)

        groups = ev.group_layers(records, patterns)
        assert len(groups) == 2
        k1 = GroupKey("decode", "MLA+MoE")
        k2 = GroupKey("prefill", "MLA+FC")
        assert k1 in groups
        assert k2 in groups
        assert len(groups[k1][0]) == 5
        assert len(groups[k1][1]) == 3
        assert len(groups[k2][0]) == 3
        assert len(groups[k2][1]) == 0


class TestGroupEvaluation:
    def test_skip_small_group(self):
        ev = ParsingEvaluator()
        records = _make_records("prefill", "MLA+MoE", 2)
        gm = ev._evaluate_group(GroupKey("prefill", "MLA+MoE"), records, [])
        assert gm.skipped is True

    def test_perfect_group(self):
        ev = ParsingEvaluator()
        records = _make_records("decode", "MLA+MoE", 20, time_us=100.0, kernel_count=27)
        patterns = _make_patterns("decode", "MLA+MoE", 20)
        gm = ev._evaluate_group(GroupKey("decode", "MLA+MoE"), records, patterns)
        assert gm.skipped is False
        assert gm.composite_score > 90


class TestBuildReport:
    def test_full_report(self):
        ev = ParsingEvaluator(threshold=70.0)
        overall = OverallStats(10000.0, 500.0, 9500.0, 0.0)
        records = (
            _make_records("prefill", "MLA+MoE", 10, time_us=5000.0, kernel_count=27, start_idx=0)
            + _make_records("decode", "MLA+MoE", 50, time_us=190.0, kernel_count=27, start_idx=10)
        )
        patterns = _make_patterns("decode", "MLA+MoE", 50, start_idx=10)

        report = ev._build_report("test.xlsx", overall, records, patterns)
        assert report.total_layers == 60
        assert report.grade in ("A", "B", "C", "F")
        assert isinstance(report.passed, bool)
        assert isinstance(report.structural_metrics, StructuralMetrics)
        assert report.structural_score >= 0
        assert report.overall_score == report.structural_score

    def test_report_with_multiple_groups(self):
        ev = ParsingEvaluator()
        overall = OverallStats(10000.0, 1000.0, 9000.0, 0.0)
        records = (
            _make_records("prefill", "MLA+MoE", 10, time_us=10000.0, kernel_count=27, start_idx=0)
            + _make_records("decode", "MLA+MoE", 30, time_us=200.0, kernel_count=27, start_idx=10)
            + _make_records("decode", "MLA+FC", 10, time_us=150.0, kernel_count=15, start_idx=40)
        )
        patterns = _make_patterns("decode", "MLA+MoE", 10, start_idx=10)

        report = ev._build_report("test.xlsx", overall, records, patterns)
        assert report.group_count == 3  # prefill MoE + decode MoE + decode FC
        assert len(report.group_metrics) == 3
        assert report.overall_score > 0


# ── Output format tests ─────────────────────────────────────────────────────


class TestOutputFormats:
    def _make_report(self):
        ev = ParsingEvaluator()
        overall = OverallStats(10000.0, 500.0, 9500.0, 0.0)
        records = (
            _make_records("prefill", "MLA+MoE", 10, time_us=5000.0, kernel_count=27, start_idx=0)
            + _make_records("decode", "MLA+MoE", 20, time_us=100.0, kernel_count=27, start_idx=10)
        )
        patterns = _make_patterns("decode", "MLA+MoE", 20, start_idx=10)
        return ev, ev._build_report("test.xlsx", overall, records, patterns)

    def test_terminal_report(self):
        ev, report = self._make_report()
        text = ev.format_terminal_report(report)
        assert "Trace Parsing Quality Report" in text
        assert "test.xlsx" in text
        assert "PASS" in text or "FAIL" in text
        assert "S1 Prefill Ordering" in text
        assert "S2 Architecture Sig" in text
        assert "S3 Round Consistency" in text
        assert "S4 Type Sequence" in text

    def test_json_report(self):
        ev, report = self._make_report()
        text = ev.format_json_report(report)
        data = json.loads(text)
        assert "overall_score" in data
        assert "grade" in data
        assert "passed" in data
        assert "groups" in data
        assert "structural_rules" in data
        assert "s1_prefill_ordering" in data["structural_rules"]
        assert "s4_type_sequence" in data["structural_rules"]


# ── Prefill special handling tests ───────────────────────────────────────────


class TestPrefillSubGrouping:
    def test_prefill_subgrouping_triggered(self):
        """Prefill layers with >3 distinct kernel counts should be sub-grouped."""
        ev = ParsingEvaluator()
        records = []
        for kc in [10, 20, 30, 40, 50]:
            records.extend(_make_records("prefill", "MLA+MoE", 5, kernel_count=kc, start_idx=kc * 10))

        gm = ev._evaluate_group(GroupKey("prefill", "MLA+MoE"), records, [])
        assert not gm.skipped
        assert any("sub-grouped" in n.lower() for n in gm.notes)

    def test_decode_no_subgrouping(self):
        """Decode layers should never be sub-grouped."""
        ev = ParsingEvaluator()
        records = []
        for kc in [10, 20, 30, 40, 50]:
            records.extend(_make_records("decode", "MLA+MoE", 5, kernel_count=kc, start_idx=kc * 10))

        gm = ev._evaluate_group(GroupKey("decode", "MLA+MoE"), records, [])
        assert not gm.skipped
        assert not any("sub-grouped" in n.lower() for n in gm.notes)
