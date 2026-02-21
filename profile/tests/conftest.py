"""Pytest configuration and fixtures for trace_analyzer regression tests."""

import json
import os
import pickle
import sys

import pytest

# Add parent directory so we can import trace_analyzer,
# and tests directory so we can import generate_baselines
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_baselines import TRACE_REGISTRY, compute_metrics
from trace_analyzer import TraceAnalyzer

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.join(TESTS_DIR, "baselines")
SNAPSHOTS_DIR = os.path.join(TESTS_DIR, "snapshots")


def pytest_addoption(parser):
    parser.addoption(
        "--trace-dir",
        action="store",
        default="/home/yichiche/profile-baseline",
        help="Directory containing trace files",
    )
    parser.addoption(
        "--regen-snapshots",
        action="store_true",
        default=False,
        help="Force re-parse traces instead of using cached pickles",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: re-parses full traces")
    config.addinivalue_line("markers", "known_broken: models with known parsing issues")


def pytest_collection_modifyitems(config, items):
    """Auto-apply xfail to tests for known_broken models."""
    for item in items:
        model_name = None
        if hasattr(item, "callspec") and "model_name" in item.callspec.params:
            model_name = item.callspec.params["model_name"]

        if model_name and model_name in TRACE_REGISTRY:
            status = TRACE_REGISTRY[model_name]["status"]
            if status == "known_broken":
                item.add_marker(
                    pytest.mark.xfail(
                        reason=f"Model {model_name} has status=known_broken",
                        strict=False,
                    )
                )


# All model names from the registry
ALL_MODELS = list(TRACE_REGISTRY.keys())


@pytest.fixture(scope="session")
def trace_dir(request):
    return request.config.getoption("--trace-dir")


@pytest.fixture(scope="session")
def regen_snapshots(request):
    return request.config.getoption("--regen-snapshots")


@pytest.fixture(scope="session")
def analysis_results(trace_dir, regen_snapshots):
    """Session-scoped cache of AnalysisResult objects keyed by model name.

    Loads from pickle cache if available, otherwise parses the trace file
    and saves the pickle.
    """
    results = {}

    for model_name, entry in TRACE_REGISTRY.items():
        pickle_path = os.path.join(SNAPSHOTS_DIR, f"{model_name}.pkl")

        # Try loading from pickle cache
        if not regen_snapshots and os.path.exists(pickle_path):
            try:
                with open(pickle_path, "rb") as f:
                    results[model_name] = pickle.load(f)
                continue
            except Exception:
                pass

        # Parse from trace file
        trace_path = os.path.join(trace_dir, entry["trace_file"])
        if not os.path.exists(trace_path):
            continue

        analyzer = TraceAnalyzer(
            grid_threshold=entry["grid_threshold"],
            mtp_qseqlen_decode=entry["mtp_qseqlen_decode"],
        )
        trace_data = analyzer.load_trace(trace_path)
        result = analyzer.analyze(trace_data, detect_layers=True)

        # Cache the result
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
        with open(pickle_path, "wb") as f:
            pickle.dump(result, f)

        results[model_name] = result

    return results


@pytest.fixture
def model_name(request):
    """Indirect fixture to receive model_name from parametrize."""
    return request.param


@pytest.fixture
def analysis_result(model_name, analysis_results):
    """Per-test fixture that returns the AnalysisResult for the current model."""
    if model_name not in analysis_results:
        pytest.skip(f"No analysis result for {model_name} (trace file missing?)")
    return analysis_results[model_name]


@pytest.fixture
def baseline(model_name):
    """Load golden baseline JSON for the current model."""
    baseline_path = os.path.join(BASELINES_DIR, f"{model_name}.json")
    if not os.path.exists(baseline_path):
        pytest.skip(f"No baseline file for {model_name}")

    with open(baseline_path) as f:
        return json.load(f)
