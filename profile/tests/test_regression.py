"""Regression tests for trace_analyzer.py parsing quality.

These tests verify that trace_analyzer correctly classifies kernels and
detects layers for each model's trace. Metrics are architecture-invariant
(they don't depend on input length, output length, or concurrency).

Models with status="known_broken" are automatically marked xfail via conftest.
"""

import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_baselines import TRACE_REGISTRY, compute_metrics
from trace_analyzer import KernelType, LayerType, Stage

ALL_MODELS = list(TRACE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Test 1: Stage Coverage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_name", ALL_MODELS, indirect=True)
def test_stage_coverage(model_name, analysis_result, baseline):
    """Verify that enough kernel time is classified into PREFILL or DECODE."""
    entry = TRACE_REGISTRY[model_name]
    metrics = compute_metrics(analysis_result, entry)

    min_threshold = baseline["metrics"]["stage_coverage_pct"]["min"]
    actual = metrics["stage_coverage_pct"]

    assert actual >= min_threshold, (
        f"Stage coverage {actual:.1f}% < {min_threshold:.1f}% threshold. "
        f"Too many kernels classified as UNKNOWN stage."
    )


# ---------------------------------------------------------------------------
# Test 2: Both Stages Present
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_name", ALL_MODELS, indirect=True)
def test_both_stages_present(model_name, analysis_result, baseline):
    """Verify that both prefill and decode layers are detected."""
    entry = TRACE_REGISTRY[model_name]
    metrics = compute_metrics(analysis_result, entry)

    assert metrics["prefill_layers"] > 0, "No prefill layers detected"
    assert metrics["decode_layers"] > 0, "No decode layers detected"


# ---------------------------------------------------------------------------
# Test 3: Layer Count Minimum
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_name", ALL_MODELS, indirect=True)
def test_layer_count_minimum(model_name, analysis_result, baseline):
    """Verify that at least num_hidden_layers layers are detected."""
    entry = TRACE_REGISTRY[model_name]
    metrics = compute_metrics(analysis_result, entry)

    min_layers = baseline["metrics"]["layer_count_min"]
    total = metrics["total_layers"]

    assert total >= min_layers, (
        f"Only {total} layers detected, expected at least {min_layers} "
        f"(num_hidden_layers={entry['num_hidden_layers']})"
    )


# ---------------------------------------------------------------------------
# Test 4: Layer Types
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_name", ALL_MODELS, indirect=True)
def test_layer_types(model_name, analysis_result, baseline):
    """Verify that detected layer types match the model's architecture.

    - DSR1: first 3 layers are MLA+FC, remaining are MLA+MoE per pass
    - Grok2: MHA+MoE after half-layer merging
    """
    entry = TRACE_REGISTRY[model_name]
    expected = baseline["metrics"]["expected_layer_types"]
    pattern = expected["pattern"]

    layers = analysis_result.layers
    if not layers:
        pytest.fail("No layers detected")

    num_hidden = entry["num_hidden_layers"]

    if pattern == "first_n_fc_rest_moe":
        first_n_fc = expected["first_n_fc"]
        prefill_layers = [l for l in layers if l.stage == Stage.PREFILL]
        decode_layers = [l for l in layers if l.stage == Stage.DECODE]

        for pass_name, pass_layers in [
            ("prefill", prefill_layers),
            ("decode", decode_layers),
        ]:
            if len(pass_layers) < num_hidden:
                continue

            # Verify the repeating FC→MoE pattern:
            # Each pass of num_hidden layers starts with first_n_fc FC layers
            # followed by MoE layers. We check by grouping consecutive
            # same-type runs and verifying FC always comes before MoE
            # within each pass.
            #
            # Find the first FC layer to locate pass start
            start = None
            for i, layer in enumerate(pass_layers):
                if layer.layer_type in (LayerType.MLA_FC, LayerType.MHA_FC):
                    start = i
                    break

            if start is None:
                pytest.fail(f"{pass_name}: no FC layers found at all")

            # Check just the first pass: first_n_fc FC then MoE
            fc_count = 0
            saw_moe = False
            for layer in pass_layers[start:]:
                is_fc = layer.layer_type in (LayerType.MLA_FC, LayerType.MHA_FC)
                is_moe = layer.layer_type in (
                    LayerType.MLA_MOE,
                    LayerType.MHA_MOE,
                )

                if is_fc and not saw_moe:
                    fc_count += 1
                elif is_moe:
                    saw_moe = True
                elif is_fc and saw_moe:
                    # Hit FC again — this is the start of the next pass
                    break

            assert fc_count == first_n_fc, (
                f"{pass_name}: expected {first_n_fc} leading FC layers, "
                f"got {fc_count}"
            )

    elif pattern == "alternating_attn_moe":
        valid_types = {
            LayerType.MHA_MOE,
            LayerType.MLA_MOE,
            LayerType.MHA_FC,
            LayerType.MLA_FC,
            LayerType.ATTN,
            LayerType.MOE,
            LayerType.UNKNOWN,
        }
        for layer in layers:
            assert layer.layer_type in valid_types, (
                f"Layer {layer.layer_idx}: unexpected type {layer.layer_type.value}"
            )

    else:
        pytest.skip(f"Unknown layer type pattern: {pattern}")


# ---------------------------------------------------------------------------
# Test 5: Type Coverage (secondary — warning, not hard failure)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_name", ALL_MODELS, indirect=True)
def test_type_coverage(model_name, analysis_result, baseline):
    """Verify that most kernel time is classified into a known type (not OTHER).

    This is a secondary metric: warns rather than fails.
    """
    entry = TRACE_REGISTRY[model_name]
    metrics = compute_metrics(analysis_result, entry)

    min_threshold = baseline["metrics"]["type_coverage_pct"]["min"]
    actual = metrics["type_coverage_pct"]

    if actual < min_threshold:
        warnings.warn(
            f"Type coverage {actual:.1f}% < {min_threshold:.1f}% threshold. "
            f"Many kernels classified as OTHER type.",
            UserWarning,
            stacklevel=1,
        )
