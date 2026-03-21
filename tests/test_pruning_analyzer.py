"""Tests for pruning_analyzer.py."""

import pytest

from cola_coder.features.pruning_analyzer import PruningAnalyzer, PruningReport


@pytest.fixture
def analyzer():
    return PruningAnalyzer(dead_neuron_threshold=1e-3, low_head_threshold=1e-2)


def _make_dense_weight(rows: int, cols: int) -> list[list[float]]:
    """Create a weight matrix with non-zero values."""
    return [[float(i * cols + j + 1) * 0.01 for j in range(cols)] for i in range(rows)]


def _make_zero_row_weight(rows: int, cols: int, zero_row: int) -> list[list[float]]:
    """Create a weight matrix with one near-zero row."""
    mat = _make_dense_weight(rows, cols)
    mat[zero_row] = [1e-6] * cols
    return mat


def test_feature_enabled():
    from cola_coder.features.pruning_analyzer import FEATURE_ENABLED, is_enabled

    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_empty_state_dict(analyzer):
    report = analyzer.analyze({})
    assert report.total_params == 0
    assert report.total_layers_analyzed == 0
    assert isinstance(report, PruningReport)


def test_dense_weights_no_dead_neurons(analyzer):
    state_dict = {
        "layer.weight": _make_dense_weight(8, 16),
    }
    report = analyzer.analyze(state_dict)
    assert len(report.dead_neurons) == 0


def test_dead_neuron_detected(analyzer):
    state_dict = {
        "layer.weight": _make_zero_row_weight(8, 16, zero_row=3),
    }
    report = analyzer.analyze(state_dict)
    assert any(d.neuron_index == 3 for d in report.dead_neurons)


def test_low_mag_head_detected(analyzer):
    # Use a matrix where head_dim=32 so num_heads=2 (rows=64, rows//32=2)
    rows, cols = 64, 32
    mat = _make_dense_weight(rows, cols)
    # Zero out the entire first head (first 32 rows = head_size when num_heads=2)
    head_size = rows // 2  # 32 rows per head
    for i in range(head_size):
        mat[i] = [1e-6] * cols
    state_dict = {"attn.q_proj.weight": mat}
    report = analyzer.analyze(state_dict)
    assert len(report.low_mag_heads) >= 1


def test_total_params_counted(analyzer):
    state_dict = {
        "layer1.weight": _make_dense_weight(4, 8),  # 32 params
        "layer2.weight": _make_dense_weight(2, 4),  # 8 params
    }
    report = analyzer.analyze(state_dict)
    assert report.total_params == 40


def test_prunable_fraction_in_range(analyzer):
    state_dict = {
        "layer.weight": _make_zero_row_weight(8, 16, zero_row=0),
    }
    report = analyzer.analyze(state_dict)
    assert 0.0 <= report.prunable_param_fraction <= 1.0


def test_estimated_speedup_gte_one(analyzer):
    state_dict = {"layer.weight": _make_dense_weight(4, 4)}
    report = analyzer.analyze(state_dict)
    assert report.estimated_speedup >= 1.0


def test_summary_returns_string(analyzer):
    state_dict = {"layer.weight": _make_dense_weight(4, 4)}
    report = analyzer.analyze(state_dict)
    s = report.summary()
    assert isinstance(s, str)
    assert "params=" in s
