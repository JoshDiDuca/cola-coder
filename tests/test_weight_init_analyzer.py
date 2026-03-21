"""Tests for weight_init_analyzer.py (feature 43)."""

from __future__ import annotations

import math


from cola_coder.features.weight_init_analyzer import (
    FEATURE_ENABLED,
    WeightInitAnalyzer,
    _compute_fan,
    _he_std,
    _xavier_std,
    is_enabled,
    make_test_tensor,
)


def test_feature_enabled():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_fan_computation_2d():
    fan_in, fan_out = _compute_fan((128, 64))
    assert fan_in == 64
    assert fan_out == 128


def test_fan_computation_1d():
    fan_in, fan_out = _compute_fan((32,))
    assert fan_in == 32
    assert fan_out == 32


def test_fan_computation_conv():
    # Conv2d: (out_ch, in_ch, kH, kW)
    fan_in, fan_out = _compute_fan((16, 8, 3, 3))
    assert fan_in == 8 * 9
    assert fan_out == 16 * 9


def test_xavier_std_formula():
    std = _xavier_std(fan_in=64, fan_out=128)
    expected = math.sqrt(2.0 / (64 + 128))
    assert abs(std - expected) < 1e-9


def test_he_std_formula():
    std = _he_std(fan_in=64)
    expected = math.sqrt(2.0 / 64)
    assert abs(std - expected) < 1e-9


def _make_xavier_weights(fan_in: int, fan_out: int, n: int = 200) -> list:
    """Generate values with Xavier std."""
    std = _xavier_std(fan_in, fan_out)
    import random
    rng = random.Random(42)
    return [rng.gauss(0, std) for _ in range(n)]


def test_good_xavier_init_not_suspicious():
    analyzer = WeightInitAnalyzer()
    fan_in, fan_out = 64, 128
    vals = _make_xavier_weights(fan_in, fan_out, n=1000)
    t = make_test_tensor(vals, (fan_out, fan_in))
    report = analyzer.analyze({"layer.weight": t})
    assert report.num_suspicious == 0
    assert report.overall_ok


def test_zero_init_flagged():
    analyzer = WeightInitAnalyzer()
    t = make_test_tensor([0.0] * 100, (10, 10))
    report = analyzer.analyze({"weight": t})
    assert report.num_suspicious > 0
    assert not report.overall_ok


def test_exploding_init_flagged():
    analyzer = WeightInitAnalyzer()
    vals = [50.0, -50.0] * 50
    t = make_test_tensor(vals, (10, 10))
    report = analyzer.analyze({"weight": t})
    assert report.num_suspicious > 0


def test_bias_not_flagged_when_zero():
    """Biases are legitimately zero-initialized — should not be suspicious."""
    analyzer = WeightInitAnalyzer()
    t = make_test_tensor([0.0] * 64, (64,))
    report = analyzer.analyze({"layer.bias": t})
    # Zero bias should NOT be flagged as suspicious
    assert report.num_suspicious == 0


def test_num_zero_init_counts_weights_only():
    analyzer = WeightInitAnalyzer()
    state = {
        "layer.weight": make_test_tensor([0.0] * 100, (10, 10)),
        "layer.bias": make_test_tensor([0.0] * 10, (10,)),  # bias — not counted
    }
    report = analyzer.analyze(state)
    # layer.weight is zero-init weight, layer.bias is bias (excluded from zero_init count)
    assert report.num_zero_init == 1


def test_summary_contains_warning():
    analyzer = WeightInitAnalyzer()
    vals = [100.0, -100.0] * 50
    t = make_test_tensor(vals, (10, 10))
    report = analyzer.analyze({"big_weight": t})
    s = report.summary()
    assert "WARN" in s or "suspicious" in s.lower()


def test_layer_report_summary():
    analyzer = WeightInitAnalyzer()
    t = make_test_tensor(_make_xavier_weights(32, 64, 500), (64, 32))
    report = analyzer.analyze({"fc.weight": t})
    assert report.layers
    s = report.layers[0].summary()
    assert "fc.weight" in s
    assert "std=" in s


def test_empty_state_dict():
    analyzer = WeightInitAnalyzer()
    report = analyzer.analyze({})
    assert report.layers == []
    assert report.overall_ok


def test_multiple_layers():
    analyzer = WeightInitAnalyzer()
    state = {
        "l1.weight": make_test_tensor(_make_xavier_weights(64, 128, 1000), (128, 64)),
        "l1.bias": make_test_tensor([0.0] * 128, (128,)),
        "l2.weight": make_test_tensor(_make_xavier_weights(128, 64, 1000), (64, 128)),
    }
    report = analyzer.analyze(state)
    assert len(report.layers) == 3
    assert report.overall_ok
