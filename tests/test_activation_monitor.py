"""Tests for activation_monitor.py (feature 44)."""

from __future__ import annotations


from cola_coder.features.activation_monitor import (
    FEATURE_ENABLED,
    ActivationMonitor,
    _compute_stats,
    is_enabled,
)


def test_feature_enabled():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_basic_record_and_report():
    mon = ActivationMonitor()
    mon.record("layer1", [0.5, 1.0, -0.5, 2.0])
    report = mon.report(step=1)
    assert "layer1" in report.layer_stats
    stats = report.layer_stats["layer1"]
    assert stats.count == 4
    assert abs(stats.mean - 0.75) < 1e-9


def test_dead_relu_detection():
    mon = ActivationMonitor(dead_relu_threshold=0.5)
    # 80% zeros → dead
    vals = [0.0] * 80 + [1.0] * 20
    mon.record("relu1", vals)
    report = mon.report()
    assert report.layer_stats["relu1"].is_dead
    assert "relu1" in report.dead_layers()


def test_exploding_activation_detection():
    mon = ActivationMonitor(explode_threshold=100.0)
    vals = [200.0, 300.0, 250.0, 180.0]
    mon.record("layer2", vals)
    report = mon.report()
    assert report.layer_stats["layer2"].is_exploding
    assert "layer2" in report.exploding_layers()


def test_vanishing_activation_detection():
    # Use dead_zero_eps=1e-9 so values around 1e-6 don't trigger dead detection
    mon = ActivationMonitor(vanish_threshold=1e-4, dead_zero_eps=1e-9)
    vals = [1e-6, 2e-6, 5e-7]
    mon.record("layerV", vals)
    report = mon.report()
    assert report.layer_stats["layerV"].is_vanishing
    assert "layerV" in report.vanishing_layers()


def test_healthy_layer_has_no_issues():
    mon = ActivationMonitor()
    vals = [i * 0.1 - 0.5 for i in range(100)]
    mon.record("fc", vals)
    report = mon.report()
    assert not report.layer_stats["fc"].is_dead
    assert not report.layer_stats["fc"].is_exploding
    assert not report.layer_stats["fc"].is_vanishing
    assert not report.has_issues()


def test_percentiles_ordered():
    mon = ActivationMonitor()
    mon.record("l", list(range(100)))
    report = mon.report()
    s = report.layer_stats["l"]
    assert s.p5 < s.p25 < s.p50 < s.p75 < s.p95


def test_window_limits_buffered_snapshots():
    mon = ActivationMonitor(window=3)
    for i in range(10):
        mon.record("l", [float(i)])
    # Only last 3 snapshots retained: [7, 8, 9] → sum=24, mean=8.0
    report = mon.report()
    assert abs(report.layer_stats["l"].mean - 8.0) < 0.5


def test_reset_all():
    mon = ActivationMonitor()
    mon.record("a", [1.0, 2.0])
    mon.record("b", [3.0, 4.0])
    mon.reset()
    assert mon.tracked_layers == []


def test_reset_single_layer():
    mon = ActivationMonitor()
    mon.record("a", [1.0])
    mon.record("b", [2.0])
    mon.reset("a")
    assert "a" not in mon.tracked_layers
    assert "b" in mon.tracked_layers


def test_multiple_layers_tracked():
    mon = ActivationMonitor()
    mon.record("l1", [1.0, 2.0])
    mon.record("l2", [3.0, 4.0])
    report = mon.report(step=5)
    assert len(report.layer_stats) == 2
    assert report.step == 5


def test_summary_contains_warn_for_issues():
    mon = ActivationMonitor(explode_threshold=10.0)
    mon.record("big", [100.0, 200.0])
    report = mon.report()
    s = report.summary()
    assert "WARN" in s or "EXPLODING" in s


def test_empty_layer_values():
    stats = _compute_stats(
        "empty", [], dead_relu_threshold=0.5,
        explode_threshold=100.0, vanish_threshold=1e-4, dead_zero_eps=1e-6
    )
    assert stats.count == 0
    assert not stats.is_dead
    assert not stats.is_exploding


def test_stats_summary_format():
    mon = ActivationMonitor()
    mon.record("fc", [0.1, 0.2, 0.3])
    report = mon.report()
    s = report.layer_stats["fc"].summary()
    assert "fc" in s
    assert "mean=" in s
    assert "std=" in s
