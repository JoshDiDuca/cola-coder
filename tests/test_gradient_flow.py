"""Tests for features/gradient_flow.py — Feature 99.

All tests are CPU-only, no model weights, no I/O.
"""

from __future__ import annotations


import pytest

from cola_coder.features.gradient_flow import (
    FEATURE_ENABLED,
    GradientFlowReport,
    GradientFlowTracker,
    GradientHealth,
    LayerGradRecord,
    classify_gradient,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


def test_feature_enabled():
    assert FEATURE_ENABLED is True


def test_is_enabled():
    assert is_enabled() is True


# ---------------------------------------------------------------------------
# classify_gradient
# ---------------------------------------------------------------------------


def test_classify_vanishing():
    assert classify_gradient(1e-10) == GradientHealth.VANISHING


def test_classify_weak():
    assert classify_gradient(1e-5) == GradientHealth.WEAK


def test_classify_healthy():
    assert classify_gradient(1.0) == GradientHealth.HEALTHY


def test_classify_elevated():
    assert classify_gradient(50.0) == GradientHealth.ELEVATED


def test_classify_exploding():
    assert classify_gradient(10000.0) == GradientHealth.EXPLODING


def test_classify_nan_exploding():
    assert classify_gradient(float("nan")) == GradientHealth.EXPLODING


def test_classify_inf_exploding():
    assert classify_gradient(float("inf")) == GradientHealth.EXPLODING


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker():
    return GradientFlowTracker()


def _fill_tracker(tracker, steps=5):
    for i in range(steps):
        tracker.record_step(
            step=i,
            layer_norms={
                "layer1": 1.0 + i * 0.1,
                "layer2": 0.5,
            },
            global_norm=2.0,
        )


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def test_n_steps_zero(tracker):
    assert tracker.n_steps == 0


def test_record_increments_steps(tracker):
    tracker.record_step(0, {"layer1": 1.0})
    assert tracker.n_steps == 1


def test_layer_names_populated(tracker):
    tracker.record_step(0, {"attn": 0.5, "ffn": 0.8})
    assert set(tracker.layer_names) == {"attn", "ffn"}


def test_clear_resets(tracker):
    _fill_tracker(tracker)
    tracker.clear()
    assert tracker.n_steps == 0
    assert tracker.layer_names == []


# ---------------------------------------------------------------------------
# LayerGradRecord
# ---------------------------------------------------------------------------


def test_layer_record_latest(tracker):
    tracker.record_step(0, {"l": 1.0})
    tracker.record_step(1, {"l": 2.0})
    rec = tracker._records["l"]
    assert rec.latest == pytest.approx(2.0)


def test_layer_record_mean(tracker):
    tracker.record_step(0, {"l": 1.0})
    tracker.record_step(1, {"l": 3.0})
    rec = tracker._records["l"]
    assert rec.mean == pytest.approx(2.0)


def test_layer_record_health():
    rec = LayerGradRecord(name="test", norms=[1.0])
    assert rec.health == GradientHealth.HEALTHY


def test_layer_record_health_none_when_empty():
    rec = LayerGradRecord(name="test")
    assert rec.health is None


def test_layer_record_trend_increasing():
    rec = LayerGradRecord(name="test", norms=[0.1, 0.5, 1.0])
    assert rec.trend == "increasing"


def test_layer_record_trend_decreasing():
    rec = LayerGradRecord(name="test", norms=[1.0, 0.5, 0.1])
    assert rec.trend == "decreasing"


def test_layer_record_as_dict():
    rec = LayerGradRecord(name="ffn", norms=[1.0, 1.1])
    d = rec.as_dict()
    assert d["name"] == "ffn"
    assert "health" in d


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_report_returns_report(tracker):
    _fill_tracker(tracker)
    report = tracker.report()
    assert isinstance(report, GradientFlowReport)


def test_report_global_norm(tracker):
    tracker.record_step(0, {"l": 1.0}, global_norm=5.0)
    report = tracker.report()
    assert report.global_norm == pytest.approx(5.0)


def test_report_layer_access(tracker):
    _fill_tracker(tracker)
    report = tracker.report()
    rec = report.layer("layer1")
    assert rec is not None
    assert rec.name == "layer1"


def test_report_is_healthy(tracker):
    tracker.record_step(0, {"l1": 1.0, "l2": 0.5})
    report = tracker.report()
    assert report.is_healthy is True


def test_report_n_vanishing(tracker):
    tracker.record_step(0, {"vanish_layer": 1e-10})
    report = tracker.report()
    assert report.n_vanishing >= 1


def test_report_as_dict(tracker):
    _fill_tracker(tracker)
    d = tracker.report().as_dict()
    assert "step" in d
    assert "is_healthy" in d
    assert "layers" in d


# ---------------------------------------------------------------------------
# Vanishing / exploding detection
# ---------------------------------------------------------------------------


def test_vanishing_layers_detected(tracker):
    tracker.record_step(0, {"ok": 1.0, "bad": 1e-9})
    assert "bad" in tracker.vanishing_layers()
    assert "ok" not in tracker.vanishing_layers()


def test_exploding_layers_detected(tracker):
    tracker.record_step(0, {"ok": 1.0, "boom": 999999.0})
    assert "boom" in tracker.exploding_layers()


# ---------------------------------------------------------------------------
# gradient_history
# ---------------------------------------------------------------------------


def test_gradient_history_returns_list(tracker):
    _fill_tracker(tracker, steps=3)
    hist = tracker.gradient_history("layer1")
    assert isinstance(hist, list)
    assert len(hist) == 3


def test_gradient_history_unknown_layer(tracker):
    hist = tracker.gradient_history("nonexistent")
    assert hist == []


# ---------------------------------------------------------------------------
# diagnostic_data
# ---------------------------------------------------------------------------


def test_diagnostic_data_keys(tracker):
    _fill_tracker(tracker)
    data = tracker.diagnostic_data()
    assert "steps" in data
    assert "global_norms" in data
    assert "layers" in data
