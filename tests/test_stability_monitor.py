"""Tests for features/stability_monitor.py — Feature 94.

All tests are CPU-only, no model weights, no I/O.
"""

from __future__ import annotations


import pytest

from cola_coder.features.stability_monitor import (
    FEATURE_ENABLED,
    AlertSeverity,
    StabilityAlert,
    StabilitySnapshot,
    StabilityThresholds,
    TrainingStabilityMonitor,
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mon():
    return TrainingStabilityMonitor()


def _fill(mon, losses, grad_norms=None, lrs=None):
    """Helper to bulk-record steps."""
    for i, loss in enumerate(losses):
        gn = grad_norms[i] if grad_norms else 1.0
        lr = lrs[i] if lrs else 1e-4
        mon.record(step=i, loss=loss, grad_norm=gn, lr=lr)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def test_n_steps_zero_initially(mon):
    assert mon.n_steps == 0


def test_record_increments_steps(mon):
    mon.record(0, loss=2.0, grad_norm=1.0)
    assert mon.n_steps == 1


def test_latest_loss(mon):
    mon.record(0, loss=3.5, grad_norm=1.0)
    assert mon.latest_loss == pytest.approx(3.5)


def test_latest_grad_norm(mon):
    mon.record(0, loss=2.0, grad_norm=0.5)
    assert mon.latest_grad_norm == pytest.approx(0.5)


def test_clear_resets_state(mon):
    mon.record(0, loss=2.0, grad_norm=1.0)
    mon.clear()
    assert mon.n_steps == 0
    assert mon.latest_loss is None


# ---------------------------------------------------------------------------
# Grad norm variance
# ---------------------------------------------------------------------------


def test_grad_norm_variance_none_on_single_step(mon):
    mon.record(0, loss=2.0, grad_norm=1.0)
    assert mon.grad_norm_variance() is None


def test_grad_norm_variance_computed(mon):
    _fill(mon, [2.0] * 5, grad_norms=[1.0, 10.0, 1.0, 10.0, 1.0])
    var = mon.grad_norm_variance()
    assert var is not None and var > 0


def test_grad_norm_variance_window(mon):
    _fill(mon, [2.0] * 10, grad_norms=[1.0] * 8 + [100.0, 200.0])
    var_full = mon.grad_norm_variance()
    var_window = mon.grad_norm_variance(window=2)
    # Window-2 variance should be different from full-history variance
    assert var_window != var_full


# ---------------------------------------------------------------------------
# Loss oscillation
# ---------------------------------------------------------------------------


def test_loss_oscillation_none_on_single(mon):
    mon.record(0, loss=2.0, grad_norm=1.0)
    assert mon.loss_oscillation() is None


def test_loss_oscillation_stable(mon):
    _fill(mon, [2.0, 2.0, 2.0, 2.0, 2.0])
    lo = mon.loss_oscillation()
    assert lo == pytest.approx(0.0)


def test_loss_oscillation_high_on_zigzag(mon):
    _fill(mon, [1.0, 5.0, 1.0, 5.0, 1.0])
    lo = mon.loss_oscillation()
    assert lo is not None and lo > 1.0


# ---------------------------------------------------------------------------
# NaN / Inf detection
# ---------------------------------------------------------------------------


def test_no_nan_inf_on_clean(mon):
    _fill(mon, [2.0, 1.9, 1.8])
    assert mon.has_nan_inf() is False


def test_nan_loss_detected(mon):
    mon.record(0, loss=float("nan"), grad_norm=1.0)
    assert mon.has_nan_inf() is True


def test_inf_grad_norm_detected(mon):
    mon.record(0, loss=2.0, grad_norm=float("inf"))
    assert mon.has_nan_inf() is True


# ---------------------------------------------------------------------------
# assess()
# ---------------------------------------------------------------------------


def test_assess_returns_snapshot(mon):
    _fill(mon, [2.0, 2.0, 2.0])
    snap = mon.assess()
    assert isinstance(snap, StabilitySnapshot)


def test_assess_stable_on_clean(mon):
    _fill(mon, [2.0, 1.9, 1.85, 1.8, 1.75])
    snap = mon.assess()
    assert snap.is_stable is True


def test_assess_critical_on_nan(mon):
    mon.record(0, loss=float("nan"), grad_norm=1.0)
    snap = mon.assess()
    assert snap.is_stable is False
    severities = [a.severity for a in snap.alerts]
    assert AlertSeverity.CRITICAL in severities


def test_assess_warning_on_high_oscillation(mon):
    t = StabilityThresholds(loss_oscillation_warning=0.3)
    mon2 = TrainingStabilityMonitor(thresholds=t)
    _fill(mon2, [1.0, 5.0, 1.0, 5.0, 1.0])
    snap = mon2.assess()
    signals = [a.signal for a in snap.alerts]
    assert "loss_oscillation" in signals


def test_assess_as_dict(mon):
    _fill(mon, [2.0, 1.9, 1.8])
    d = mon.assess().as_dict()
    assert "step" in d
    assert "is_stable" in d


# ---------------------------------------------------------------------------
# StabilityAlert __str__
# ---------------------------------------------------------------------------


def test_alert_str_contains_signal():
    alert = StabilityAlert(
        severity=AlertSeverity.WARNING,
        signal="loss_oscillation",
        message="Elevated oscillation",
        value=0.8,
        threshold=0.5,
    )
    s = str(alert)
    assert "loss_oscillation" in s
    assert "WARNING" in s
