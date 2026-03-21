"""Tests for training_anomaly_detector.py (feature 60)."""

import pytest

from cola_coder.features.training_anomaly_detector import (
    FEATURE_ENABLED,
    AnomalyType,
    DetectorConfig,
    TrainingAnomalyDetector,
    _rolling_stats,
    _z_score,
    is_enabled,
)


@pytest.fixture
def detector():
    cfg = DetectorConfig(
        window_size=10,
        z_score_threshold=3.0,
        grad_norm_threshold=50.0,
        lr_jump_factor=5.0,
        stagnation_steps=10,
    )
    return TrainingAnomalyDetector(config=cfg)


def _feed_normal_loss(detector, n=15, base=2.0, step_offset=0):
    """Feed stable decreasing loss to warm up the rolling window."""
    for i in range(n):
        detector.update(step_offset + i, {"loss": base - i * 0.01})


def test_feature_flag():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_rolling_stats_basic():
    from collections import deque
    w = deque([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, std = _rolling_stats(w)
    assert mean == pytest.approx(3.0)
    assert std > 0


def test_rolling_stats_empty():
    from collections import deque
    mean, std = _rolling_stats(deque())
    assert mean == 0.0
    assert std > 0


def test_z_score_basic():
    z = _z_score(10.0, 5.0, 2.0)
    assert z == pytest.approx(2.5)


def test_no_anomaly_stable_loss(detector):
    _feed_normal_loss(detector, n=20)
    assert len(detector.all_anomalies) == 0


def test_nan_detected(detector):
    anomalies = detector.update(0, {"loss": float("nan")})
    assert len(anomalies) == 1
    assert anomalies[0].anomaly_type == AnomalyType.NAN_INF
    assert anomalies[0].severity == "critical"


def test_inf_detected(detector):
    anomalies = detector.update(0, {"loss": float("inf")})
    assert len(anomalies) == 1
    assert anomalies[0].anomaly_type == AnomalyType.NAN_INF


def test_loss_spike_detected(detector):
    _feed_normal_loss(detector, n=15)
    # Inject a big spike
    anomalies = detector.update(15, {"loss": 100.0})
    types = [a.anomaly_type for a in anomalies]
    assert AnomalyType.LOSS_SPIKE in types


def test_gradient_explosion_absolute(detector):
    # grad_norm_threshold = 50.0
    anomalies = detector.update(0, {"grad_norm": 200.0})
    types = [a.anomaly_type for a in anomalies]
    assert AnomalyType.GRADIENT_EXPLOSION in types
    assert any(a.severity == "critical" for a in anomalies)


def test_gradient_explosion_z_score(detector):
    # Feed 15 normal grad norms then spike
    for i in range(15):
        detector.update(i, {"grad_norm": 1.0})
    anomalies = detector.update(15, {"grad_norm": 500.0})
    types = [a.anomaly_type for a in anomalies]
    assert AnomalyType.GRADIENT_EXPLOSION in types


def test_lr_jump_detected(detector):
    detector.update(0, {"learning_rate": 0.001})
    # Big jump beyond lr_jump_factor=5
    anomalies = detector.update(1, {"learning_rate": 0.1})
    types = [a.anomaly_type for a in anomalies]
    assert AnomalyType.LR_ANOMALY in types


def test_lr_drop_detected(detector):
    detector.update(0, {"learning_rate": 0.01})
    anomalies = detector.update(1, {"learning_rate": 0.0001})
    types = [a.anomaly_type for a in anomalies]
    assert AnomalyType.LR_ANOMALY in types


def test_stagnation_detected(detector):
    # Feed loss stuck at 2.0 for stagnation_steps=10 steps
    # First warm up with a window
    for i in range(10):
        detector.update(i, {"loss": 2.0})
    # Now keep feeding the same value
    for i in range(10, 20):
        detector.update(i, {"loss": 2.0})
    stag = detector.get_anomalies_by_type(AnomalyType.STAGNATION)
    assert len(stag) >= 1


def test_no_stagnation_improving_loss(detector):
    for i in range(25):
        detector.update(i, {"loss": 2.0 - i * 0.1})
    stag = detector.get_anomalies_by_type(AnomalyType.STAGNATION)
    assert len(stag) == 0


def test_summary_counts(detector):
    detector.update(0, {"loss": float("nan")})
    detector.update(1, {"grad_norm": 200.0})
    s = detector.summary()
    assert s["total_anomalies"] >= 2
    assert s["critical_count"] >= 1


def test_reset(detector):
    detector.update(0, {"loss": float("nan")})
    assert len(detector.all_anomalies) > 0
    detector.reset()
    assert len(detector.all_anomalies) == 0
    assert detector._states == {}


def test_get_anomalies_by_type(detector):
    detector.update(0, {"loss": float("nan")})
    detector.update(1, {"grad_norm": 999.0})
    nan_anoms = detector.get_anomalies_by_type(AnomalyType.NAN_INF)
    assert len(nan_anoms) >= 1


def test_get_anomalies_for_metric(detector):
    detector.update(0, {"loss": float("nan")})
    detector.update(1, {"val_loss": float("inf")})
    loss_anoms = detector.get_anomalies_for_metric("loss")
    assert all(a.metric == "loss" for a in loss_anoms)


def test_anomaly_as_dict(detector):
    anomalies = detector.update(0, {"loss": float("nan")})
    d = anomalies[0].as_dict()
    assert "step" in d
    assert "metric" in d
    assert "type" in d
    assert "severity" in d


def test_multiple_metrics_per_step(detector):
    anomalies = detector.update(0, {
        "loss": float("nan"),
        "val_loss": float("inf"),
        "grad_norm": 1000.0,
    })
    assert len(anomalies) >= 3
