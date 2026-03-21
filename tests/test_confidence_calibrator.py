"""Tests for features/confidence_calibrator.py — Feature 92.

All tests are CPU-only, no model weights, no I/O.
"""

from __future__ import annotations

import math

import pytest

from cola_coder.features.confidence_calibrator import (
    FEATURE_ENABLED,
    CalibrationBin,
    CalibrationResult,
    ConfidenceCalibrator,
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
def cal():
    return ConfidenceCalibrator(temperature=1.0, n_bins=10)


# ---------------------------------------------------------------------------
# Softmax / probability computation
# ---------------------------------------------------------------------------


def test_logits_sum_to_one(cal):
    logits = [1.0, 2.0, 0.5, -1.0]
    probs = cal.logits_to_probs(logits)
    assert abs(sum(probs) - 1.0) < 1e-9


def test_higher_logit_gets_higher_prob(cal):
    probs = cal.logits_to_probs([0.0, 5.0])
    assert probs[1] > probs[0]


def test_temperature_scaling_sharpens(cal):
    logits = [1.0, 2.0]
    probs_hot = cal.logits_to_probs(logits)
    cal.temperature = 0.1
    probs_cold = cal.logits_to_probs(logits)
    # Lower T → more peaked distribution
    assert probs_cold[1] > probs_hot[1]


def test_log_probs_are_negative(cal):
    log_probs = cal.logits_to_log_probs([1.0, 2.0])
    assert all(lp <= 0.0 for lp in log_probs)


def test_log_probs_consistent_with_probs(cal):
    logits = [0.5, 1.5, -0.5]
    probs = cal.logits_to_probs(logits)
    log_probs = cal.logits_to_log_probs(logits)
    for p, lp in zip(probs, log_probs):
        assert abs(math.exp(lp) - p) < 1e-9


def test_top_confidence_returns_argmax(cal):
    idx, conf = cal.top_confidence([0.0, 5.0, 1.0])
    assert idx == 1
    assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------


def test_ece_empty_input(cal):
    result = cal.compute_ece([], [])
    assert result.ece == 0.0
    assert result.n_samples == 0


def test_ece_returns_calibration_result(cal):
    confs = [0.9] * 10
    correct = [1.0] * 10
    result = cal.compute_ece(confs, correct)
    assert isinstance(result, CalibrationResult)


def test_ece_perfect_calibration(cal):
    # 10 samples all with confidence 0.9 and all correct → ECE ≈ 0.1
    confs = [0.9] * 10
    correct = [1.0] * 10
    result = cal.compute_ece(confs, correct)
    # avg_confidence=0.9, accuracy=1.0 → gap=0.1
    assert abs(result.ece - 0.1) < 0.01


def test_ece_always_non_negative(cal):
    import random
    rng = random.Random(42)
    confs = [rng.random() for _ in range(50)]
    correct = [float(rng.random() > 0.5) for _ in range(50)]
    result = cal.compute_ece(confs, correct)
    assert result.ece >= 0.0


def test_ece_length_mismatch_raises(cal):
    with pytest.raises(ValueError):
        cal.compute_ece([0.5, 0.6], [1.0])


def test_n_bins_matches(cal):
    confs = [i / 10 for i in range(10)]
    correct = [1.0] * 10
    result = cal.compute_ece(confs, correct)
    assert result.n_bins == 10
    assert len(result.bins) == 10


# ---------------------------------------------------------------------------
# CalibrationBin properties
# ---------------------------------------------------------------------------


def test_bin_avg_confidence_empty():
    b = CalibrationBin(lower=0.0, upper=0.1)
    assert b.avg_confidence == 0.0


def test_bin_accuracy_empty():
    b = CalibrationBin(lower=0.0, upper=0.1)
    assert b.accuracy == 0.0


def test_bin_calibration_gap():
    b = CalibrationBin(lower=0.8, upper=0.9, count=2, total_confidence=1.8, total_correct=2.0)
    # avg_confidence=0.9, accuracy=1.0 → gap=0.1
    assert abs(b.calibration_gap - 0.1) < 1e-9


# ---------------------------------------------------------------------------
# Temperature search
# ---------------------------------------------------------------------------


def test_find_temperature_returns_float(cal):
    confs = [0.9] * 20
    correct = [1.0 if i < 15 else 0.0 for i in range(20)]
    t = cal.find_temperature(confs, correct, temps=[0.5, 1.0, 2.0])
    assert isinstance(t, float)
    assert t > 0


def test_find_temperature_does_not_mutate_self(cal):
    original_t = cal.temperature
    cal.find_temperature([0.7] * 5, [1.0] * 5, temps=[0.5, 1.5])
    assert cal.temperature == original_t


# ---------------------------------------------------------------------------
# Reliability diagram data
# ---------------------------------------------------------------------------


def test_reliability_diagram_keys(cal):
    confs = [0.3, 0.7, 0.9]
    correct = [0.0, 1.0, 1.0]
    data = cal.reliability_diagram_data(confs, correct)
    assert set(data.keys()) == {"bin_centers", "accuracies", "avg_confidences", "counts"}


def test_reliability_diagram_lengths(cal):
    confs = [0.2, 0.8]
    correct = [1.0, 1.0]
    data = cal.reliability_diagram_data(confs, correct)
    assert len(data["bin_centers"]) == cal.n_bins


# ---------------------------------------------------------------------------
# CalibrationResult summary
# ---------------------------------------------------------------------------


def test_summary_has_ece():
    r = CalibrationResult(
        ece=0.05,
        temperature=1.0,
        n_bins=10,
        bins=[],
        n_samples=100,
        overconfidence=0.3,
        underconfidence=0.2,
    )
    s = r.summary()
    assert "ece" in s
    assert s["ece"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Invalid constructor args
# ---------------------------------------------------------------------------


def test_invalid_temperature_raises():
    with pytest.raises(ValueError):
        ConfidenceCalibrator(temperature=0.0)


def test_invalid_bins_raises():
    with pytest.raises(ValueError):
        ConfidenceCalibrator(n_bins=1)
