"""Tests for gradient_noise.py (feature 41)."""

from __future__ import annotations



from cola_coder.features.gradient_noise import (
    FEATURE_ENABLED,
    GradientNoiseEstimator,
    GradientNoiseReport,
    estimate_noise_scale,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


def test_feature_enabled():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


# ---------------------------------------------------------------------------
# GradientNoiseEstimator
# ---------------------------------------------------------------------------


def test_no_estimate_before_min_samples():
    est = GradientNoiseEstimator(window=100, min_samples=20)
    for i in range(19):
        est.record(1.0, step=i)
    assert est.estimate() is None


def test_estimate_after_min_samples():
    est = GradientNoiseEstimator(window=100, min_samples=20)
    for i in range(30):
        est.record(1.0 + i * 0.01, step=i)
    report = est.estimate(step=29)
    assert report is not None
    assert isinstance(report, GradientNoiseReport)


def test_low_noise_high_snr():
    """Constant gradient norms → very low noise, high SNR."""
    norms = [1.0] * 50
    report = estimate_noise_scale(norms, target_snr=10.0)
    assert report is not None
    # Variance should be near zero
    assert report.grad_variance < 1e-6
    # SNR should be very high
    assert report.snr > 1000.0


def test_high_noise_low_snr():
    """Highly variable gradient norms → high noise, low SNR."""
    import random

    random.seed(42)
    norms = [random.uniform(0.01, 10.0) for _ in range(100)]
    report = estimate_noise_scale(norms, target_snr=10.0)
    assert report is not None
    assert report.snr < 10.0  # noisy gradients
    assert report.grad_variance > 0.0


def test_batch_multiplier_clamped():
    """Batch multiplier must be within [0.1, 10.0]."""
    norms = [0.001] * 5 + [100.0] * 95  # very noisy
    report = estimate_noise_scale(norms, target_snr=10.0)
    assert report is not None
    assert 0.1 <= report.recommended_batch_multiplier <= 10.0


def test_is_batch_too_small():
    # Simulate very noisy gradients → batch multiplier > 1.5
    import random

    random.seed(0)
    norms = [random.uniform(0.1, 5.0) for _ in range(100)]
    report = estimate_noise_scale(norms, target_snr=100.0)
    assert report is not None
    # With very high target SNR, multiplier > 1.5
    if report.recommended_batch_multiplier > 1.5:
        assert report.is_batch_too_small(threshold=1.5)


def test_is_batch_too_large():
    # Very stable gradients → SNR >> target → multiplier < 0.5
    norms = [1.0001] * 100  # almost constant
    report = estimate_noise_scale(norms, target_snr=0.001)
    assert report is not None
    assert report.is_batch_too_large(threshold=0.5)


def test_nan_grad_ignored():
    est = GradientNoiseEstimator(window=50, min_samples=5)
    for i in range(20):
        est.record(1.0, step=i)
    est.record(float("nan"), step=20)
    assert est.num_samples == 20  # nan not added


def test_inf_grad_ignored():
    est = GradientNoiseEstimator(window=50, min_samples=5)
    for i in range(20):
        est.record(1.0, step=i)
    est.record(float("inf"), step=20)
    assert est.num_samples == 20


def test_window_limits_samples():
    est = GradientNoiseEstimator(window=10, min_samples=5)
    for i in range(100):
        est.record(float(i), step=i)
    assert est.num_samples == 10  # window caps at 10


def test_reset_clears_state():
    est = GradientNoiseEstimator(window=50, min_samples=5)
    for i in range(30):
        est.record(1.0, step=i)
    est.reset()
    assert est.num_samples == 0
    assert est.estimate() is None


def test_report_summary_format():
    report = GradientNoiseReport(
        noise_scale=0.5,
        grad_variance=0.25,
        grad_mean=1.0,
        snr=4.0,
        recommended_batch_multiplier=1.58,
        step=100,
    )
    s = report.summary()
    assert "step=100" in s
    assert "noise_scale=" in s
    assert "snr=" in s
    assert "batch_mult=" in s


def test_step_auto_increments():
    est = GradientNoiseEstimator(window=50, min_samples=5)
    for _ in range(25):
        est.record(1.0)  # no explicit step
    report = est.estimate()
    assert report is not None
    # step should equal number of records (auto-incremented each record call)
    assert report.step == 25
