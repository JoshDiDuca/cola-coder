"""Tests for progress_estimator.py."""

from __future__ import annotations

import math

from cola_coder.features.progress_estimator import (
    FEATURE_ENABLED,
    EstimationResult,
    ProgressEstimator,
    _exp_decay,
    fit_exponential_decay,
    is_enabled,
    lr_adjustment_factor,
)


class TestIsEnabled:
    def test_constant(self):
        assert FEATURE_ENABLED is True

    def test_function(self):
        assert is_enabled() is True


class TestExpDecay:
    def test_at_zero(self):
        # a*exp(0) + c = a + c
        assert abs(_exp_decay(0, 3.0, 0.01, 1.0) - 4.0) < 1e-6

    def test_decreasing_with_positive_b(self):
        v0 = _exp_decay(0, 2.0, 0.01, 0.0)
        v1 = _exp_decay(100, 2.0, 0.01, 0.0)
        assert v0 > v1

    def test_asymptote(self):
        large_step = 1_000_000
        v = _exp_decay(large_step, 2.0, 0.01, 1.0)
        assert abs(v - 1.0) < 0.01


class TestFitExponentialDecay:
    def _make_data(self, a=2.0, b=0.001, c=1.0, n=20):
        steps = [i * 100 for i in range(n)]
        losses = [_exp_decay(s, a, b, c) for s in steps]
        return steps, losses

    def test_returns_three_params(self):
        steps, losses = self._make_data()
        params = fit_exponential_decay(steps, losses)
        assert params is not None
        assert len(params) == 3

    def test_fits_clean_data(self):
        steps, losses = self._make_data(a=3.0, b=0.002, c=0.5)
        params = fit_exponential_decay(steps, losses)
        assert params is not None
        a, b, c = params
        # Check predictions are close
        for s, loss_val in zip(steps[:5], losses[:5]):
            pred = _exp_decay(s, a, b, c)
            assert abs(pred - loss_val) < 0.5  # tolerant fit

    def test_too_few_points_returns_none(self):
        result = fit_exponential_decay([0, 100], [3.0, 2.0])
        assert result is None

    def test_b_positive(self):
        steps, losses = self._make_data()
        params = fit_exponential_decay(steps, losses)
        assert params is not None
        assert params[1] > 0


class TestLRAdjustmentFactor:
    def test_warmup_linear(self):
        f = lr_adjustment_factor(50, warmup_steps=100, total_steps=1000)
        assert abs(f - 0.5) < 0.01

    def test_after_warmup_decreases(self):
        f0 = lr_adjustment_factor(100, warmup_steps=100, total_steps=1000)
        f1 = lr_adjustment_factor(500, warmup_steps=100, total_steps=1000)
        assert f0 >= f1

    def test_cosine_schedule_bounded(self):
        for step in [0, 100, 500, 999]:
            f = lr_adjustment_factor(step, warmup_steps=0, total_steps=1000, schedule="cosine")
            assert 0.0 <= f <= 1.0


class TestProgressEstimator:
    def _make_estimator_with_data(self, target=1.5):
        est = ProgressEstimator(target_loss=target, total_steps=10000)
        # Simulate decaying loss
        for i in range(20):
            step = i * 100
            loss = 2.0 * math.exp(-0.001 * step) + 1.0
            est.record(step, loss)
        return est

    def test_returns_estimation_result(self):
        est = self._make_estimator_with_data()
        result = est.estimate()
        assert isinstance(result, EstimationResult)

    def test_too_few_observations_returns_none(self):
        est = ProgressEstimator(target_loss=1.0)
        est.record(0, 3.0)
        est.record(100, 2.5)
        result = est.estimate()
        assert result is None

    def test_converged_flag_set(self):
        est = ProgressEstimator(target_loss=5.0)  # target above all losses
        for i in range(5):
            est.record(i * 100, 3.0)
        result = est.estimate()
        assert result is not None
        assert result.is_converged is True

    def test_predicted_step_positive(self):
        est = self._make_estimator_with_data(target=1.2)
        result = est.estimate()
        assert result is not None
        if result.predicted_step is not None:
            assert result.predicted_step > 0

    def test_predict_loss_at_future_step(self):
        est = self._make_estimator_with_data()
        pred = est.predict_loss_at(5000)
        assert pred is None or isinstance(pred, float)

    def test_explicit_steps_losses(self):
        est = ProgressEstimator(target_loss=1.5)
        steps = [i * 100 for i in range(10)]
        losses = [3.0 * math.exp(-0.005 * s) + 1.0 for s in steps]
        result = est.estimate(steps=steps, losses=losses)
        assert result is not None
        assert isinstance(result, EstimationResult)

    def test_clear_removes_observations(self):
        est = self._make_estimator_with_data()
        est.clear()
        assert len(est.observations) == 0

    def test_summary_str(self):
        est = self._make_estimator_with_data()
        result = est.estimate()
        assert result is not None
        s = result.summary()
        assert "EstimationResult" in s

    def test_confidence_in_range(self):
        est = self._make_estimator_with_data()
        result = est.estimate()
        assert result is not None
        assert 0.0 <= result.confidence <= 1.0
