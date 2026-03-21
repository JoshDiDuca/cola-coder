"""Tests for lr_range_test.py (feature 42)."""

from __future__ import annotations

import math

import pytest

from cola_coder.features.lr_range_test import (
    FEATURE_ENABLED,
    LRRangeResult,
    LRRangeTest,
    find_optimal_lr,
    is_enabled,
)


def test_feature_enabled():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_schedule_is_log_linear():
    test = LRRangeTest(min_lr=1e-6, max_lr=1.0, num_steps=10)
    lrs = test._schedule
    assert len(lrs) == 10
    assert abs(lrs[0] - 1e-6) < 1e-9
    assert abs(lrs[-1] - 1.0) < 1e-9
    # Check log-linear: ratios between consecutive steps should be constant
    ratios = [math.log(lrs[i + 1] / lrs[i]) for i in range(len(lrs) - 1)]
    assert max(ratios) - min(ratios) < 1e-9


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        LRRangeTest(min_lr=-1e-4, max_lr=1.0)
    with pytest.raises(ValueError):
        LRRangeTest(min_lr=1.0, max_lr=0.1)  # max < min
    with pytest.raises(ValueError):
        LRRangeTest(min_lr=1e-7, max_lr=1.0, num_steps=1)


def test_simulation_with_parabolic_loss():
    """Parabolic loss: minimum at lr=0.01."""

    def loss_fn(lr):
        return (math.log10(lr) + 2) ** 2 + 0.5

    test = LRRangeTest(min_lr=1e-5, max_lr=1.0, num_steps=200)
    result = test.run(loss_fn)
    assert result.is_valid()
    assert result.suggested_max_lr is not None
    assert result.suggested_min_lr is not None
    assert result.suggested_min_lr < result.suggested_max_lr


def test_simulation_records_all_steps():
    def flat_loss(lr):
        return 1.0

    test = LRRangeTest(min_lr=1e-6, max_lr=1.0, num_steps=50)
    result = test.run(flat_loss)
    # With constant loss there's no divergence, so all 50 steps complete
    assert result.num_steps == 50


def test_divergence_stops_early():
    """Loss that immediately explodes should stop the sweep early."""

    def exploding_loss(lr):
        if lr > 0.01:
            return 1e9
        return 1.0

    test = LRRangeTest(min_lr=1e-7, max_lr=1.0, num_steps=200, diverge_threshold=5.0)
    result = test.run(exploding_loss)
    # Sweep should stop before all 200 steps
    assert len(result.lr_values) < 200


def test_nan_loss_triggers_diverge():
    test = LRRangeTest(min_lr=1e-7, max_lr=1.0, num_steps=20)
    for _ in range(5):
        test.next_lr()  # advance schedule (return value not needed here)
        test.record_loss(1.0)
    test.record_loss(float("nan"))
    assert test.has_diverged()


def test_find_optimal_lr_returns_tuple():
    def loss_fn(lr):
        return (math.log10(lr) + 3) ** 2

    lo, hi = find_optimal_lr(loss_fn, min_lr=1e-7, max_lr=1.0, num_steps=100)
    assert lo is not None
    assert hi is not None
    assert lo < hi


def test_result_summary_valid():
    result = LRRangeResult(
        lr_values=[1e-5, 1e-4, 1e-3],
        losses=[2.0, 1.0, 3.0],
        suggested_min_lr=1e-5,
        suggested_max_lr=1e-4,
        num_steps=3,
    )
    assert result.is_valid()
    s = result.summary()
    assert "min_lr" in s
    assert "max_lr" in s


def test_result_invalid_when_empty():
    result = LRRangeResult()
    assert not result.is_valid()
    assert "insufficient" in result.summary()


def test_iterator_mode():
    """Test the step-by-step iterator API."""
    test = LRRangeTest(min_lr=1e-6, max_lr=1.0, num_steps=30)
    for _ in range(30):
        lr = test.next_lr()
        assert lr > 0
        test.record_loss(1.0 + lr)
    result = test.finish()
    assert len(result.lr_values) == 30


def test_monotone_decreasing_loss_no_diverge():
    """If loss always decreases, no divergence should be flagged."""
    test = LRRangeTest(min_lr=1e-7, max_lr=0.1, num_steps=50, diverge_threshold=5.0)
    for i in range(50):
        test.record_loss(5.0 - i * 0.09)
    assert not test.has_diverged()
