"""Tests for features/cost_estimator.py.

No GPU or model weights required.
"""

from __future__ import annotations

import pytest

from cola_coder.features.cost_estimator import (
    FEATURE_ENABLED,
    CostEstimator,
    CostReport,
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


TINY_CFG = {
    "model": {
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 4,
        "n_kv_heads": 4,
        "vocab_size": 1000,
        "d_ffn": 1024,
    }
}

SMALL_CFG = {
    "model": {
        "d_model": 768,
        "n_layers": 12,
        "n_heads": 12,
        "n_kv_heads": 12,
        "vocab_size": 32_000,
        "d_ffn": 3072,
    }
}


@pytest.fixture
def estimator():
    return CostEstimator()


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------


def test_available_gpus_not_empty(estimator):
    gpus = estimator.available_gpus()
    assert len(gpus) > 0


def test_available_gpus_contains_a100(estimator):
    assert "a100_40gb" in estimator.available_gpus()


def test_available_gpus_contains_h100(estimator):
    assert "h100" in estimator.available_gpus()


def test_unknown_gpu_raises(estimator):
    with pytest.raises(ValueError, match="Unknown GPU"):
        estimator.estimate(TINY_CFG, tokens=1_000_000, gpu_type="rtx_9090")


def test_estimate_returns_cost_report(estimator):
    report = estimator.estimate(TINY_CFG, tokens=1_000_000, gpu_type="a100_40gb")
    assert isinstance(report, CostReport)


def test_estimate_training_hours_positive(estimator):
    report = estimator.estimate(TINY_CFG, tokens=10_000_000, gpu_type="a100_40gb")
    assert report.training_hours > 0


def test_estimate_electricity_cost_positive(estimator):
    report = estimator.estimate(TINY_CFG, tokens=10_000_000, gpu_type="a100_40gb")
    assert report.electricity_cost_usd > 0


def test_estimate_cloud_prices_populated(estimator):
    report = estimator.estimate(TINY_CFG, tokens=10_000_000, gpu_type="a100_40gb")
    assert len(report.cloud_estimates) > 0


def test_estimate_cloud_cost_positive(estimator):
    report = estimator.estimate(TINY_CFG, tokens=10_000_000, gpu_type="a100_40gb")
    for provider, cost in report.cloud_estimates.items():
        assert cost >= 0, f"{provider} has negative cost"


def test_h100_faster_than_a100(estimator):
    tokens = 1_000_000_000
    a100 = estimator.estimate(SMALL_CFG, tokens=tokens, gpu_type="a100_40gb")
    h100 = estimator.estimate(SMALL_CFG, tokens=tokens, gpu_type="h100")
    assert h100.training_hours < a100.training_hours


def test_summary_contains_gpu_name(estimator):
    report = estimator.estimate(TINY_CFG, tokens=1_000_000, gpu_type="a100_40gb")
    summary = report.summary()
    assert "A100" in summary


def test_compare_returns_sorted(estimator):
    reports = estimator.compare(TINY_CFG, tokens=1_000_000, gpu_types=["h100", "rtx_3080"])
    # Should be sorted fastest first
    assert reports[0].training_hours <= reports[-1].training_hours


def test_custom_electricity_price():
    estimator = CostEstimator(electricity_usd_kwh=0.20)
    report = estimator.estimate(TINY_CFG, tokens=10_000_000, gpu_type="a100_40gb")
    estimator2 = CostEstimator(electricity_usd_kwh=0.10)
    report2 = estimator2.estimate(TINY_CFG, tokens=10_000_000, gpu_type="a100_40gb")
    assert report.electricity_cost_usd > report2.electricity_cost_usd


def test_more_tokens_costs_more(estimator):
    small = estimator.estimate(TINY_CFG, tokens=1_000_000, gpu_type="a100_40gb")
    large = estimator.estimate(TINY_CFG, tokens=1_000_000_000, gpu_type="a100_40gb")
    assert large.training_hours > small.training_hours


def test_notes_not_empty(estimator):
    report = estimator.estimate(TINY_CFG, tokens=10_000_000, gpu_type="a100_40gb")
    assert len(report.notes) > 0
