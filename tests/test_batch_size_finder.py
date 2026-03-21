"""Tests for batch_size_finder.py (feature 52)."""

import pytest

from cola_coder.features.batch_size_finder import (
    FEATURE_ENABLED,
    BatchSizeFinder,
    ModelConfig,
    _compute_breakdown,
    is_enabled,
)


@pytest.fixture
def small_config():
    """A small model config: ~10M params, short sequences."""
    return ModelConfig(
        n_params=10_000_000,
        dtype_bytes=2,
        seq_len=128,
        hidden_size=256,
        n_layers=4,
        n_heads=4,
        vocab_size=8000,
    )


@pytest.fixture
def finder():
    return BatchSizeFinder()


def test_feature_flag():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_breakdown_positive_values(small_config):
    bd = _compute_breakdown(small_config, batch_size=4)
    assert bd.parameters_mb > 0
    assert bd.gradients_mb > 0
    assert bd.optimizer_state_mb > 0
    assert bd.activations_mb > 0
    assert bd.workspace_mb > 0
    assert bd.total_mb > 0


def test_breakdown_batch_scales_activations(small_config):
    bd1 = _compute_breakdown(small_config, batch_size=1)
    bd8 = _compute_breakdown(small_config, batch_size=8)
    # Only activations should scale; params/grads/optim are fixed
    assert bd8.activations_mb == pytest.approx(bd1.activations_mb * 8)
    assert bd8.parameters_mb == pytest.approx(bd1.parameters_mb)


def test_breakdown_summary_contains_batch(small_config):
    bd = _compute_breakdown(small_config, batch_size=16)
    assert "batch=16" in bd.summary()


def test_find_returns_valid_batch(finder, small_config):
    result = finder.find(small_config, memory_limit_mb=4096)
    assert result.optimal_batch_size >= 1
    # The returned batch must actually fit
    assert result.breakdown.total_mb <= 4096


def test_find_power_of_two(finder, small_config):
    result = finder.find(small_config, memory_limit_mb=4096, power_of_two=True)
    bs = result.optimal_batch_size
    assert bs > 0
    # Power of two check
    assert (bs & (bs - 1)) == 0


def test_find_infeasible_returns_zero(finder):
    # 1 byte of memory — nothing can fit
    big_config = ModelConfig(n_params=1_000_000_000, dtype_bytes=4, seq_len=512)
    result = finder.find(big_config, memory_limit_mb=0.001)
    assert result.optimal_batch_size == 0
    assert len(result.notes) > 0


def test_find_tight_budget_batch_one(finder, small_config):
    # Budget just enough for batch=1 but not batch=2
    bd1 = _compute_breakdown(small_config, batch_size=1)
    bd2 = _compute_breakdown(small_config, batch_size=2)
    # Set limit between the two
    limit = (bd1.total_mb + bd2.total_mb) / 2
    result = finder.find(small_config, memory_limit_mb=limit, power_of_two=True)
    assert result.optimal_batch_size == 1


def test_find_larger_budget_gives_larger_batch(finder, small_config):
    r1 = finder.find(small_config, memory_limit_mb=512)
    r2 = finder.find(small_config, memory_limit_mb=2048)
    assert r2.optimal_batch_size >= r1.optimal_batch_size


def test_find_non_power_of_two(finder, small_config):
    result = finder.find(small_config, memory_limit_mb=4096, power_of_two=False)
    assert result.optimal_batch_size >= 1


def test_candidates_tried_non_empty(finder, small_config):
    result = finder.find(small_config, memory_limit_mb=1024)
    assert len(result.candidate_sizes_tried) > 0


def test_estimate_memory_matches_compute(finder, small_config):
    bd_direct = _compute_breakdown(small_config, batch_size=8)
    bd_method = finder.estimate_memory(small_config, batch_size=8)
    assert bd_direct.total_mb == pytest.approx(bd_method.total_mb)


def test_gradient_accumulation_feasible(finder, small_config):
    rec = finder.recommend_gradient_accumulation(
        small_config, memory_limit_mb=4096, target_effective_batch=256
    )
    assert rec["feasible"] is True
    assert rec["micro_batch"] >= 1
    assert rec["accumulation_steps"] >= 1
    assert rec["effective_batch"] >= rec["micro_batch"]


def test_gradient_accumulation_infeasible(finder):
    huge = ModelConfig(n_params=10_000_000_000, dtype_bytes=4, seq_len=2048)
    rec = finder.recommend_gradient_accumulation(huge, memory_limit_mb=1, target_effective_batch=64)
    assert rec["feasible"] is False
    assert rec["micro_batch"] == 0
