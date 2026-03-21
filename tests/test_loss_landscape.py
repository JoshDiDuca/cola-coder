"""Tests for loss_landscape.py (feature 53)."""

import math
import pytest

from cola_coder.features.loss_landscape import (
    FEATURE_ENABLED,
    LossLandscapeAnalyzer,
    _add_scaled,
    _finite_diff_curvature,
    _normalize,
    _random_unit_vector,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Simple analytic loss functions for testing
# ---------------------------------------------------------------------------

def quadratic_loss(theta):
    """L(θ) = ||θ||² — convex bowl, sharp at origin."""
    return sum(x * x for x in theta)


def flat_loss(theta):
    """L(θ) = 1.0 — completely flat landscape."""
    return 1.0


def asymmetric_loss(theta):
    """L(θ) = θ[0]² + 10*θ[1]²  — sharp in dim 1, flatter in dim 0."""
    return theta[0] ** 2 + 10 * theta[1] ** 2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_feature_flag():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_normalize_unit():
    import random
    rng = random.Random(0)
    v = [rng.gauss(0, 1) for _ in range(8)]
    n = _normalize(v)
    length = math.sqrt(sum(x * x for x in n))
    assert abs(length - 1.0) < 1e-9


def test_normalize_zero_vector():
    v = [0.0, 0.0, 0.0]
    n = _normalize(v)
    assert n == v  # Returns unchanged for zero vector


def test_add_scaled():
    base = [1.0, 2.0, 3.0]
    direction = [1.0, 0.0, 0.0]
    result = _add_scaled(base, direction, 2.0)
    assert result == [3.0, 2.0, 3.0]


def test_random_unit_vector_unit_norm():
    import random
    rng = random.Random(7)
    v = _random_unit_vector(16, rng)
    length = math.sqrt(sum(x * x for x in v))
    assert abs(length - 1.0) < 1e-9


def test_finite_diff_curvature_quadratic():
    # For L=||θ||², curvature along any unit direction is 2
    theta = [0.5, 0.5, 0.5, 0.5]
    direction = _normalize([1.0, 0.0, 0.0, 0.0])
    curv = _finite_diff_curvature(quadratic_loss, theta, direction, eps=1e-4)
    assert abs(curv - 2.0) < 0.01


def test_finite_diff_curvature_flat():
    theta = [1.0, 2.0, 3.0]
    direction = _normalize([1.0, 0.0, 0.0])
    curv = _finite_diff_curvature(flat_loss, theta, direction)
    assert abs(curv) < 1e-6


def test_sample_direction_count():
    theta = [0.0, 0.0, 0.0, 0.0]
    analyzer = LossLandscapeAnalyzer(quadratic_loss, theta)
    profile = analyzer.sample_direction(n_points=7)
    assert len(profile.samples) == 7


def test_sample_direction_base_loss():
    theta = [1.0, 0.0]
    analyzer = LossLandscapeAnalyzer(quadratic_loss, theta)
    profile = analyzer.sample_direction()
    assert profile.base_loss == pytest.approx(1.0)


def test_sample_direction_min_max():
    theta = [0.0, 0.0]
    analyzer = LossLandscapeAnalyzer(quadratic_loss, theta)
    profile = analyzer.sample_direction(n_points=11)
    assert profile.min_loss <= profile.base_loss
    # max should be away from origin
    assert profile.max_loss >= profile.min_loss


def test_sample_direction_as_dict_list():
    theta = [0.0, 0.0]
    analyzer = LossLandscapeAnalyzer(flat_loss, theta)
    profile = analyzer.sample_direction(n_points=5)
    records = profile.as_dict_list()
    assert len(records) == 5
    assert all("alpha" in r and "loss" in r for r in records)


def test_sharpness_flat():
    theta = [1.0, 2.0, 3.0, 4.0]
    analyzer = LossLandscapeAnalyzer(flat_loss, theta)
    metrics = analyzer.sharpness(n_directions=8)
    assert metrics.classification == "flat"
    assert abs(metrics.mean_curvature) < 0.01
    assert metrics.n_directions == 8


def test_sharpness_sharp_quadratic():
    theta = [0.1] * 4
    analyzer = LossLandscapeAnalyzer(quadratic_loss, theta, seed=0)
    metrics = analyzer.sharpness(n_directions=20, eps=1e-4)
    # quadratic bowl has curvature ~2 per dimension
    assert metrics.mean_curvature > 0


def test_sharpness_summary_contains_class():
    theta = [0.0, 0.0]
    analyzer = LossLandscapeAnalyzer(flat_loss, theta)
    metrics = analyzer.sharpness(n_directions=4)
    s = metrics.summary()
    assert "class=" in s
    assert "trace=" in s


def test_multi_direction_profiles():
    theta = [0.0] * 4
    analyzer = LossLandscapeAnalyzer(quadratic_loss, theta)
    profiles = analyzer.multi_direction_profiles(n_directions=4, n_points=5)
    assert len(profiles) == 4
    for p in profiles:
        assert len(p.samples) == 5


def test_flatness_ratio_flat_function():
    theta = [0.5, 0.5]
    analyzer = LossLandscapeAnalyzer(flat_loss, theta)
    ratio = analyzer.flatness_ratio(n_directions=10)
    assert ratio == pytest.approx(1.0)


def test_flatness_ratio_sharp_function():
    # Very sharp: perturbation at alpha=10 causes huge loss increase
    theta = [0.0, 0.0]
    analyzer = LossLandscapeAnalyzer(quadratic_loss, theta, seed=5)
    ratio = analyzer.flatness_ratio(n_directions=10, alpha=10.0)
    # At alpha=10 the loss increases significantly, so flat ratio should be 0
    assert ratio < 0.5
