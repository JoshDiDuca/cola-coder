"""Tests for distillation_helper.py."""

from __future__ import annotations

import math
import pytest

from cola_coder.features.distillation_helper import (
    FEATURE_ENABLED,
    AlignmentReport,
    DistillationLoss,
    compute_alignment,
    compute_distillation_loss,
    find_optimal_temperature,
    generate_soft_labels,
    is_enabled,
    js_divergence,
    kl_divergence,
    softmax,
)


class TestIsEnabled:
    def test_constant(self):
        assert FEATURE_ENABLED is True

    def test_function(self):
        assert is_enabled() is True


class TestSoftmax:
    def test_sums_to_one(self):
        logits = [1.0, 2.0, 3.0]
        probs = softmax(logits)
        assert abs(sum(probs) - 1.0) < 1e-6

    def test_higher_logit_higher_prob(self):
        logits = [1.0, 5.0, 0.0]
        probs = softmax(logits)
        assert probs[1] > probs[0] > probs[2]

    def test_temperature_1_is_standard(self):
        logits = [1.0, 2.0, 3.0]
        assert softmax(logits, temperature=1.0) == softmax(logits)

    def test_high_temperature_more_uniform(self):
        logits = [1.0, 10.0, 0.0]
        sharp = softmax(logits, temperature=0.1)
        soft = softmax(logits, temperature=10.0)
        # Soft should have higher entropy (closer to uniform)
        assert max(soft) < max(sharp)

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError):
            softmax([1.0, 2.0], temperature=0.0)

    def test_empty_returns_empty(self):
        assert softmax([]) == []


class TestKLDivergence:
    def test_same_distribution_zero(self):
        p = [0.25, 0.25, 0.25, 0.25]
        assert kl_divergence(p, p) < 1e-6

    def test_asymmetric(self):
        p = [0.9, 0.1]
        q = [0.5, 0.5]
        assert kl_divergence(p, q) != kl_divergence(q, p)

    def test_nonnegative(self):
        p = [0.6, 0.4]
        q = [0.3, 0.7]
        assert kl_divergence(p, q) >= 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            kl_divergence([0.5, 0.5], [0.3, 0.4, 0.3])


class TestJSDivergence:
    def test_same_distribution_zero(self):
        p = [0.5, 0.5]
        assert js_divergence(p, p) < 1e-6

    def test_symmetric(self):
        p = [0.8, 0.2]
        q = [0.2, 0.8]
        assert abs(js_divergence(p, q) - js_divergence(q, p)) < 1e-9

    def test_bounded_by_log2_nats(self):
        # JS divergence in nats ≤ log(2)
        p = [1.0, 0.0]
        q = [0.0, 1.0]
        assert js_divergence(p, q) <= math.log(2) + 1e-9


class TestSoftLabels:
    def test_soft_labels_sum_to_one(self):
        teacher_logits = [[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]
        soft = generate_soft_labels(teacher_logits, temperature=2.0)
        for row in soft:
            assert abs(sum(row) - 1.0) < 1e-6

    def test_high_temp_more_uniform(self):
        logits = [[10.0, 1.0, 1.0]]
        sharp = generate_soft_labels(logits, temperature=0.5)
        soft = generate_soft_labels(logits, temperature=5.0)
        assert max(sharp[0]) > max(soft[0])


class TestAlignment:
    def _identical_logits(self, n: int = 4) -> list[list[float]]:
        return [[1.0, 2.0, 3.0, 0.5]] * n

    def test_returns_alignment_report(self):
        logits = self._identical_logits()
        result = compute_alignment(logits, logits)
        assert isinstance(result, AlignmentReport)

    def test_identical_distributions_high_agreement(self):
        logits = self._identical_logits()
        result = compute_alignment(logits, logits)
        assert result.top1_agreement == 1.0

    def test_summary_str(self):
        logits = self._identical_logits()
        result = compute_alignment(logits, logits)
        s = result.summary()
        assert "KL=" in s

    def test_batch_size_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_alignment([[1.0, 2.0]], [[1.0, 2.0], [3.0, 4.0]])


class TestDistillationLoss:
    def test_returns_named_tuple(self):
        result = compute_distillation_loss(
            student_logits=[1.0, 2.0, 3.0],
            teacher_logits=[1.5, 2.5, 0.5],
            true_label=2,
        )
        assert isinstance(result, DistillationLoss)

    def test_all_components_finite(self):
        result = compute_distillation_loss([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], true_label=1)
        assert math.isfinite(result.hard_loss)
        assert math.isfinite(result.soft_loss)
        assert math.isfinite(result.total_loss)

    def test_alpha_zero_uses_soft_only(self):
        result = compute_distillation_loss(
            [1.0, 2.0], [1.0, 2.0], true_label=1, alpha=0.0
        )
        assert abs(result.total_loss - result.soft_loss) < 1e-4

    def test_alpha_one_uses_hard_only(self):
        result = compute_distillation_loss(
            [1.0, 2.0], [1.0, 2.0], true_label=1, alpha=1.0
        )
        assert abs(result.total_loss - result.hard_loss) < 1e-4


class TestTemperatureSearch:
    def test_returns_float(self):
        logits = [[1.0, 2.0, 0.5], [0.5, 0.5, 2.0]]
        labels = [1, 2]
        temp = find_optimal_temperature(logits, labels)
        assert isinstance(temp, float)
        assert temp > 0
