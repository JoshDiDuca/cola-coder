"""Tests for pass^k (consistency / all-k-pass) — the reliability mirror of pass@k.

Mirrors the structure of the pass@k tests in ``test_inference.py``: small,
deterministic, arithmetic. pass^k is the unbiased estimator of "all k sampled
solutions pass", and the capability-reliability gap is ``pass@k − pass^k``.
"""

from math import comb

import pytest

from cola_coder.evaluation.metrics import (
    ProblemResult,
    capability_reliability_gap,
    compute_pass_hat_k,
    format_results,
    pass_at_k,
    pass_hat_k,
)


class TestPassHatK:
    """Unbiased pass^k estimator: pass^k = C(c, k) / C(n, k)."""

    def test_matches_combinatorial_definition(self):
        """pass^k == comb(c, k) / comb(n, k) for several small (n, c, k)."""
        for n in range(1, 12):
            for c in range(0, n + 1):
                for k in range(1, n + 1):
                    expected = comb(c, k) / comb(n, k)  # 0 when c < k
                    assert abs(pass_hat_k(n, c, k) - expected) < 1e-12, (n, c, k)

    def test_all_correct_is_one(self):
        """c == n -> every draw is correct -> 1.0."""
        assert pass_hat_k(n=10, c=10, k=1) == 1.0
        assert pass_hat_k(n=10, c=10, k=5) == 1.0
        assert pass_hat_k(n=10, c=10, k=10) == 1.0

    def test_fewer_correct_than_k_is_zero(self):
        """c < k -> cannot draw k correct -> 0.0."""
        assert pass_hat_k(n=10, c=2, k=5) == 0.0
        assert pass_hat_k(n=10, c=0, k=1) == 0.0

    def test_k1_equals_pass_at_1(self):
        """pass^1 == c/n == pass@1."""
        for c in range(0, 11):
            assert abs(pass_hat_k(10, c, 1) - c / 10) < 1e-12
            assert abs(pass_hat_k(10, c, 1) - pass_at_k(10, c, 1)) < 1e-12

    def test_kn_is_one_only_when_all_correct(self):
        """k == n -> 1.0 iff c == n, else 0.0 (need every sample correct)."""
        assert pass_hat_k(n=8, c=8, k=8) == 1.0
        assert pass_hat_k(n=8, c=7, k=8) == 0.0

    def test_reliability_never_exceeds_capability(self):
        """Monotonicity: pass^k <= pass@k for the same (n, c, k)."""
        for n in range(1, 12):
            for c in range(0, n + 1):
                for k in range(1, n + 1):
                    assert pass_hat_k(n, c, k) <= pass_at_k(n, c, k) + 1e-12, (n, c, k)


class TestPassHatKValidation:
    """The estimator is undefined for out-of-range args and must raise."""

    def test_k_greater_than_n_raises(self):
        with pytest.raises(ValueError):
            pass_hat_k(n=5, c=3, k=6)

    def test_nonpositive_n_raises(self):
        with pytest.raises(ValueError):
            pass_hat_k(n=0, c=0, k=1)

    def test_negative_c_raises(self):
        with pytest.raises(ValueError):
            pass_hat_k(n=5, c=-1, k=1)

    def test_c_greater_than_n_raises(self):
        with pytest.raises(ValueError):
            pass_hat_k(n=5, c=6, k=1)

    def test_k_below_one_raises(self):
        with pytest.raises(ValueError):
            pass_hat_k(n=5, c=3, k=0)


class TestComputePassHatK:
    """Aggregation across problems mirrors compute_pass_at_k."""

    def test_mean_over_problems(self):
        results = [
            ProblemResult(task_id="a", num_samples=10, num_correct=10),  # pass^5 = 1.0
            ProblemResult(task_id="b", num_samples=10, num_correct=0),   # pass^5 = 0.0
        ]
        m = compute_pass_hat_k(results, k_values=[5])
        assert abs(m["pass^5"] - 0.5) < 1e-12

    def test_not_estimable_returns_none(self):
        """No problem has >= k samples -> None, not 0.0 (mirrors pass@k)."""
        results = [ProblemResult(task_id="a", num_samples=2, num_correct=2)]
        assert compute_pass_hat_k(results, k_values=[5])["pass^5"] is None

    def test_excludes_problems_with_too_few_samples(self):
        results = [
            ProblemResult(task_id="a", num_samples=10, num_correct=10),  # eligible
            ProblemResult(task_id="b", num_samples=2, num_correct=2),    # excluded for k=5
        ]
        # Only "a" (pass^5 = 1.0) contributes.
        assert abs(compute_pass_hat_k(results, [5])["pass^5"] - 1.0) < 1e-12


class TestCapabilityReliabilityGap:
    """gap = pass@k − pass^k, non-negative by construction."""

    def test_gap_equals_difference(self):
        cap = pass_at_k(n=10, c=5, k=3)
        rel = pass_hat_k(n=10, c=5, k=3)
        assert capability_reliability_gap(cap, rel) == cap - rel

    def test_gap_non_negative(self):
        for n in range(1, 12):
            for c in range(0, n + 1):
                for k in range(1, n + 1):
                    cap = pass_at_k(n, c, k)
                    rel = pass_hat_k(n, c, k)
                    assert capability_reliability_gap(cap, rel) >= -1e-12, (n, c, k)

    def test_gap_zero_at_extremes(self):
        # All correct: capability == reliability == 1.0 -> gap 0.
        assert capability_reliability_gap(pass_at_k(10, 10, 5), pass_hat_k(10, 10, 5)) == 0.0
        # None correct: both 0.0 -> gap 0.
        assert capability_reliability_gap(pass_at_k(10, 0, 5), pass_hat_k(10, 0, 5)) == 0.0


class TestFormatResultsReliability:
    """format_results surfaces pass^k + gap under each pass@k>1."""

    def test_shows_pass_hat_k_for_k_above_one(self):
        results = [ProblemResult(task_id="a", num_samples=10, num_correct=5)]
        out = format_results(results, k_values=[1, 5], bootstrap=False)
        assert "pass^5" in out
        assert "gap" in out
        # k=1 line is plain pass@1, no pass^1 row.
        assert "pass^1" not in out

    def test_no_pass_hat_when_only_k1(self):
        results = [ProblemResult(task_id="a", num_samples=10, num_correct=5)]
        out = format_results(results, k_values=[1], bootstrap=False)
        assert "pass^" not in out
