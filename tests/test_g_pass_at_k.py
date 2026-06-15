"""Tests for G-Pass@k_τ — the generalized consistency metric (arXiv:2412.13147).

G-Pass@k_τ = P(at least ⌈τ·k⌉ of k drawn samples are correct). It must interpolate
exactly between the two extremes the project already ships: pass_at_k (≥1 correct)
at the low-τ end, and pass_hat_k (all-k correct) at τ = 1.
"""

from __future__ import annotations

import math

import pytest

from cola_coder.evaluation.metrics import g_pass_at_k, pass_at_k, pass_hat_k


class TestReducesToExtremes:
    @pytest.mark.parametrize(("n", "c", "k"), [(10, 4, 5), (8, 8, 3), (6, 0, 2), (20, 7, 10)])
    def test_tau_one_equals_pass_hat_k(self, n: int, c: int, k: int) -> None:
        # threshold = ceil(1.0 * k) = k → "all k correct" → pass^k
        assert g_pass_at_k(n, c, k, tau=1.0) == pytest.approx(pass_hat_k(n, c, k))

    @pytest.mark.parametrize(("n", "c", "k"), [(10, 4, 5), (8, 1, 3), (6, 3, 2), (20, 7, 10)])
    def test_low_tau_equals_pass_at_k(self, n: int, c: int, k: int) -> None:
        # tau <= 1/k → threshold = ceil(tau*k) = 1 → "≥1 correct" → pass@k
        assert g_pass_at_k(n, c, k, tau=1.0 / k) == pytest.approx(pass_at_k(n, c, k))


class TestMonotonicityAndBounds:
    def test_monotonic_decreasing_in_tau(self) -> None:
        # Demanding MORE consistency cannot increase the score.
        n, c, k = 12, 6, 6
        taus = [1 / k, 0.34, 0.5, 0.67, 0.84, 1.0]
        vals = [g_pass_at_k(n, c, k, t) for t in taus]
        for a, b in zip(vals, vals[1:]):
            assert a + 1e-12 >= b

    def test_in_unit_interval(self) -> None:
        for n, c, k, tau in [(10, 5, 4, 0.5), (15, 3, 7, 0.25), (9, 9, 9, 1.0)]:
            v = g_pass_at_k(n, c, k, tau)
            assert 0.0 <= v <= 1.0

    def test_all_correct_is_one(self) -> None:
        assert g_pass_at_k(7, 7, 4, tau=0.75) == pytest.approx(1.0)

    def test_matches_explicit_hypergeometric(self) -> None:
        # n=10, c=5, k=4, tau=0.5 → threshold=2 → P(>=2 of 4 correct).
        n, c, k = 10, 5, 4
        denom = math.comb(n, k)
        expected = sum(
            math.comb(c, j) * math.comb(n - c, k - j) / denom for j in (2, 3, 4)
        )
        assert g_pass_at_k(n, c, k, tau=0.5) == pytest.approx(expected)


class TestValidation:
    @pytest.mark.parametrize(
        ("n", "c", "k", "tau"),
        [(0, 0, 1, 0.5), (10, 5, 0, 0.5), (10, 5, 11, 0.5), (10, 11, 5, 0.5),
         (10, -1, 5, 0.5), (10, 5, 5, 0.0), (10, 5, 5, 1.5)],
    )
    def test_invalid_inputs_raise(self, n: int, c: int, k: int, tau: float) -> None:
        with pytest.raises(ValueError):
            g_pass_at_k(n, c, k, tau)
