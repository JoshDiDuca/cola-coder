"""Input validation for the unbiased ``pass_at_k`` estimator (EVAL-041).

``pass_at_k`` previously returned a spurious ``1.0`` for ``n < k`` (the
estimator is undefined there — you cannot draw k samples from fewer than k).
It now validates its inputs and raises ``ValueError``, exactly like its
siblings ``pass_hat_k`` / ``g_pass_at_k``. These tests pin that behavior and
confirm valid edge cases are unaffected.
"""

import pytest

from cola_coder.evaluation.metrics import pass_at_k


class TestPassAtKValidation:
    def test_raises_when_n_below_k(self) -> None:
        # The footgun: 0 correct out of 3 samples must never read as pass@5 = 1.0.
        with pytest.raises(ValueError):
            pass_at_k(n=3, c=0, k=5)

    def test_raises_on_non_positive_n(self) -> None:
        with pytest.raises(ValueError):
            pass_at_k(n=0, c=0, k=1)
        with pytest.raises(ValueError):
            pass_at_k(n=-1, c=0, k=1)

    def test_raises_on_k_below_one(self) -> None:
        with pytest.raises(ValueError):
            pass_at_k(n=10, c=3, k=0)

    def test_raises_on_k_above_n(self) -> None:
        with pytest.raises(ValueError):
            pass_at_k(n=4, c=2, k=5)

    def test_raises_on_c_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            pass_at_k(n=10, c=11, k=1)
        with pytest.raises(ValueError):
            pass_at_k(n=10, c=-1, k=1)


class TestPassAtKValidInputsUnchanged:
    def test_all_correct_is_one(self) -> None:
        assert pass_at_k(n=10, c=10, k=1) == 1.0
        assert pass_at_k(n=10, c=10, k=5) == 1.0

    def test_none_correct_is_zero(self) -> None:
        assert pass_at_k(n=10, c=0, k=1) == 0.0
        assert pass_at_k(n=10, c=0, k=5) == 0.0

    def test_boundary_n_equals_k(self) -> None:
        # n == k is valid (draw all samples); 1 correct of 5 -> 1/5 chance the
        # single all-5 draw includes it... actually all 5 drawn -> certain.
        assert pass_at_k(n=5, c=1, k=5) == 1.0

    def test_monotonic_in_k(self) -> None:
        p1 = pass_at_k(n=10, c=3, k=1)
        p5 = pass_at_k(n=10, c=3, k=5)
        p10 = pass_at_k(n=10, c=3, k=10)
        assert p1 < p5 < p10
        assert p10 == 1.0  # n - c = 7 < 10 -> any 10-draw includes a correct one
