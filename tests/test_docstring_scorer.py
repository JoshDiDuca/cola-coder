"""Tests for DocstringScorer (features/docstring_scorer.py)."""

from __future__ import annotations

import pytest

from cola_coder.features.docstring_scorer import DocstringScorer


GOOD_CODE = '''\
def add(a: int, b: int) -> int:
    """Add two integers and return the result.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        The sum of a and b.

    Example:
        >>> add(1, 2)
        3
    """
    return a + b
'''

MINIMAL_CODE = '''\
def add(a, b):
    """Add."""
    return a + b
'''

NO_DOCSTRING = '''\
def add(a, b):
    return a + b
'''

NO_PARAMS = '''\
def greet() -> str:
    """Return a greeting message.

    Returns:
        A greeting string.
    """
    return "Hello!"
'''


@pytest.fixture()
def scorer() -> DocstringScorer:
    return DocstringScorer()


class TestIsEnabled:
    def test_feature_enabled(self):
        from cola_coder.features.docstring_scorer import FEATURE_ENABLED, is_enabled

        assert FEATURE_ENABLED is True
        assert is_enabled() is True


class TestScoreBasic:
    def test_good_code_high_score(self, scorer):
        score = scorer.score(GOOD_CODE)
        assert score >= 0.6, f"Expected >= 0.6 but got {score}"

    def test_no_docstring_zero(self, scorer):
        score = scorer.score(NO_DOCSTRING)
        assert score == 0.0

    def test_minimal_doc_low_score(self, scorer):
        score = scorer.score(MINIMAL_CODE)
        assert 0.0 < score < 0.8

    def test_score_between_0_and_1(self, scorer):
        for code in [GOOD_CODE, MINIMAL_CODE, NO_DOCSTRING, NO_PARAMS]:
            s = scorer.score(code)
            assert 0.0 <= s <= 1.0, f"Score out of range: {s}"


class TestScoreDetailed:
    def test_param_coverage_full(self, scorer):
        detail = scorer.score_detailed(GOOD_CODE)
        assert detail.parameter_coverage == 1.0

    def test_param_coverage_zero(self, scorer):
        detail = scorer.score_detailed(MINIMAL_CODE)
        assert detail.parameter_coverage < 1.0

    def test_return_mentioned_good(self, scorer):
        detail = scorer.score_detailed(GOOD_CODE)
        assert detail.return_mentioned is True

    def test_return_not_mentioned(self, scorer):
        detail = scorer.score_detailed(MINIMAL_CODE)
        assert detail.return_mentioned is False

    def test_example_detected(self, scorer):
        detail = scorer.score_detailed(GOOD_CODE)
        assert detail.example_included is True

    def test_no_example(self, scorer):
        detail = scorer.score_detailed(MINIMAL_CODE)
        assert detail.example_included is False

    def test_no_params_full_coverage(self, scorer):
        detail = scorer.score_detailed(NO_PARAMS)
        assert detail.parameter_coverage == 1.0  # no params = full credit


class TestScoreRawDocstring:
    def test_direct_docstring(self, scorer):
        docstring = (
            "Compute the sum.\n\n"
            "Args:\n    a: first arg\n    b: second arg\n\n"
            "Returns:\n    The sum.\n\n"
            "Example:\n    >>> f(1, 2)\n    3\n"
        )
        detail = scorer.score_raw_docstring(docstring, ["a", "b"])
        assert detail.parameter_coverage == 1.0
        assert detail.return_mentioned is True
        assert detail.example_included is True
        assert detail.overall >= 0.6

    def test_empty_docstring(self, scorer):
        detail = scorer.score_raw_docstring("", [])
        assert detail.overall == 0.0


class TestWeights:
    def test_custom_weights(self):
        # Disable example weight, boost param weight
        scorer = DocstringScorer(weights={"params": 1.0, "return": 0.0, "example": 0.0, "desc": 0.0})
        # Code with full param coverage should score 1.0
        detail = scorer.score_detailed(GOOD_CODE)
        assert abs(detail.overall - 1.0) < 0.01

    def test_syntax_error_code(self, scorer):
        # Should not raise, just return 0.0 or low score
        score = scorer.score("def broken(:\n    pass")
        assert 0.0 <= score <= 1.0
