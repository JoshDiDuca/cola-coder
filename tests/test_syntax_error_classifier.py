"""Tests for syntax_error_classifier.py (feature 56)."""

import pytest

from cola_coder.features.syntax_error_classifier import (
    FEATURE_ENABLED,
    ErrorCategory,
    ErrorStats,
    SyntaxErrorClassifier,
    _check_missing_colon,
    _check_unmatched_brackets,
    is_enabled,
)


@pytest.fixture
def clf():
    return SyntaxErrorClassifier()


VALID_CODE = """\
def add(a, b):
    return a + b

result = add(1, 2)
"""

MISSING_COLON = """\
def bad()
    return 1
"""

UNMATCHED_PAREN = "x = (1 + 2\ny = 3\n"

INDENT_ERROR = "def f():\nx = 1\n"

EMPTY_CODE = "   "


def test_feature_flag():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_valid_code(clf):
    result = clf.classify(VALID_CODE)
    assert result.is_valid is True
    assert result.category == ErrorCategory.NONE


def test_missing_colon_detected(clf):
    result = clf.classify(MISSING_COLON)
    assert result.is_valid is False
    assert result.category == ErrorCategory.MISSING_COLON


def test_unmatched_bracket(clf):
    result = clf.classify(UNMATCHED_PAREN)
    assert result.is_valid is False
    assert result.category in (ErrorCategory.UNMATCHED_BRACKET, ErrorCategory.INCOMPLETE_EXPRESSION, ErrorCategory.GENERAL_SYNTAX)


def test_indentation_error(clf):
    result = clf.classify(INDENT_ERROR)
    assert result.is_valid is False
    assert result.category == ErrorCategory.INDENTATION_ERROR


def test_empty_code(clf):
    result = clf.classify(EMPTY_CODE)
    assert result.is_valid is False
    assert result.category == ErrorCategory.INCOMPLETE_EXPRESSION


def test_result_as_dict(clf):
    result = clf.classify(VALID_CODE)
    d = result.as_dict()
    assert "is_valid" in d
    assert "category" in d
    assert "message" in d


def test_check_missing_colon_none_for_valid():
    code = "def foo():\n    pass\n"
    assert _check_missing_colon(code) is None


def test_check_missing_colon_returns_lineno():
    code = "def foo()\n    pass\n"
    lineno = _check_missing_colon(code)
    assert lineno == 1


def test_check_unmatched_brackets_ok():
    assert _check_unmatched_brackets("x = (1 + 2)") is None


def test_check_unmatched_brackets_unclosed():
    msg = _check_unmatched_brackets("x = (1 + 2")
    assert msg is not None
    assert "(" in msg or "Unclosed" in msg


def test_check_unmatched_brackets_extra_close():
    msg = _check_unmatched_brackets("x = 1 + 2)")
    assert msg is not None


def test_classify_many(clf):
    samples = [VALID_CODE, MISSING_COLON, EMPTY_CODE]
    results = clf.classify_many(samples)
    assert len(results) == 3
    assert results[0].is_valid is True
    assert results[1].is_valid is False
    assert results[2].is_valid is False


def test_track_errors_accumulates(clf):
    samples = [VALID_CODE, MISSING_COLON, INDENT_ERROR, VALID_CODE]
    stats = clf.track_errors(samples)
    assert stats.total_samples == 4
    assert stats.valid_count == 2
    assert sum(stats.error_counts.values()) == 2


def test_track_errors_incremental(clf):
    stats = ErrorStats()
    clf.track_errors([VALID_CODE], stats)
    clf.track_errors([MISSING_COLON], stats)
    assert stats.total_samples == 2
    assert stats.valid_count == 1


def test_validity_rate(clf):
    samples = [VALID_CODE] * 3 + [MISSING_COLON]
    stats = clf.track_errors(samples)
    assert stats.validity_rate == pytest.approx(0.75)


def test_most_common_error(clf):
    samples = [MISSING_COLON, MISSING_COLON, INDENT_ERROR]
    stats = clf.track_errors(samples)
    assert stats.most_common_error() == ErrorCategory.MISSING_COLON.value


def test_stats_summary(clf):
    stats = clf.track_errors([VALID_CODE, MISSING_COLON])
    s = stats.summary()
    assert "total=" in s
    assert "valid=" in s


def test_find_undefined_names(clf):
    code = "result = undefined_var + 1\n"
    undefined = clf.find_undefined_names(code)
    assert "undefined_var" in undefined
