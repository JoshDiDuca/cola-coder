"""Tests for TestGenEvaluator (features/test_gen_evaluator.py)."""

from __future__ import annotations

import pytest

from cola_coder.features.test_gen_evaluator import (
    FEATURE_ENABLED,
    TestQualityReport,
    evaluate_tests,
    is_enabled,
)

# ---------------------------------------------------------------------------
# Sample snippets
# ---------------------------------------------------------------------------

GOOD_TESTS = '''\
import pytest


def test_add_positive_numbers():
    """Test adding two positive integers."""
    assert 1 + 1 == 2


def test_add_zero():
    assert 0 + 5 == 5


def test_add_negative():
    assert -1 + -1 == -2


def test_edge_empty_list():
    items = []
    assert len(items) == 0


def test_edge_none_input():
    result = str(None)
    assert result == "None"
'''

BAD_TESTS = '''\
def testAddition():
    pass


def testSubtract():
    x = 1 - 1
'''

ISOLATION_VIOLATION = '''\
import requests

def test_api():
    resp = requests.get("http://example.com")
    assert resp.status_code == 200
'''

WITH_MOCKS = '''\
from unittest.mock import MagicMock, patch


def test_with_mock():
    m = MagicMock()
    m.return_value = 42
    assert m() == 42
'''

NO_TESTS = '''\
def helper(x):
    return x + 1
'''

SYNTAX_ERROR = '''\
def test_foo(
    pass
'''


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


class TestIsEnabled:
    def test_constant(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled(self):
        assert is_enabled() is True


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


class TestBasicStructure:
    def test_returns_report(self):
        r = evaluate_tests(GOOD_TESTS)
        assert isinstance(r, TestQualityReport)

    def test_overall_score_in_range(self):
        r = evaluate_tests(GOOD_TESTS)
        assert 0.0 <= r.overall_score <= 1.0

    def test_good_tests_high_score(self):
        r = evaluate_tests(GOOD_TESTS)
        assert r.overall_score >= 0.5

    def test_no_tests_zero_score(self):
        r = evaluate_tests(NO_TESTS)
        assert r.overall_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


class TestCounting:
    def test_total_tests(self):
        r = evaluate_tests(GOOD_TESTS)
        assert r.total_tests == 5

    def test_well_named_tests(self):
        r = evaluate_tests(GOOD_TESTS)
        assert r.well_named_tests == 5

    def test_badly_named_tests(self):
        r = evaluate_tests(BAD_TESTS)
        # testAddition is not snake_case test_* format
        assert r.well_named_tests < r.total_tests

    def test_tests_with_asserts(self):
        r = evaluate_tests(GOOD_TESTS)
        assert r.tests_with_asserts >= 4

    def test_bad_tests_no_asserts(self):
        r = evaluate_tests(BAD_TESTS)
        assert r.tests_with_asserts == 0


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


class TestScores:
    def test_naming_score_good_tests(self):
        r = evaluate_tests(GOOD_TESTS)
        assert r.naming_score == pytest.approx(1.0)

    def test_naming_score_bad_tests(self):
        r = evaluate_tests(BAD_TESTS)
        assert r.naming_score < 1.0

    def test_assert_score_good_tests(self):
        r = evaluate_tests(GOOD_TESTS)
        assert r.assert_score >= 0.8

    def test_edge_case_score(self):
        r = evaluate_tests(GOOD_TESTS)
        assert r.edge_case_score > 0.0


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


class TestIsolation:
    def test_isolation_violation_detected(self):
        r = evaluate_tests(ISOLATION_VIOLATION)
        assert r.isolation_violations >= 1

    def test_isolation_lowers_score(self):
        r_clean = evaluate_tests(GOOD_TESTS)
        r_dirty = evaluate_tests(ISOLATION_VIOLATION)
        assert r_clean.isolation_score >= r_dirty.isolation_score


# ---------------------------------------------------------------------------
# Mock usage
# ---------------------------------------------------------------------------


class TestMockUsage:
    def test_mock_usage_detected(self):
        r = evaluate_tests(WITH_MOCKS)
        assert r.tests_using_mocks >= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_source(self):
        r = evaluate_tests("")
        assert r.overall_score == pytest.approx(0.0)

    def test_syntax_error_returns_report(self):
        r = evaluate_tests(SYNTAX_ERROR)
        assert isinstance(r, TestQualityReport)
        assert any("syntax" in i.lower() for i in r.issues)

    def test_issues_populated_for_bad_tests(self):
        r = evaluate_tests(BAD_TESTS)
        assert len(r.issues) > 0
