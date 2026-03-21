"""Tests for CoverageEstimator (features/coverage_estimator.py)."""

from __future__ import annotations


from cola_coder.features.coverage_estimator import (
    FEATURE_ENABLED,
    CoverageEstimator,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Sample snippets
# ---------------------------------------------------------------------------

SIMPLE_CODE = """\
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""

BRANCHY_CODE = """\
def classify(value):
    if value > 0:
        return "positive"
    elif value < 0:
        return "negative"
    else:
        return "zero"

def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        return None
    return result
"""

LOOP_CODE = """\
def sum_list(items):
    total = 0
    for item in items:
        total += item
    return total
"""

TEST_CODE = """\
def test_add():
    assert add(1, 2) == 3

def test_subtract():
    assert subtract(5, 3) == 2
"""

SYNTAX_ERROR = "def broken(:\n    pass"

UNREACHABLE_CODE = """\
def has_unreachable():
    x = 1
    return x
    y = 2
"""


class TestIsEnabled:
    def test_feature_enabled(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled_returns_true(self):
        assert is_enabled() is True


class TestSimpleCode:
    def test_counts_functions(self):
        estimator = CoverageEstimator()
        report = estimator.estimate(SIMPLE_CODE)
        assert report.total_functions == 2

    def test_no_test_source_means_zero_tested(self):
        estimator = CoverageEstimator()
        report = estimator.estimate(SIMPLE_CODE)
        assert report.tested_functions == 0

    def test_with_test_source_detects_tested(self):
        estimator = CoverageEstimator()
        report = estimator.estimate(SIMPLE_CODE, TEST_CODE)
        assert report.tested_functions == 2

    def test_estimated_coverage_with_tests_higher(self):
        estimator = CoverageEstimator()
        no_tests = estimator.estimate(SIMPLE_CODE)
        with_tests = estimator.estimate(SIMPLE_CODE, TEST_CODE)
        assert with_tests.estimated_coverage > no_tests.estimated_coverage


class TestBranchPoints:
    def test_detects_if_branches(self):
        estimator = CoverageEstimator()
        report = estimator.estimate(BRANCHY_CODE)
        if_branches = [bp for bp in report.branch_points if bp.node_type == "if"]
        assert len(if_branches) >= 1

    def test_detects_try_branches(self):
        estimator = CoverageEstimator()
        report = estimator.estimate(BRANCHY_CODE)
        try_branches = [bp for bp in report.branch_points if bp.node_type == "try"]
        assert len(try_branches) >= 1

    def test_detects_loop_branches(self):
        estimator = CoverageEstimator()
        report = estimator.estimate(LOOP_CODE)
        loop_branches = [bp for bp in report.branch_points if bp.node_type == "loop"]
        assert len(loop_branches) >= 1

    def test_total_branches_positive(self):
        estimator = CoverageEstimator()
        report = estimator.estimate(BRANCHY_CODE)
        assert report.total_branches > 0


class TestUnreachable:
    def test_detects_unreachable_lines(self):
        estimator = CoverageEstimator()
        report = estimator.estimate(UNREACHABLE_CODE)
        assert len(report.unreachable_lines) >= 1

    def test_unreachable_are_line_numbers(self):
        estimator = CoverageEstimator()
        report = estimator.estimate(UNREACHABLE_CODE)
        for ln in report.unreachable_lines:
            assert isinstance(ln, int) and ln > 0


class TestEdgeCases:
    def test_syntax_error_returns_empty(self):
        estimator = CoverageEstimator()
        report = estimator.estimate(SYNTAX_ERROR)
        assert report.total_functions == 0

    def test_empty_source_perfect_coverage(self):
        estimator = CoverageEstimator()
        report = estimator.estimate("# empty\n")
        assert report.estimated_coverage == 1.0

    def test_summary_is_string(self):
        estimator = CoverageEstimator()
        report = estimator.estimate(BRANCHY_CODE)
        assert isinstance(report.summary, str)
        assert "%" in report.summary

    def test_untested_functions(self):
        estimator = CoverageEstimator()
        report = estimator.estimate(SIMPLE_CODE)
        assert report.untested_functions == 2

    def test_coverage_bounds(self):
        estimator = CoverageEstimator()
        for code in [SIMPLE_CODE, BRANCHY_CODE, LOOP_CODE]:
            report = estimator.estimate(code)
            assert 0.0 <= report.estimated_coverage <= 1.0
