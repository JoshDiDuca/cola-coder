"""Tests for CommentQualityEvaluator (features/comment_quality.py)."""

from __future__ import annotations

import pytest

from cola_coder.features.comment_quality import (
    FEATURE_ENABLED,
    CommentQualityEvaluator,
    CommentQualityReport,
    evaluate_comments,
    is_enabled,
)

# ---------------------------------------------------------------------------
# Sample snippets
# ---------------------------------------------------------------------------

WELL_COMMENTED = '''\
"""Module-level docstring explaining purpose."""

import os


def read_file(path: str) -> str:
    """Read a file and return its contents.

    Note: This uses binary mode to avoid platform line-ending issues.
    """
    # Why binary? Windows line endings would corrupt checksums
    with open(path, "rb") as f:
        return f.read().decode("utf-8")
'''

NO_COMMENTS = '''\
def add(a, b):
    return a + b


def mul(a, b):
    return a * b
'''

TRIVIAL_COMMENTS = '''\
def get_value(x):
    # get x
    return x

def set_value(x, val):
    # set val
    x = val
'''

STALE_COMMENTS = '''\
def connect():
    # TODO 2019: migrate off python 2
    pass

def legacy_api():
    # deprecated: use new_api() instead
    pass
'''

DENSE_COMMENTS = '''\
# line 1
# line 2
# line 3
# line 4
# line 5
def foo():
    # comment
    return 1  # inline
'''


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


class TestIsEnabled:
    def test_feature_enabled_constant(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled(self):
        assert is_enabled() is True


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


class TestBasicStructure:
    def test_returns_report_instance(self):
        report = evaluate_comments(WELL_COMMENTED)
        assert isinstance(report, CommentQualityReport)

    def test_overall_score_in_range(self):
        report = evaluate_comments(WELL_COMMENTED)
        assert 0.0 <= report.overall_score <= 1.0

    def test_no_comment_code_scores_low_density(self):
        report = evaluate_comments(NO_COMMENTS)
        assert report.density_score < 0.7

    def test_well_commented_has_reasonable_score(self):
        report = evaluate_comments(WELL_COMMENTED)
        assert report.overall_score >= 0.5


# ---------------------------------------------------------------------------
# Density scoring
# ---------------------------------------------------------------------------


class TestDensityScoring:
    def test_no_comments_density_zero(self):
        report = evaluate_comments(NO_COMMENTS)
        assert report.density_ratio == pytest.approx(0.0)

    def test_dense_comments_penalised(self):
        report = evaluate_comments(DENSE_COMMENTS)
        # Very high density should lower the score
        assert report.density_score < 1.0

    def test_ideal_density_scores_one(self):
        # Build a snippet right in the ideal range
        source = "\n".join(
            ["# comment"] * 2 + ["x = 1"] * 8
        )
        ev = CommentQualityEvaluator(ideal_density_low=0.1, ideal_density_high=0.3)
        report = ev.evaluate(source)
        assert report.density_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Informativeness scoring
# ---------------------------------------------------------------------------


class TestInformativenessScoring:
    def test_trivial_comments_lower_informativeness(self):
        trivial_report = evaluate_comments(TRIVIAL_COMMENTS)
        assert trivial_report.trivial_count >= 1

    def test_informative_keywords_boost_score(self):
        source = "# Why: this avoids a race condition\nx = 1\n"
        report = evaluate_comments(source)
        assert report.informative_count >= 1

    def test_no_comments_neutral_informativeness(self):
        report = evaluate_comments(NO_COMMENTS)
        assert report.informativeness_score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Staleness scoring
# ---------------------------------------------------------------------------


class TestStalenessScoring:
    def test_stale_comments_detected(self):
        report = evaluate_comments(STALE_COMMENTS)
        assert report.stale_count >= 1
        assert report.staleness_score < 1.0

    def test_fresh_comments_not_stale(self):
        source = "# This explains the algorithm\ndef foo(): pass\n"
        report = evaluate_comments(source)
        assert report.stale_count == 0
        assert report.staleness_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Issues list
# ---------------------------------------------------------------------------


class TestIssuesList:
    def test_trivial_comment_produces_issue(self):
        report = evaluate_comments(TRIVIAL_COMMENTS)
        assert any("trivial" in issue for issue in report.issues)

    def test_stale_produces_issue(self):
        report = evaluate_comments(STALE_COMMENTS)
        assert any("stale" in issue for issue in report.issues)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_source(self):
        report = evaluate_comments("")
        assert report.total_lines == 0
        assert 0.0 <= report.overall_score <= 1.0

    def test_syntax_error_code_does_not_raise(self):
        bad_code = "def foo(\n    pass\n"
        report = evaluate_comments(bad_code)
        assert isinstance(report, CommentQualityReport)

    def test_only_comments(self):
        source = "# just\n# comments\n# here\n"
        report = evaluate_comments(source)
        assert report.comment_lines == 3
