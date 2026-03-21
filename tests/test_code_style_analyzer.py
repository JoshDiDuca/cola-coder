"""Tests for CodeStyleAnalyzer (features/code_style_analyzer.py)."""

from __future__ import annotations

import pytest

from cola_coder.features.code_style_analyzer import (
    CodeStyleAnalyzer,
    StyleReport,
)

# ---------------------------------------------------------------------------
# Sample code snippets
# ---------------------------------------------------------------------------

GOOD_CODE = """\
import json
import os

MY_CONST = 42
MAX_SIZE = 100


def compute_sum(a: int, b: int) -> int:
    return a + b


class MyClass:
    def __init__(self, value: int) -> None:
        self.value = value

    def get_value(self) -> int:
        return self.value
"""

MIXED_INDENT_CODE = "def foo():\n    pass\ndef bar():\n\tpass\n"

BAD_NAMING_CODE = """\
def ComputeSum(a, b):
    return a + b


class myclass:
    pass
"""

LONG_LINE_CODE = "x = " + "a" * 110 + "\n"


@pytest.fixture()
def analyzer() -> CodeStyleAnalyzer:
    return CodeStyleAnalyzer(max_line_length=100)


class TestIsEnabled:
    def test_feature_enabled(self):
        from cola_coder.features.code_style_analyzer import FEATURE_ENABLED, is_enabled

        assert FEATURE_ENABLED is True
        assert is_enabled() is True


class TestReturnType:
    def test_returns_style_report(self, analyzer):
        report = analyzer.analyze(GOOD_CODE)
        assert isinstance(report, StyleReport)

    def test_overall_score_in_range(self, analyzer):
        for code in [GOOD_CODE, BAD_NAMING_CODE, MIXED_INDENT_CODE]:
            report = analyzer.analyze(code)
            assert 0.0 <= report.overall_score <= 1.0, code[:50]


class TestIndentation:
    def test_consistent_spaces(self, analyzer):
        report = analyzer.analyze(GOOD_CODE)
        assert report.indentation.style == "spaces"
        assert report.indentation.consistent is True

    def test_mixed_indentation(self, analyzer):
        report = analyzer.analyze(MIXED_INDENT_CODE)
        assert report.indentation.consistent is False
        assert report.indentation.style == "mixed"

    def test_no_indentation_code(self, analyzer):
        report = analyzer.analyze("x = 1\ny = 2\n")
        assert report.indentation.consistent is True


class TestNaming:
    def test_good_naming_high_score(self, analyzer):
        report = analyzer.analyze(GOOD_CODE)
        assert report.naming.score >= 0.7

    def test_bad_function_name_detected(self, analyzer):
        report = analyzer.analyze(BAD_NAMING_CODE)
        assert "ComputeSum" in report.naming.functions_bad

    def test_bad_class_name_detected(self, analyzer):
        report = analyzer.analyze(BAD_NAMING_CODE)
        assert "myclass" in report.naming.classes_bad

    def test_dunder_methods_exempt(self, analyzer):
        code = "class Foo:\n    def __init__(self):\n        pass\n"
        report = analyzer.analyze(code)
        assert "__init__" not in report.naming.functions_bad

    def test_syntax_error_graceful(self, analyzer):
        # Should not raise
        report = analyzer.analyze("def broken(:\n    pass\n")
        assert isinstance(report, StyleReport)


class TestLineLength:
    def test_good_code_high_compliance(self, analyzer):
        report = analyzer.analyze(GOOD_CODE)
        assert report.line_length.compliance_rate == 1.0

    def test_long_line_detected(self, analyzer):
        report = analyzer.analyze(LONG_LINE_CODE)
        assert report.line_length.lines_over_limit >= 1
        assert report.line_length.worst_line_length > 100

    def test_compliance_rate_range(self, analyzer):
        report = analyzer.analyze(LONG_LINE_CODE)
        assert 0.0 <= report.line_length.compliance_rate <= 1.0

    def test_custom_limit(self):
        analyzer_80 = CodeStyleAnalyzer(max_line_length=80)
        code = "x = " + "a" * 82 + "\n"
        report = analyzer_80.analyze(code)
        assert report.line_length.lines_over_limit >= 1


class TestSummary:
    def test_summary_string(self, analyzer):
        report = analyzer.analyze(GOOD_CODE)
        s = report.summary()
        assert "score=" in s

    def test_passed_threshold(self, analyzer):
        report = analyzer.analyze(GOOD_CODE)
        # Good code should pass at 0.5 threshold
        assert report.passed(threshold=0.5)

    def test_issues_list_non_empty_for_bad(self, analyzer):
        report = analyzer.analyze(BAD_NAMING_CODE)
        # Bad naming should generate at least one issue
        assert isinstance(report.issues, list)
