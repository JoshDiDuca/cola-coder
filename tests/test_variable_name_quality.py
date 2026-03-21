"""Tests for VariableNameAnalyzer (features/variable_name_quality.py)."""

from __future__ import annotations


from cola_coder.features.variable_name_quality import (
    FEATURE_ENABLED,
    VariableNameAnalyzer,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Sample code snippets
# ---------------------------------------------------------------------------

CLEAN_CODE = """\
def compute_total(item_count: int, unit_price: float) -> float:
    total_cost = item_count * unit_price
    return total_cost
"""

SINGLE_CHAR_CODE = """\
def foo(a, b, c):
    q = a + b
    return q + c
"""

ABBREV_CODE = """\
def process(buf, tmp):
    val = buf + tmp
    return val
"""

SHADOW_CODE = """\
def broken():
    list = [1, 2, 3]
    dict = {}
    return list, dict
"""

CAMEL_IN_SNAKE_FILE = """\
def process_items(itemCount, unitPrice):
    totalCost = itemCount * unitPrice
    return totalCost
"""

SNAKE_IN_CAMEL_FILE = """\
def processItems(item_count, unit_price):
    total_cost = item_count * unit_price
    return total_cost
"""

SYNTAX_ERROR_CODE = "def broken(:\n    pass\n"

EMPTY_CODE = "# just a comment\n"

ALLOWED_SINGLE_CHAR = """\
for i in range(10):
    for j in range(10):
        x = i + j
"""


class TestIsEnabled:
    def test_feature_enabled_constant(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled_returns_true(self):
        assert is_enabled() is True


class TestCleanCode:
    def test_clean_code_high_score(self):
        analyzer = VariableNameAnalyzer()
        report = analyzer.analyze(CLEAN_CODE)
        assert report.score >= 0.8

    def test_clean_code_no_single_char(self):
        analyzer = VariableNameAnalyzer()
        report = analyzer.analyze(CLEAN_CODE)
        assert report.single_char_names == []

    def test_clean_code_no_abbreviations(self):
        analyzer = VariableNameAnalyzer()
        report = analyzer.analyze(CLEAN_CODE)
        assert report.abbreviations == []

    def test_clean_code_no_shadowed_builtins(self):
        analyzer = VariableNameAnalyzer()
        report = analyzer.analyze(CLEAN_CODE)
        assert report.shadowed_builtins == []


class TestSingleCharNames:
    def test_detects_single_char_names(self):
        analyzer = VariableNameAnalyzer()
        report = analyzer.analyze(SINGLE_CHAR_CODE)
        # 'q' is not in allowed set
        assert "q" in report.single_char_names

    def test_allowed_single_char_not_flagged(self):
        analyzer = VariableNameAnalyzer()
        report = analyzer.analyze(ALLOWED_SINGLE_CHAR)
        # i, j, x are in the allowed set
        assert "i" not in report.single_char_names
        assert "j" not in report.single_char_names
        assert "x" not in report.single_char_names

    def test_score_lower_with_single_char(self):
        analyzer = VariableNameAnalyzer()
        clean_report = analyzer.analyze(CLEAN_CODE)
        bad_report = analyzer.analyze(SINGLE_CHAR_CODE)
        assert bad_report.score < clean_report.score


class TestAbbreviations:
    def test_detects_abbreviations(self):
        analyzer = VariableNameAnalyzer()
        report = analyzer.analyze(ABBREV_CODE)
        assert len(report.abbreviations) > 0
        # buf, tmp, val are all common abbreviations
        assert any(a in report.abbreviations for a in ["buf", "tmp", "val"])

    def test_abbreviations_lower_score(self):
        analyzer = VariableNameAnalyzer()
        report = analyzer.analyze(ABBREV_CODE)
        assert report.score < 1.0


class TestShadowedBuiltins:
    def test_detects_shadowed_builtins(self):
        analyzer = VariableNameAnalyzer()
        report = analyzer.analyze(SHADOW_CODE)
        assert "list" in report.shadowed_builtins
        assert "dict" in report.shadowed_builtins

    def test_shadowed_builtins_lower_score(self):
        analyzer = VariableNameAnalyzer()
        report = analyzer.analyze(SHADOW_CODE)
        assert report.score < 0.8


class TestConventionConsistency:
    def test_mixed_code_detects_violations(self):
        # Mix of snake_case function name param and camelCase vars in same file
        mixed = """\
def process_items(item_count, unit_price, totalCost):
    final_value = item_count * unit_price + totalCost
    return final_value
"""
        analyzer = VariableNameAnalyzer()
        report = analyzer.analyze(mixed)
        # snake_case dominates (item_count, unit_price, final_value vs totalCost)
        assert report.dominant_convention == "snake_case"
        assert "totalCost" in report.convention_violations

    def test_snake_convention_detection(self):
        analyzer = VariableNameAnalyzer()
        report = analyzer.analyze(CLEAN_CODE)
        assert report.dominant_convention == "snake_case"


class TestEdgeCases:
    def test_syntax_error_returns_empty_report(self):
        analyzer = VariableNameAnalyzer()
        report = analyzer.analyze(SYNTAX_ERROR_CODE)
        assert report.total_names == 0
        assert report.score == 1.0

    def test_empty_code_returns_perfect_score(self):
        analyzer = VariableNameAnalyzer()
        report = analyzer.analyze(EMPTY_CODE)
        assert report.score == 1.0

    def test_report_has_issues_property(self):
        analyzer = VariableNameAnalyzer()
        report = analyzer.analyze(SHADOW_CODE)
        issues = report.issues
        assert isinstance(issues, list)
        assert len(issues) > 0

    def test_score_clamped_to_zero_minimum(self):
        """Even extremely bad code should not produce negative score."""
        very_bad = "\n".join(f"    {c} = {i}" for i, c in enumerate("qwrtyuops"))
        very_bad = "def bad():\n" + very_bad
        analyzer = VariableNameAnalyzer()
        report = analyzer.analyze(very_bad)
        assert report.score >= 0.0
