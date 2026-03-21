"""Tests for OutputFormatValidator (features/output_format_validator.py)."""

from __future__ import annotations


from cola_coder.features.output_format_validator import (
    FEATURE_ENABLED,
    OutputFormatValidator,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Sample snippets
# ---------------------------------------------------------------------------

CLEAN_CODE = """\
def add(a: int, b: int) -> int:
    return a + b


def subtract(a: int, b: int) -> int:
    return a - b
"""

TRAILING_WHITESPACE = "def foo():  \n    pass\n"

MIXED_LINE_ENDINGS = "def foo():\r\n    pass\n    return 1\n"

LONG_LINE = "x = " + "a" * 110 + "\n"

MIXED_INDENT = "def foo():\n    x = 1\n\ty = 2\n"

NO_FINAL_NEWLINE = "def foo():\n    pass"

CRLF_ONLY = "def foo():\r\n    pass\r\n"

BOM_CODE = "\ufeffdef foo():\n    pass\n"

NULL_BYTES = "def foo():\n    pass\0\n"


class TestIsEnabled:
    def test_feature_enabled(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled_returns_true(self):
        assert is_enabled() is True


class TestCleanCode:
    def test_clean_code_is_valid(self):
        validator = OutputFormatValidator()
        report = validator.validate(CLEAN_CODE)
        assert report.is_valid

    def test_clean_code_score_one(self):
        validator = OutputFormatValidator()
        report = validator.validate(CLEAN_CODE)
        assert report.score == 1.0

    def test_clean_code_lf_endings(self):
        validator = OutputFormatValidator()
        report = validator.validate(CLEAN_CODE)
        assert report.line_ending == "lf"

    def test_clean_code_spaces_indent(self):
        validator = OutputFormatValidator()
        report = validator.validate(CLEAN_CODE)
        assert report.indentation == "spaces"


class TestTrailingWhitespace:
    def test_detects_trailing_whitespace(self):
        validator = OutputFormatValidator()
        report = validator.validate(TRAILING_WHITESPACE)
        assert not report.is_valid
        checks = [v.check for v in report.violations]
        assert "trailing_whitespace" in checks

    def test_trailing_whitespace_violation_has_line_number(self):
        validator = OutputFormatValidator()
        report = validator.validate(TRAILING_WHITESPACE)
        tw_violations = [v for v in report.violations if v.check == "trailing_whitespace"]
        assert tw_violations[0].line == 1


class TestLineEndings:
    def test_detects_mixed_line_endings(self):
        validator = OutputFormatValidator()
        report = validator.validate(MIXED_LINE_ENDINGS)
        checks = [v.check for v in report.violations]
        assert "line_endings" in checks

    def test_crlf_only_not_flagged_as_mixed(self):
        validator = OutputFormatValidator()
        report = validator.validate(CRLF_ONLY)
        assert report.line_ending == "crlf"
        checks = [v.check for v in report.violations]
        assert "line_endings" not in checks


class TestLineTooLong:
    def test_detects_long_line(self):
        validator = OutputFormatValidator(max_line_length=100)
        report = validator.validate(LONG_LINE)
        checks = [v.check for v in report.violations]
        assert "line_too_long" in checks

    def test_longest_line_tracked(self):
        validator = OutputFormatValidator()
        report = validator.validate(LONG_LINE)
        assert report.longest_line > 100

    def test_custom_max_line_length(self):
        validator = OutputFormatValidator(max_line_length=200)
        report = validator.validate(LONG_LINE)
        checks = [v.check for v in report.violations]
        # 114 chars is under 200
        assert "line_too_long" not in checks


class TestIndentation:
    def test_detects_mixed_indentation(self):
        validator = OutputFormatValidator()
        report = validator.validate(MIXED_INDENT)
        assert report.indentation == "mixed"
        checks = [v.check for v in report.violations]
        assert "indentation" in checks


class TestFinalNewline:
    def test_detects_missing_final_newline(self):
        validator = OutputFormatValidator()
        report = validator.validate(NO_FINAL_NEWLINE)
        checks = [v.check for v in report.violations]
        assert "final_newline" in checks


class TestBomAndNullBytes:
    def test_detects_bom(self):
        validator = OutputFormatValidator()
        report = validator.validate(BOM_CODE)
        checks = [v.check for v in report.violations]
        assert "bom" in checks

    def test_detects_null_bytes(self):
        validator = OutputFormatValidator()
        report = validator.validate(NULL_BYTES)
        checks = [v.check for v in report.violations]
        assert "null_bytes" in checks


class TestEdgeCases:
    def test_empty_string_is_valid(self):
        validator = OutputFormatValidator()
        report = validator.validate("")
        assert report.is_valid

    def test_summary_ok_for_clean(self):
        validator = OutputFormatValidator()
        report = validator.validate(CLEAN_CODE)
        assert "OK" in report.summary

    def test_summary_shows_violations(self):
        validator = OutputFormatValidator()
        report = validator.validate(TRAILING_WHITESPACE)
        assert "violation" in report.summary.lower()

    def test_score_decreases_with_violations(self):
        validator = OutputFormatValidator()
        clean = validator.validate(CLEAN_CODE)
        bad = validator.validate(TRAILING_WHITESPACE + LONG_LINE + MIXED_INDENT)
        assert bad.score < clean.score
