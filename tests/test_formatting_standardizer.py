"""Tests for features/formatting_standardizer.py — Feature 91.

All tests are CPU-only, no model weights, no I/O.
"""

from __future__ import annotations

import pytest

from cola_coder.features.formatting_standardizer import (
    FEATURE_ENABLED,
    FormattingStandardizer,
    FormattingStats,
    StandardizeResult,
    is_enabled,
    standardize_code,
)


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


def test_feature_enabled():
    assert FEATURE_ENABLED is True


def test_is_enabled():
    assert is_enabled() is True


# ---------------------------------------------------------------------------
# Basic interface
# ---------------------------------------------------------------------------


@pytest.fixture
def std():
    return FormattingStandardizer()


def test_returns_standardize_result(std):
    result = std.standardize("x = 1\n", "python")
    assert isinstance(result, StandardizeResult)


def test_result_has_code(std):
    result = std.standardize("x = 1\n", "python")
    assert isinstance(result.code, str)


def test_result_language_preserved(std):
    result = std.standardize("x = 1\n", "typescript")
    assert result.language == "typescript"


# ---------------------------------------------------------------------------
# CRLF normalization
# ---------------------------------------------------------------------------


def test_crlf_to_lf(std):
    code = "a = 1\r\nb = 2\r\n"
    result = std.standardize(code, "python")
    assert "\r" not in result.code


def test_crlf_count_tracked(std):
    code = "a = 1\r\nb = 2\r\n"
    result = std.standardize(code, "python")
    assert result.stats.crlf_fixes == 2


def test_bare_cr_normalized(std):
    code = "a\rb\rc\r"
    result = std.standardize(code, "python")
    assert "\r" not in result.code


# ---------------------------------------------------------------------------
# Tab → space conversion
# ---------------------------------------------------------------------------


def test_tabs_replaced_python(std):
    code = "\tdef foo():\n\t\tpass\n"
    result = std.standardize(code, "python")
    assert "\t" not in result.code
    assert result.code.startswith("    def")


def test_tabs_replaced_typescript(std):
    code = "\tfunction foo() {\n\t\treturn 1;\n\t}\n"
    result = std.standardize(code, "typescript")
    assert "\t" not in result.code
    # TypeScript uses 2-space indent
    assert result.code.startswith("  function")


def test_tab_fix_count(std):
    code = "\ta = 1\n\tb = 2\n"
    result = std.standardize(code, "python")
    assert result.stats.tab_fixes == 2


# ---------------------------------------------------------------------------
# Trailing whitespace
# ---------------------------------------------------------------------------


def test_trailing_ws_removed(std):
    code = "a = 1   \nb = 2  \n"
    result = std.standardize(code, "python")
    for line in result.code.splitlines():
        assert line == line.rstrip()


def test_trailing_ws_count(std):
    code = "a = 1   \nb = 2  \nc = 3\n"
    result = std.standardize(code, "python")
    assert result.stats.trailing_ws_fixes == 2


# ---------------------------------------------------------------------------
# Final newline
# ---------------------------------------------------------------------------


def test_missing_newline_added(std):
    code = "x = 1"
    result = std.standardize(code, "python")
    assert result.code.endswith("\n")


def test_missing_newline_tracked(std):
    result = std.standardize("x = 1", "python")
    assert result.stats.missing_final_newline_fixes == 1


def test_existing_newline_not_doubled(std):
    code = "x = 1\n"
    result = std.standardize(code, "python")
    assert result.code == "x = 1\n"


# ---------------------------------------------------------------------------
# changed flag
# ---------------------------------------------------------------------------


def test_changed_false_on_clean_code(std):
    code = "x = 1\n"
    result = std.standardize(code, "python")
    assert result.changed is False


def test_changed_true_when_tabs(std):
    result = std.standardize("\tx = 1\n", "python")
    assert result.changed is True


# ---------------------------------------------------------------------------
# Cumulative stats
# ---------------------------------------------------------------------------


def test_cumulative_stats_accumulate(std):
    std.standardize("\tx = 1\n", "python")
    std.standardize("\ty = 2\n", "python")
    assert std.cumulative_stats.files_processed == 2
    assert std.cumulative_stats.tab_fixes >= 2


def test_reset_stats(std):
    std.standardize("\tx = 1\n", "python")
    std.reset_stats()
    assert std.cumulative_stats.files_processed == 0


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------


def test_batch_returns_list(std):
    items = [("x = 1\n", "python"), ("let x = 1;\n", "typescript")]
    results = std.standardize_batch(items)
    assert len(results) == 2
    assert all(isinstance(r, StandardizeResult) for r in results)


# ---------------------------------------------------------------------------
# FormattingStats helpers
# ---------------------------------------------------------------------------


def test_stats_merge():
    a = FormattingStats(files_processed=1, tab_fixes=3)
    b = FormattingStats(files_processed=2, tab_fixes=1)
    c = a.merge(b)
    assert c.files_processed == 3
    assert c.tab_fixes == 4


def test_stats_as_dict():
    s = FormattingStats(files_processed=5, crlf_fixes=2)
    d = s.as_dict()
    assert d["files_processed"] == 5
    assert d["crlf_fixes"] == 2


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def test_module_level_standardize():
    result = standardize_code("\tx = 1\r\n", "python")
    assert isinstance(result, StandardizeResult)
    assert "\t" not in result.code
    assert "\r" not in result.code
