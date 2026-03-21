"""Tests for features/code_normalizer.py.

All tests are CPU-only, no model weights.
"""

from __future__ import annotations

import pytest

from cola_coder.features.code_normalizer import (
    FEATURE_ENABLED,
    CodeNormalizer,
    NormalizeResult,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


def test_feature_enabled():
    assert FEATURE_ENABLED is True


def test_is_enabled():
    assert is_enabled() is True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normalizer():
    return CodeNormalizer()


# ---------------------------------------------------------------------------
# General interface
# ---------------------------------------------------------------------------


def test_returns_normalize_result(normalizer):
    result = normalizer.normalize("x = 1\n", "python")
    assert isinstance(result, NormalizeResult)


def test_result_has_code(normalizer):
    result = normalizer.normalize("x = 1\n", "python")
    assert isinstance(result.code, str)


def test_result_language_set(normalizer):
    result = normalizer.normalize("x = 1\n", "python")
    assert result.language == "python"


def test_unsupported_language_raises(normalizer):
    with pytest.raises(ValueError, match="Unsupported language"):
        normalizer.normalize("var x = 1", "ruby")


def test_javascript_treated_as_typescript(normalizer):
    result = normalizer.normalize("var x = 1", "javascript")
    assert result.language == "typescript"


def test_case_insensitive_language(normalizer):
    result = normalizer.normalize("x = 1\n", "Python")
    assert result.language == "python"


# ---------------------------------------------------------------------------
# Python normaliser
# ---------------------------------------------------------------------------


def test_python_tabs_expanded(normalizer):
    code = "def f():\n\treturn 1\n"
    result = normalizer.normalize(code, "python")
    assert "\t" not in result.code
    assert "    return 1" in result.code


def test_python_trailing_whitespace_stripped(normalizer):
    code = "x = 1   \ny = 2  \n"
    result = normalizer.normalize(code, "python")
    for line in result.code.splitlines():
        assert line == line.rstrip()


def test_python_no_change_clean_code(normalizer):
    code = "def hello():\n    return 'world'\n"
    result = normalizer.normalize(code, "python")
    assert result.code == code


def test_python_collapses_excess_blank_lines(normalizer):
    code = "a = 1\n\n\n\nb = 2\n"
    result = normalizer.normalize(code, "python")
    # Should have at most 2 consecutive blank lines
    assert "\n\n\n\n" not in result.code


def test_python_adds_trailing_newline(normalizer):
    code = "x = 1"
    result = normalizer.normalize(code, "python")
    assert result.code.endswith("\n")


def test_python_changed_flag_on_tabs(normalizer):
    code = "def f():\n\tpass\n"
    result = normalizer.normalize(code, "python")
    assert result.changed


def test_python_no_change_flag_on_clean(normalizer):
    code = "x = 1\n"
    result = normalizer.normalize(code, "python")
    assert not result.changed


# ---------------------------------------------------------------------------
# TypeScript normaliser
# ---------------------------------------------------------------------------


def test_typescript_tabs_to_2_spaces(normalizer):
    code = "function f() {\n\treturn 1;\n}\n"
    result = normalizer.normalize(code, "typescript")
    assert "\t" not in result.code
    assert "  return 1;" in result.code


def test_typescript_trailing_whitespace(normalizer):
    code = "const x = 1;   \n"
    result = normalizer.normalize(code, "typescript")
    assert result.code.strip() == "const x = 1;"


def test_typescript_brace_style_krs(normalizer):
    code = "function hello()\n{\n  return 1;\n}\n"
    result = normalizer.normalize(code, "typescript")
    assert "hello() {" in result.code


def test_typescript_adds_trailing_newline(normalizer):
    code = "const x = 1;"
    result = normalizer.normalize(code, "typescript")
    assert result.code.endswith("\n")


def test_typescript_collapses_blank_lines(normalizer):
    code = "const a = 1;\n\n\n\nconst b = 2;\n"
    result = normalizer.normalize(code, "typescript")
    assert "\n\n\n" not in result.code
