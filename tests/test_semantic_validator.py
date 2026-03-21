"""Tests for features/semantic_validator.py — Feature 95.

All tests are CPU-only, no model weights, no I/O.
"""

from __future__ import annotations

import pytest

from cola_coder.features.semantic_validator import (
    FEATURE_ENABLED,
    CodeSemanticValidator,
    IssueSeverity,
    SemanticIssue,
    ValidationResult,
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
def val():
    return CodeSemanticValidator()


# ---------------------------------------------------------------------------
# Basic interface
# ---------------------------------------------------------------------------


def test_validate_returns_result(val):
    result = val.validate("x = 1\n")
    assert isinstance(result, ValidationResult)


def test_checks_run_populated(val):
    result = val.validate("x = 1\n")
    assert len(result.checks_run) > 0


def test_clean_code_is_valid(val):
    code = "def foo(a, b):\n    return a + b\n"
    result = val.validate(code)
    assert result.is_valid


def test_unknown_check_raises():
    with pytest.raises(ValueError, match="Unknown checks"):
        CodeSemanticValidator(checks=["nonexistent_check"])


# ---------------------------------------------------------------------------
# type_consistency check
# ---------------------------------------------------------------------------


def test_type_consistency_catches_int_string():
    v = CodeSemanticValidator(checks=["type_consistency"])
    code = 'x: int = "hello"\n'
    result = v.validate(code)
    checks = [i.check for i in result.issues]
    assert "type_consistency" in checks


def test_type_consistency_no_issue_on_correct():
    v = CodeSemanticValidator(checks=["type_consistency"])
    code = "x: int = 42\n"
    result = v.validate(code)
    assert result.error_count == 0


# ---------------------------------------------------------------------------
# unreachable_code check
# ---------------------------------------------------------------------------


def test_unreachable_code_detected():
    v = CodeSemanticValidator(checks=["unreachable_code"])
    code = "def foo():\n    return 1\n    x = 2\n"
    result = v.validate(code)
    checks = [i.check for i in result.issues]
    assert "unreachable_code" in checks


def test_unreachable_code_no_issue_on_clean():
    v = CodeSemanticValidator(checks=["unreachable_code"])
    code = "def foo():\n    x = 1\n    return x\n"
    result = v.validate(code)
    checks = [i.check for i in result.issues]
    assert "unreachable_code" not in checks


# ---------------------------------------------------------------------------
# unused_variables check
# ---------------------------------------------------------------------------


def test_unused_variable_flagged():
    v = CodeSemanticValidator(checks=["unused_variables"])
    code = "def foo():\n    unused_var = 42\n    return 0\n"
    result = v.validate(code)
    messages = " ".join(i.message for i in result.issues)
    assert "unused_var" in messages


def test_used_variable_not_flagged():
    v = CodeSemanticValidator(checks=["unused_variables"])
    code = "def foo():\n    result = 42\n    return result\n"
    result = v.validate(code)
    # 'result' is used, so no unused variable issue
    unused_issues = [
        i for i in result.issues
        if i.check == "unused_variables" and "result" in i.message
    ]
    assert len(unused_issues) == 0


# ---------------------------------------------------------------------------
# missing_return check
# ---------------------------------------------------------------------------


def test_missing_return_flagged():
    v = CodeSemanticValidator(checks=["missing_return"])
    code = (
        "def compute(x):\n"
        "    if x > 0:\n"
        "        return x\n"
        "    y = x * 2\n"
    )
    result = v.validate(code)
    checks = [i.check for i in result.issues]
    assert "missing_return" in checks


def test_missing_return_no_issue_unconditional():
    v = CodeSemanticValidator(checks=["missing_return"])
    code = "def foo(x):\n    return x + 1\n"
    result = v.validate(code)
    missing = [i for i in result.issues if i.check == "missing_return"]
    assert len(missing) == 0


# ---------------------------------------------------------------------------
# ValidationResult helpers
# ---------------------------------------------------------------------------


def test_error_count():
    r = ValidationResult(
        issues=[
            SemanticIssue("x", IssueSeverity.ERROR, 1, "bad"),
            SemanticIssue("y", IssueSeverity.WARNING, 2, "warn"),
        ]
    )
    assert r.error_count == 1
    assert r.warning_count == 1


def test_summary_keys(val):
    result = val.validate("x = 1\n")
    s = result.summary()
    assert "is_valid" in s
    assert "errors" in s
    assert "warnings" in s


# ---------------------------------------------------------------------------
# SemanticIssue __str__
# ---------------------------------------------------------------------------


def test_issue_str_contains_check():
    issue = SemanticIssue(
        check="type_consistency",
        severity=IssueSeverity.WARNING,
        line=5,
        message="mismatch",
    )
    s = str(issue)
    assert "type_consistency" in s
    assert "5" in s


# ---------------------------------------------------------------------------
# Subset checks
# ---------------------------------------------------------------------------


def test_subset_checks_only_runs_selected():
    v = CodeSemanticValidator(checks=["type_consistency"])
    result = v.validate("x: int = 1\n")
    assert result.checks_run == ["type_consistency"]


def test_enabled_checks_property():
    v = CodeSemanticValidator(checks=["type_consistency", "unreachable_code"])
    assert "type_consistency" in v.enabled_checks
    assert "unused_variables" not in v.enabled_checks
