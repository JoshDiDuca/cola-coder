"""Tests for ErrorRecoveryAnalyzer (features/error_recovery.py)."""

from __future__ import annotations


from cola_coder.features.error_recovery import (
    FEATURE_ENABLED,
    ErrorRecoveryAnalyzer,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Sample code snippets
# ---------------------------------------------------------------------------

GOOD_CODE = """\
def read_file(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Cannot open {path}") from exc


def parse_int(value: str) -> int:
    if not isinstance(value, str):
        raise TypeError("value must be a str")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Cannot parse '{value}' as int") from exc
"""

BARE_EXCEPT_CODE = """\
def unsafe():
    try:
        risky()
    except:
        pass
"""

BROAD_EXCEPT_CODE = """\
def broad():
    try:
        risky()
    except Exception:
        pass
"""

NO_TRY_CODE = """\
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""

WITH_FINALLY_CODE = """\
def write_data(path, data):
    fh = None
    try:
        fh = open(path, 'w')
        fh.write(data)
    finally:
        if fh:
            fh.close()
"""

RAISE_NO_MSG_CODE = """\
def strict(x):
    if x < 0:
        raise ValueError()
"""

SYNTAX_ERROR_CODE = "def bad(:\n    pass"


class TestIsEnabled:
    def test_feature_enabled(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled_returns_true(self):
        assert is_enabled() is True


class TestGoodCode:
    def test_good_code_score(self):
        analyzer = ErrorRecoveryAnalyzer()
        report = analyzer.analyze(GOOD_CODE)
        assert report.score >= 0.4

    def test_good_code_has_try_coverage(self):
        analyzer = ErrorRecoveryAnalyzer()
        report = analyzer.analyze(GOOD_CODE)
        assert report.functions_with_try >= 2

    def test_good_code_has_context_manager(self):
        analyzer = ErrorRecoveryAnalyzer()
        report = analyzer.analyze(GOOD_CODE)
        assert report.has_finally_or_context_manager is True

    def test_good_code_no_bare_except(self):
        analyzer = ErrorRecoveryAnalyzer()
        report = analyzer.analyze(GOOD_CODE)
        assert report.bare_except_count == 0


class TestBareExcept:
    def test_detects_bare_except(self):
        analyzer = ErrorRecoveryAnalyzer()
        report = analyzer.analyze(BARE_EXCEPT_CODE)
        assert report.bare_except_count == 1

    def test_bare_except_lowers_score(self):
        analyzer = ErrorRecoveryAnalyzer()
        good = analyzer.analyze(GOOD_CODE)
        bare = analyzer.analyze(BARE_EXCEPT_CODE)
        assert bare.score < good.score

    def test_bare_except_in_issues(self):
        analyzer = ErrorRecoveryAnalyzer()
        report = analyzer.analyze(BARE_EXCEPT_CODE)
        assert any("bare" in issue.lower() for issue in report.issues)


class TestBroadExcept:
    def test_detects_broad_except(self):
        analyzer = ErrorRecoveryAnalyzer()
        report = analyzer.analyze(BROAD_EXCEPT_CODE)
        assert report.broad_except_count == 1
        assert report.bare_except_count == 0


class TestNoTry:
    def test_no_try_low_coverage(self):
        analyzer = ErrorRecoveryAnalyzer()
        report = analyzer.analyze(NO_TRY_CODE)
        assert report.try_coverage == 0.0

    def test_no_try_issues_reported(self):
        analyzer = ErrorRecoveryAnalyzer()
        report = analyzer.analyze(NO_TRY_CODE)
        assert any("coverage" in i.lower() for i in report.issues)


class TestFinally:
    def test_detects_finally(self):
        analyzer = ErrorRecoveryAnalyzer()
        report = analyzer.analyze(WITH_FINALLY_CODE)
        assert report.has_finally_or_context_manager is True


class TestRaiseWithoutMessage:
    def test_detects_raise_no_msg(self):
        analyzer = ErrorRecoveryAnalyzer()
        report = analyzer.analyze(RAISE_NO_MSG_CODE)
        assert report.raises_without_message >= 1


class TestEdgeCases:
    def test_syntax_error_returns_report(self):
        analyzer = ErrorRecoveryAnalyzer()
        report = analyzer.analyze(SYNTAX_ERROR_CODE)
        assert report.total_functions == 0
        assert report.score == 0.5

    def test_try_coverage_property(self):
        analyzer = ErrorRecoveryAnalyzer()
        report = analyzer.analyze(GOOD_CODE)
        assert 0.0 <= report.try_coverage <= 1.0

    def test_score_bounds(self):
        analyzer = ErrorRecoveryAnalyzer()
        for code in [GOOD_CODE, BARE_EXCEPT_CODE, NO_TRY_CODE, RAISE_NO_MSG_CODE]:
            report = analyzer.analyze(code)
            assert 0.0 <= report.score <= 1.0
