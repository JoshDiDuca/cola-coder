"""Test Generation Evaluator (improvement #69).

Evaluates the quality of model-generated (or human-written) test code:
  - assert coverage (are there meaningful assertions?)
  - edge case handling (None, empty, boundary values)
  - mock usage (appropriate isolation)
  - naming conventions (test_* functions, descriptive names)
  - test isolation (no shared mutable state, no network/file calls)

TypeScript analogy: like a quality plugin for Jest/Vitest — imagine
a linter that scores your test file on coverage patterns instead of
just syntax errors.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Feature toggle (project convention)
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if test generation evaluation is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_EDGE_CASE_PATTERNS = [
    re.compile(r"\bNone\b"),
    re.compile(r'""|\'\'\s*[,)]'),        # empty string
    re.compile(r"\[\s*\]"),               # empty list
    re.compile(r"\{\s*\}"),              # empty dict
    re.compile(r"\b0\b"),                # zero
    re.compile(r"\b-\d"),               # negative numbers
    re.compile(r"\bfloat\(\"inf\"\)", re.I),
    re.compile(r"\bmath\.inf\b"),
    re.compile(r"\boverflow\b", re.I),
    re.compile(r"\bboundary\b|\bedge\b", re.I),
]

_MOCK_PATTERNS = [
    re.compile(r"\bMock\b|\bMagicMock\b"),
    re.compile(r"\bpatch\b"),
    re.compile(r"mocker\b"),
    re.compile(r"monkeypatch\b"),
    re.compile(r"@pytest\.fixture"),
]

_ISOLATION_VIOLATION_PATTERNS = [
    re.compile(r"\bopen\s*\("),
    re.compile(r"\burllib\b|\brequests\.\b|\bhttpx\.\b"),
    re.compile(r"\bsubprocess\b"),
    re.compile(r"\bos\.environ\b"),
    re.compile(r"\bglobal\s+\w"),
]

_ASSERT_PATTERNS = [
    re.compile(r"\bassert\s"),
    re.compile(r"\.assert\w+\("),           # mock.assert_called, etc.
    re.compile(r"pytest\.raises"),
    re.compile(r"pytest\.approx"),
]

_GOOD_NAME_RE = re.compile(r"^test_[a-z][a-z0-9_]*$")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TestFunctionRecord:
    """Record for a single test function found in the source."""

    name: str
    line_no: int
    has_good_name: bool
    assert_count: int
    edge_case_count: int
    uses_mock: bool
    isolation_violation: bool
    docstring: Optional[str] = None


@dataclass
class TestQualityReport:
    """Quality report for a test file / generated test snippet."""

    total_tests: int = 0
    well_named_tests: int = 0
    tests_with_asserts: int = 0
    tests_with_edge_cases: int = 0
    tests_using_mocks: int = 0
    isolation_violations: int = 0

    test_records: List[TestFunctionRecord] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)

    naming_score: float = 0.0
    assert_score: float = 0.0
    edge_case_score: float = 0.0
    isolation_score: float = 0.0
    overall_score: float = 0.0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class TestGenEvaluator:
    """Evaluate quality of test code (generated or human-written)."""

    def __init__(
        self,
        require_docstrings: bool = False,
        mock_bonus: float = 0.1,
    ) -> None:
        self.require_docstrings = require_docstrings
        self.mock_bonus = mock_bonus

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, source: str) -> TestQualityReport:
        """Evaluate a test source file and return a quality report."""
        report = TestQualityReport()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            report.issues.append("Source has a syntax error — cannot parse")
            return report

        lines = source.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test"):
                    rec = self._analyse_test_function(node, lines)
                    report.test_records.append(rec)

        self._aggregate(report)
        self._compute_scores(report)
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _analyse_test_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        lines: List[str],
    ) -> TestFunctionRecord:
        start = node.lineno - 1
        end = (node.end_lineno or node.lineno)
        func_source = "\n".join(lines[start:end])

        has_good_name = bool(_GOOD_NAME_RE.match(node.name))
        assert_count = sum(1 for p in _ASSERT_PATTERNS if p.search(func_source))
        edge_count = sum(1 for p in _EDGE_CASE_PATTERNS if p.search(func_source))
        uses_mock = any(p.search(func_source) for p in _MOCK_PATTERNS)
        isolation_violation = any(
            p.search(func_source) for p in _ISOLATION_VIOLATION_PATTERNS
        )

        docstring: Optional[str] = None
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            docstring = node.body[0].value.value

        return TestFunctionRecord(
            name=node.name,
            line_no=node.lineno,
            has_good_name=has_good_name,
            assert_count=assert_count,
            edge_case_count=edge_count,
            uses_mock=uses_mock,
            isolation_violation=isolation_violation,
            docstring=docstring,
        )

    def _aggregate(self, report: TestQualityReport) -> None:
        report.total_tests = len(report.test_records)
        for rec in report.test_records:
            if rec.has_good_name:
                report.well_named_tests += 1
            else:
                report.issues.append(f"Test '{rec.name}' has a non-standard name")
            if rec.assert_count > 0:
                report.tests_with_asserts += 1
            else:
                report.issues.append(f"Test '{rec.name}' has no assertions")
            if rec.edge_case_count > 0:
                report.tests_with_edge_cases += 1
            if rec.uses_mock:
                report.tests_using_mocks += 1
            if rec.isolation_violation:
                report.isolation_violations += 1
                report.issues.append(
                    f"Test '{rec.name}' may violate isolation (network/file/env access)"
                )
            if self.require_docstrings and rec.docstring is None:
                report.issues.append(f"Test '{rec.name}' missing docstring")

    def _compute_scores(self, report: TestQualityReport) -> None:
        n = max(report.total_tests, 1)

        report.naming_score = report.well_named_tests / n
        report.assert_score = report.tests_with_asserts / n
        report.edge_case_score = min(report.tests_with_edge_cases / n, 1.0)
        isolation_violations_ratio = report.isolation_violations / n
        report.isolation_score = max(0.0, 1.0 - isolation_violations_ratio)

        mock_bonus = self.mock_bonus if report.tests_using_mocks > 0 else 0.0

        report.overall_score = min(
            1.0,
            0.25 * report.naming_score
            + 0.35 * report.assert_score
            + 0.20 * report.edge_case_score
            + 0.20 * report.isolation_score
            + mock_bonus,
        )

        if report.total_tests == 0:
            report.issues.append("No test functions found (functions must start with 'test')")
            report.overall_score = 0.0


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def evaluate_tests(source: str) -> TestQualityReport:
    """Evaluate test quality with default settings."""
    return TestGenEvaluator().evaluate(source)
