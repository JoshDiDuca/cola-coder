"""Tests for TypeAnnotationScorer (features/type_annotation_scorer.py)."""

from __future__ import annotations

import pytest

from cola_coder.features.type_annotation_scorer import (
    FEATURE_ENABLED,
    TypeAnnotationReport,
    is_enabled,
    score_annotations,
)

# ---------------------------------------------------------------------------
# Snippets
# ---------------------------------------------------------------------------

FULLY_TYPED = """\
def add(a: int, b: int) -> int:
    return a + b

def greet(name: str) -> str:
    return f"Hello {name}"
"""

UNTYPED = """\
def add(a, b):
    return a + b

def greet(name):
    return f"Hello {name}"
"""

PARTIAL = """\
def add(a: int, b) -> int:
    return a + b
"""

USES_ANY = """\
from typing import Any

def process(data: Any) -> Any:
    return data
"""

BARE_GENERICS = """\
from typing import List, Dict

def take_list(items: List) -> Dict:
    return {}
"""

ASYNC_TYPED = """\
async def fetch(url: str) -> bytes:
    return b""
"""

CLASS_METHODS = """\
class Foo:
    def __init__(self, x: int) -> None:
        self.x = x

    def get(self) -> int:
        return self.x
"""


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


class TestIsEnabled:
    def test_constant(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled(self):
        assert is_enabled() is True


# ---------------------------------------------------------------------------
# Basic scoring
# ---------------------------------------------------------------------------


class TestBasicScoring:
    def test_returns_report(self):
        r = score_annotations(FULLY_TYPED)
        assert isinstance(r, TypeAnnotationReport)

    def test_fully_typed_high_score(self):
        r = score_annotations(FULLY_TYPED)
        assert r.overall_score >= 0.9

    def test_untyped_low_score(self):
        r = score_annotations(UNTYPED)
        assert r.overall_score < 0.5

    def test_partial_between(self):
        r = score_annotations(PARTIAL)
        full = score_annotations(FULLY_TYPED)
        untyped = score_annotations(UNTYPED)
        assert untyped.overall_score <= r.overall_score <= full.overall_score + 0.1


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


class TestCounting:
    def test_total_functions(self):
        r = score_annotations(FULLY_TYPED)
        assert r.total_functions == 2

    def test_annotated_functions_fully_typed(self):
        r = score_annotations(FULLY_TYPED)
        assert r.annotated_functions == 2

    def test_unannotated_functions(self):
        r = score_annotations(UNTYPED)
        assert r.unannotated_functions == 2

    def test_missing_return_count(self):
        r = score_annotations(UNTYPED)
        assert r.missing_return_count == 2

    def test_total_params(self):
        r = score_annotations(FULLY_TYPED)
        # add(a, b) + greet(name) = 3 params
        assert r.total_params == 3

    def test_annotated_params(self):
        r = score_annotations(FULLY_TYPED)
        assert r.annotated_params == 3


# ---------------------------------------------------------------------------
# Quality issues
# ---------------------------------------------------------------------------


class TestQualityIssues:
    def test_any_usage_detected(self):
        r = score_annotations(USES_ANY)
        assert r.any_usage_count >= 1

    def test_any_lowers_quality_score(self):
        r_any = score_annotations(USES_ANY)
        r_typed = score_annotations(FULLY_TYPED)
        assert r_any.quality_score < r_typed.quality_score

    def test_bare_generic_detected(self):
        r = score_annotations(BARE_GENERICS)
        assert r.incomplete_generic_count >= 1

    def test_issues_list_populated_for_untyped(self):
        r = score_annotations(UNTYPED)
        assert len(r.issues) > 0
        kinds = {i.kind for i in r.issues}
        assert "untyped_param" in kinds or "missing_return" in kinds


# ---------------------------------------------------------------------------
# Special cases
# ---------------------------------------------------------------------------


class TestSpecialCases:
    def test_async_function(self):
        r = score_annotations(ASYNC_TYPED)
        assert r.annotated_functions == 1

    def test_self_cls_skipped(self):
        r = score_annotations(CLASS_METHODS)
        # self should not count as untyped param
        assert r.unannotated_functions == 0

    def test_empty_source(self):
        r = score_annotations("")
        assert r.total_functions == 0
        assert r.overall_score == pytest.approx(1.0)

    def test_syntax_error_returns_empty_report(self):
        r = score_annotations("def foo(\n    pass\n")
        assert r.total_functions == 0
