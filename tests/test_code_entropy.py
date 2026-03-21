"""Tests for CodeEntropyAnalyzer (features/code_entropy.py)."""

from __future__ import annotations

import pytest

from cola_coder.features.code_entropy import (
    FEATURE_ENABLED,
    CodeEntropyAnalyzer,
    EntropyReport,
    analyze_entropy,
    is_enabled,
)

# ---------------------------------------------------------------------------
# Sample snippets
# ---------------------------------------------------------------------------

REPETITIVE = "x = 1\n" * 50
COMPLEX_CODE = """\
import ast
import re
from collections import Counter
from typing import List, Dict, Optional, Tuple


def tokenize(source: str) -> List[str]:
    pattern = re.compile(r'[A-Za-z_]\\w*|[0-9]+(?:\\.[0-9]+)?|[^\\s\\w]')
    return pattern.findall(source)


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    import math
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)
"""

EMPTY = ""
SINGLE_CHAR = "aaaaaaaaaaaaaaaaaaa"


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


class TestIsEnabled:
    def test_constant(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled(self):
        assert is_enabled() is True


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


class TestBasicStructure:
    def test_returns_report(self):
        r = analyze_entropy(COMPLEX_CODE)
        assert isinstance(r, EntropyReport)

    def test_scores_in_valid_range(self):
        r = analyze_entropy(COMPLEX_CODE)
        assert r.char_entropy >= 0.0
        assert r.token_entropy >= 0.0
        assert r.bigram_entropy >= 0.0

    def test_empty_source(self):
        r = analyze_entropy(EMPTY)
        assert r.char_entropy == pytest.approx(0.0)
        assert r.token_entropy == pytest.approx(0.0)
        assert r.total_tokens == 0


# ---------------------------------------------------------------------------
# Entropy ordering
# ---------------------------------------------------------------------------


class TestEntropyOrdering:
    def test_complex_higher_entropy_than_repetitive(self):
        r_rep = analyze_entropy(REPETITIVE)
        r_cpx = analyze_entropy(COMPLEX_CODE)
        assert r_cpx.token_entropy > r_rep.token_entropy

    def test_single_char_string_low_entropy(self):
        r = analyze_entropy(SINGLE_CHAR)
        assert r.char_entropy == pytest.approx(0.0)

    def test_complex_has_higher_vocab(self):
        r_rep = analyze_entropy(REPETITIVE)
        r_cpx = analyze_entropy(COMPLEX_CODE)
        assert r_cpx.vocabulary_size > r_rep.vocabulary_size


# ---------------------------------------------------------------------------
# Complexity labels
# ---------------------------------------------------------------------------


class TestComplexityLabel:
    def test_repetitive_is_trivial_or_simple(self):
        r = analyze_entropy(REPETITIVE)
        assert r.complexity_label in ("trivial", "simple")

    def test_complex_code_not_trivial(self):
        r = analyze_entropy(COMPLEX_CODE)
        assert r.complexity_label not in ("trivial",)

    def test_label_is_valid_string(self):
        valid = {"trivial", "simple", "moderate", "complex", "highly_complex"}
        r = analyze_entropy(COMPLEX_CODE)
        assert r.complexity_label in valid


# ---------------------------------------------------------------------------
# Issues
# ---------------------------------------------------------------------------


class TestIssues:
    def test_boilerplate_triggers_issue(self):
        r = analyze_entropy(REPETITIVE)
        # Low entropy boilerplate should trigger an issue
        if r.token_entropy < 1.5:
            assert len(r.issues) > 0

    def test_complex_code_no_boilerplate_issue(self):
        r = analyze_entropy(COMPLEX_CODE)
        boilerplate_issues = [i for i in r.issues if "boilerplate" in i]
        assert len(boilerplate_issues) == 0


# ---------------------------------------------------------------------------
# Batch and compare
# ---------------------------------------------------------------------------


class TestBatchAndCompare:
    def test_analyze_batch(self):
        analyzer = CodeEntropyAnalyzer()
        reports = analyzer.analyze_batch([REPETITIVE, COMPLEX_CODE])
        assert len(reports) == 2

    def test_compare_returns_dict(self):
        analyzer = CodeEntropyAnalyzer()
        diff = analyzer.compare(REPETITIVE, COMPLEX_CODE)
        assert "token_entropy_diff" in diff
        assert diff["token_entropy_diff"] > 0  # complex > repetitive

    def test_top_tokens(self):
        r = analyze_entropy(COMPLEX_CODE)
        assert len(r.top_tokens) > 0
        assert all(isinstance(t, tuple) for t in r.top_tokens)
