"""Tests for features/perplexity_analyzer.py.

Uses a tiny mock model and tokenizer — no GPU, no real weights.
"""

from __future__ import annotations

import math

import pytest

from cola_coder.features.perplexity_analyzer import (
    FEATURE_ENABLED,
    PerplexityAnalyzer,
    PerplexityReport,
    TokenPerplexity,
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
# Mock tokenizer
# ---------------------------------------------------------------------------


class MockTokenizer:
    """Tiny tokenizer: each character is its own token."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 256 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) if 32 <= i < 127 else "?" for i in ids)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer():
    return MockTokenizer()


@pytest.fixture
def analyzer():
    return PerplexityAnalyzer(top_k=3)


# ---------------------------------------------------------------------------
# analyze_from_log_probs (no model needed)
# ---------------------------------------------------------------------------


def test_from_log_probs_basic(analyzer, tokenizer):
    text = "hello"
    token_ids = tokenizer.encode(text)  # [104, 101, 108, 108, 111]
    # Uniform log-probs at -2.0
    log_probs = [-2.0] * (len(token_ids) - 1)
    report = analyzer.analyze_from_log_probs(text, token_ids, log_probs, tokenizer)
    assert isinstance(report, PerplexityReport)


def test_from_log_probs_mean(analyzer, tokenizer):
    text = "abc"
    token_ids = tokenizer.encode(text)
    log_probs = [-1.0, -3.0]
    report = analyzer.analyze_from_log_probs(text, token_ids, log_probs, tokenizer)
    expected_ppls = [math.exp(1.0), math.exp(3.0)]
    assert abs(report.mean_perplexity - sum(expected_ppls) / 2) < 1e-6


def test_from_log_probs_min_max(analyzer, tokenizer):
    text = "xy"
    token_ids = tokenizer.encode(text)
    log_probs = [-0.5]
    report = analyzer.analyze_from_log_probs(text, token_ids, log_probs, tokenizer)
    assert abs(report.min_perplexity - math.exp(0.5)) < 1e-6
    assert abs(report.max_perplexity - math.exp(0.5)) < 1e-6


def test_from_log_probs_top_k(analyzer, tokenizer):
    text = "abcde"
    token_ids = tokenizer.encode(text)
    # Varied log-probs
    log_probs = [-0.1, -5.0, -1.0, -3.0]
    report = analyzer.analyze_from_log_probs(text, token_ids, log_probs, tokenizer)
    assert len(report.most_confident) <= 3
    assert len(report.least_confident) <= 3


def test_from_log_probs_most_confident_ordering(analyzer, tokenizer):
    text = "abc"
    token_ids = tokenizer.encode(text)
    log_probs = [-0.1, -5.0]  # first token is more confident
    report = analyzer.analyze_from_log_probs(text, token_ids, log_probs, tokenizer)
    if len(report.most_confident) >= 2:
        assert report.most_confident[0].perplexity <= report.most_confident[1].perplexity


def test_from_log_probs_empty(analyzer, tokenizer):
    text = "a"
    token_ids = tokenizer.encode(text)
    report = analyzer.analyze_from_log_probs(text, token_ids, [], tokenizer)
    assert report.tokens == []
    assert report.mean_perplexity == 0.0


def test_token_perplexity_fields(analyzer, tokenizer):
    text = "hi"
    token_ids = tokenizer.encode(text)
    log_probs = [-2.0]
    report = analyzer.analyze_from_log_probs(text, token_ids, log_probs, tokenizer)
    assert len(report.tokens) == 1
    tok = report.tokens[0]
    assert isinstance(tok, TokenPerplexity)
    assert tok.token_id == token_ids[1]
    assert abs(tok.log_prob - (-2.0)) < 1e-6
    assert abs(tok.perplexity - math.exp(2.0)) < 1e-6


def test_line_perplexity_built(analyzer, tokenizer):
    text = "line1\nline2\n"
    token_ids = tokenizer.encode(text)
    log_probs = [-1.0] * (len(token_ids) - 1)
    report = analyzer.analyze_from_log_probs(text, token_ids, log_probs, tokenizer)
    assert len(report.lines) >= 1


def test_summary_contains_stats(analyzer, tokenizer):
    text = "def f():\n    pass\n"
    token_ids = tokenizer.encode(text)
    log_probs = [-1.5] * (len(token_ids) - 1)
    report = analyzer.analyze_from_log_probs(text, token_ids, log_probs, tokenizer)
    summary = report.summary()
    assert "Mean" in summary
    assert "Median" in summary


def test_median_single_token(analyzer, tokenizer):
    text = "ab"
    token_ids = tokenizer.encode(text)
    log_probs = [-2.0]
    report = analyzer.analyze_from_log_probs(text, token_ids, log_probs, tokenizer)
    assert abs(report.median_perplexity - math.exp(2.0)) < 1e-6
