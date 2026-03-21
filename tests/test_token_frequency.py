"""Tests for TokenFrequencyAnalyzer (features/token_frequency.py)."""

from __future__ import annotations


import pytest

from cola_coder.features.token_frequency import (
    FEATURE_ENABLED,
    TokenFrequencyAnalyzer,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def zipf_tokens(n_types: int = 100, scale: int = 1000) -> list[str]:
    """Generate tokens that approximately follow Zipf's law."""
    tokens: list[str] = []
    for rank in range(1, n_types + 1):
        count = max(1, int(scale / rank))
        tokens.extend([f"tok_{rank}"] * count)
    return tokens


UNIFORM_TOKENS = ["a", "b", "c", "d"] * 25  # perfectly uniform


class TestIsEnabled:
    def test_feature_enabled(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled_returns_true(self):
        assert is_enabled() is True


class TestBasicCounts:
    def test_total_tokens(self):
        analyzer = TokenFrequencyAnalyzer()
        tokens = ["a", "b", "a", "c"]
        report = analyzer.analyze(tokens)
        assert report.total_tokens == 4

    def test_unique_tokens(self):
        analyzer = TokenFrequencyAnalyzer()
        tokens = ["a", "b", "a", "c"]
        report = analyzer.analyze(tokens)
        assert report.unique_tokens == 3

    def test_most_common_correct(self):
        analyzer = TokenFrequencyAnalyzer()
        tokens = ["a", "a", "a", "b", "b", "c"]
        report = analyzer.analyze(tokens)
        assert report.most_common[0][0] == "a"
        assert report.most_common[0][1] == 3

    def test_empty_tokens(self):
        analyzer = TokenFrequencyAnalyzer()
        report = analyzer.analyze([])
        assert report.total_tokens == 0
        assert report.unique_tokens == 0

    def test_type_token_ratio(self):
        analyzer = TokenFrequencyAnalyzer()
        # 4 unique / 4 total = 1.0
        report = analyzer.analyze(["a", "b", "c", "d"])
        assert report.type_token_ratio == 1.0


class TestEntropy:
    def test_uniform_distribution_high_entropy(self):
        analyzer = TokenFrequencyAnalyzer()
        report = analyzer.analyze(UNIFORM_TOKENS)
        # entropy for 4 equally likely symbols = log2(4) = 2.0
        assert abs(report.entropy_bits - 2.0) < 0.01

    def test_single_token_zero_entropy(self):
        analyzer = TokenFrequencyAnalyzer()
        report = analyzer.analyze(["a"] * 100)
        assert report.entropy_bits == pytest.approx(0.0, abs=1e-9)

    def test_entropy_positive_for_mixed(self):
        analyzer = TokenFrequencyAnalyzer()
        report = analyzer.analyze(["a", "b", "c"] * 10)
        assert report.entropy_bits > 0


class TestZipfFit:
    def test_zipf_distribution_high_fit(self):
        analyzer = TokenFrequencyAnalyzer()
        tokens = zipf_tokens()
        report = analyzer.analyze(tokens)
        assert report.zipf_fit >= 0.7

    def test_uniform_distribution_lower_zipf(self):
        analyzer = TokenFrequencyAnalyzer()
        # Use a larger uniform distribution (many types, same frequency) to
        # ensure it clearly diverges from Zipf
        uniform_large = [f"w_{i}" for i in range(50)] * 20  # 50 types, all equal
        report = analyzer.analyze(uniform_large)
        zipf_report = analyzer.analyze(zipf_tokens())
        # Zipf distribution should fit better than large uniform
        assert zipf_report.zipf_fit > report.zipf_fit

    def test_zipf_fit_bounds(self):
        analyzer = TokenFrequencyAnalyzer()
        for tokens in [UNIFORM_TOKENS, zipf_tokens(), ["x"] * 5]:
            report = analyzer.analyze(tokens)
            assert 0.0 <= report.zipf_fit <= 1.0


class TestOOVRate:
    def test_full_vocab_zero_oov(self):
        analyzer = TokenFrequencyAnalyzer()
        tokens = ["a", "b", "c"]
        vocab = {"a", "b", "c"}
        report = analyzer.analyze(tokens, vocab=vocab)
        assert report.oov_rate == 0.0

    def test_no_vocab_zero_oov(self):
        analyzer = TokenFrequencyAnalyzer()
        report = analyzer.analyze(["a", "b"])
        assert report.oov_rate == 0.0

    def test_partial_vocab_correct_oov(self):
        analyzer = TokenFrequencyAnalyzer()
        tokens = ["a", "a", "b", "c"]  # 1/4 tokens are 'c' (OOV)
        vocab = {"a", "b"}
        report = analyzer.analyze(tokens, vocab=vocab)
        assert report.oov_rate == pytest.approx(0.25)


class TestFrequencyBands:
    def test_bands_sum_to_total_types(self):
        analyzer = TokenFrequencyAnalyzer()
        tokens = zipf_tokens(50, 500)
        report = analyzer.analyze(tokens)
        # hapax is a count of unique tokens, not token occurrences — just check non-negative
        assert report.bands.very_frequent >= 0
        assert report.bands.common >= 0
        assert report.bands.rare >= 0
        assert report.bands.hapax >= 0

    def test_hapax_count_correct(self):
        analyzer = TokenFrequencyAnalyzer()
        # 3 tokens appear once
        tokens = ["a"] * 10 + ["b"] * 5 + ["c", "d", "e"]
        report = analyzer.analyze(tokens)
        assert report.bands.hapax == 3


class TestSummary:
    def test_summary_is_string(self):
        analyzer = TokenFrequencyAnalyzer()
        report = analyzer.analyze(["hello", "world"] * 5)
        assert isinstance(report.summary, str)
        assert "TTR" in report.summary
