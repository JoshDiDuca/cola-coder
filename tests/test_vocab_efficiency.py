"""Tests for vocab_efficiency.py (feature 45)."""

from __future__ import annotations


from cola_coder.features.vocab_efficiency import (
    FEATURE_ENABLED,
    VocabEfficiencyAnalyzer,
    is_enabled,
    make_char_tokenizer,
)


def test_feature_enabled():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vocab(*tokens: str) -> dict:
    return {t: i for i, t in enumerate(tokens)}


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------


def test_basic_analysis():
    vocab = _make_vocab("hello", "world", "foo", "bar")
    analyzer = VocabEfficiencyAnalyzer()
    report = analyzer.analyze(vocab, ["hello world"])
    assert report.vocab_size == 4
    assert report.total_tokens_seen == 2


def test_unused_tokens_detected():
    vocab = _make_vocab("hello", "world", "unused_token", "another_unused")
    analyzer = VocabEfficiencyAnalyzer()
    report = analyzer.analyze(vocab, ["hello world"] * 10)
    assert "unused_token" in report.unused_tokens
    assert "another_unused" in report.unused_tokens
    assert report.num_unused == 2


def test_rare_tokens_detected():
    vocab = _make_vocab("hello", "world", "rare")
    # "hello" appears 1000 times, "rare" appears once
    samples = ["hello"] * 1000 + ["rare"]
    analyzer = VocabEfficiencyAnalyzer(rare_threshold=0.001)
    report = analyzer.analyze(vocab, samples)
    # "rare" appears 1 time out of 1001 total = 0.001 fraction → rare
    assert "rare" in report.rare_tokens


def test_no_rare_tokens_when_balanced():
    vocab = _make_vocab("a", "b", "c")
    samples = ["a b c"] * 100
    analyzer = VocabEfficiencyAnalyzer(rare_threshold=0.0001)
    report = analyzer.analyze(vocab, samples)
    assert report.num_rare == 0


def test_fragmentation_ratio_whitespace():
    """Whitespace tokenizer: ratio should be 1.0 (one token per word)."""
    vocab = _make_vocab("hello", "world")
    analyzer = VocabEfficiencyAnalyzer()
    report = analyzer.analyze(vocab, ["hello world", "hello world"])
    assert abs(report.fragmentation_ratio - 1.0) < 1e-9


def test_fragmentation_ratio_char_tokenizer():
    """Char tokenizer on 5-char words → ratio ≈ 5."""
    vocab = {chr(c): i for i, c in enumerate(range(ord("a"), ord("z") + 1))}
    char_tok = make_char_tokenizer()
    analyzer = VocabEfficiencyAnalyzer(tokenize_fn=char_tok)
    # "hello" is 5 chars, so 5 tokens per 1 word
    report = analyzer.analyze(vocab, ["hello world"] * 10)
    assert report.fragmentation_ratio >= 4.5  # hello=5, world=5 → avg 5


def test_single_token_coverage():
    """Each word in samples should map to one whitespace token = 100% coverage."""
    vocab = _make_vocab("hello", "world", "foo")
    analyzer = VocabEfficiencyAnalyzer()
    report = analyzer.analyze(vocab, ["hello world foo"])
    assert report.single_token_coverage == 1.0


def test_low_single_token_coverage_char_tok():
    """Char tokenizer → multi-token words → low coverage."""
    vocab = {chr(c): i for i, c in enumerate(range(ord("a"), ord("z") + 1))}
    char_tok = make_char_tokenizer()
    analyzer = VocabEfficiencyAnalyzer(tokenize_fn=char_tok)
    report = analyzer.analyze(vocab, ["hello world foo bar"])
    # All words are >1 char, so single-token coverage = 0
    assert report.single_token_coverage == 0.0


def test_efficiency_score_range():
    vocab = _make_vocab("hello", "world")
    analyzer = VocabEfficiencyAnalyzer()
    report = analyzer.analyze(vocab, ["hello world"] * 50)
    score = report.efficiency_score()
    assert 0.0 <= score <= 1.0


def test_efficiency_score_high_when_good():
    """Good vocab: no unused, no rare, low fragmentation, high coverage."""
    vocab = _make_vocab("hello", "world", "foo")
    analyzer = VocabEfficiencyAnalyzer()
    report = analyzer.analyze(vocab, ["hello world foo"] * 100)
    assert report.efficiency_score() >= 0.6


def test_summary_format():
    vocab = _make_vocab("a", "b")
    analyzer = VocabEfficiencyAnalyzer()
    report = analyzer.analyze(vocab, ["a b"] * 5)
    s = report.summary()
    assert "vocab=" in s
    assert "frag=" in s
    assert "score=" in s


def test_empty_samples():
    vocab = _make_vocab("hello", "world")
    analyzer = VocabEfficiencyAnalyzer()
    report = analyzer.analyze(vocab, [])
    assert report.total_tokens_seen == 0
    assert report.num_unused == 2


def test_empty_vocab():
    analyzer = VocabEfficiencyAnalyzer()
    report = analyzer.analyze({}, ["hello world"])
    assert report.vocab_size == 0
    assert report.efficiency_score() == 0.0
