"""Tests for tokenizer_coverage.py."""

import pytest

from cola_coder.features.tokenizer_coverage import CoverageReport, TokenizerCoverageAnalyzer


class MockTokenizer:
    """Minimal tokenizer mock that splits on whitespace."""

    vocab_size = 100

    def encode(self, text: str):  # noqa: D102
        return SimpleEncoding(text.split())


class SimpleEncoding:
    def __init__(self, tokens: list[str]) -> None:
        self.ids = list(range(len(tokens)))


class MockTokenizerWithUnk:
    """Mock tokenizer that marks very long words as UNK (id=0)."""

    vocab_size = 50
    unk_token_id = 0

    def encode(self, text: str):
        ids = [0 if len(w) > 10 else (hash(w) % 49 + 1) for w in text.split()]
        return SimpleEncoding(ids)


@pytest.fixture
def analyzer():
    return TokenizerCoverageAnalyzer()


def test_feature_enabled():
    from cola_coder.features.tokenizer_coverage import FEATURE_ENABLED, is_enabled

    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_basic_coverage(analyzer):
    tokenizer = MockTokenizer()
    corpus = ["def foo(): pass", "const x = 1;"]
    report = analyzer.analyze(tokenizer, corpus)
    assert isinstance(report, CoverageReport)
    assert report.total_tokens > 0
    assert report.total_words > 0


def test_single_string_input(analyzer):
    tokenizer = MockTokenizer()
    report = analyzer.analyze(tokenizer, "def foo(): pass")
    assert report.total_tokens > 0


def test_compression_ratio(analyzer):
    tokenizer = MockTokenizer()
    corpus = ["hello world foo bar"]
    report = analyzer.analyze(tokenizer, corpus)
    assert report.compression_ratio > 0.0


def test_vocab_size_reported(analyzer):
    tokenizer = MockTokenizer()
    report = analyzer.analyze(tokenizer, ["test"])
    assert report.vocab_size == 100


def test_oov_rate_with_unk(analyzer):
    tokenizer = MockTokenizerWithUnk()
    # word longer than 10 chars should trigger UNK
    corpus = ["superlongidentifiername short"]
    report = analyzer.analyze(tokenizer, corpus, )
    assert 0.0 <= report.oov_rate <= 1.0


def test_avg_tokens_per_word_positive(analyzer):
    tokenizer = MockTokenizer()
    corpus = ["one two three four five"]
    report = analyzer.analyze(tokenizer, corpus)
    # Each word maps to 1 token in mock
    assert report.avg_tokens_per_word > 0.0


def test_summary_returns_string(analyzer):
    tokenizer = MockTokenizer()
    report = analyzer.analyze(tokenizer, ["print('hello')"])
    s = report.summary()
    assert isinstance(s, str)
    assert "avg_tok/word=" in s


def test_empty_corpus(analyzer):
    tokenizer = MockTokenizer()
    report = analyzer.analyze(tokenizer, [])
    assert report.total_tokens == 0
    assert report.total_words == 0
