"""Tests for output_diversity.py (feature 47)."""

from __future__ import annotations



from cola_coder.features.output_diversity import (
    FEATURE_ENABLED,
    OutputDiversityScorer,
    _distinct_n,
    _ngrams,
    _self_bleu,
    _token_entropy,
    is_enabled,
)


def test_feature_enabled():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_ngrams_unigrams():
    tokens = ["a", "b", "c"]
    result = _ngrams(tokens, 1)
    assert result == [("a",), ("b",), ("c",)]


def test_ngrams_bigrams():
    tokens = ["a", "b", "c"]
    result = _ngrams(tokens, 2)
    assert result == [("a", "b"), ("b", "c")]


def test_ngrams_empty():
    assert _ngrams([], 2) == []


def test_distinct_n_all_unique():
    tokenized = [["a", "b", "c"], ["d", "e", "f"]]
    d1 = _distinct_n(tokenized, 1)
    assert abs(d1 - 1.0) < 1e-9


def test_distinct_n_all_same():
    tokenized = [["x", "x", "x"], ["x", "x"]]
    d1 = _distinct_n(tokenized, 1)
    assert abs(d1 - 1 / 5) < 1e-9  # 1 unique / 5 total


def test_token_entropy_uniform():
    """Uniform distribution → maximum entropy."""
    tokenized = [["a"], ["b"], ["c"], ["d"]]
    H = _token_entropy(tokenized)
    assert abs(H - 2.0) < 1e-9  # log2(4) = 2


def test_token_entropy_constant():
    tokenized = [["a", "a", "a"]]
    H = _token_entropy(tokenized)
    assert abs(H - 0.0) < 1e-9


def test_self_bleu_identical_outputs_high():
    identical = [["def", "foo", "return", "1"]] * 5
    sb = _self_bleu(identical)
    assert sb > 0.8


def test_self_bleu_diverse_outputs_low():
    diverse = [
        ["def", "foo", "return", "1"],
        ["class", "Bar", "init", "self"],
        ["import", "os", "path", "join"],
        ["for", "i", "in", "range", "10"],
    ]
    sb = _self_bleu(diverse)
    assert sb < 0.5


# ---------------------------------------------------------------------------
# DiversityScorer
# ---------------------------------------------------------------------------


def test_score_empty_outputs():
    scorer = OutputDiversityScorer()
    report = scorer.score([])
    assert report.num_outputs == 0
    assert report.diversity_score() == 0.0


def test_score_single_output():
    scorer = OutputDiversityScorer()
    report = scorer.score(["hello world"])
    assert report.num_outputs == 1
    assert report.distinct_1 >= 0.0


def test_score_diverse_outputs_high_score():
    scorer = OutputDiversityScorer()
    outputs = [f"function_{i}() {{ return {i}; }}" for i in range(20)]
    report = scorer.score(outputs)
    assert report.diversity_score() > 0.3


def test_score_repetitive_outputs_low_diversity():
    scorer = OutputDiversityScorer(collapse_threshold=0.1)
    outputs = ["the same text repeated"] * 20
    report = scorer.score(outputs)
    # distinct-1 should be low for repetitive outputs
    assert report.distinct_1 < 0.5


def test_mode_collapse_detection():
    scorer = OutputDiversityScorer(collapse_threshold=0.5)
    # All identical outputs → distinct-1 = 1/N which is very low
    outputs = ["x x x x x"] * 10
    report = scorer.score(outputs)
    assert report.is_collapsed


def test_summary_contains_key_metrics():
    scorer = OutputDiversityScorer()
    report = scorer.score(["hello world", "foo bar baz", "abc def ghi"])
    s = report.summary()
    assert "dist-1=" in s
    assert "self-BLEU=" in s
    assert "entropy=" in s
    assert "score=" in s


def test_custom_tokenizer():
    scorer = OutputDiversityScorer()
    char_tok = list  # splits into individual characters
    outputs = ["abc", "def", "ghi"]
    report = scorer.score(outputs, tokenize_fn=char_tok)
    assert report.total_tokens == 9  # 3 chars each


def test_diversity_score_range():
    scorer = OutputDiversityScorer()
    outputs = ["foo bar baz " * i for i in range(1, 10)]
    report = scorer.score(outputs)
    score = report.diversity_score()
    assert 0.0 <= score <= 1.0
